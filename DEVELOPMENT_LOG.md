# 协同仿真平台开发日志（AI 移交文档）

> 记录从环境搭建到 validation 实验的全部工作，供后续 AI 接手。

---

## 一、项目概述

本项目实现 **大语言模型（LLM）推理/训练负载** 的端到端协同仿真：

```
NeuSight (算子时间预测) → GPUSim (计算仿真) ↔ FNCS (时间同步) ↔ NS-3 (网络仿真)
```

### 仓库结构（`/Users/taco/Desktop/cosim/`）

| 目录/文件 | 说明 |
|----------|------|
| `fncs/` | FNCS broker + client 库（C++, autotools），已编译到 `fncs/builds/local/` |
| `GPUsim/` | 计算仿真器（Java + CloudSim），JNI 调用 libfncs |
| `ns3/` | NS-3 网络仿真器，FNCS 模块在 `src/fncs/` |
| `NeuSight/` | MLP 性能预测器（PyTorch），预测各 GPU 上的 kernel 延迟 |
| `atlahs/` | ATLAHS 网络仿真工具链（LogGOPSim + htsim），提供 ground truth |
| `neusight2cosim.py` | NeuSight CSV → GPUSim 输入转换器 |
| `goal2cosim.py` | ATLAHS GOAL trace → GPUSim 输入转换器 |
| `run_case_study.py` | Case Study 实验自动化脚本 |
| `case_study_results/` | 实验结果和报告 |

---

## 二、环境搭建（macOS arm64）

### 已安装的依赖

```bash
# Homebrew 包
brew install zeromq czmq cmake autoconf automake libtool openjdk gengetopt re2c
# 国内镜像加速: HOMEBREW_BOTTLE_DOMAIN=https://mirrors.ustc.edu.cn/homebrew-bottles brew install ...

# Python 包
pip3 install --user torch pandas numpy tqdm scipy transformers==4.38.1
```

### 编译的组件

**1. FNCS broker + libfncs**
```bash
cd fncs
./configure --prefix=$PWD/builds/local CXXFLAGS="-I/opt/homebrew/include" LDFLAGS="-L/opt/homebrew/lib"
make -j$(sysctl -n hw.ncpu) && make install
# 产出: fncs/builds/local/bin/fncs_broker, fncs/builds/local/lib/libfncs.1.dylib
```

**2. NS-3 (fncs-example)**
```bash
cd ns3/build
cmake .. -DNS3_EXAMPLES=ON -DNS3_WARNINGS_AS_ERRORS=OFF -DNS3_LOG=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --target fncs-example -j$(sysctl -n hw.ncpu)
# 产出: ns3/build/src/fncs/examples/ns3-dev-fncs-example
```

NS-3 FNCS 模块的 CMakeLists 修改了 FNCS 路径（`src/fncs/CMakeLists.txt`）：
- `include_directories` → `/Users/taco/Desktop/cosim/fncs/builds/local/include`
- `FNCS_LIB_PATH` → `/Users/taco/Desktop/cosim/fncs/builds/local/lib/libfncs.dylib`
- 新增 `${libinternet}` 链接（为 `GetRoutableAddress` 所需）

**3. GPUSim (Java)**
```bash
cd GPUsim
export PATH="/opt/homebrew/opt/openjdk/bin:$PATH"
find src -name "*.java" > /tmp/sources.txt
javac -J--add-opens=jdk.compiler/com.sun.tools.javac.processing=ALL-UNNAMED \
  ... (10个 --add-opens) ...
  -d out/production/gpuworkflowsim -cp "out/production/gpuworkflowsim:jars/*" @/tmp/sources.txt
```

JNI 库路径修正（`install_name_tool`）：
```bash
install_name_tool -change /Users/davidt/FNCS_install/lib/libfncs.1.dylib \
  /Users/taco/Desktop/cosim/fncs/builds/local/lib/libfncs.1.dylib \
  GPUsim/lib/libJNIfncs.dylib
```

**4. ATLAHS LogGOPSim**
```bash
cd atlahs/sim/LogGOPSim
gengetopt < simulator.ggo
g++ -O3 -g -c LogGOPSim.cpp cmdline.c
g++ LogGOPSim.o cmdline.o -o LogGOPSim
# 注: txt2bin 有编译错误（C 兼容性），但不需要（ATLAHS 提供 .bin 文件）
```

**5. NeuSight**
```bash
# 无需安装，通过 PYTHONPATH 使用
PYTHONPATH=/Users/taco/Desktop/cosim/NeuSight python3 scripts/pred.py ...
```
macOS 无 CUDA 的适配：
- `neusight/Dataset/collect.py`: `torch.cuda.Event` 创建加 `cuda.is_available()` 保护
- `neusight/Tracing/analysis.py`: 同上
- `neusight/Prediction/aggregator.py`: 添加 `"llama" in model_name` 和 `"mistral" in model_name` 到 GPT 分支
- `neusight/Prediction/predictor.py`: `n_layer` 读取支持多种 key 名

---

## 三、核心代码修改

### 3.1 通信延迟真实建模（最重要的修改）

**问题**：原始 CommEngine 在任务完成后立即本地消费所有网络包（`while (!job.ifReceiveAll()) { job.receivePacket(); }`），NS-3 的网络延迟为 0%。

**修改的文件**：

#### `GPUsim/src/comm/CommEngine.java`
- `processCloudletReturn()`: 有 packets 的任务放入 `waittingJobs` 等待 NS-3 回包，不再本地短路
- `processOtherEvent(RECEIVE_EVENT)`: 收到 NS-3 回包后 `receivePacket()`，全部收齐则 `deliverReadyChildren()`

#### `GPUsim/src/comm/Event.java`
- `toEvent()`: 修复解析 NS-3 回包格式。NS-3 发回 `"task_name\0\0...=sendend"`（含 null 填充），需要 `.replace("\0", "").trim()` 后按 `=` 分割。

#### `GPUsim/src/cloudsim/core/CloudSim.java`
- **FNCS 消息发布顺序**：在 `timeRequest()` 之前调用 `truePublish()`（确保 broker 先看到消息再授时给 NS-3）。同时保留处理后的 `truePublish()`（首次发布需要）。
- **FNCS 事件接收**：`getEvents()` + `getValue()` 的 split("/") 在 `timeRequest` 返回后立即处理，不等本地事件到期。

#### `GPUsim/src/api/info/JobInfo.java`
- 只为有 `src_host` 和 `dst_host` 的 children 创建 Packet（同主机依赖不创建 packet）。

#### `GPUsim/src/comm/Api.java`
- 新增 `getValues()` 方法（调用 `JNIfncs.get_values()`），支持 FNCS `list=true` 模式。

### 3.2 NS-3 FNCS 模块修改

#### `ns3/src/fncs/model/fncs-application.cc`
- **HandleRead 缓冲**：不再每次 HandleRead 都 `fncs::publish("finish", ...)`。改为缓冲到 `GetFinishBuffer()`，在 `FlushFinishBuffer()` 中合并为一条 `/` 分隔的消息一次性发布（解决同一时间步多个 finish 覆盖问题）。
- **null 字节清理**：HandleRead 中 `value` 包含 NS-3 packet buffer 的 null 填充，发布前 strip。
- **GetRoutableAddress**：新增方法，从发送方节点查找与目标节点直连的子网 IP（解决 p2p 多接口路由问题）。Send() 中用 `to->GetRoutableAddress(GetNode())` 替代 `to->GetLocalInet()`。
- 新增 `#include "ns3/ipv4.h"` 支持 Ipv4 对象访问。

#### `ns3/src/fncs/model/fncs-simulator-impl.cc`
- 在 `time_request` 之前调用 `FlushFinishBuffer()`（确保 finish 消息在授时前发送）。

#### `ns3/src/fncs/model/fncs-application.h`
- 新增 `GetRoutableAddress(Ptr<Node> fromNode)` 声明。
- 新增 `#include "ns3/node.h"`。

#### `ns3/src/applications/helper/fncs-application-helper.cc`
- socket 绑定改为 `Ipv4Address::GetAny()` (0.0.0.0)，接收所有接口的包（p2p 多接口场景必需）。

#### `ns3/src/fncs/CMakeLists.txt`
- 新增 `${libinternet}` 链接。

### 3.3 编译错误修复

- `fncs-application.cc`: `~f_name.empty()` → `!f_name.empty()`（bitwise NOT → logical NOT）
- `fncs-application.cc`: `char_to_uint8_t` 未使用函数加 `__attribute__((unused))`
- NS-3 编译加 `-DNS3_WARNINGS_AS_ERRORS=OFF`

---

## 四、已解决的关键问题

### 4.1 NS-3 stop time 过小（最后发现的 bug）

**症状**：4-rank 仿真 7/14 任务完成，rank3 事件永远不到达。
**根因**：`topology.yaml` 的 `sim: stop: 600s`，而 rank3 的 FNCS 时间 = 614s。NS-3 在 600s 退出。
**修复**：`stop: 86400s`。

### 4.2 FNCS publish-after-time_request 时序

**症状**：GPUSim 的 `truePublish` 在 `timeRequest` 之后调用，NS-3 已被 broker 授予超前时间。
**修复**：在 `timeRequest` 之前也调用一次 `truePublish`。两处都需要保留——前者处理跨 tick 消息，后者处理首次发布。

### 4.3 FNCS 同一 topic 同一时间步多值覆盖

**症状**：多个 HandleRead 在同一 NS-3 时间步发布 `fncs::publish("finish", ...)`，只有最后一个值保留。
**修复**：NS-3 端缓冲 finish 消息，合并为 `/` 分隔的单条消息一次性发布。GPUSim 端 `getValue().split("/")` 解析。

### 4.4 p2p 多接口 UDP socket 绑定

**症状**：4-node p2p ring 中 host3 收不到来自 host4 的包（socket 绑定在 interface 1 IP，包从 interface 2 到达）。
**修复**：FncsApplicationHelper 绑定 `0.0.0.0:1234`；Send 时用 `GetRoutableAddress` 查找正确子网 IP。

### 4.5 CSMA ARP 延迟过大

**症状**：CSMA switch 拓扑中首包 ARP 解析需要 ~8ms NS-3 时间 = 数百万次 FNCS 同步，wall-clock 分钟级。
**规避**：使用 p2p 链路（无 ARP）。2-host 用 p2p 直连，4-host 用 p2p 线性链。

### 4.6 NeuSight Llama/MoE 模型适配

**症状**：`aggregator.py` 的 `replicate_layer()` 不识别 Llama 模型名，层边界标记 `add_15` 在 SiLU 激活中不唯一。
**部分修复**：添加 `"llama" in model_name` 分支。完整修复需调整层边界检测逻辑。

---

## 五、实验现状

### 5.1 Case Study（二、仿真扩展能力）— 全部完成

| 实验 | 配置 | 状态 |
|------|------|------|
| 1. GPU 规格变化 | GPT-3 27层 TP=2, 6种GPU | ✅ 完成 |
| 2. 并行方式变化 | TP=2 vs PP=2 | ✅ 完成 |
| 3. 拓扑变化 | star vs direct_p2p | ✅ 完成 |
| 4. 带宽变化 | 400/200/100/50 Gbps | ✅ 完成 |

结果在 `case_study_results/实验报告.md`。

### 5.2 Validation（一、仿真准确度）— 进行中

**ATLAHS trace 数据**：已下载到 `atlahs/data/ai/`

| Trace | 下载 | LGS 时间 | Co-sim |
|-------|------|---------|--------|
| Llama7B 16GPU | ✅ | 2.120s | **1.129s** (14/14 tasks) |
| Llama7B 128GPU | ✅ | SEGV (macOS) | 未跑 |
| Llama70B 256GPU | ✅ | SEGV (macOS) | 未跑 |
| Mistral8x7B 64GPU | ✅ | 3.523s | 未跑 |
| MoE8x13B 128GPU | ✅ | SEGV (macOS) | 未跑 |
| MoE8x70B 256GPU | ✅ | SEGV (macOS) | 未跑 |

> 128/256 GPU 的 LogGOPSim 在 macOS 上 SEGFAULT (exit 139)，需在 Linux Docker 环境运行。

**Llama7B 16GPU 验证结果**：
- Co-sim: 1.129s, ATLAHS: 2.120s, 误差 46.8%
- 差异来源：`goal2cosim.py` 将 1.1M 细粒度操作聚合为 14 个粗粒度任务（4 rank × 3-4 training step phases），AllReduce 通信量大幅压缩（原始 75GB → 10KB/phase）

---

## 六、启动协同仿真的标准流程

```bash
# 1. 生成输入（NeuSight CSV 路径 或 ATLAHS GOAL 路径）
python3 neusight2cosim.py --csv ... --tp 2 --pp 1 --num-layers 27 --hidden-dim 2560 --output-dir /tmp/test
# 或
python3 goal2cosim.py --goal ... --bandwidth 200 --output-dir /tmp/test

# 2. 生成 ZPL 文件
cat > /tmp/test/gpusim_fncs.zpl << 'EOF'
name = gpusim
time_delta = 1ns
broker = tcp://localhost:5570
values
    finish
        topic = ns3/finish
        default = ""
        type = string
        list = false
EOF
cat > /tmp/test/ns3_fncs.zpl << 'EOF'
name = ns3
time_delta = 1ns
broker = tcp://localhost:5570
values
    cloudsim/transfer
        topic = gpusim/cloudsim/transfer
        default = ""
        type = string
        list = false
    cloudsim/end
        topic = gpusim/cloudsim/end
EOF

# 3. 启动（顺序: broker → NS-3 → GPUSim）
fncs/builds/local/bin/fncs_broker 2 &
sleep 1

FNCS_CONFIG_FILE=/tmp/test/ns3_fncs.zpl FNCS_BROKER=tcp://localhost:5570 \
FNCS_NAME=ns3 FNCS_TIME_DELTA=1ns FNCS_FATAL=yes \
  ns3/build/src/fncs/examples/ns3-dev-fncs-example --topo=/tmp/test/topology.yaml &
sleep 1

export PATH="/opt/homebrew/opt/openjdk/bin:$PATH"
FNCS_CONFIG_FILE=/tmp/test/gpusim_fncs.zpl FNCS_BROKER=tcp://localhost:5570 \
FNCS_NAME=gpusim FNCS_TIME_DELTA=1ns FNCS_FATAL=yes \
java --enable-native-access=ALL-UNNAMED -Djava.library.path=GPUsim/lib \
  -cp "GPUsim/out/production/gpuworkflowsim:GPUsim/jars/*" \
  backend.SimEngine /tmp/test/output /tmp/test/hosts.json /tmp/test/jobs.json /tmp/test/faults.json \
  -1 false false 0

# 4. 结果在 /tmp/test/output/jobRun.xml
```

### 关键注意事项

1. **必须用 `export` 或 `VAR=val command` 设置环境变量**（macOS SIP 会 strip `DYLD_LIBRARY_PATH`）
2. **Java PATH**：每次新 shell 都要 `export PATH="/opt/homebrew/opt/openjdk/bin:$PATH"`
3. **NS-3 stop time** 必须大于最大 FNCS 时间（`topology.yaml` 中 `sim: stop`）
4. **p2p 拓扑**：2-host 用 p2p 直连；多 host 用 p2p 线性链（不要回环，不要 CSMA）
5. **编译 NS-3 后**需确认 dylib 更新：`rm -f ns3/build/lib/libns3-dev-fncs.dylib && cmake --build .`
6. **编译 GPUSim**：必须 `rm -f out/.../CloudSim.class` 后再 javac（否则可能用缓存）
7. **AllReduce 建模**：不创建跨 rank 的 children 依赖（避免死锁），只创建 packet（虚拟 child 名 `__ar_...`）

---

## 七、待完成的工作

### 高优先级

1. **Validation 实验补全**：
   - 在 Linux Docker 环境运行 128/256 GPU 的 LogGOPSim
   - `goal2cosim.py` 需要更精细的聚合策略（当前 1.1M ops → 14 tasks 损失太多信息）
   - 4-rank 协同仿真已通，可对 Llama7B 和 Mistral 跑完整验证

2. **NeuSight Llama/MoE 适配**：
   - `aggregator.py` 的 `replicate_layer()` 层边界检测需修改（SiLU 产生多个 `add` 操作使 `add_15` 不唯一）
   - 修复后可用 NeuSight 直接预测 Llama/Mistral/MoE 各 GPU 的 kernel 时间

### 中优先级

3. **通信量还原**：当前 AllReduce 包大小压缩为 10KB/phase。需要按 GOAL trace 中的实际通信量生成 packets（原始 ~75GB/rank），或实现分块传输。

4. **实验报告最终版**：将 validation 数据填入 `case_study_results/实验报告.md` 的表格。

### 低优先级

5. **NS-3 RDMA 模型**：当前 p2p 链路的协议栈开销 ~1.4ms >> 真实 RDMA ~1μs。可用 NS-3 的 RDMA 模块改善。
6. **多请求推理场景**：当前为单次前向传播，未涉及 Continuous Batching / KV Cache。

---

## 八、第二项工作：纵向与横向性能优化（2026-09-24）

本阶段将性能机制统一为两个优化轴：纵向事件聚合减少 Federate 内部事件，横向
rank-visible 切分与 Active-dependency 减少 Federate 之间不必要的时间同步。

实现方面，工作流优化器改为一次遍历批量收缩所有互不重叠的认证区域；编排器只在计算在途、
网络在途或终止状态变化时发布依赖 epoch；broker 将依赖图编译为整数索引和传递闭包，并在
调度轮之间复用状态与缓冲区。有限 idle lookahead 继续作为动态任务派发的因果安全边界。

完整模型补充实验使用三个 HPC 应用（LULESH-64、HPCG-8、ICON-8）和一个大模型训练负载
（Grok-314B-256）。纵向单开加速分别为 2.234×、1.305×、1.290× 和 2.081×。
Grok-256 rank-visible 二因素主消融结果为：纵向
单开 2.121×，横向单开 2.153×，双开 2.615×。在三 Federate 完整模型中，纵向单开
2.081×，横向单开 1.005×；差异表明横向收益需要将逐-rank 时间线暴露给 broker。

设计文档位于 `docs/WORK_2_DESIGN.md`，实验索引位于 `experiments/README.md`。原始 GOAL、
生成输入、逐次运行日志和本机构建产物不进入 Git。
