# 单卡常驻 MoE 切分预测

基于 ComposeDES `main` 的 `a999a9f` 和其固定的 FNCS `c71ce2c`。编排器使用
[graph_optimizer.py](../../fncs/orchestrator/graph_optimizer.py) 的 DAG、关键路径证书和
闭合区域收缩，以及 [workflow_orchestrator.py](../../fncs/orchestrator/workflow_orchestrator.py)
的 dispatch/completion 状态机。

## 输入和图

- 输入为单 H20 的 Qwen3.5-35B-A3B、全显存 GGUF、并发 2/4/8 的 13,440 条独立
  gate/up 与 down 投影计时，以及 28 个完整留出 forward 的路由负载。
  `loads` 按 kernel 的专家首次出现次序排列，每项是该专家接收的 token 数。
- 每个投影分别按量化格式、阶段 CTA 数、线程数和行块宽度校准服务时间。
  服务模型使用活跃权重字节数、CTA 最大工作量和 tile 波次。候选的完整 forward
  计时只在最后的 `compare` 步骤读取。
- 并行候选将每个投影 tile 转成事件。在同一逻辑 CTA 内，tile 按 GPU 实现的
  strided 顺序串行。每个专家的 gate tiles 都完成后发布就绪事件，才释放该专家
  的 down tiles。两投影有独立 CTA 配额；整段串行与切分模型使用同一校准和
  各自的 launch、屏障计费。层与层串行，逐层模拟时延之和形成路由专家路径的
  forward 分数。
- 用训练拟合残差的 P90 为每类投影服务时间生成经验上下界，并由 ComposeDES
  沿 DAG 传播。它刻画固定图中服务时间变化的影响，不能覆盖 GPU 运行时的模型
  偏差，也不是统计置信区间。一次预测区间完全分离但仍选错的真实例子见
  `comparison.csv` 的 B2 CTA 配额。
- 固定预算模型用 CTA 顺序和依赖直接编码容量约束。完整模型运行中的 L2/HBM
  争用、动态 CTA 角色迁移、FP8 WGMMA 服务数据、跨节点链路以及 GPU 内在线选择
  尚未接入。当前预测值只估计路由专家路径，用来比较候选排名。

## 用已有 H20 数据验证

同一批数据此前用于验证手写事件模型。这是 ComposeDES 的等价移植和回归验证。
两个方法在七个不同的分块候选上都选中实测最快的整段融合。只看就绪流水候选的 CTA 配额：

| 并发 | 实测最优 gate/down CTA | 加和模型选择 | ComposeDES 事件图选择 | 加和误选损失 | 事件图误选损失 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 26/52 | 39/39 | 39/39 | 1.80% | 1.80% |
| 4 | 26/52 | 39/39 | 26/52 | 3.45% | 0.00% |
| 8 | 26/52 | 39/39 | 26/52 | 9.13% | 0.00% |

原版 llama.cpp 的显存常驻性能仍领先，作为外部基线参与计时，当前未进入七个
候选的同算术排名。预测模型不包含原版投影的独立服务参数。
[原始汇总](comparison.csv)

### 本地和 FNCS 回放

抽取三个实际输入的单层图，本地事件回放与 ComposeDES 扫描所得关键路径逐纳秒一致。
对 launch→setup 使用 ComposeDES 的精确收缩后，各图少一个节点，关键路径和回放一致。
同样三张图在实际编译的 FNCS broker 与原 orchestrator 中再次跑完，采用本实验的
固定服务时间 worker。Broker 完成时刻比无消息递交开销的关键路径长 22–44 ns；
这个差值包含 1 ns 模拟时间推进和 broker 消息交接。此回放使用一个计算参与者，单卡
工作流内没有跨主机通信。 [回放汇总](federation-summary.csv)

## 换机后复算

在仓库根目录执行，只需 Python 3.10+ 和 NumPy。`replay-data/` 保存候选清单、已聚合
的路由与服务时间、冻结预测、逐轮 CSV 和 FP8 对照输出。

```sh
git submodule update --init fncs
python3 -m venv deps/predictor-venv
. deps/predictor-venv/bin/activate
p=experiments/megakernel_predictor
python -m pip install -r "$p/requirements.txt"
mkdir -p "$p/runs"
python "$p/predict.py" predict --inputs "$p/replay-data" --output "$p/runs/forecast.json" --check-dag
python "$p/predict.py" compare --prediction "$p/runs/forecast.json" \
  --measurements "$p/replay-data/rounds.csv" --output "$p/runs/comparison.csv"
python "$p/b2.py" "$p/replay-data" --routes routes-unseen-b2.csv \
  --calibration contention-v4.json --forecast ../runs/b2-unseen-forecast.rebuilt.json --upstream
python "$p/h20/summary.py" "$p/replay-data/unseen" "$p/replay-data/fp8-forecast-eager.json" \
  "$p/runs/b2-pool-eager.csv" --fp8 "$p/replay-data/fp8-heldout-eager"
python "$p/h20/verify_fp8.py" "$p/replay-data/fp8-heldout-eager"
```

如需使用 native broker，编译 libfncs 并将其安装路径传给：

```sh
python "$p/federation.py" run --workflow "$p/runs/example-b2.json" \
  --broker /path/to/fncs_broker --library /path/to/libfncs.so \
  --output "$p/runs/federation-b2"
```

[H20 实验结果](h20/RESULTS.md) · [CUDA 源码和复跑步骤](h20/runtime/README.md)
