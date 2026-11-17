# 完整 Qwen3-8B 推理中的 CUDA Graph 调度实验

## 1. 当前状态

**严格数值校验未通过。根据用户后续“只关心性能，继续”的要求，另行增加性能模式，完整模型计时及其限制见 [性能实验报告](PERFORMANCE.md)。下面保留最初中止的原因和独立诊断证据，不将性能模式的完成等同于数值正确性通过。**

本目录将此前的合成 DAG 实验扩展到完整 Qwen3-8B。2026-11-17 在新 A100 PCIe 40GB 实例上加载了全部 **8,190,735,360 个参数和 36 个 Transformer 层**，执行了原生 Transformers 的完整前向和短序列生成。随后，改写执行图的严格 logits 校验失败。诊断进一步发现原生模型重复执行也不稳定，且只读的 GPU 权重出现变化。不能将此次运行视为有效的性能实验，更不能沿用合成 DAG 的加速数字作为完整模型结论。

源码、校验器和计划中的独立验证协议已保存；完整执行器在健康 GPU 上仍需重新通过数值校验。本目录不是已验证可用的推理框架集成。

## 2. 显存错误的独立证据

`memory_check.cu` 不使用 PyTorch、FlashInfer、模型或 CUDA Graph。它申请 80 个 256 MiB 缓冲区，共 20 GiB，写入由数组下标和缓冲区编号确定的 32 位模式，然后只读核对。每个进程读四轮，间隔一秒。后续版本还将首个错误位置直接拷回 CPU，核对读出的数值。

| 独立进程 | 四轮检查中的错误字数 |
|---|---|
| 首次检查 | 2、2、2、2 |
| 增加 CPU 核对后的首次检查 | 0、0、0、0 |
| 重复 1 | 0、0、0、0 |
| 重复 2 | 415、427、427、427 |
| 重复 3 | 360、370、370、370 |
| 重复 4 | 390、404、404、404 |
| 重复 5 | 363、375、375、375 |

重复 2 的一个具体错误为：预期 `0xa5871304`，拷回 CPU 后为 `0xa5870304`。因此不只是 GPU 计数器报告异常；确实读回了不同的位。错误具有间歇性，不能用一次正常检查排除。

同时，`nvidia-smi` 报告：

- ECC 当前关闭。
- `Remapped Rows / Uncorrectable Error: 8`。
- `Remapping Failure Occurred: Yes`。
- 一个 bank 的 remap availability 为 `None`。

这些字段与独立读写错误共同表明当前实例存在必须由平台处理的显存完整性问题。本实验不自行诊断具体损坏部件。没有更改 ECC、重置 GPU 或调整时钟。

在完整模型诊断中，MLP 的 `up_proj` 输入保持相同，而原 GPU 权重相对其副本的差异从 5 个元素增长到 33 个。副本上的同样矩阵乘保持稳定。无 attention 的一次模型加载与矩阵乘检查正常，也予以保留；单次正常不否定上述间歇性错误。

完整证据见 [汇总](evidence/summary.json)、[GPU 状态](evidence/gpu-memory-health.txt)、[重复 2 原始检查](evidence/standalone-memory-check-repeat-2.txt) 和 [权重变化诊断](evidence/full-qwen-diagnose4.txt)。原始诊断日志另保存在仓库的本地 `experiments/megakernel_predictor/runs/full-qwen-diagnostics/`，不纳入性能分析。

### 2.1 用户要求的再次复查（2026-11-17 22:48 起）

重新编译检查器，在同一 GPU UUID、没有其他可见计算进程时，启动三个独立进程，每次检查 20 GiB：

| 路径 | 进程 1 | 进程 2 | 进程 3 |
|---|---|---|---|
| GPU kernel 写入与检查，错误位置另拷回 CPU | 0/0/0/0 | 295/309/309/309 | 351/355/355/355 |
| CPU 生成，H2D 后全部 D2H，由 CPU 核对，不启动自定义 GPU kernel | 0/0 | 0/0 | 0/0 |

表中数字为同一进程各轮错误的 32 位字数，不应跨轮相加当作独立错误总数。新错误样例：预期 `0xa55d2d5c`，CPU 实际读回 `0xa55d2c5c`，异或为 `0x100`。NVML 的重映射失败状态及 8 行不可纠正错误记录未变；该状态可能包含历史记录，不能将 8 行解释为本次新增错误。

新增的 [纯主机校验实现](memory_check_host.cu) 三次通过，应与失败结果一并报告。它说明纯拷贝路径在这轮测试中正常，**不能将当前结论扩大为所有内存路径都失败，也不能仅凭这些检查定位具体故障部件**。目前可重复确认的是 GPU kernel 读写路径的数据完整性错误；结合此前模型权重变化，完整模型性能结论仍不可信。尚未区分物理显存、缓存、驱动或其他执行环境因素。

本次全部原始记录位于 [recheck-20261117](evidence/recheck-20261117/)，没有继续运行模型性能测量。

## 3. 完整模型实验的范围

研究问题仍参考 [Optimizing CUDA Graph Scheduling with Reinforcement Learning: A Case Study in SSTA Propagation](https://tsung-wei-huang.github.io/papers/2025-MLCAD.pdf)：在不减少原始计算、不移除功能依赖的前提下，添加辅助依赖边或改变图构建顺序，能否改进真实 GPU 执行。

本轮使用完整 Qwen3-8B 的 BF16 权重，所有权重驻留 GPU。长前缀 KV 保存在锁页 CPU 内存；一次请求读回此前缀，为新增输入执行全部 36 层计算，经过最终归一化和 lm_head 产生首 token，再自回归生成后续 token。模型不是单层替身，也未跳过 embedding、MLP 或输出头。attention 使用 FlashInfer FA2，普通矩阵乘使用 PyTorch/cuBLAS。

该工作负载是 CPU 前缀缓存命中后的完整模型推理，不是专家卸载，也不是写回到 CPU。另设 KV 已在 GPU 的对照。它是独立推理执行器，**不是原生 SGLang、vLLM 或在线服务请求实验**。

原生 Transformers 与实验执行器使用相同权重、输入、位置编码和 attention 实现，以隔离模型拆分与调度的正确性。每个候选运行前，在计时之外将 GPU KV 和中间状态填为 NaN。校验检查完整 logits 的有限性、相对 L2 误差和整个生成 token 序列，不能仅凭首 token 相同认定正确。当前相对 L2 阈值为 0.005；发现错误后未放宽阈值以通过实验。

## 4. 执行图与对照

模型分为 146 个逻辑阶段：embedding；每层的前缀读回、QKV 投影、attention、输出投影与 MLP；最后的归一化和输出头。每个逻辑阶段包含实际捕获的多个 CUDA 节点，146 不是 CUDA kernel 数。

保留完整的层间计算依赖、KV 数据依赖和读回链。所有节点先按指定顺序创建，再添加候选辅助依赖，避免把加边效果和创建顺序变化混为一谈。每个片段保留独立捕获流和相应资源生命周期。

候选包括原图、独立相同原图（A/A）、先创建读回节点、先创建计算节点、预取最多领先 1/2/4/8/16 层，以及八个仅加 2 或 4 条合法辅助边的候选。共 17 个显式图候选。限制预取领先距离通过增加“较早层计算完成 → 后续层读回”的依赖实现，输入、权重、读回字节数和原始运算保持不变。

另外设置三项对照：

1. 普通完整前向 CUDA Graph：完整模型与双流逐层读回，不使用额外阶段状态复制。
2. 普通分阶段 CUDA Graph：与显式图使用相同阶段缓冲区，以分离图构造和阶段拆分成本。
3. KV 全驻留 GPU 的完整前向 CUDA Graph：不计读回成本的参考。

只有改进原图而不能超过普通完整前向，不能证明实际推理加速。A/A 的差值也不能算优化收益。

## 5. ComposeDES 与计划中的验证

`predict.py` 使用 ComposeDES 的 `WorkflowController` 维护依赖与完成事件，将阶段独立耗时与读回/计算共跑代价组合。第一版只有一条计算链和一个 H2D 服务资源，使用共享的干扰、提交间隔和优先级参数。这是待验证模型，不是已经识别出的 CUDA 调度规律。

独立阶段数据用于标定；完整候选的目标测量用于评价。协议先用 `(CPU 前缀 token 数, 新增输入 token 数)=(2048,64)、(4096,256)` 标定，并保存冻结参数及 SHA256。随后使用 `(4096,128)、(8192,64)、(8192,512)`，每种形状在两个独立进程中使用不同 prompt。每个候选生成 32 个 token，进行 24 轮随机顺序首 token 测量和 4 轮完整生成测量。最初由于硬件检查失败未执行；后续按用户要求在性能模式下运行，结果与正确性结论分开报告。

这不是原论文 GNN+DQN 的完整模型训练复现。第一步检验完整推理中是否存在值得学习的加边收益及 ComposeDES 的排序能力；不能以准备了候选就声称复现了强化学习优于搜索的结论。

## 6. 计时边界与应报告的指标

- TTFT：从执行图提交前开始，包括 CPU KV 读回、完整模型对新增输入的计算和首 token 返回 CPU。分词、前缀缓存建立、图构建和模型加载均在计时外；没有服务器排队或网络延迟。
- GPU 时间：只在整图前后记录事件，性能图内不增加阶段计时事件。
- TPOT：后续 token 的逐次完整 decode 回放和 token 返回 CPU 的时间。本轮所有候选共享 decode 路径，不能将其波动归因于首 token 图优化。
- 完整生成时间与输出 token/s：包含 TTFT 及后续 decode。
- 同时记录 GPU 分配/保留显存、设备用量、输入 token ID、生成结果、候选顺序、原始逐次测量和标定成本。

选择成本应单独报告；当前 Python ComposeDES 枚举不是在线请求路径。阶段测量可能改变竞争状态的问题仍存在，需要通过未插桩的完整运行验证。24 轮不能支撑可靠 P99 结论。

## 7. 换机恢复

先请平台更换或修复 GPU。只重启当前容器不能证明显存已恢复。健康 GPU 上首先检查 `nvidia-smi -q` 并重复独立显存测试，再执行严格模型校验。检查通过也不代替模型数值验证。

环境为 Python 3.12、torch 2.12.1+cu130、CUDA Toolkit 13.0.88，其余依赖见 `requirements.txt`。官方 Qwen3-8B 文件可通过 `download_model.py` 从 ModelScope 下载并生成 SHA256 清单；不将权重提交 GitHub。当前服务器模型目录为 `/root/autodl-tmp/models/Qwen3-8B`。

```bash
git submodule update --init fncs
cd experiments/megakernel_predictor/a100/full_inference
export PATH=/usr/local/cuda/bin:/root/miniconda3/bin:$PATH
export CUDA_HOME=/usr/local/cuda MAX_JOBS=4
PYTHON=python NVCC=nvcc MODEL=/path/to/Qwen3-8B bash run.sh
```

默认 `run.sh` 在显存重映射失败、独立检查失败或模型校验失败时立即停止。设置 `PERFORMANCE_ONLY=1` 会显式绕过这些阻断，仍保留真实计算、数据传输、结构检查和 CUDA 执行错误检查；所有结果标注未验证数值正确性。性能模式示例见 `PERFORMANCE.md`。目前已确认 Python/Shell 语法、CUDA 编译、17 个候选保留原依赖且无环，以及 ComposeDES 的事件推进检查。CPU 协议检查可运行 `python verify_protocol.py`。
