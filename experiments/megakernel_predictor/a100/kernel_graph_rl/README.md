# 完整 Qwen3-8B 实际 CUDA kernel 图的辅助边优化

本轮已完成。两个留出形状中，最终 512 次目标测量预算下选择的配对收益区间均包含零，未观察到稳定加速。详见 [结论与研究判断](DISCUSSION.md)、[结果表与预算曲线](RESULTS.md)；全部原始计时和冻结权重已归档。

## 研究目标与复现边界

本实验依照 [Optimizing CUDA Graph Scheduling with Reinforcement Learning: A Case Study in SSTA Propagation](https://tsung-wei-huang.github.io/papers/2025-MLCAD.pdf) 的公开方法，在完整 Qwen3-8B 上实现节点拓扑层级调整、GCN 与 DQN，并与随机策略和 ComposeDES 比较。应用、GPU 和 CUDA 版本与论文不同。

本轮从实际 CUDA Graph 提取 kernel、拷贝及初始化节点，直接给原图的克隆增加依赖。权重和前缀 KV 均驻留 GPU，所有 36 层、embedding 和 lm_head 均参与执行。按用户要求只评价性能，数值差异不阻断实验。已知 GPU 数据完整性异常仍保留在上一轮诊断记录中。

目前的代码是根据论文公开描述进行的重实现。没有找到与该论文对应的公开作者代码。原文未完全指定 GCN 隐层维度、归一化方式、目标节点选择和每轮动作数，这些实现选择明确列在 `PROTOCOL.json` 中，不能声称逐项复现了作者未公开的实现。

## 实际图提取

`capture_model.py` 捕获完整 Qwen3-8B 的执行过程。每层保留 Q/K/V 投影的独立分支以及 MLP 中 gate/up 两路投影的独立分支，分支汇合严格保留数据依赖。串行对照使用相同运算表达式。

`kernel_runtime.cu` 通过 CUDA Graph API 导出实际节点与依赖，包括 kernel 名、线程数、block 数、寄存器数、动态共享内存和占用率上限。候选使用 `cudaGraphClone` 克隆原始图，通过节点映射添加辅助边；不替换 kernel，不删除原依赖，不改变原节点的构建过程。独立进程复测前核对整个源图的指纹。

这与上一轮将多 kernel 阶段作为搜索单位不同：本轮动作的目标是实际 kernel 节点。拷贝与初始化节点同样保留在依赖图和性能模型中。

## 动作与学习方法

动作 0 保持节点层级；动作 1/2 分别将其拓扑层级增加 1/2。新增前驱从目标层级的前一层选择，并排除目标自身及其全部后继，保证无环。层级指 DAG 的拓扑层级，不是 Transformer 层编号。

GCN 使用论文的六项节点特征和两跳信息传播；DQN 的全连接网络为 3072→256→64→3。折扣因子 0.99、目标网络更新率 0.005、epsilon 从 1.0 以 0.997 衰减至 0.05、batch size 16、Adam 学习率 0.0001、Huber loss、100 个训练 epoch，第 10 个 epoch 后冻结 GCN。奖励是按当轮原图时间归一化后的执行时间下降。

未公开的细节在此固定为：GCN 6→64→3072；带自环的有向入边传播与对称度归一化；均匀选择当前可移动 kernel；每 epoch 31 个动作。目标节点的两跳表示在局部计算，其结果与对整个图执行相同两层 GCN 一致，`test_methods.py` 对这一等价关系进行数值检查。该检查是 CPU 上算法实现的测试，不是要求异常 GPU 的模型数值校验通过。

## 等预算协议

协议在训练前固定于 `PROTOCOL.json`。

1. **收益空间探索**：512-token 前缀、1-token 完整前向，测量 10,000 个不同随机加边图。每图增加 1/2/4/8/16/32 条边，定期插入原图测量。搜索后冻结前 16 个候选，再随机交错复测。该开发数据不用于学习参数。
2. **训练**：独立的 768-token 前缀输入。随机策略、GCN+DQN 和 ComposeDES 各 3,200 次完整图测量，包含每 epoch 的原图对照。每次查询固定 1 次预热、3 次计时回放，因此每方法 12,800 次图回放。
3. **留出**：2048/1 和 1024/64 两个新输入形状。各方法每形状 512 次查询，使用相同动作空间和相同回放次数。报告 8/32/128/512 查询预算下选中的图。
4. **独立确认**：重建模型，使用新 prompt，检查源图指纹；在同一进程中交错比较三个方法各预算的选择、原图、A/A 和串行对照，32 轮。确认数据不反馈到选择器。
5. **完整生成**：测首个优化 forward 和后续 31 次完整 decode，总计 32 个 token。后续 decode 使用共同路径，因此本轮只能把首个 forward 的改动归因于辅助边。

每个方法都单独记录公共准备、kernel 标定、完整图查询和 CPU 选择耗时。标定包含实际节点的独立计时，不隐藏在“零测量”表述中。零目标反馈只表示选择时未使用该目标完整图的计时反馈，ComposeDES 仍使用目标节点标定。

论文式冻结策略的直接选择与等预算搜索分开报告。留出时 DQN 权重冻结，直接选择用 epsilon=0，预算搜索用 epsilon=0.05；ComposeDES 在等预算观测下更新共享参数权重。这项预算比较是本实验的扩展，不应冒充论文原有评价协议。

GCN/DQN 的训练和决策在 CPU 上进行，GPU 只执行被测模型。选择属于离线图优化；完整生成时直接回放已经选定的图，不重新运行选择器。离线搜索耗时与 GPU 前向时间分别报告。

## ComposeDES

`admission.cpp` 实现编译后的离散事件资源模型，依据真实 kernel 的独立耗时、grid、occupancy 上限及原始依赖，模拟准入与并发干扰。`predict.py` 用实际计时更新共享参数权重，能够改变未测候选的相对排序，而非只修正共同偏移。

为降低大量候选的 Python 调用成本，候选排序使用编译后的事件循环；选中图的完成事件同时通过 ComposeDES `WorkflowController` 回放，核对原始任务依赖与终止状态。该资源模型是可检验的近似，不是 CUDA 驱动调度器的精确还原。

## 运行

沿用上一轮的 PyTorch、Transformers 和 FlashInfer 环境；模型目录为 `/root/autodl-tmp/models/Qwen3-8B`。初始化 ComposeDES 的 `fncs` 子模块后，在本目录执行：

```bash
export PATH=/usr/local/cuda/bin:/root/miniconda3/bin:$PATH
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.0 FLASHINFER_CUDA_ARCH_LIST=8.0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
nvcc -std=c++17 -O3 -arch=sm_80 -shared -Xcompiler -fPIC kernel_runtime.cu -lcuda -o libkernel_graph.so
g++ -std=c++17 -O3 -shared -fPIC admission.cpp -o libadmission.so
python test_methods.py
python probe.py --out ../../runs/kernel-graph-rl/discovery --samples 10000
python run_comparison.py ../../runs/kernel-graph-rl
python analyze.py ../../runs/kernel-graph-rl --out reference/summary.json
python report.py reference/summary.json
python archive.py ../../runs/kernel-graph-rl --out reference
```

`run_comparison.py` 支持继续已经完成的训练或确认任务。CUDA 执行错误、非法依赖及不同进程源图结构不一致仍会停止任务；输出数值不同不会停止性能实验。最终结果必须从冻结方案的独立确认中取得，不能只报告随机搜索的最小值。

结果表、完整生成测量及预算曲线见 `RESULTS.md`。`reference/kernel-graph-raw.tar.gz` 保存全部原始计时、真实图导出、训练权重和冻结选择，`reference/artifacts.json` 提供逐文件 SHA-256。实验环境在归档内的 `environment.json` 中。归档不包含模型权重，可从 Qwen/Qwen3-8B 单独获取。
