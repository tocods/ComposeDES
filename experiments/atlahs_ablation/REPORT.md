# ATLAHS LULESH-64 纵向 × 横向消融实验

实验日期：2026-09-24（Asia/Shanghai）

## 结论

在本实验的 64-rank LULESH 派生工作流中，Critical-path event
acceleration 将计算事件从 8,192 个精确收缩为 1,024 个宏事件。它使墙钟运行时间的
中位数从 13.947 s 降至 8.089 s，缩短 42.0%，相当于 1.724 倍加速；调度轮次减少
18.9%，FNCS 时间授权次数减少 19.8%。

Active-dependency 时间协调单独开启时，授权次数减少 1.13%，墙钟时间中位数只变化
0.36%。这个差异小于三次运行的波动，不能据此认为它在此负载上有稳定的性能收益。
与事件收缩同时开启时，中位数为 8.189 s，仍比全关闭快 1.703 倍，但比只开启事件
收缩慢约 1.24%。

因此，这组实验支持“事件聚合能显著提升执行性能”；它没有证明 active-dependency
在当前三联邦成员、可运行任务较多的负载上能进一步提速。后者更可能在联邦成员更多、
阻塞关系更稀疏且保守时间授权占主导的场景中体现收益。

## 数据集与映射

原始数据来自 ATLAHS 官方 trace collection 的 LULESH-64 GOAL trace：

- 文件：`/home/sdic/atlahs/data/hpc/lulesh/lulesh_64/lulesh_64.goal`
- 大小：665,077,560 bytes（约 634 MiB）
- SHA-256：`d6f955de615cf2c4a57dc95b1c88c5828f03cfd31b75cc50ad044242284b9a3f`
- 64 ranks，11,438,062 条 GOAL 操作
- 7,284,064 个 `calc`，2,076,999 个 `send`，2,076,999 个 `recv`

转换器按每个 rank 的操作顺序划分为 16 个 phase，并使用全部操作累计计算时间与通信量。
每个 phase 被拆为 8 个同主机、线性、零通信的计算事件，形成 1,024 个满足闭包约束的
精确收缩区域。通信按 `(source rank, phase, destination rank)` 聚合，并连接到下一 phase：

- 基线计算任务：8,192
- 收缩后计算任务：1,024
- 模型化网络边：17,032
- 模型化发送量：5,747,240,448 bytes
- 网络：64-host chain，56 Gbit/s，单链路延迟 1,000 ns
- ns-3 使用 store-and-forward flow macro 模式

最后一个 phase 的发送没有下一 phase 可连接，因此未纳入模型；原始 trace 的全部发送量为
6,131,284,152 bytes。数据转换细节和计数保存在 `lulesh64/dataset.json`。

## 实验设计

两个布尔变量形成 2×2 组合：

1. 纵向优化（Critical-path event acceleration）：通过工作流 `acceleration.regions` 开关控制；开启时执行
   1,024 个 exact closed-region contractions。
2. 横向优化（Active-dependency 时间协调）：同时通过 broker 的 `FNCS_ACTIVE_DEPENDENCY` 与编排器的
   `--active-dependency-coordination` 控制。

每个组合运行 3 次，轮换组合顺序。墙钟时间从 broker 启动前计时到整个 federation 退出，
包含工作流读取和关键路径优化。实验没有开启 backend-local chains，以免引入第三个变量。

## 结果

下表中的墙钟时间为三次运行中位数；括号中为均值 ± 样本标准差。

| 纵向事件聚合 | 横向 Active 协调 | 墙钟时间 / s | 相对基线 | 计算事件 | 调度轮次 | FNCS grants | 控制面记录 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 关 | 关 | 13.947 (14.046 ± 0.173) | 1.000× | 8,192 | 64,051 | 64,834 | 34,296 |
| 关 | 开 | 13.896 (13.963 ± 0.116) | 1.004× | 8,192 | 64,083 | 64,101 | 34,494 |
| 开 | 关 | 8.089 (8.091 ± 0.048) | 1.724× | 1,024 | 51,974 | 51,989 | 19,978 |
| 开 | 开 | 8.189 (8.206 ± 0.076) | 1.703× | 1,024 | 52,018 | 52,034 | 20,174 |

四组都成功完成 17,032 个网络事件。将宏任务名称归一化为 rank/phase 边界后，四组网络
传输语义 SHA-256 相同。模拟 makespan 分别落在 5,617,368,443–5,617,370,535 ns，
跨度 2,092 ns，即 0.372 ppm；这满足报告采用的 1 ppm 等价阈值。精确时间戳哈希不同，
原因是不同协调策略改变了纳秒级授权顺序。

## 优化器实现发现

首次运行暴露了一个实现瓶颈：原优化器逐区域调用 `_contract`，每次都深拷贝并重新索引
整张图。1,024 个区域使开启事件收缩的冷启动时间达到 264–265 s，虽然调度轮次已经下降。

`fncs/orchestrator/graph_optimizer.py` 现改为对互不重叠区域做一次批量收缩。修正后，完整
端到端时间降为约 8.1 s，且仍包含优化步骤。原始慢实现的首轮结果保存在
`lulesh64/summary_pre_batch_optimizer.json` 和 `lulesh64/runs_pre_batch_optimizer/`。
新增的相邻多区域测试与原有测试共 5 项均通过，`tests/control_plane_alpha4_small/run.sh`
端到端回归也通过。

## 复现

在 `/home/sdic/ComposeDES/main` 下运行：

```bash
python3 experiments/atlahs_ablation/run_ablation.py --prepare-only
python3 experiments/atlahs_ablation/run_ablation.py --run-only --repeats 3 --timeout 900
```

完整机器可读汇总位于 `lulesh64/summary.json`，每次运行的事件日志、组件日志、协调指标和
优化报告位于 `lulesh64/runs/`。根文件系统当前仍有约 407 GiB 可用空间。

## 适用范围

这是使用大型官方 trace 构造的机制实验，不是 11,438,062 条 GOAL 操作的一对一重放。
通信被按 phase/peer 聚合，网络使用 flow macro，链式拓扑也不是原 ATLAHS 实验机器的真实
拓扑。三次重复足以看出 42% 量级的事件收缩收益，但不足以分辨 active-dependency 的亚百分比
差异。若用于论文，应增加重复次数，并补充更多 federation 成员、稀疏依赖工作流、真实拓扑
与 packet-level 网络模型。

## 来源

- ATLAHS trace collection: https://spcl.inf.ethz.ch/Research/Scalable_Networking/ATLAHS/
- ATLAHS paper: https://spcl.inf.ethz.ch/Publications/.pdf/2025_atlahs.pdf
