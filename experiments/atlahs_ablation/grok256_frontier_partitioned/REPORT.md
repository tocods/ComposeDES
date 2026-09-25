# Grok-314B-256：frontier 与长期 Federate 优化后的二因素消融

实验日期：2026-09-25（Asia/Shanghai）。每个单元独立运行三次，报告墙钟中位数。

## 变量与实现

实验仍只改变两个变量：纵向为 Critical-path event acceleration；横向为完整的 Federate
切分与 Active-dependency 时间协调。优化后的横向实现包含三项运行时机制：

1. 编排器根据所有在途计算和网络命令的持续时间下界，发布不会更早收到新任务的安全
   `frontier`；缺少任一下界时立即撤销 frontier。
2. 空闲 worker 请求最大 FNCS 时间作为可撤销长期 lease。Active 依赖在第一个生产者完成时
   唤醒编排器，因此不再每 1 ms 轮询。
3. 256 ranks 按连续编号放入 8 个长期存活的 GPUSim Federates，每个承载 32 ranks，避免
   一 rank 一 JVM。`--compute-partition-map` 也可按独立作业或 DAG 分量显式分组。

网络仍使用一个共享 ns-3 时间线。当前 Grok trace 是单作业，通信共享同一链路模型，没有拆分
网络时间线；只有多个作业的网络拓扑和资源也相互独立时，网络分区才是安全的。

## 三次重复结果

| 纵向 | 横向 | 计算 Federates | Ranks/Federate | 墙钟中位数 / s | 均值 ± 标准差 / s | 相对基线 | 调度轮次 |
|---|---|---:|---:|---:|---:|---:|---:|
| 关 | 关 | 1 | 256 | 90.052 | 90.028 ± 0.280 | 1.000× | 1,117,671 |
| 关 | 开 | 8 | 32 | 60.030 | 60.173 ± 0.381 | **1.500×** | 115,613 |
| 开 | 关 | 1 | 256 | 37.817 | 37.904 ± 0.185 | **2.381×** | 1,069,305 |
| 开 | 开 | 8 | 32 | 10.503 | 10.508 ± 0.071 | **8.574×** | 56,223 |

横向单开把调度轮次减少 89.66%，墙钟减少 33.34%。组合项把轮次减少 94.97%，墙钟减少
88.34%。两个方向存在正交之外的协同：纵向先减少派发事件，frontier 再消除这些事件之间的
空轮询；因此组合加速高于两个单项加速的简单乘积。

旧实现使用 256 个 JVM、每个 rank 一个 Federate，并保留短周期请求。它在同一 Grok 模型上
横向单开只有 0.720×、组合为 1.577×。新结果说明性能恶化来自切分粒度和时间协调协议，而
不是 Grok 数据缺少并行性。

## 分区粒度扫描

扫描固定纵向和横向都开启，每个点运行一次；8 ranks/Federate 点使用最终的长期 lease 实现。

| Ranks/Federate | 计算 Federates | 墙钟 / s | 调度轮次 |
|---:|---:|---:|---:|
| 1 | 256 | 30.908 | 51,212 |
| 2 | 128 | 23.174 | 51,628 |
| 4 | 64 | 18.241 | 52,417 |
| 8 | 32 | 14.875 | 53,231 |
| 16 | 16 | 12.125 | 54,457 |
| 32 | 8 | **10.440** | 56,223 |

增加每个 Federate 承载的 ranks 会略增协调轮次，却显著减少 JVM、FNCS 连接和序列化开销。
64 ranks/Federate 的试验触发 GPUSim/CloudSim `Past event detected`：单进程内大量并发事件无法
安全接受如此长的时间跳跃。因此当前经过验证的上限为 32 ranks/Federate，而不是继续合并到
4 个 Federates。

## 正确性

四个单元都完成 4,758 个网络事件，归一化网络完成语义 SHA-256 相同。模拟 makespan 最大
跨度为 20,994 ns，相对跨度 0.0209 ppm，低于 1 ppm 验证阈值。计算下界为声明下界减去
10,000 ns 的 GPUSim 时间转换安全余量；网络下界只使用传播延迟的保守下界。任何没有证书的
在途事件都会关闭 frontier，回退到 Active 依赖安全条件。

机器可读结果位于 [`results.csv`](results.csv)、[`summary.json`](summary.json) 和
[`partition_sweep.csv`](partition_sweep.csv)。原始逐事件日志和 8.4 GB GOAL 文件不进入 Git。

## 复现

```bash
python3 experiments/atlahs_ablation/run_ablation.py \
  --goal /path/to/grok.goal \
  --output /tmp/grok256-frontier \
  --phases 16 --events-per-phase 8 --ranks-per-federate 32 \
  --repeats 3 --timeout 3600 --quiet-worker-logs
```

多作业可提供完整的 host 到 DAG 分区映射：

```bash
python3 experiments/atlahs_ablation/run_ablation.py \
  --output /tmp/multi-job --run-only --ranks-per-federate 32 \
  --compute-partition-map job-partitions.json
```
