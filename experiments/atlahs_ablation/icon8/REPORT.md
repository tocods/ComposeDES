# ATLAHS ICON-8 消融实验

> 本报告保留一 rank 一 JVM、短周期请求的优化前结果。安全 frontier、长期 lease 和
> 4 ranks/Federate 的当前结果见
> [`../icon8_frontier_partitioned/REPORT.md`](../icon8_frontier_partitioned/REPORT.md)。

实验日期：2026-09-25（Asia/Shanghai）

组件版本：FNCS `48dd83bd1`、GPUSim `6ab9fc71f`、ns-3 `2f049f0d4`

## 数据与方法

ICON 是天气与气候模拟 HPC 应用。
该 8-rank GOAL trace 包含 295,072 条操作：160,856 个 `calc`、67,108 个 `send` 和 67,108 个 `recv`。
源文件 SHA-256 为 `46b5ac70e60d099e3457f6ad2153cebb4a0ba63c8fbbc83edd8672c07eee9885`。

转换后基线有 1,024 个计算事件，纵向优化收缩为 128 个宏事件；模型包含 707 个网络边。横向关闭时使用一个集中式 GPUSim，横向开启时使用每 rank 一个 GPUSim 并开启 Active-dependency。四个单元各运行三次。

## 结果

| 纵向 | 横向 | 计算 Federates | 墙钟中位数 / s | 相对基线 | 计算事件 | 调度轮次 | Grants |
|---|---|---:|---:|---:|---:|---:|---:|
| 关 | 关 | 1 | 2.1120 | 1.000× | 1,024 | 23,908 | 24,051 |
| 关 | 开 | 8 | 2.7506 | 0.768× | 1,024 | 23,692 | 23,725 |
| 开 | 关 | 1 | 1.6803 | 1.257× | 128 | 22,543 | 22,554 |
| 开 | 开 | 8 | 1.9834 | 1.065× | 128 | 22,326 | 22,359 |

纵向单独获得 1.257×。完整横向把调度轮次减少 0.90%，但进程和通信成本使墙钟变为基线的 0.768×。两者组合仍获得 1.065×，但低于仅纵向。

四组均完成 707 个网络事件，归一化网络语义哈希一致。模拟 makespan 最大跨度为 9,702 ns，即 0.489 ppm。

## 复现

```bash
python3 experiments/atlahs_ablation/run_ablation.py --goal /home/sdic/atlahs/data/hpc/icon/icon_8/icon_8.goal \
  --output experiments/atlahs_ablation/icon8 --phases 16 --events-per-phase 8 --prepare-only
python3 experiments/atlahs_ablation/run_ablation.py --output experiments/atlahs_ablation/icon8 \
  --run-only --repeats 3 --timeout 1800 --quiet-worker-logs
```

机器可读结果位于 `results.csv` 和 `summary.json`。
