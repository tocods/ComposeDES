# ATLAHS HPCG-8 消融实验

> 本报告保留一 rank 一 JVM、短周期请求的优化前结果。安全 frontier、长期 lease 和
> 4 ranks/Federate 的当前结果见
> [`../hpcg8_frontier_partitioned/REPORT.md`](../hpcg8_frontier_partitioned/REPORT.md)。

实验日期：2026-09-25（Asia/Shanghai）

组件版本：FNCS `48dd83bd1`、GPUSim `6ab9fc71f`、ns-3 `2f049f0d4`

## 数据与方法

HPCG 是 HPC 多重网格与稀疏线性代数基准。
该 8-rank GOAL trace 包含 626,216 条操作：402,680 个 `calc`、111,768 个 `send` 和 111,768 个 `recv`。
源文件 SHA-256 为 `3e9c5aff427e3173264cceee411081f4d9046c7bd870be1c362f8505e80d5ce9`。

转换后基线有 1,024 个计算事件，纵向优化收缩为 128 个宏事件；模型包含 840 个网络边。横向关闭时使用一个集中式 GPUSim，横向开启时使用每 rank 一个 GPUSim 并开启 Active-dependency。四个单元各运行三次。

## 结果

| 纵向 | 横向 | 计算 Federates | 墙钟中位数 / s | 相对基线 | 计算事件 | 调度轮次 | Grants |
|---|---|---:|---:|---:|---:|---:|---:|
| 关 | 关 | 1 | 1.8304 | 1.000× | 1,024 | 9,315 | 9,435 |
| 关 | 开 | 8 | 2.4858 | 0.736× | 1,024 | 9,071 | 9,108 |
| 开 | 关 | 1 | 1.4297 | 1.280× | 128 | 7,898 | 7,913 |
| 开 | 开 | 8 | 1.7337 | 1.056× | 128 | 7,660 | 7,697 |

纵向单独获得 1.280×。完整横向把调度轮次减少 2.62%，但进程和通信成本使墙钟变为基线的 0.736×。两者组合仍获得 1.056×，但低于仅纵向。

四组均完成 840 个网络事件，归一化网络语义哈希一致。模拟 makespan 最大跨度为 4,703 ns，即 0.998 ppm。

## 复现

```bash
python3 experiments/atlahs_ablation/run_ablation.py --goal /home/sdic/atlahs/data/hpc/hpcg/hpcg_8/hpcg_8.goal \
  --output experiments/atlahs_ablation/hpcg8 --phases 16 --events-per-phase 8 --prepare-only
python3 experiments/atlahs_ablation/run_ablation.py --output experiments/atlahs_ablation/hpcg8 \
  --run-only --repeats 3 --timeout 1800 --quiet-worker-logs
```

机器可读结果位于 `results.csv` 和 `summary.json`。
