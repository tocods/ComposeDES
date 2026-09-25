# ATLAHS LULESH-64 消融实验

实验日期：2026-09-25（Asia/Shanghai）

组件版本：FNCS `48dd83bd1`、GPUSim `6ab9fc71f`、ns-3 `2f049f0d4`

## 数据与方法

LULESH 是 HPC 冲击流体动力学代理应用。
该 64-rank GOAL trace 包含 11,438,062 条操作：7,284,064 个 `calc`、2,076,999 个 `send` 和 2,076,999 个 `recv`。
源文件 SHA-256 为 `d6f955de615cf2c4a57dc95b1c88c5828f03cfd31b75cc50ad044242284b9a3f`。

转换后基线有 8,192 个计算事件，纵向优化收缩为 1,024 个宏事件；模型包含 17,032 个网络边。横向关闭时使用一个集中式 GPUSim，横向开启时使用每 rank 一个 GPUSim 并开启 Active-dependency。四个单元各运行三次。

## 结果

| 纵向 | 横向 | 计算 Federates | 墙钟中位数 / s | 相对基线 | 计算事件 | 调度轮次 | Grants |
|---|---|---:|---:|---:|---:|---:|---:|
| 关 | 关 | 1 | 12.7955 | 1.000× | 8,192 | 64,051 | 64,834 |
| 关 | 开 | 64 | 23.2066 | 0.551× | 8,192 | 61,637 | 61,842 |
| 开 | 关 | 1 | 5.2851 | 2.421× | 1,024 | 51,974 | 51,989 |
| 开 | 开 | 64 | 11.2422 | 1.138× | 1,024 | 49,554 | 49,759 |

纵向单独获得 2.421×。完整横向把调度轮次减少 3.77%，但进程和通信成本使墙钟变为基线的 0.551×。两者组合仍获得 1.138×，但低于仅纵向。

四组均完成 17,032 个网络事件，归一化网络语义哈希一致。模拟 makespan 最大跨度为 24,252 ns，即 4.317 ppm。

## 复现

```bash
python3 experiments/atlahs_ablation/run_ablation.py --goal /home/sdic/atlahs/data/hpc/lulesh/lulesh_64/lulesh_64.goal \
  --output experiments/atlahs_ablation/lulesh64 --phases 16 --events-per-phase 8 --prepare-only
python3 experiments/atlahs_ablation/run_ablation.py --output experiments/atlahs_ablation/lulesh64 \
  --run-only --repeats 3 --timeout 1800 --quiet-worker-logs
```

机器可读结果位于 `results.csv` 和 `summary.json`。
