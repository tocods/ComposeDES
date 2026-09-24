# ATLAHS LULESH-64 当前版本消融实验

实验日期：2026-09-24（Asia/Shanghai）

代码版本：FNCS `700d7d55a`、GPUSim `09861db84`、ns-3 `2f049f0d4`

## 数据与设计

LULESH 是传统 HPC 冲击流体动力学代理应用。本实验使用 64-rank GOAL trace，共
11,438,062 条操作：7,284,064 个 `calc`、2,076,999 个 `send` 和 2,076,999 个 `recv`。
源文件 SHA-256 为 `d6f955de615cf2c4a57dc95b1c88c5828f03cfd31b75cc50ad044242284b9a3f`。

工作流按 rank 划分 16 个 phase，每 phase 8 个计算事件。纵向优化把 8,192 个计算事件
精确收缩为 1,024 个 phase 宏事件；横向优化开启 Active-dependency。网络包含 17,032 个
聚合边和 5,747,240,448 modeled bytes，采用 56 Gbit/s、1,000 ns 链路的 ns-3
store-and-forward flow 模型。四个组合各运行三次。

## 结果

| 纵向优化 | 横向优化 | 墙钟中位数 / s | 相对双关 | 计算事件 | 调度轮次 | Grants |
|---|---|---:|---:|---:|---:|---:|
| 关 | 关 | 13.5972 | 1.000× | 8,192 | 64,051 | 64,834 |
| 关 | 开 | 13.6464 | 0.996× | 8,192 | 64,083 | 64,101 |
| 开 | 关 | 6.0869 | 2.234× | 1,024 | 51,974 | 51,989 |
| 开 | 开 | 6.0864 | **2.234×** | 1,024 | 52,018 | 52,034 |

当前版本的纵向聚合稳定获得约 2.234× 加速。横向 Active-dependency 在固定三 Federate
架构中没有可分辨的墙钟收益；双开与纵向单开基本相同。

四组均完成 17,032 个网络事件，归一化网络语义 SHA-256 相同。模拟 makespan 跨度为
2,092 ns，即 0.372 ppm，满足 1 ppm 等价阈值。

## 与旧结果的区别

旧报告使用 Active 高频路径和批量图收缩最终优化完成前的运行，纵向单开为 1.724×。
本报告全部 12 次运行均由 FNCS `700d7d55a` 重新执行；批量区域收缩减少了工作流准备开销，
纵向单开更新为 2.234×。旧结果不再作为当前版本结论。

## 复现

```bash
python3 experiments/atlahs_ablation/run_ablation.py \
  --goal /path/to/lulesh_64.goal \
  --output experiments/atlahs_ablation/lulesh64 \
  --phases 16 --events-per-phase 8 --prepare-only

python3 experiments/atlahs_ablation/run_ablation.py \
  --output experiments/atlahs_ablation/lulesh64 \
  --run-only --repeats 3 --timeout 900 --quiet-worker-logs
```

机器可读结果位于 `lulesh64/results.csv` 和 `lulesh64/summary.json`。
