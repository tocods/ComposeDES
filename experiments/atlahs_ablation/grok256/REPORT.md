# ATLAHS Grok-314B 256-GPU 二因素消融实验

实验日期：2026-09-25（Asia/Shanghai）

组件版本：FNCS `48dd83bd1`、GPUSim `6ab9fc71f`、ns-3 `2f049f0d4`

## 数据集

实验使用 `Grok314B_N64_GPU256_TP4_PP1_CP1_VP1_EP8_ETP4_GBS512/grok.goal`：

- 文件大小 8,410,598,614 bytes，SHA-256 为 `dec250a489ef317757de0aa58c1d0becb7063c8b7d2927bf349d6b2d54830c06`；
- 256 ranks，114,000,750 条 GOAL 操作；
- 59,710,454 个 `calc`、27,145,148 个 `send`、27,145,148 个 `recv`；
- 转换后有 32,768 个基线计算事件、4,096 个纵向宏事件和 4,758 个网络完成事件。

转换采用每 rank 16 个 phase、每 phase 8 个计算事件。网络模型是 256-host chain、56 Gbit/s、
1,000 ns 单链路延迟的 ns-3 store-and-forward flow macro。

## 二因素定义

纵向开启表示 Critical-path event acceleration。横向关闭时，256 个 host 由一个 GPUSim
Federate 承载并采用静态保守协调；横向开启时，脚本启动 256 个 GPUSim Federates，编排器
按 host 定向派发任务，并启用 Active-dependency。横向结果因此包含完整切分及其真实进程成本。

## 三次重复结果

| 纵向 | 横向 | 计算 Federates | 墙钟中位数 / s | 均值 ± 标准差 / s | 相对基线 | 调度轮次 | Grants | 控制面记录 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 关 | 关 | 1 | 91.153 | 92.237 ± 3.558 | 1.000× | 1,117,671 | 1,119,194 | 73,554 |
| 关 | 开 | 256 | 126.527 | 126.808 ± 0.934 | 0.720× | 1,100,923 | 1,102,190 | 80,954 |
| 开 | 关 | 1 | 38.028 | 39.846 ± 3.281 | **2.397×** | 1,069,305 | 1,069,493 | 16,274 |
| 开 | 开 | 256 | 57.805 | 57.777 ± 0.212 | 1.577× | 1,056,096 | 1,057,337 | 23,610 |

纵向收缩减少 87.5% 的计算派发，单独获得 2.397×。完整横向单独减少 16,748 个调度轮次
（1.50%），但启动 256 个 JVM、维持 256 条 FNCS 时间线和跨进程序列化的成本更大，因此
墙钟性能降为 0.720×。组合项仍比基线快 1.577×，但比仅纵向慢。

这与此前的 rank-local 纯协调结果并不矛盾。纯协调模型排除了 GPUSim、ns-3 和 JVM 成本，
256 Federates 下横向单开为 2.153×，代表调度层上界；本报告是包含计算、网络和真实进程拓扑
的端到端结果。当前瓶颈是切分粒度和运行时承载方式，而不是 Grok 数据缺少 rank 并行性。

## 正确性

四组均完成 4,758 个网络事件，归一化网络语义 SHA-256 一致。模拟 makespan 范围为
1,006,360,860,972–1,006,360,889,632 ns，最大跨度 28,660 ns，即 0.0285 ppm。精确完成
时间哈希不同，来自不同 Federate 授权顺序的纳秒级偏移。

## 复现

```bash
python3 experiments/atlahs_ablation/run_ablation.py \
  --goal /home/sdic/atlahs/data/ai/grok/Grok314B_N64_GPU256_TP4_PP1_CP1_VP1_EP8_ETP4_GBS512/grok.goal \
  --output experiments/atlahs_ablation/grok256 \
  --phases 16 --events-per-phase 8 --prepare-only

python3 experiments/atlahs_ablation/run_ablation.py \
  --output experiments/atlahs_ablation/grok256 \
  --run-only --repeats 3 --timeout 3600 --quiet-worker-logs
```

中断后可用 `--repeat-start N` 从第 N 次继续。机器可读结果位于 `results.csv` 和
`summary.json`。实验按 phase/peer 聚合原始通信量，并非 1.14 亿操作的一对一 packet 重放；
链式拓扑也不代表生产 Grok 集群的实际拓扑，因此结果用于比较 ComposeDES 控制面方案。
