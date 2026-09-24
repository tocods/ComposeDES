# ATLAHS Grok-314B 256-GPU 消融实验

实验日期：2026-09-24（Asia/Shanghai）

## 结论

在 256-rank Grok-314B 派生工作流中，Critical-path event acceleration 将计算事件从
32,768 个精确收缩到 4,096 个。关闭组件 stdout 落盘后，三次运行的墙钟中位数从
99.562 s 降至 47.689 s，缩短 52.1%，即 2.088 倍加速。

Active-dependency 时间协调单独开启时，时间授权次数减少 0.65%，但墙钟中位数增加
11.1%。双开时为 55.748 s，比全关闭快 1.786 倍，但比只开启事件加速慢 16.9%。
在当前只有 broker、ns-3、GPUSim 和 orchestrator 参与的三联邦成员协调模型中，新增的
1,019 次依赖更新没有抵消其处理成本。

## 数据集

- 官方文件：`grok.goal`
- 本地路径：`/home/sdic/atlahs/data/ai/grok/Grok314B_N64_GPU256_TP4_PP1_CP1_VP1_EP8_ETP4_GBS512/grok.goal`
- 文件大小：8,410,598,614 bytes（7.833 GiB）
- SHA-256：`dec250a489ef317757de0aa58c1d0becb7063c8b7d2927bf349d6b2d54830c06`
- 256 ranks，114,000,750 条 GOAL 操作
- 59,710,454 个 `calc`、27,145,148 个 `send`、27,145,148 个 `recv`

转换参数与 LULESH-64 实验一致：每 rank 16 个 phase，每 phase 8 个计算事件。每个 phase
构成一个同主机、线性、零通信的 exact closed region。

- 基线计算事件：32,768
- 加速后计算事件：4,096
- exact regions：4,096
- 聚合网络边：4,758
- 模型化发送量：9,487,384,518,016 bytes
- 原始 trace 发送量：10,481,131,527,552 bytes
- 256-host chain，56 Gbit/s，单链路延迟 1,000 ns
- ns-3 store-and-forward flow macro

最后一个 phase 的发送没有下一 phase 可连接，因此没有纳入模型化发送量。数据准备耗时
160.35 s，采用两遍流式扫描，未将 1.14 亿条操作整体装入内存。

## 三次重复结果

组件 stdout 使用 `--quiet-worker-logs` 丢弃，控制面事件、协调指标和优化报告仍完整保留。
墙钟列为中位数，括号内为均值 ± 样本标准差。

| Event acceleration | Active dependency | 墙钟 / s | 相对基线 | 计算事件 | 调度轮次 | FNCS grants | 控制面记录 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 关 | 关 | 99.562 (99.878 ± 0.777) | 1.000× | 32,768 | 1,117,671 | 1,119,194 | 73,554 |
| 关 | 开 | 110.625 (110.644 ± 0.625) | 0.900× | 32,768 | 1,111,667 | 1,111,885 | 74,579 |
| 开 | 关 | 47.689 (47.722 ± 0.106) | **2.088×** | 4,096 | 1,069,305 | 1,069,493 | 16,274 |
| 开 | 开 | 55.748 (55.815 ± 0.256) | 1.786× | 4,096 | 1,066,834 | 1,067,048 | 17,297 |

事件加速使计算事件减少 87.5%、控制面记录减少 77.9%，但调度轮次只减少 4.33%。这说明
Grok 的百万级轮次主要来自长逻辑时间和网络协调；墙钟收益主要来自减少计算后端事件、
序列化和控制面处理。

主动依赖单独使调度轮次减少 0.54%、grants 减少 0.65%，同时新增 1,019 次依赖更新，
最终墙钟时间反而上升。这个机制是否有效取决于联邦成员数量、阻塞稀疏度以及保守授权占比，
不能由任务或 rank 数量本身保证。

## 正确性检查

- 四组均完成 4,758 个网络事件。
- 将宏任务名称归一化为 rank/phase 边界后，网络传输语义 SHA-256 完全一致。
- 模拟 makespan 位于 1,006,360,860,972–1,006,360,868,792 ns。
- 最大跨度 7,820 ns，即 0.00777 ppm，满足 1 ppm 等价阈值。
- 精确完成时间哈希不同，来自不同时间授权顺序的纳秒级偏移。

## 默认日志首轮

保留 worker stdout 时，每个组合产生约 1.3–1.4 GiB 日志。首轮结果为 103.72、51.84、
115.98 和 59.91 s，对应事件加速 2.001 倍、双开 1.731 倍。安静模式基线只快约 4%，
且事件加速比提高到 2.088 倍，因此大量日志不是加速结论的主要来源。

默认日志结果位于 `summary_logged_smoke.json` 与 `runs_logged_smoke/`。

## 与 LULESH-64 对比

| 指标 | LULESH-64 | Grok-256 |
|---|---:|---:|
| 原始文件 | 665 MB | 8.41 GB |
| 原始 GOAL 操作 | 11,438,062 | 114,000,750 |
| 基线计算事件 | 8,192 | 32,768 |
| 基线墙钟中位数 | 13.947 s | 99.562 s |
| 事件加速比 | 1.724× | 2.088× |
| Active-dependency 单独效果 | 无稳定收益 | 慢 11.1% |

Grok-256 将原始操作规模扩大约 10 倍、仿真计算事件扩大 4 倍，并强化了事件聚合有效、
主动依赖在三成员 federation 中收益有限的结论。

## 复现

```bash
python3 experiments/atlahs_ablation/run_ablation.py \
  --goal /home/sdic/atlahs/data/ai/grok/Grok314B_N64_GPU256_TP4_PP1_CP1_VP1_EP8_ETP4_GBS512/grok.goal \
  --output experiments/atlahs_ablation/grok256 \
  --phases 16 --events-per-phase 8 --prepare-only

python3 experiments/atlahs_ablation/run_ablation.py \
  --output experiments/atlahs_ablation/grok256 \
  --run-only --repeats 3 --timeout 1800 --quiet-worker-logs
```

## 适用范围

实验使用全部原始操作累计计算时间和通信量，但按 phase/peer 聚合通信，并非 1.14 亿操作的
一对一重放。链式网络和 flow macro 也不是 Grok 训练集群的真实 Dragonfly 拓扑。当前结果
适合评价 ComposeDES 控制面机制的规模趋势；若评价绝对网络性能，需要接入原始拓扑并使用
更细粒度网络模型。

官方来源：

- https://spcl.inf.ethz.ch/Research/Scalable_Networking/ATLAHS/
- http://storage2.spcl.ethz.ch/traces/ai/grok/Grok314B_N64_GPU256_TP4_PP1_CP1_VP1_EP8_ETP4_GBS512/
