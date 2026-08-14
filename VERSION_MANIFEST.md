# 协同仿真组件版本清单

更新日期：2026-08-14

本文件记录 DAG 控制面重构前的组件来源和本地基线。源码 commit 以各子仓库为准；大型 trace、模型输出、构建产物和运行日志不进入 Git。

| 组件 | 上游基点 | 重构前基线 | v2 alpha.3 | 重构分支 | 职责 |
|---|---|---|---|---|---|
| FNCS | `790ec3de` | `d0896f2` | `0593a82` | `cosim/dag-control-plane-v2` | broker、FNCS client library、workflow orchestrator |
| GPUSim | `bc84bdd6` | `0b4f104` | `55476aa` | `cosim/dag-control-plane-v2` | 计算仿真 worker |
| ns-3 | `78b53d4b3` | `c4cead46f` | `72934a130` | `cosim/dag-control-plane-v2` | 网络仿真 worker |
| NeuSight | `6945927d` | 未提交本地实验改动 | 不参与运行时改造 | 保持当前分支 | 算子时间预测和输入生成 |
| ATLAHS | `fb51a99f` | 上游 commit | 不参与运行时改造 | 保持当前分支 | validation trace 与 LogGOPSim 基准 |

## 基线规则

- 核心三个仓库的首个重构提交只冻结当前协同通信源码，不混入 DAG 控制面实现。
- tag `cosim-v1-pre-orchestrator` 分别指向 FNCS `d0896f2`、GPUSim `0b4f104`、ns-3 `c4cead46f`。
- tag `v2.0.0-alpha.3` 分别指向 FNCS `0593a82`、GPUSim `55476aa`、ns-3 `72934a130`。
- 后续提交按组件独立演进，禁止跨仓库使用同一个模糊提交说明。
- 顶层 integration 仓库记录确切 commit 组合；端到端结果必须同时记录 manifest 版本。
- 不提交 `build/`、`out/`、`.dylib`、`.so`、trace `.bin/.goal`、运行 XML/log 和 NeuSight 预测输出。

## 当前可复现基线

- FNCS：本机 arm64 构建，broker 位于 `fncs/builds/local/bin/fncs_broker`。
- GPUSim：OpenJDK + JNI FNCS，本地 class 输出位于 `GPUsim/out/production/gpuworkflowsim`。
- ns-3：本机 arm64 构建，FNCS example 位于 `ns3/build/src/fncs/examples/ns3-dev-fncs-example`。
- 4-rank 聚合 case：14/14 计算任务完成，13 个网络事件，协同仿真约 1.129s。
- 对照 ATLAHS LogGOPSim：约 2.120s。当前 46.8% 差异来自 trace 聚合和通信量简化，不能作为准确度通过结论。

## v2 控制面验证

- 命令：`tests/control_plane_smoke/run.sh`
- 拓扑：2 hosts、10Gbps p2p、10us propagation delay。
- DAG：`producer(host1) -> 120KB transfer -> consumer(host2)`。
- 结果：producer 100001ns 完成；两个 60KB 数据报分别在 246770ns、295514ns 被 ns-3 目标应用收到；consumer 395516ns 完成。
- 控制面守恒：2 个 `SUCCEEDED`，`inflight_compute=0`，`inflight_network=0`。
- 协议依赖：ns-3 vendored `nlohmann/json` v3.11.3。

## 版本发布约定

- `v2.0.0-alpha.1`：orchestrator + fake workers 完成状态机测试（已完成）。
- `v2.0.0-alpha.2`：GPUSim worker 接入，legacy/worker 双模式（已完成）。
- `v2.0.0-alpha.3`：ns-3 worker 接入，worker 路径移除 synthetic completion（已完成）。
- `v2.0.0-rc.1`：端到端控制权切换，完成因果与守恒测试。
- `v2.0.0`：删除正式路径中的 GPUSim DAG 控制逻辑，实验脚本默认使用 orchestrator。
