# 协同仿真组件版本清单

更新日期：2026-09-09

本文件记录 DAG 控制面重构前的组件来源和本地基线。源码 commit 以各子仓库为准；大型 trace、模型输出、构建产物和运行日志不进入 Git。

| 组件 | 上游基点 | 重构前基线 | v2 alpha.3 | v2 alpha.4 | v2 alpha.4.1 | 重构分支 | 职责 |
|---|---|---|---|---|---|---|---|
| FNCS | `790ec3de` | `d0896f2` | `0593a82` | `3c32900` | `e3e0d95` | `cosim/dag-control-plane-v2` | broker、FNCS client library、workflow orchestrator |
| GPUSim | `bc84bdd6` | `0b4f104` | `55476aa` | `1601a6c` | `25081e7` | `cosim/dag-control-plane-v2` | 计算仿真 worker |
| ns-3 | `78b53d4b3` | `c4cead46f` | `72934a130` | `4f42d2468` | `2f049f0d4` | `cosim/dag-control-plane-v2` | 网络仿真 worker |
| NeuSight | `6945927d` | 未提交本地实验改动 | 不参与运行时改造 | 不参与运行时改造 | 不参与运行时改造 | 保持当前分支 | 算子时间预测和输入生成 |
| ATLAHS | `fb51a99f` | 上游 commit | 不参与运行时改造 | 不参与运行时改造 | 不参与运行时改造 | 保持当前分支 | validation trace 与 LogGOPSim 基准 |

## 基线规则

- 核心三个仓库的首个重构提交只冻结当前协同通信源码，不混入 DAG 控制面实现。
- tag `cosim-v1-pre-orchestrator` 分别指向 FNCS `d0896f2`、GPUSim `0b4f104`、ns-3 `c4cead46f`。
- tag `v2.0.0-alpha.3` 分别指向 FNCS `0593a82`、GPUSim `55476aa`、ns-3 `72934a130`。
- tag `v2.0.0-alpha.4` 分别指向 FNCS `3c32900`、GPUSim `1601a6c`、ns-3 `4f42d2468`。
- tag `v2.0.0-alpha.4.1` 分别指向 FNCS `e3e0d95`、GPUSim `25081e7`、ns-3 `2f049f0d4`。
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

## v2 集合通信验证

- 命令：`tests/control_plane_collective/run.sh`。
- 工作流：两个根计算任务、2-rank Ring AllReduce、两个后继计算任务。
- AllReduce 输入：每 rank 120KB；中间件展开为 2 个 ring step，每步 2 条 60KB ns-3 flow。
- 根计算均在 10001ns 完成；step 0 流在 78418ns、156770ns 完成；step 1 流在 287574ns、305676ns 完成。
- 两个后继任务在最终 barrier 后的 305677ns 才下发，并在 315678ns 完成。
- 控制面守恒：4 个 `SUCCEEDED`、4 个 network attempt、0 个在途计算、0 个在途网络。
- `v2.0.0-alpha.4` tag 仍使用 60KB UDP 分段的精确包级路径；以下机制扩展在后续补丁版本中加入。

## alpha.4.1 机制扩展

- FNCS broker 支持由 orchestrator 原子发布的动态主动依赖图，并保留全局最小值回退路径和因果越界检查。
- orchestrator 支持带四类闭包条件的精确区域收缩，以及根据关键路径上下界和误差预算进行选择性细化。
- ns-3 worker 新增可选 store-and-forward flow 宏事件。模型按 chain、star、dual-switch 或 direct-p2p 路径逐跳预留有向链路，显式计算排队、序列化和传播时延；默认 packet 路径继续保留。
- 大规模 case study 使用 flow 宏事件，packet 模式只用于小型协议栈校准；二者属于不同精度层级，结果不应直接宣称等价。
- 小型回归：主动依赖 rounds 由 14 降为 10；精确收缩将 compute dispatch 从 3 降为 2 且 makespan 不变；选择性细化只展开关键区域并将证书 gap 收敛到 0；单流宏模型与理论值同为 146000ns。
- 全量实验与理论校验：`python3 run_case_study.py 1 2 3 4` 和 `python3 case_study_results/validate_alpha4_results.py`。

## 版本发布约定

- `v2.0.0-alpha.1`：orchestrator + fake workers 完成状态机测试（已完成）。
- `v2.0.0-alpha.2`：GPUSim worker 接入，legacy/worker 双模式（已完成）。
- `v2.0.0-alpha.3`：ns-3 worker 接入，worker 路径移除 synthetic completion（已完成）。
- `v2.0.0-alpha.4`：中间件 Ring collective、失败重试、跨组件幂等与批量回传（已完成）。
- `v2.0.0-rc.1`：端到端控制权切换，完成因果与守恒测试。
- `v2.0.0`：删除正式路径中的 GPUSim DAG 控制逻辑，实验脚本默认使用 orchestrator。
