# FNCS Workflow Orchestrator

The orchestrator is the workflow control-plane federate for co-simulation v2. It owns DAG dependency state, dispatches compute and network commands, and decides global completion. FNCS broker remains a generic pub/sub and conservative-time coordinator.

## Run

Start a broker for three federates, then start ns-3 and GPUSim with their `fncs.worker.zpl` files. Start the orchestrator with:

```bash
FNCS_CONFIG_FILE=fncs/orchestrator/fncs.zpl \
COSIM_RUN_ID=my-run \
python3 fncs/orchestrator/workflow_orchestrator.py jobs.json \
  --active-dependency-coordination \
  --event-log output/control-plane.jsonl
```

The workflow input may be the legacy jobs array or an object with `tasks` and
`collectives`. In worker mode GPUSim does not read that file; each task
specification is sent in `compute.dispatch` only when its dependencies are
ready. The orchestrator currently expands ring `allreduce`, `allgather`,
`reduce_scatter`, and `alltoall` operations into point-to-point ns-3 flows and
enforces a barrier between ring steps.

`bytes_per_rank` is the rank's input buffer size. Ring AllReduce and
ReduceScatter send `ceil(bytes_per_rank / ranks)` per rank per step; AllGather
sends one rank input buffer per step; AllToAll sends
`ceil(bytes_per_rank / ranks)` to each non-local peer.

```json
{
  "tasks": [],
  "collectives": [
    {
      "collective_id": "allreduce0",
      "type": "allreduce",
      "algorithm": "ring",
      "bytes_per_rank": 120000,
      "participants": [
        {"src_task_id": "pre0", "dst_task_id": "post0", "host": "host1"},
        {"src_task_id": "pre1", "dst_task_id": "post1", "host": "host2"}
      ]
    }
  ]
}
```

Failed compute and network attempts can be retried by passing
`--max-compute-retries N` and `--max-network-retries N`. A task may override
the compute default with its own `max_retries` field. Every worker event must
carry a unique `event_id`; duplicate events and dispatches are idempotent.

## Coordination and acceleration

With `FNCS_ACTIVE_DEPENDENCY=yes`, the broker accepts atomic dependency graph
updates from the orchestrator on the reserved `__fncs/active_dependencies`
topic. A federate can advance to the transitive lower bound of only the
federates that can currently affect it; the broker falls back to conservative
global-min scheduling if no usable graph has been published. Set
`FNCS_COORDINATION_METRICS=/path/metrics.json` to record grants and rounds.

ComposeDES describes these mechanisms along two optimization axes:

- **Vertical optimization** contracts certified local event chains inside one
  federate. It reduces the number of compute events, dispatches, and grants.
- **Horizontal optimization** exposes independent timelines through federate
  partitioning and uses Active-dependency coordination to advance them without
  waiting for unrelated federates.

Federate partitioning exposes concurrency; Active-dependency exploits that
concurrency. They belong to the same horizontal axis but remain separate
architecture and scheduling decisions. A controlled Active-dependency ablation
must therefore hold the partition fixed.

The orchestrator publishes a new dependency epoch only when a worker event
changes one of three relevant states: compute in flight, network in flight, or
terminal status. The broker translates federate names to integer indexes and
builds the transitive dependency closure once per epoch. Normal scheduling
rounds reuse indexed state and scratch buffers. Invalid or cyclic graphs retain
the conservative global-minimum fallback.

The orchestrator keeps a finite idle request horizon. Requesting terminal time
while it has no current work is unsafe because a later worker completion may
cause it to dispatch another task to a federate that has already reached the
terminal grant.

An object workflow may contain an `acceleration` section. Regions are eligible
for contraction only when all four closure declarations (`causal`, `state`,
`temporal`, and `resource_non_interference`) are true, the members form a
single-host linear chain, and the region contains no network or collective
boundary. Exact regions must match the detailed duration exactly. Approximate
regions carry lower/upper duration bounds; the optimizer refines only regions
whose upper path can still affect the critical path until the requested
`error_budget_ns` is met. `--optimization-report` writes the certificate and
the selected refinements.

Backend-local chains are disabled by default. Enable them with
`--backend-local-chains` or `COSIM_BACKEND_LOCAL_CHAINS=yes`; limit one plan
with `--backend-local-chain-max-tasks N` or
`COSIM_BACKEND_LOCAL_CHAIN_MAX_TASKS` (default 32, 0 means unlimited). The
coordinator only creates a plan for a same-host, zero-network, single-entry
linear chain when every task using that host is statically ordered. Collective
participants and tasks with retries are excluded. GPUSim still executes every
original `GpuJob` in order and returns every task's original start/finish
timestamp in one boundary batch, so this option removes coordination traffic
without replacing the compute model or changing the boundary trace.

## Test

```bash
python3 -m unittest discover -s fncs/orchestrator/tests -v
tests/active_dependency_smoke/run.sh
tests/control_plane_alpha4_small/run.sh
python3 tests/backend_local_chain_gpu_compare.py
```
