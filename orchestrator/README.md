# FNCS Workflow Orchestrator

The orchestrator is the workflow control-plane federate for co-simulation v2. It owns DAG dependency state, dispatches compute and network commands, and decides global completion. FNCS broker remains a generic pub/sub and conservative-time coordinator.

## Run

Start a broker for three federates, then start ns-3 and GPUSim with their `fncs.worker.zpl` files. Start the orchestrator with:

```bash
FNCS_CONFIG_FILE=fncs/orchestrator/fncs.zpl \
COSIM_RUN_ID=my-run \
python3 fncs/orchestrator/workflow_orchestrator.py jobs.json \
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

## Test

```bash
python3 -m unittest discover -s fncs/orchestrator/tests -v
tests/control_plane_smoke/run.sh
tests/control_plane_collective/run.sh
```
