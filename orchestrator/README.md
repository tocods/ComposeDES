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

The workflow input is the existing jobs array. In worker mode GPUSim does not read that file; each task specification is sent in `compute.dispatch` only when its dependencies are ready.

## Test

```bash
python3 -m unittest discover -s fncs/orchestrator/tests -v
tests/control_plane_smoke/run.sh
```
