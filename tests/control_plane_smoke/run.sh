#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
JAVA_BIN="${JAVA_BIN:-$(command -v java)}"
export LD_LIBRARY_PATH="$ROOT/fncs/builds/local/lib:$ROOT/deps/local/lib:${LD_LIBRARY_PATH:-}"
export FNCS_LIBRARY="$ROOT/fncs/builds/local/lib/libfncs.so"
CASE_DIR="$ROOT/tests/control_plane_smoke"
OUTPUT_DIR="$CASE_DIR/output"
RUN_ID="control-plane-smoke-v2"

mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_DIR"/*.log(N) "$OUTPUT_DIR"/*.jsonl(N) "$OUTPUT_DIR"/*.xml(N)

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

FNCS_BROKER=tcp://localhost:5570 \
  "$ROOT/fncs/builds/local/bin/fncs_broker" 3 \
  >"$OUTPUT_DIR/broker.log" 2>&1 &
PIDS+=("$!")
sleep 1

FNCS_CONFIG_FILE="$ROOT/ns3/fncs.worker.zpl" \
FNCS_BROKER=tcp://localhost:5570 \
FNCS_NAME=ns3 FNCS_TIME_DELTA=1ns FNCS_FATAL=yes \
  "$ROOT/ns3/build/src/fncs/examples/ns3-dev-fncs-example" \
  --topo="$CASE_DIR/topology.yaml" \
  >"$OUTPUT_DIR/ns3.log" 2>&1 &
PIDS+=("$!")

FNCS_CONFIG_FILE="$ROOT/GPUsim/fncs.worker.zpl" \
FNCS_BROKER=tcp://localhost:5570 \
FNCS_NAME=gpusim FNCS_TIME_DELTA=1ns FNCS_FATAL=yes \
  "$JAVA_BIN" --enable-native-access=ALL-UNNAMED \
  -Djava.library.path="$ROOT/GPUsim/lib" \
  -cp "$ROOT/GPUsim/out/production/gpuworkflowsim:$ROOT/GPUsim/jars/*" \
  backend.SimEngine "$OUTPUT_DIR" "$CASE_DIR/hosts.json" \
  "$CASE_DIR/empty.json" "$CASE_DIR/empty.json" -1 true false 0 worker \
  >"$OUTPUT_DIR/gpusim.stdout.log" 2>&1 &
PIDS+=("$!")

FNCS_CONFIG_FILE="$ROOT/fncs/orchestrator/fncs.zpl" \
FNCS_BROKER=tcp://localhost:5570 \
FNCS_NAME=orchestrator FNCS_TIME_DELTA=1ns FNCS_FATAL=yes \
COSIM_RUN_ID="$RUN_ID" \
  python3 "$ROOT/fncs/orchestrator/workflow_orchestrator.py" \
  "$CASE_DIR/workflow.json" \
  --event-log "$OUTPUT_DIR/control-plane.jsonl" \
  >"$OUTPUT_DIR/orchestrator.log" 2>&1

wait "${PIDS[1]}"
wait "${PIDS[2]}"
wait "${PIDS[3]}" 2>/dev/null || true
trap - EXIT INT TERM

grep -q '"done": true' "$OUTPUT_DIR/orchestrator.log"
grep -q '"task_count": 2' "$OUTPUT_DIR/orchestrator.log"
grep -q 'network.completed' "$OUTPUT_DIR/control-plane.jsonl"
echo "control-plane smoke test passed"
