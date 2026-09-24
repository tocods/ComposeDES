#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
JAVA_BIN="${JAVA_BIN:-$(command -v java)}"
export LD_LIBRARY_PATH="$ROOT/fncs/builds/local/lib:$ROOT/deps/local/lib:${LD_LIBRARY_PATH:-}"
export FNCS_LIBRARY="$ROOT/fncs/builds/local/lib/libfncs.so"
CASE_DIR="$ROOT/tests/control_plane_alpha4_small"
OUTPUT_DIR="$CASE_DIR/output"
HOSTS="$ROOT/tests/control_plane_smoke/hosts.json"
TOPOLOGY="$ROOT/tests/control_plane_smoke/topology.yaml"
EMPTY="$ROOT/tests/control_plane_smoke/empty.json"

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

run_case() {
  local mode="$1"
  local port="$2"
  local workflow="$3"
  local flow_macro="${4:-no}"
  local local_chains="${5:-no}"
  local result="$OUTPUT_DIR/$mode"
  mkdir -p "$result"
  rm -f "$result"/*.log(N) "$result"/*.json(N) "$result"/*.jsonl(N) "$result"/*.xml(N)
  PIDS=()

  FNCS_ACTIVE_DEPENDENCY=yes \
  FNCS_COORDINATION_METRICS="$result/coordination.json" \
  FNCS_BROKER="tcp://localhost:$port" \
    "$ROOT/fncs/builds/local/bin/fncs_broker" 3 \
    >"$result/broker.log" 2>&1 &
  PIDS+=("$!")
  sleep 0.5

  COSIM_NS3_FLOW_MACRO="$flow_macro" \
  COSIM_NS3_TOPOLOGY=direct_p2p COSIM_NS3_HOST_COUNT=2 \
  COSIM_NS3_BANDWIDTH_GBPS=10 COSIM_NS3_DELAY_NS=10000 \
  FNCS_BROKER="tcp://localhost:$port" FNCS_NAME=ns3 FNCS_TIME_DELTA=1ns FNCS_FATAL=yes \
  FNCS_CONFIG_FILE="$ROOT/ns3/fncs.worker.zpl" \
    "$ROOT/ns3/build/src/fncs/examples/ns3-dev-fncs-example" --topo="$TOPOLOGY" \
    >"$result/ns3.log" 2>&1 &
  PIDS+=("$!")

  FNCS_CONFIG_FILE="$ROOT/GPUsim/fncs.worker.zpl" \
  FNCS_BROKER="tcp://localhost:$port" FNCS_NAME=gpusim FNCS_TIME_DELTA=1ns FNCS_FATAL=yes \
    "$JAVA_BIN" --enable-native-access=ALL-UNNAMED \
    -Djava.library.path="$ROOT/GPUsim/lib" \
    -cp "$ROOT/GPUsim/out/production/gpuworkflowsim:$ROOT/GPUsim/jars/*" \
    backend.SimEngine "$result" "$HOSTS" "$EMPTY" "$EMPTY" -1 true false 0 worker \
    >"$result/gpusim.log" 2>&1 &
  PIDS+=("$!")

  FNCS_CONFIG_FILE="$ROOT/fncs/orchestrator/fncs.zpl" \
  FNCS_BROKER="tcp://localhost:$port" FNCS_NAME=orchestrator FNCS_TIME_DELTA=1ns FNCS_FATAL=yes \
  COSIM_RUN_ID="alpha4-small-$mode" \
  COSIM_BACKEND_LOCAL_CHAINS="$local_chains" \
    python3 "$ROOT/fncs/orchestrator/workflow_orchestrator.py" "$workflow" \
      --active-dependency-coordination \
      --event-log "$result/control-plane.jsonl" \
      --optimization-report "$result/optimization.json" \
      >"$result/orchestrator.log" 2>&1

  wait "${PIDS[1]}"
  wait "${PIDS[2]}"
  wait "${PIDS[3]}"
  PIDS=()
  grep -q '"done": true' "$result/orchestrator.log"
}

mkdir -p "$OUTPUT_DIR"
run_case baseline 5591 "$CASE_DIR/workflow-baseline.json"
run_case optimized 5592 "$CASE_DIR/workflow-optimized.json"
run_case flow-macro 5593 "$CASE_DIR/workflow-baseline.json" yes
run_case refinement 5594 "$CASE_DIR/workflow-refinement.json"
run_case local-chains 5595 "$CASE_DIR/workflow-baseline.json" no yes
python3 "$CASE_DIR/verify.py" "$OUTPUT_DIR"
trap - EXIT INT TERM
