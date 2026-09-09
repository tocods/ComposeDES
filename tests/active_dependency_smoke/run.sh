#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CASE_DIR="$ROOT/tests/active_dependency_smoke"
OUTPUT_DIR="$CASE_DIR/output"
BROKER="$ROOT/fncs/fncs_broker"
LIBRARY="$ROOT/fncs/builds/local/lib/libfncs.dylib"

mkdir -p "$OUTPUT_DIR"

run_mode() {
  local mode="$1"
  local port="$2"
  local active="$3"
  local result="$OUTPUT_DIR/$mode"
  mkdir -p "$result"
  rm -f "$result"/*.json(N) "$result"/*.log(N)

  FNCS_ACTIVE_DEPENDENCY="$active" \
  FNCS_COORDINATION_METRICS="$result/metrics.json" \
  FNCS_BROKER="tcp://localhost:$port" \
    "$BROKER" 3 >"$result/broker.log" 2>&1 &
  local broker_pid="$!"
  sleep 0.5

  FNCS_CONFIG_FILE="$CASE_DIR/federate.zpl" FNCS_BROKER="tcp://localhost:$port" \
  FNCS_NAME=worker_a FNCS_TIME_DELTA=1ns FNCS_FATAL=yes FNCS_LIBRARY="$LIBRARY" \
    python3 "$CASE_DIR/probe.py" --schedule 10,20,30,40,50,60,70,80,90,100 \
      --output "$result/worker_a.json" >"$result/worker_a.log" 2>&1 &
  local worker_a_pid="$!"

  FNCS_CONFIG_FILE="$CASE_DIR/federate.zpl" FNCS_BROKER="tcp://localhost:$port" \
  FNCS_NAME=worker_b FNCS_TIME_DELTA=1ns FNCS_FATAL=yes FNCS_LIBRARY="$LIBRARY" \
    python3 "$CASE_DIR/probe.py" --schedule 15,30,45,60,75,90 \
      --output "$result/worker_b.json" >"$result/worker_b.log" 2>&1 &
  local worker_b_pid="$!"

  FNCS_CONFIG_FILE="$CASE_DIR/federate.zpl" FNCS_BROKER="tcp://localhost:$port" \
  FNCS_NAME=orchestrator FNCS_TIME_DELTA=1ns FNCS_FATAL=yes FNCS_LIBRARY="$LIBRARY" \
    python3 "$CASE_DIR/probe.py" --schedule 25,50,75,100 \
      --publish-dependencies --output "$result/orchestrator.json" \
      >"$result/orchestrator.log" 2>&1 &
  local orchestrator_pid="$!"

  wait "$worker_a_pid"
  wait "$worker_b_pid"
  wait "$orchestrator_pid"
  wait "$broker_pid"
}

run_mode global 5581 no
run_mode active 5582 yes

MESSAGE_DIR="$OUTPUT_DIR/message"
mkdir -p "$MESSAGE_DIR"
rm -f "$MESSAGE_DIR"/*.json(N) "$MESSAGE_DIR"/*.log(N)
FNCS_ACTIVE_DEPENDENCY=yes \
FNCS_COORDINATION_METRICS="$MESSAGE_DIR/metrics.json" \
FNCS_BROKER=tcp://localhost:5583 \
  "$BROKER" 2 >"$MESSAGE_DIR/broker.log" 2>&1 &
message_broker_pid="$!"
sleep 0.5
FNCS_CONFIG_FILE="$CASE_DIR/subscriber.zpl" FNCS_BROKER=tcp://localhost:5583 \
FNCS_NAME=consumer FNCS_TIME_DELTA=1ns FNCS_FATAL=yes FNCS_LIBRARY="$LIBRARY" \
  python3 "$CASE_DIR/subscriber.py" "$MESSAGE_DIR/consumer.json" \
  >"$MESSAGE_DIR/consumer.log" 2>&1 &
consumer_pid="$!"
FNCS_CONFIG_FILE="$CASE_DIR/publisher.zpl" FNCS_BROKER=tcp://localhost:5583 \
FNCS_NAME=orchestrator FNCS_TIME_DELTA=1ns FNCS_FATAL=yes FNCS_LIBRARY="$LIBRARY" \
  python3 "$CASE_DIR/publisher.py" "$MESSAGE_DIR/orchestrator.json" \
  >"$MESSAGE_DIR/orchestrator.log" 2>&1 &
publisher_pid="$!"
wait "$consumer_pid"
wait "$publisher_pid"
wait "$message_broker_pid"

python3 "$CASE_DIR/verify.py" "$OUTPUT_DIR"
