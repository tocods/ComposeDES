#!/usr/bin/env python3
"""Replay one exported predictor graph with the native ComposeDES FNCS broker."""
import argparse
from contextlib import ExitStack
from heapq import heappush, heappop
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from workflow import ORCHESTRATOR, COMPUTE_DISPATCH, longest_path_certificate
from fncs_client import FncsClient
from workflow_orchestrator import BatchEncoder, CONTROL, MAX_TIME_NS, parse_batch


def worker(run_id):
    client = FncsClient()
    client.initialize()
    pending = []
    sequence = 0
    now = 0
    encoder = BatchEncoder(run_id)
    try:
        while True:
            for topic in client.get_events():
                batch = parse_batch(client.get_value(topic), run_id)
                if topic == CONTROL:
                    if any(event['kind'] == 'control.end' for event in batch['events']):
                        if pending:
                            raise RuntimeError('simulation ended with pending tile work')
                        return
                    continue
                if topic != COMPUTE_DISPATCH:
                    raise RuntimeError(f'unexpected FNCS topic {topic}')
                for event in batch['events']:
                    payload = event['payload']
                    name = payload['task_id']
                    duration = payload['task']['estimated_duration_ns']
                    sequence += 1
                    completion = {'event_id': f'complete-{sequence}', 'kind': 'compute.completed',
                                  'correlation_id': event['correlation_id'],
                                  'payload': {'task_id': name, 'attempt': payload['attempt'], 'status': 'SUCCEEDED'}}
                    heappush(pending, (now + duration, name, completion))
            completions = []
            while pending and pending[0][0] <= now:
                completions.append(heappop(pending)[2])
            if completions:
                client.publish('compute/completed', encoder.encode(sequence, now, completions))
            # FNCS delivery granularity is 1 ns, including zero-cost nodes.
            now = client.time_request(max(now + 1, pending[0][0]) if pending else MAX_TIME_NS)
    finally:
        client.finalize()


def run(workflow, broker, library, output):
    output.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        address = f'tcp://127.0.0.1:{sock.getsockname()[1]}'
    worker_config = output / 'worker.zpl'
    worker_config.write_text(f"""name = gpusim
time_delta = 1ns
broker = {address}
values
    compute/dispatch
        topic = orchestrator/compute/dispatch
        default = ""
        type = string
        list = false
    control
        topic = orchestrator/control
        default = ""
        type = string
        list = false
""")
    run_id = 'megakernel-federation'
    env = dict(os.environ, FNCS_BROKER=address, FNCS_LIBRARY=str(library), FNCS_FATAL='yes',
               FNCS_TIME_DELTA='1ns', COSIM_ACTIVE_DEPENDENCIES='no', COSIM_BACKEND_LOCAL_CHAINS='no')
    official = ORCHESTRATOR / 'workflow_orchestrator.py'
    events = output / 'control.jsonl'
    with ExitStack() as stack:
        def start(name, args, **extra):
            log = stack.enter_context((output / f'{name}.log').open('w'))
            proc = subprocess.Popen(args, env=dict(env, **extra), stdout=log, stderr=subprocess.STDOUT)
            stack.callback(stop, proc)
            return proc

        processes = [start('broker', [str(broker), '2'])]
        processes.append(start('worker', [sys.executable, __file__, 'worker', '--run-id', run_id],
                               FNCS_NAME='gpusim', FNCS_CONFIG_FILE=str(worker_config)))
        processes.append(start('orchestrator', [sys.executable, str(official), str(workflow), '--run-id', run_id,
                                                '--event-log', str(events), '--idle-grant-ns', '100'],
                               FNCS_NAME='orchestrator', FNCS_CONFIG_FILE=str(ORCHESTRATOR / 'fncs.zpl')))
        deadline = time.monotonic() + 30
        for process in reversed(processes):
            if process.wait(timeout=max(.1, deadline - time.monotonic())):
                raise RuntimeError(f'FNCS process failed, see {output}')
    report = json.loads((output / 'orchestrator.log').read_text().splitlines()[-1])
    if not report['done']:
        raise RuntimeError(f'FNCS orchestrator did not finish: {report}')
    final = json.loads(events.read_text().splitlines()[-1])
    theoretical = longest_path_certificate(json.loads(workflow.read_text())['tasks'])['lower_ns']
    completion = final['logical_time_ns']
    if completion < theoretical:
        raise RuntimeError('broker replay completed before the service-time critical path')
    result = dict(tasks=report['task_count'], critical_path_ns=theoretical,
                  broker_completion_ns=completion, broker_delta_ns=completion - theoretical,
                  final_event=final)
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


def stop(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest='command', required=True)
    child = commands.add_parser('worker')
    child.add_argument('--run-id', required=True)
    parent = commands.add_parser('run')
    for name in ('workflow', 'broker', 'library', 'output'):
        parent.add_argument('--' + name, type=Path, required=True)
    args = cli.parse_args()
    if args.command == 'worker':
        worker(args.run_id)
    else:
        run(args.workflow.resolve(), args.broker.resolve(), args.library.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
