import copy
import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))
SPEC = importlib.util.spec_from_file_location(
    "workflow_orchestrator", MODULE_DIR / "workflow_orchestrator.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class WorkflowControllerTest(unittest.TestCase):
    event_sequence = 0

    def setUp(self):
        self.tasks = [
            {
                "name": "a",
                "host": "host1",
                "cpu_task": {"pes_number": 1, "length": 10, "ram": 0},
                "gpu_task": {"kernels": []},
                "children": [
                    {
                        "child": "b",
                        "size": 1024,
                        "src_host": "host1",
                        "dst_host": "host2",
                    },
                    {
                        "child": "c",
                        "size": 0,
                        "src_host": "host1",
                        "dst_host": "host1",
                    },
                ],
            },
            {
                "name": "b",
                "host": "host2",
                "cpu_task": {"pes_number": 1, "length": 10, "ram": 0},
                "gpu_task": {"kernels": []},
                "children": [{"child": "d", "size": 0}],
            },
            {
                "name": "c",
                "host": "host1",
                "cpu_task": {"pes_number": 1, "length": 10, "ram": 0},
                "gpu_task": {"kernels": []},
                "children": [{"child": "d", "size": 0}],
            },
            {
                "name": "d",
                "host": "host2",
                "cpu_task": {"pes_number": 1, "length": 10, "ram": 0},
                "gpu_task": {"kernels": []},
                "children": [],
            },
        ]

    @staticmethod
    def completed(task_id, attempt=1):
        WorkflowControllerTest.event_sequence += 1
        return {
            "event_id": f"compute-result-{WorkflowControllerTest.event_sequence}",
            "kind": "compute.completed",
            "correlation_id": f"task:{task_id}:attempt:{attempt}",
            "payload": {"task_id": task_id, "attempt": attempt, "status": "SUCCEEDED"},
        }

    @classmethod
    def network_completed(cls, transfer_id, status="SUCCEEDED", event_id=None):
        cls.event_sequence += 1
        return {
            "event_id": event_id or f"network-result-{cls.event_sequence}",
            "kind": "network.completed",
            "correlation_id": transfer_id,
            "payload": {"transfer_id": transfer_id, "status": status},
        }

    def test_diamond_with_network_edge(self):
        controller = MODULE.WorkflowController(self.tasks, "test-run")
        commands = controller.initial_commands()
        self.assertEqual(["a"], [c.event["payload"]["task_id"] for c in commands])

        commands = controller.handle_event(MODULE.COMPUTE_COMPLETED, self.completed("a"))
        network = [c for c in commands if c.topic == MODULE.NETWORK_DISPATCH]
        compute = [c for c in commands if c.topic == MODULE.COMPUTE_DISPATCH]
        self.assertEqual(1, len(network))
        self.assertEqual(["c"], [c.event["payload"]["task_id"] for c in compute])

        transfer_id = network[0].event["payload"]["transfer_id"]
        commands = controller.handle_event(
            MODULE.NETWORK_COMPLETED,
            self.network_completed(transfer_id),
        )
        self.assertEqual(["b"], [c.event["payload"]["task_id"] for c in commands])

        self.assertEqual([], controller.handle_event(MODULE.COMPUTE_COMPLETED, self.completed("b")))
        commands = controller.handle_event(MODULE.COMPUTE_COMPLETED, self.completed("c"))
        self.assertEqual(["d"], [c.event["payload"]["task_id"] for c in commands])
        controller.handle_event(MODULE.COMPUTE_COMPLETED, self.completed("d"))
        self.assertTrue(controller.done)

    def test_batch_encoder_preserves_wire_schema(self):
        events = [
            {
                "event_id": "evt-1",
                "kind": "compute.dispatch",
                "correlation_id": "task:a:attempt:1",
                "payload": {"task_id": "a", "attempt": 1},
            }
        ]
        encoded = MODULE.BatchEncoder("run/quoted").encode(7, 1234, events, 2)
        self.assertEqual(
            MODULE.make_batch("run/quoted", 7, 1234, events, 2), encoded
        )
        self.assertEqual(
            {
                "schema_version": MODULE.SCHEMA_VERSION,
                "run_id": "run/quoted",
                "batch_id": "batch-000000007",
                "logical_time_ns": 1234,
                "microstep": 2,
                "events": events,
            },
            MODULE.parse_batch(encoded, "run/quoted"),
        )

    def test_dispatch_task_payload_is_precompiled_without_dag_children(self):
        controller = MODULE.WorkflowController(self.tasks, "test-run")
        commands = controller.initial_commands()
        dispatched = commands[0].event["payload"]["task"]
        self.assertEqual([], dispatched["children"])
        self.assertEqual(2, len(controller.tasks["a"]["children"]))
        self.assertIs(dispatched, controller.dispatch_task_specs["a"])

    def test_duplicate_completion_is_idempotent(self):
        task = dict(self.tasks[0])
        task["children"] = []
        controller = MODULE.WorkflowController([task], "test-run")
        controller.initial_commands()
        result = self.completed("a")
        controller.handle_event(MODULE.COMPUTE_COMPLETED, result)
        self.assertEqual([], controller.handle_event(MODULE.COMPUTE_COMPLETED, result))

    def test_compute_failure_retries_with_new_attempt(self):
        task = dict(self.tasks[0])
        task["children"] = []
        controller = MODULE.WorkflowController([task], "test-run", max_compute_retries=1)
        controller.initial_commands()
        failed = self.completed("a")
        failed["payload"]["status"] = "FAILED"
        commands = controller.handle_event(MODULE.COMPUTE_COMPLETED, failed)
        self.assertEqual(2, commands[0].event["payload"]["attempt"])
        controller.handle_event(MODULE.COMPUTE_COMPLETED, self.completed("a", attempt=2))
        self.assertTrue(controller.done)

    def test_network_failure_retries_without_releasing_child(self):
        task_a = dict(self.tasks[0])
        task_a["children"] = [dict(self.tasks[0]["children"][0])]
        task_b = dict(self.tasks[1])
        task_b["children"] = []
        controller = MODULE.WorkflowController(
            [task_a, task_b], "test-run", max_network_retries=1
        )
        controller.initial_commands()
        commands = controller.handle_event(MODULE.COMPUTE_COMPLETED, self.completed("a"))
        network = [command for command in commands if command.topic == MODULE.NETWORK_DISPATCH]
        first_id = network[0].event["payload"]["transfer_id"]
        retry = controller.handle_event(
            MODULE.NETWORK_COMPLETED, self.network_completed(first_id, "FAILED")
        )
        self.assertEqual(1, len(retry))
        second_id = retry[0].event["payload"]["transfer_id"]
        self.assertNotEqual(first_id, second_id)
        self.assertEqual("BLOCKED", controller.state["b"])
        commands = controller.handle_event(
            MODULE.NETWORK_COMPLETED, self.network_completed(second_id)
        )
        self.assertEqual(["b"], [c.event["payload"]["task_id"] for c in commands])

    def test_two_rank_ring_allreduce_is_a_barrier(self):
        tasks = [
            {"name": "pre0", "host": "host1", "children": []},
            {"name": "pre1", "host": "host2", "children": []},
            {"name": "post0", "host": "host1", "children": []},
            {"name": "post1", "host": "host2", "children": []},
        ]
        collectives = [
            {
                "collective_id": "ar0",
                "type": "allreduce",
                "algorithm": "ring",
                "bytes_per_rank": 120000,
                "participants": [
                    {"src_task_id": "pre0", "dst_task_id": "post0", "host": "host1"},
                    {"src_task_id": "pre1", "dst_task_id": "post1", "host": "host2"},
                ],
            }
        ]
        controller = MODULE.WorkflowController(
            tasks, "test-run", collective_specs=collectives
        )
        initial = controller.initial_commands()
        self.assertEqual(
            ["pre0", "pre1"],
            [command.event["payload"]["task_id"] for command in initial],
        )
        self.assertEqual(
            [], controller.handle_event(MODULE.COMPUTE_COMPLETED, self.completed("pre0"))
        )
        first_step = controller.handle_event(
            MODULE.COMPUTE_COMPLETED, self.completed("pre1")
        )
        self.assertEqual(2, len(first_step))
        self.assertTrue(all(c.event["payload"]["bytes"] == 60000 for c in first_step))
        self.assertTrue(all(c.event["payload"]["step"] == 0 for c in first_step))

        self.assertEqual(
            [],
            controller.handle_event(
                MODULE.NETWORK_COMPLETED,
                self.network_completed(first_step[0].event["payload"]["transfer_id"]),
            ),
        )
        second_step = controller.handle_event(
            MODULE.NETWORK_COMPLETED,
            self.network_completed(first_step[1].event["payload"]["transfer_id"]),
        )
        self.assertEqual(2, len(second_step))
        self.assertTrue(all(c.event["payload"]["step"] == 1 for c in second_step))

        controller.handle_event(
            MODULE.NETWORK_COMPLETED,
            self.network_completed(second_step[0].event["payload"]["transfer_id"]),
        )
        post = controller.handle_event(
            MODULE.NETWORK_COMPLETED,
            self.network_completed(second_step[1].event["payload"]["transfer_id"]),
        )
        self.assertEqual(
            ["post0", "post1"],
            [command.event["payload"]["task_id"] for command in post],
        )
        self.assertEqual("COMPLETED", controller.collective_state["ar0"])

    def test_collective_cycle_is_rejected(self):
        tasks = [
            {"name": "a", "host": "host1", "children": []},
            {"name": "b", "host": "host2", "children": []},
        ]
        collective = {
            "id": "bad",
            "type": "allreduce",
            "bytes": 1,
            "participants": [
                {"src_task_id": "a", "dst_task_id": "a", "host": "host1"},
                {"src_task_id": "b", "dst_task_id": "b", "host": "host2"},
            ],
        }
        with self.assertRaises(MODULE.WorkflowError):
            MODULE.WorkflowController(tasks, "test-run", collective_specs=[collective])

    def test_collective_flow_sizes_match_ring_algorithms(self):
        tasks = [
            {"name": f"pre{rank}", "host": f"host{rank}", "children": []}
            for rank in range(4)
        ] + [
            {"name": f"post{rank}", "host": f"host{rank}", "children": []}
            for rank in range(4)
        ]
        participants = [
            {
                "src_task_id": f"pre{rank}",
                "dst_task_id": f"post{rank}",
                "host": f"host{rank}",
            }
            for rank in range(4)
        ]
        expected = {
            "allreduce": (6, 250),
            "reduce_scatter": (3, 250),
            "allgather": (3, 1000),
            "alltoall": (3, 250),
        }
        for collective_type, (step_count, bytes_per_flow) in expected.items():
            with self.subTest(collective_type=collective_type):
                controller = MODULE.WorkflowController(
                    tasks,
                    "test-run",
                    collective_specs=[
                        {
                            "id": "collective0",
                            "type": collective_type,
                            "bytes_per_rank": 1000,
                            "participants": participants,
                        }
                    ],
                )
                controller.initial_commands()
                commands = []
                for rank in range(4):
                    commands = controller.handle_event(
                        MODULE.COMPUTE_COMPLETED, self.completed(f"pre{rank}")
                    )
                self.assertEqual(step_count, controller._collective_step_count("collective0"))
                self.assertEqual(4, len(commands))
                self.assertTrue(
                    all(command.event["payload"]["bytes"] == bytes_per_flow for command in commands)
                )

    def test_cycle_is_rejected(self):
        tasks = [
            {"name": "a", "children": [{"child": "b", "size": 0}]},
            {"name": "b", "children": [{"child": "a", "size": 0}]},
        ]
        with self.assertRaises(MODULE.WorkflowError):
            MODULE.WorkflowController(tasks, "test-run")

    def test_batch_round_trip(self):
        value = MODULE.make_batch(
            "test-run",
            1,
            42,
            [
                {
                    "event_id": "e1",
                    "kind": "compute.dispatch",
                    "correlation_id": "task:a:attempt:1",
                    "payload": {},
                }
            ],
        )
        parsed = MODULE.parse_batch(value, "test-run")
        self.assertEqual(42, parsed["logical_time_ns"])
        self.assertEqual(0, parsed["microstep"])
        self.assertEqual("e1", parsed["events"][0]["event_id"])

    def test_batch_preserves_microstep(self):
        event = {
            "event_id": "e1",
            "kind": "compute.dispatch",
            "correlation_id": "task:a:attempt:1",
            "payload": {},
        }
        parsed = MODULE.parse_batch(
            MODULE.make_batch("test-run", 1, 7, [event], microstep=3),
            "test-run",
        )
        self.assertEqual(3, parsed["microstep"])

    def test_active_dependencies_follow_inflight_owners(self):
        task = copy.deepcopy(self.tasks[0])
        task["children"] = []
        controller = MODULE.WorkflowController([task], "test-run")
        controller.initial_commands()
        self.assertEqual(
            (("gpusim",), False, True), controller.simulator_dependency_state()
        )
        self.assertEqual(
            {
                "orchestrator": {"gpusim"},
                "gpusim": {"orchestrator"},
                "ns3": {"orchestrator"},
            },
            controller.simulator_dependencies(),
        )
        controller.handle_event(MODULE.COMPUTE_COMPLETED, self.completed("a"))
        self.assertEqual(((), False, False), controller.simulator_dependency_state())
        self.assertEqual(
            {"orchestrator": set(), "gpusim": set(), "ns3": set()},
            controller.simulator_dependencies(),
        )

    def test_partitioned_compute_dependencies_follow_active_hosts(self):
        tasks = [
            {
                "name": "left",
                "host": "host1",
                "cpu_task": {"pes_number": 1, "length": 10, "ram": 0},
                "gpu_task": {"kernels": []},
                "children": [],
            },
            {
                "name": "right",
                "host": "host2",
                "cpu_task": {"pes_number": 1, "length": 10, "ram": 0},
                "gpu_task": {"kernels": []},
                "children": [],
            },
        ]
        controller = MODULE.WorkflowController(tasks, "partitioned")
        controller.initial_commands()
        workers = {"host1": "gpusim-rank-000", "host2": "gpusim-rank-001"}
        self.assertEqual(
            (("gpusim-rank-000", "gpusim-rank-001"), False, True),
            controller.simulator_dependency_state(workers),
        )
        self.assertEqual(
            {
                "orchestrator": {"gpusim-rank-000", "gpusim-rank-001"},
                "gpusim-rank-000": {"orchestrator"},
                "gpusim-rank-001": {"orchestrator"},
                "ns3": {"orchestrator"},
            },
            controller.simulator_dependencies(compute_worker_by_host=workers),
        )

    def test_backend_local_chain_dispatch_preserves_task_completions(self):
        tasks = [
            {
                "name": "a",
                "host": "host1",
                "cpu_task": {"pes_number": 1, "length": 10, "ram": 0},
                "gpu_task": {"kernels": []},
                "children": [{"child": "b", "size": 0}],
            },
            {
                "name": "b",
                "host": "host1",
                "cpu_task": {"pes_number": 1, "length": 20, "ram": 0},
                "gpu_task": {"kernels": []},
                "children": [],
            },
        ]
        controller = MODULE.WorkflowController(
            tasks, "test-run", backend_local_chains=True
        )
        commands = controller.initial_commands()
        self.assertEqual(1, len(commands))
        self.assertEqual("compute.plan.dispatch", commands[0].event["kind"])
        self.assertEqual(
            ["a", "b"],
            [item["task_id"] for item in commands[0].event["payload"]["tasks"]],
        )
        self.assertEqual([], controller.handle_event(
            MODULE.COMPUTE_COMPLETED, self.completed("a")
        ))
        self.assertEqual("DISPATCHED", controller.state["b"])
        self.assertEqual([], controller.handle_event(
            MODULE.COMPUTE_COMPLETED, self.completed("b")
        ))
        self.assertTrue(controller.done)
        self.assertEqual(1, controller.summary()["backend_local_chains"]["dispatches"])

    def test_backend_local_chain_rejects_possible_host_interleaving(self):
        tasks = [
            {
                "name": name,
                "host": "host1",
                "cpu_task": {"pes_number": 1, "length": 10, "ram": 0},
                "gpu_task": {"kernels": []},
                "children": [],
            }
            for name in ("independent-a", "independent-b")
        ]
        controller = MODULE.WorkflowController(
            tasks, "test-run", backend_local_chains=True
        )
        commands = controller.initial_commands()
        self.assertEqual(2, len(commands))
        self.assertTrue(all(command.event["kind"] == "compute.dispatch" for command in commands))
        self.assertEqual({}, controller.local_plans)

    def test_dependency_update_is_deterministic_and_atomic(self):
        encoded = MODULE.encode_dependency_update(
            9,
            {
                "ns3": {"orchestrator"},
                "orchestrator": {"ns3", "gpusim"},
                "gpusim": {"orchestrator"},
            },
        )
        self.assertEqual(
            "epoch=9\n"
            "gpusim=orchestrator\n"
            "ns3=orchestrator\n"
            "orchestrator=gpusim,ns3",
            encoded,
        )

        bounded = MODULE.encode_dependency_update(
            10,
            {"orchestrator": {"worker"}, "worker": {"orchestrator"}},
            {"worker": 123456},
        )
        self.assertEqual(
            "epoch=10\n"
            "orchestrator=worker\n"
            "worker=orchestrator\n"
            "frontier.worker=123456",
            bounded,
        )

    def test_command_completion_lower_bounds_use_certificates(self):
        compute = MODULE.Command(
            MODULE.COMPUTE_DISPATCH,
            {
                "kind": "compute.dispatch",
                "payload": {
                    "task_id": "task-a",
                    "task": {"duration_bounds_ns": [50, 80]},
                },
            },
        )
        self.assertEqual(
            {"compute:task-a": 150},
            MODULE.command_completion_lower_bounds(compute, 100),
        )
        self.assertEqual(
            {"compute:task-a": 140},
            MODULE.command_completion_lower_bounds(compute, 100, 10),
        )

        network = MODULE.Command(
            MODULE.NETWORK_DISPATCH,
            {
                "kind": "network.dispatch",
                "payload": {
                    "transfer_id": "transfer-a",
                    "latency_bounds_ns": [20, 40],
                },
            },
        )
        self.assertEqual(
            {"network:transfer-a": 120},
            MODULE.command_completion_lower_bounds(network, 100),
        )


if __name__ == "__main__":
    unittest.main()
