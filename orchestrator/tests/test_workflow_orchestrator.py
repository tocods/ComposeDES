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
        return {
            "kind": "compute.completed",
            "payload": {"task_id": task_id, "attempt": attempt, "status": "SUCCEEDED"},
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
            {
                "kind": "network.completed",
                "payload": {"transfer_id": transfer_id, "status": "SUCCEEDED"},
            },
        )
        self.assertEqual(["b"], [c.event["payload"]["task_id"] for c in commands])

        self.assertEqual([], controller.handle_event(MODULE.COMPUTE_COMPLETED, self.completed("b")))
        commands = controller.handle_event(MODULE.COMPUTE_COMPLETED, self.completed("c"))
        self.assertEqual(["d"], [c.event["payload"]["task_id"] for c in commands])
        controller.handle_event(MODULE.COMPUTE_COMPLETED, self.completed("d"))
        self.assertTrue(controller.done)

    def test_duplicate_completion_is_idempotent(self):
        task = dict(self.tasks[0])
        task["children"] = []
        controller = MODULE.WorkflowController([task], "test-run")
        controller.initial_commands()
        controller.handle_event(MODULE.COMPUTE_COMPLETED, self.completed("a"))
        self.assertEqual([], controller.handle_event(MODULE.COMPUTE_COMPLETED, self.completed("a")))

    def test_cycle_is_rejected(self):
        tasks = [
            {"name": "a", "children": [{"child": "b", "size": 0}]},
            {"name": "b", "children": [{"child": "a", "size": 0}]},
        ]
        with self.assertRaises(MODULE.WorkflowError):
            MODULE.WorkflowController(tasks, "test-run")

    def test_batch_round_trip(self):
        value = MODULE.make_batch(
            "test-run", 1, 42, [{"event_id": "e1", "kind": "compute.dispatch"}]
        )
        parsed = MODULE.parse_batch(value, "test-run")
        self.assertEqual(42, parsed["logical_time_ns"])
        self.assertEqual("e1", parsed["events"][0]["event_id"])


if __name__ == "__main__":
    unittest.main()
