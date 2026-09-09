import copy
import sys
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from graph_optimizer import OptimizationError, optimize_workflow


def task(name, duration, children=None, host="host1"):
    return {
        "name": name,
        "host": host,
        "estimated_duration_ns": duration,
        "duration_bounds_ns": [duration, duration],
        "simulated_duration_ns": duration,
        "cpu_task": {
            "pes_number": 1,
            "length": duration,
            "ram": 0,
            "fileSize": 0,
            "outputSize": 0,
        },
        "gpu_task": {"kernels": []},
        "children": children or [],
    }


def closure():
    return {
        "causal": True,
        "state": True,
        "temporal": True,
        "resource_non_interference": True,
    }


class GraphOptimizerTest(unittest.TestCase):
    def test_exact_chain_contraction_preserves_boundary_and_duration(self):
        tasks = [
            task("pre", 3, [{"child": "a", "size": 0}]),
            task("a", 7, [{"child": "b", "size": 0}]),
            task("b", 11, [{"child": "post", "size": 0}]),
            task("post", 5),
        ]
        optimized, report = optimize_workflow(
            tasks,
            [],
            {
                "error_budget_ns": 0,
                "regions": [
                    {
                        "id": "exact-chain",
                        "mode": "exact",
                        "members": ["a", "b"],
                        "closure": closure(),
                        "duration_bounds_ns": [18, 18],
                    }
                ],
            },
        )
        indexed = {item["name"]: item for item in optimized}
        self.assertEqual(3, len(optimized))
        self.assertEqual("macro:exact-chain", indexed["pre"]["children"][0]["child"])
        self.assertEqual("post", indexed["macro:exact-chain"]["children"][0]["child"])
        self.assertEqual(18, indexed["macro:exact-chain"]["cpu_task"]["length"])
        self.assertEqual({"lower_ns": 26, "upper_ns": 26, "gap_ns": 0}, report["final_certificate"])

    def test_only_possible_critical_region_is_refined(self):
        tasks = [
            task("critical-a", 14, [{"child": "critical-b", "size": 0}]),
            task("critical-b", 16, [{"child": "join", "size": 0}]),
            task("side-a", 2, [{"child": "side-b", "size": 0}]),
            task("side-b", 3, [{"child": "join", "size": 0}]),
            task("join", 1),
        ]
        regions = [
            {
                "id": "critical",
                "mode": "approximate",
                "members": ["critical-a", "critical-b"],
                "closure": closure(),
                "duration_bounds_ns": [20, 40],
                "estimated_duration_ns": 30,
                "refine_cost": 2,
            },
            {
                "id": "side",
                "mode": "approximate",
                "members": ["side-a", "side-b"],
                "closure": closure(),
                "duration_bounds_ns": [4, 8],
                "estimated_duration_ns": 5,
                "refine_cost": 1,
            },
        ]
        optimized, report = optimize_workflow(
            tasks, [], {"error_budget_ns": 0, "regions": regions}
        )
        names = {item["name"] for item in optimized}
        self.assertEqual(["critical"], report["refined_regions"])
        self.assertEqual(["side"], report["retained_regions"])
        self.assertTrue(report["budget_satisfied"])
        self.assertEqual({"lower_ns": 31, "upper_ns": 31, "gap_ns": 0}, report["final_certificate"])
        self.assertIn("macro:side", names)
        self.assertNotIn("macro:critical", names)

    def test_rejects_interior_boundary_edge(self):
        tasks = [
            task("external", 1, [{"child": "b", "size": 0}]),
            task("a", 1, [{"child": "b", "size": 0}]),
            task("b", 1),
        ]
        with self.assertRaises(OptimizationError):
            optimize_workflow(
                tasks,
                [],
                {
                    "regions": [
                        {
                            "id": "bad",
                            "mode": "exact",
                            "members": ["a", "b"],
                            "closure": closure(),
                            "duration_bounds_ns": [2, 2],
                        }
                    ]
                },
            )

    def test_rejects_network_edge_without_latency_certificate(self):
        tasks = [task("a", 1, [{"child": "b", "size": 10}]), task("b", 1)]
        with self.assertRaises(OptimizationError):
            optimize_workflow(tasks, [], {"regions": []})


if __name__ == "__main__":
    unittest.main()
