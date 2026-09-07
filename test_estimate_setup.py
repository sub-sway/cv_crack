import ast
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from estimate_setup import DEFAULT_INPUTS, DEPTH_SCENARIOS_CM, EstimateSetupStore, estimate_scenario, estimate_depth_scenarios, validate_inputs
from repair_costs import METHODS, present_record, summarize_costs


def setup_api(store):
    """Run the actual route body without loading camera/ML dependencies."""
    source = ast.parse(Path(__file__).with_name("crack_monitor.py").read_text(encoding="utf-8-sig"))
    node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "api_estimate_setup")
    node.decorator_list = []
    request = SimpleNamespace(method="GET", get_json=lambda **kw: {})
    namespace = {"request": request, "jsonify": lambda value: value,
                 "estimate_setup_store": store, "DEFAULT_INPUTS": DEFAULT_INPUTS, "METHODS": METHODS,
                 "DEPTH_SCENARIOS_CM": DEPTH_SCENARIOS_CM}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "setup_route", "exec"), namespace)
    return namespace["api_estimate_setup"], request


class SetupTests(unittest.TestCase):
    def test_depth_comparison_has_four_alternatives_and_no_single_total(self):
        setup = {"configured": True, "inputs": copy.deepcopy(DEFAULT_INPUTS)}
        result = estimate_depth_scenarios(1000, 0.8, setup, True, True)
        self.assertIsNone(result["estimated_cost_krw"])
        self.assertEqual([s["assumed_depth_cm"] for s in result["scenarios"]], [1, 3, 5, 10])
        amounts = [s["estimated_cost_krw"] for s in result["scenarios"]]
        self.assertEqual(amounts, sorted(amounts))
        self.assertAlmostEqual(result["scenarios"][-1]["material_kg_per_m"], 10 * result["scenarios"][0]["material_kg_per_m"])
        setup["inputs"]["depth_mm"] = 180
        self.assertEqual(result["scenarios"], estimate_depth_scenarios(1000, 0.8, setup, True, True)["scenarios"])
        self.assertNotIn("assumed_depth_mm", result["basis"])

    def test_comparison_totals_are_per_depth_and_missing_is_not_zero(self):
        result = estimate_depth_scenarios(1000, 0.8, {"configured": True, "inputs": DEFAULT_INPUTS}, True, True)
        record = {"cost_model_version": result["model_version"], "cost_scenarios": result["scenarios"]}
        self.assertEqual(present_record(record), record)
        totals = summarize_costs([record, record])["depth_scenarios"]
        for total, item in zip(totals, result["scenarios"]):
            self.assertEqual(total["estimated_cost_krw"], round(item["estimated_cost_krw"] * 2))
        partial = summarize_costs([record, {}])["depth_scenarios"]
        self.assertTrue(all(row["estimated_cost_krw"] is None and row["unpriced_count"] == 1 for row in partial))

    def test_surface_comparison_is_depth_independent(self):
        result = estimate_depth_scenarios(1000, None, {"configured": True, "inputs": {**DEFAULT_INPUTS, "method": "surface_treatment"}}, True, False)
        self.assertEqual(len(result["scenarios"]), 1)
        self.assertIsNone(result["scenarios"][0]["assumed_depth_cm"])
        total = summarize_costs([{"cost_model_version": result["model_version"], "cost_scenarios": result["scenarios"]}])
        self.assertEqual(len(total["depth_scenarios"]), 1)
        self.assertIsNone(total["depth_scenarios"][0]["assumed_depth_cm"])

    def test_first_visit_save_reload_and_no_automatic_save(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "setup.json"
            store = EstimateSetupStore(path)
            state = store.snapshot()
            self.assertFalse(state["configured"])
            self.assertFalse(path.exists())
            state["inputs"]["depth_mm"] = 80
            store.save(state["inputs"])
            restored = EstimateSetupStore(path).snapshot()
            self.assertTrue(restored["configured"])
            self.assertEqual(restored["inputs"]["depth_mm"], 80)
            self.assertEqual(DEFAULT_INPUTS["depth_mm"], 50)

    def test_validation_and_failed_write_preserve_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EstimateSetupStore(Path(directory) / "setup.json")
            store.save(DEFAULT_INPUTS)
            for value in (None, -1, float("nan"), float("inf"), True, 2001):
                with self.assertRaises(ValueError):
                    store.save({**DEFAULT_INPUTS, "depth_mm": value})
            with patch("estimate_setup.os.replace", side_effect=OSError("test")):
                with self.assertRaises(OSError):
                    store.save({**DEFAULT_INPUTS, "depth_mm": 90})
            self.assertEqual(store.snapshot()["inputs"]["depth_mm"], 50)

    def test_injection_quantity_and_depth_changes_only_material_cost(self):
        setup = {"configured": True, "inputs": copy.deepcopy(DEFAULT_INPUTS)}
        first = estimate_scenario(1000, 0.8, setup, True, False)
        self.assertAlmostEqual(first["basis"]["material_kg_per_m"], 0.0198)
        setup["inputs"]["depth_mm"] = 100
        second = estimate_scenario(1000, 0.8, setup, True, False)
        self.assertEqual(first["breakdown"]["labor"], second["breakdown"]["labor"])
        self.assertAlmostEqual(second["breakdown"]["material"], 2 * first["breakdown"]["material"])
        self.assertEqual(first["basis"]["scenario_inputs"]["depth_mm"], 50)

    def test_unconfigured_invalid_length_and_observed_width(self):
        self.assertIsNone(estimate_scenario(1000, 1, {"configured": False}, True, True)["estimated_cost_krw"])
        setup = {"configured": True, "inputs": {**DEFAULT_INPUTS, "width_mode": "observed"}}
        self.assertIsNone(estimate_scenario(1000, 0.1, setup, True, False)["estimated_cost_krw"])
        self.assertIsNone(estimate_scenario(1000, 1, setup, False, True)["estimated_cost_krw"])
        result = estimate_scenario(1000, 0.8, setup, True, True)
        self.assertEqual(result["basis"]["width_used_mm"], 0.8)

    def test_surface_ignores_depth_and_width_and_assumptions_survive_report(self):
        setup = {"configured": True, "inputs": {**DEFAULT_INPUTS, "method": "surface_treatment"}}
        first = estimate_scenario(1000, None, setup, True, False)
        setup["inputs"]["depth_mm"] = 100
        second = estimate_scenario(1000, None, setup, True, False)
        self.assertEqual(first["estimated_cost_krw"], second["estimated_cost_krw"])
        record = {"cost_model_version": first["model_version"], "estimated_repair_cost_krw": first["estimated_cost_krw"]}
        self.assertEqual(present_record(record), record)
        self.assertEqual(summarize_costs([record])["assumed_count"], 1)

    def test_api_get_post_validation_and_storage_error(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EstimateSetupStore(Path(directory) / "setup.json")
            route, request = setup_api(store)
            self.assertFalse(route()["configured"])
            request.method = "POST"
            request.get_json = lambda **kw: {"inputs": {**DEFAULT_INPUTS, "depth_mm": 75}}
            self.assertEqual(route()["inputs"]["depth_mm"], 75)
            request.method = "GET"
            self.assertTrue(route()["configured"])
            request.method = "POST"
            request.get_json = lambda **kw: {"inputs": {}}
            self.assertEqual(route()[1], 400)
            request.get_json = lambda **kw: {"inputs": DEFAULT_INPUTS}
            with patch.object(store, "save", side_effect=OSError()):
                self.assertEqual(route()[1], 500)


if __name__ == "__main__":
    unittest.main()
