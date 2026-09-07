import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from repair_costs import MODEL_VERSION, estimate_repair_cost, present_record, summarize_costs


def fixture(method="injection"):
    # Synthetic arithmetic fixtures, NOT actual wages, quantities or quotations.
    source = {"title": "TEST ONLY", "reference": "synthetic fixture", "date": "2026-01-01"}
    scope = {"bridge_id": "TEST", "member_label": "P1", "member_position": "front"}
    config = {
        "schema_version": 2, "currency": "KRW",
        "labor_rates": {name: {"krw_per_day": value, "source": dict(source)}
                        for name, value in {"plasterer": 110000, "skilled_laborer": 200000,
                                            "general_laborer": 160000}.items()},
        "methods": {method: {"scope": dict(scope), "material": {
            "name": "TEST", "kg_per_m": 0.1, "krw_per_kg": 10000,
            "quantity_source": dict(source), "price_source": dict(source)},
            "misc_material_rate": 0.05, "misc_material_source": dict(source)}},
    }
    context = {**scope, "member_material": "rc", "repair_method": method,
               "repair_review_note": "TEST ONLY"}
    return config, context


class RepairCostTests(unittest.TestCase):
    def test_injection_breakdown_and_evidence_snapshot(self):
        config, context = fixture()
        cost = estimate_repair_cost(2000, 0.8, config, context, True)
        # Crew/day = 560000, output = 28m -> labor 20000/m.
        self.assertEqual(cost["breakdown"], {"labor": 40000, "tools": 800,
                                            "material": 2000, "misc_material": 100})
        self.assertEqual(cost["estimated_cost_krw"], 42900)
        config["methods"]["injection"]["material"]["kg_per_m"] = 999
        self.assertEqual(cost["basis"]["profile"]["material"]["kg_per_m"], 0.1)

    def test_surface_method_is_selected_not_inferred_from_width(self):
        config, context = fixture("surface_treatment")
        self.assertEqual(estimate_repair_cost(1000, 0.8, config, context, True)["estimated_cost_krw"], 2070)
        context["repair_method"] = ""
        self.assertIsNone(estimate_repair_cost(1000, 0.8, config, context, True)["estimated_cost_krw"])

    def test_default_config_and_legacy_config_cannot_produce_money(self):
        config = json.loads(Path(__file__).with_name("repair_cost_config.json").read_text(encoding="utf-8"))
        _, context = fixture()
        for settings in (config, {"methods": {"injection": {"unit_price_krw_per_m": 25000}}}):
            self.assertIsNone(estimate_repair_cost(1000, 1, settings, context, True)["estimated_cost_krw"])

    def test_missing_or_invalid_evidence_blocks_estimate(self):
        changes = [
            lambda c: c["labor_rates"]["skilled_laborer"].update(krw_per_day=float("nan")),
            lambda c: c["labor_rates"]["skilled_laborer"]["source"].update(date="not a date"),
            lambda c: c["methods"]["injection"]["scope"].update(member_position="back"),
            lambda c: c["methods"]["injection"]["material"].update(kg_per_m=None),
            lambda c: c["methods"]["injection"]["material"].update(krw_per_kg=True),
            lambda c: c["methods"]["injection"].update(misc_material_rate=0.06),
            lambda c: c["methods"]["injection"].update(misc_material_source={}),
        ]
        for change in changes:
            config, context = fixture()
            change(config)
            with self.subTest(change=change):
                self.assertIsNone(estimate_repair_cost(1000, 1, config, context, True)["estimated_cost_krw"])

    def test_measurement_limits_and_scope(self):
        config, context = fixture()
        for length, width, valid in [(1000, 1, False), (0, 1, True), (1000, 11, True), (1000, float("inf"), True)]:
            self.assertIsNone(estimate_repair_cost(length, width, config, context, valid)["estimated_cost_krw"])
        context["member_material"] = "steel"
        self.assertIsNone(estimate_repair_cost(1000, 1, config, context, True)["estimated_cost_krw"])

    def test_legacy_preserved_but_not_aggregated_and_partial_total_label(self):
        old = {"estimated_repair_cost_krw": 25000}
        displayed = present_record(old)
        self.assertIsNone(displayed["estimated_repair_cost_krw"])
        self.assertEqual(old["estimated_repair_cost_krw"], 25000)
        current = {"cost_model_version": MODEL_VERSION, "estimated_repair_cost_krw": 42900}
        summary = summarize_costs([current, displayed])
        self.assertIsNone(summary["estimated_cost_krw"])
        self.assertEqual(summary["priced_subtotal_krw"], 42900)
        self.assertEqual(summary["unpriced_count"], 1)
        self.assertEqual(summarize_costs([current])["estimated_cost_krw"], 42900)
        self.assertIsNone(summarize_costs([])["estimated_cost_krw"])

    def test_invalid_config_does_not_keep_previous_price(self):
        source = ast.parse(Path(__file__).with_name("crack_monitor.py").read_text(encoding="utf-8-sig"))
        node = next(n for n in source.body if isinstance(n, ast.FunctionDef)
                    and n.name == "load_repair_cost_config")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            # Deliberately invalid input in an isolated test directory.
            path.write_text("{", encoding="utf-8")
            namespace = {"Any": object, "json": json, "S": SimpleNamespace(repair_cost_config_path=path),
                         "repair_cost_config": {"schema_version": 2}, "_repair_config_mtime": None,
                         "print": lambda *args: None}
            exec(compile(ast.Module(body=[node], type_ignores=[]), "loader", "exec"), namespace)
            self.assertNotIn("schema_version", namespace["load_repair_cost_config"]())


if __name__ == "__main__":
    unittest.main()
