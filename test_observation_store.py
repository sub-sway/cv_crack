import ast
import tempfile
import unittest
from pathlib import Path

from observation_store import ObservationStore, ObservationTracker


def detection(box=(10, 10, 40, 80), context="P1"):
    return {"bbox": list(box), "inspection_key": context,
            "timestamp": "2026-09-06 12:00:00", "image_url": None,
            "estimated_repair_cost_krw": 1000}


class ObservationTests(unittest.TestCase):
    def test_continuous_observation_updates_one_record_and_survives_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observations.sqlite3"
            store, tracker = ObservationStore(path), ObservationTracker()
            for second in range(60):
                records = [detection((10 + second % 2, 10, 40, 80))]
                tracker.observe(records, second)
                store.save(records)
            restored = ObservationStore(path).records()
            self.assertEqual(len(restored), 1)
            self.assertEqual(restored[0]["damage_no"], "D-01")
            self.assertEqual(sum(r["estimated_repair_cost_krw"] for r in restored), 1000)

    def test_matching_is_one_to_one_and_context_and_timeout_separate(self):
        tracker = ObservationTracker()
        first = [detection(), detection((12, 10, 42, 80))]
        tracker.observe(first, 0)
        second = [detection(), detection((12, 10, 42, 80))]
        tracker.observe(second, 1)
        self.assertEqual({r["observation_id"] for r in first},
                         {r["observation_id"] for r in second})
        self.assertEqual(len({r["observation_id"] for r in second}), 2)
        other = [detection(context="P2")]
        tracker.observe(other, 2)
        self.assertNotIn(other[0]["observation_id"], {r["observation_id"] for r in first})
        returned = [detection()]
        tracker.observe(returned, 10)
        self.assertNotIn(returned[0]["observation_id"], {r["observation_id"] for r in first})

    def test_more_than_500_records_images_and_clear(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observations.sqlite3"
            store = ObservationStore(path)
            records = [dict(detection(), observation_id=str(i)) for i in range(510)]
            saved = store.save(records)
            store.attach_image(saved[:1], "/captures/test.png")
            store.save(records[:1])
            restored = ObservationStore(path).records()
            self.assertEqual(len(restored), 510)
            self.assertEqual(restored[-1]["image_url"], "/captures/test.png")
            self.assertEqual(len(store.records(20)), 20)
            store.clear()
            store.attach_image(saved, "/captures/late.png")
            self.assertEqual(ObservationStore(path).records(), [])

    def test_resolution_boundary_defers_grade(self):
        # Execute the actual pure grading function without requiring camera/YOLO imports.
        source = ast.parse(Path(__file__).with_name("crack_monitor.py").read_text(encoding="utf-8-sig"))
        names = {"CRACK_GRADE_TABLES", "GRADE_FALLBACK", "MATERIAL_LABELS",
                 "GRADE_ACTIONS", "GRADE_RISK_CODES"}
        nodes = [node for node in source.body
                 if (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                     and node.target.id in names)
                 or (isinstance(node, ast.FunctionDef) and node.name == "grade_crack")]
        namespace = {"Any": object}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "grade_crack", "exec"), namespace)
        grade = namespace["grade_crack"]
        for width in (0.2, 0.4):
            result = grade(width, 0.2)
            self.assertIsNone(result["grade"])
            self.assertEqual(result["risk_code"], "pending")
            self.assertTrue(result["below_resolution"])
        self.assertEqual(grade(0.41, 0.2)["grade"], "c")
        self.assertIsNone(grade(0.4, None)["grade"])
        self.assertIsNone(grade(0.4, 0.01, "steel")["grade"])


if __name__ == "__main__":
    unittest.main()
