import importlib.util
import tempfile
import unittest
from pathlib import Path

from observation_store import ObservationStore
from width_observations import WIDTH_VERSION, record_width_change


def observation(width=0.35, **values):
    return {"observation_id": "test-crack", "timestamp": "2026-09-06 12:00:00",
            "width_profile_version": WIDTH_VERSION, "width_p95_mm": width,
            "raw_max_width_mm": width + 0.1, "mean_width_mm": width * 0.8,
            "dimension_scale_source": "aruco", "mm_per_pixel": 0.01,
            "aruco_marker_id": 0, "scale_stale": False, "grade": "c", **values}


class WidthHistoryTests(unittest.TestCase):
    def test_delta_from_initial_and_previous_are_different(self):
        first = record_width_change(observation())
        second = record_width_change(observation(0.5), first)
        third = record_width_change(observation(0.45), second)
        self.assertEqual(third["width_delta_mm"], 0.1)
        self.assertEqual(third["width_previous_delta_mm"], -0.05)
        self.assertEqual(third["width_sample_count"], 3)
        self.assertIn("증가", second["width_change_status"])

    def test_changed_scale_marker_or_stale_data_is_not_compared(self):
        first = record_width_change(observation())
        for change in ({"mm_per_pixel": 0.02}, {"dimension_scale_source": "depth_approximation"},
                       {"aruco_marker_id": 1}, {"scale_stale": True}):
            current = record_width_change(observation(0.8, **change), first)
            self.assertIsNone(current["width_delta_mm"])
            self.assertEqual(current["width_baseline"], first["width_baseline"])

    def test_unresolved_first_frame_does_not_become_baseline(self):
        unresolved = record_width_change(observation(0.01))
        self.assertIsNone(unresolved["width_baseline"])
        good = record_width_change(observation(0.35), unresolved)
        self.assertEqual(good["width_delta_mm"], 0)
        self.assertEqual(good["width_change_status"], "초기 기준 관측")
        noisy = record_width_change(observation(0.351), good)
        self.assertIn("비교 한계 이내", noisy["width_change_status"])

    def test_samples_persist_while_damage_count_stays_one(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observations.sqlite3"
            store = ObservationStore(path)
            for index in range(30):
                store.save([observation(0.35 + index * 0.001)])
            restored = ObservationStore(path)
            rows = restored.records()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["width_sample_count"], 30)
            self.assertEqual(len(rows[0]["width_history"]), 20)
            with restored.connect() as db:
                self.assertEqual(db.execute("SELECT count(*) FROM width_samples").fetchone()[0], 30)
            restored.clear()
            with restored.connect() as db:
                self.assertEqual(db.execute("SELECT count(*) FROM width_samples").fetchone()[0], 0)


@unittest.skipUnless(importlib.util.find_spec("cv2"), "camera environment required")
class WidthGeometryTests(unittest.TestCase):
    def test_widest_point_and_orientation(self):
        import numpy as np
        from crack_monitor import calculate_crack_metrics, build_wall_mask
        mask = np.zeros((160, 80), dtype=np.uint8)
        mask[10:150, 30:34] = 255
        mask[75:85, 25:39] = 255
        def measure(binary):
            return calculate_crack_metrics(binary, np.zeros_like(binary, dtype=np.uint16),
                                           build_wall_mask(binary), 500, 500, 0.1)
        metrics = measure(mask)
        self.assertGreater(metrics["raw_max_width_mm"], metrics["max_width_mm"])
        x, y = metrics["widest_point_roi"]
        self.assertGreater(mask[y, x], 0)
        rotated = measure(np.ascontiguousarray(np.rot90(mask)))
        self.assertAlmostEqual(metrics["raw_max_width_mm"], rotated["raw_max_width_mm"], places=4)
        self.assertAlmostEqual(metrics["max_width_mm"], rotated["max_width_mm"], places=4)


if __name__ == "__main__":
    unittest.main()
