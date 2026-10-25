import tempfile
import io
import json
import math
import unittest
from pathlib import Path
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from adr.budget import check_receipt
from adr.io import ensure_disjoint, digest_file
from adr.prepare import event_of, crop_box, stratified_train_sample, validate_prepare_classes
from adr.review import uncertainty, disagreement, order, ideal_review, classification
from adr.remotezip import RangeFile, ByteBudget


class ProtocolTests(unittest.TestCase):
    def test_event_with_underscore(self):
        self.assertEqual(event_of("target/la_palma-volcano_00000001_instance_damage.json"), "la_palma-volcano")

    def test_event_leakage(self):
        with self.assertRaises(ValueError):
            ensure_disjoint({"train": ["A"], "val": ["B"], "test": ["A"]})

    def test_crop_geometry(self):
        self.assertEqual(crop_box([0, 0, 10, 20], 100, 100, 1), [0, 0, 15, 20])
        with self.assertRaises(ValueError):
            crop_box([95, 0, 20, 20], 100, 100, 1)

    def test_confidence_extremes(self):
        self.assertAlmostEqual(uncertainty([1, 0, 0]), 0)
        self.assertAlmostEqual(uncertainty([1/3] * 3), 1)
        self.assertAlmostEqual(disagreement([1, 0, 0], [0, 1, 0]), 1)
        self.assertAlmostEqual(disagreement([0.2, 0.4, 0.4], [0.2, 0.4, 0.4]), 0)

    def fixture(self):
        rows = [{"id": "A", "fusion": [0.9, 0.05, 0.05], "optical": [0.9, 0.05, 0.05], "sar": [0.1, 0.8, 0.1]},
                {"id": "B", "fusion": [0.34, 0.33, 0.33], "optical": [0.34, 0.33, 0.33], "sar": [0.34, 0.33, 0.33]}]
        return rows, {"A": 2, "B": 0}

    def test_review_finds_confident_miss(self):
        rows, labels = self.fixture()
        ranking = order(rows, "combined", alpha=0)
        result = ideal_review(rows, labels, ranking, 1)
        self.assertEqual(result["damaged_misses_found"], 1)
        self.assertEqual(result["remaining_errors"], 0)

    def test_entropy_can_miss_confident_error(self):
        rows, labels = self.fixture()
        result = ideal_review(rows, labels, order(rows, "uncertainty"), 1)
        self.assertEqual(result["remaining_damaged_as_intact"], 1)
        self.assertEqual(result["remaining_unreviewed_error_rate"], 1)

    def test_random_reproducible(self):
        rows, _ = self.fixture()
        self.assertEqual(order(rows, "random", seed=22), order(rows, "random", seed=22))

    def test_equal_budget_and_bad_ranking(self):
        rows, labels = self.fixture()
        for method in ["random", "uncertainty", "combined"]:
            self.assertEqual(ideal_review(rows, labels, order(rows, method), 1)["review_count"], 1)
        with self.assertRaises(ValueError):
            ideal_review(rows, labels, ["A", "A"], 1)

    def test_absent_class_reported(self):
        rows, labels = self.fixture()
        self.assertIsNone(classification(rows, labels, "fusion")["class_recall"][1])

    def test_zero_and_full_review(self):
        rows, labels = self.fixture()
        self.assertEqual(ideal_review(rows, labels, ["A", "B"], 0)["remaining_errors"], 1)
        self.assertIsNone(ideal_review(rows, labels, ["A", "B"], 2)["remaining_unreviewed_error_rate"])

    def test_stratified_train_sampling_prioritizes_rare_classes(self):
        rows = [
            {"id": f"r{i}", "label": 0, "event": "bata-explosion"} for i in range(20)
        ] + [
            {"id": f"d{i}", "label": 1, "event": "bata-explosion"} for i in range(6)
        ] + [
            {"id": f"x{i}", "label": 2, "event": "beirut-explosion"} for i in range(3)
        ]
        sample = stratified_train_sample(rows, seed=20261025, target_min_per_class=5, intact_cap=8)
        self.assertLessEqual(len(sample), 20)
        self.assertGreaterEqual(sum(1 for r in sample if r["label"] == 1), 5)
        self.assertGreaterEqual(sum(1 for r in sample if r["label"] == 2), 3)
        self.assertLessEqual(sum(1 for r in sample if r["label"] == 0), 8)

    def test_missing_val_classes_can_warn_without_fabrication(self):
        rows = [{"split": "train", "label": 0}, {"split": "train", "label": 1}, {"split": "train", "label": 2},
                {"split": "val", "label": 0}, {"split": "val", "label": 1}]
        with tempfile.TemporaryDirectory() as tmp_name:
            with self.assertRaises(ValueError):
                validate_prepare_classes(rows, Path(tmp_name), allow_missing_val_classes=False)
            validate_prepare_classes(rows, Path(tmp_name), allow_missing_val_classes=True)

    def test_continue_run_allows_explicit_missing_val_classes(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            rows = []
            for row_id, split, event, label in [
                ("t1", "train", "a", 0),
                ("t2", "train", "a", 1),
                ("t3", "train", "a", 2),
                ("v1", "val", "b", 0),
                ("v2", "val", "b", 1),
                ("te1", "test", "c", None),
            ]:
                item = {"id": row_id, "split": split, "event": event, "label": label,
                        "optical_path": f"crops/{row_id}_optical.png", "sar_path": f"crops/{row_id}_sar.png"}
                for mode in ("optical", "sar"):
                    path = tmp / item[f"{mode}_path"]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes((row_id + mode).encode())
                    item["file_sha256"] = {**item.get("file_sha256", {}), mode: digest_file(path)}
                rows.append(item)
            manifest = tmp / "manifest.json"
            manifest.write_text(json.dumps(rows), encoding="utf-8")
            config = tmp / "config.json"
            config.write_text(json.dumps({"allow_missing_val_classes": True}), encoding="utf-8")
            prep = tmp / "preparation_complete.json"
            prep.write_text(json.dumps({"manifest_sha256": digest_file(manifest), "config_sha256": digest_file(config), "allow_missing_val_classes": True}), encoding="utf-8")
            weight_dir = tmp / "cache/torch/hub/checkpoints"
            weight_dir.mkdir(parents=True, exist_ok=True)
            weight = weight_dir / "resnet18-f37072fd.pth"
            weight.write_bytes(b"x" * (1_000_001))
            from adr.continue_run import validate
            self.assertEqual(validate(tmp, config)["total_buildings"], 6)
    

    def receipt(self):
        now = datetime.now(timezone.utc)
        return {"balance_verified": True, "multiplier_verified": True, "auto_stop_verified": True,
                "persistent_storage_verified": True, "billing_unit": "points", "previous_cloud_points": 0,
                "previous_gpu_minutes": 0, "points_per_minute": 3, "available_points": 600,
                "session_minutes": 10, "started_at_utc": now.isoformat(),
                "auto_stop_at_utc": (now + timedelta(minutes=10)).isoformat()}, now

    def test_unverified_platform_blocks(self):
        r, now = self.receipt()
        r["auto_stop_verified"] = False
        with self.assertRaises(ValueError):
            check_receipt(r, now)

    def test_budget_hard_limits(self):
        r, now = self.receipt()
        self.assertAlmostEqual(check_receipt(r, now), 510, places=2)
        r["previous_cloud_points"] = 520
        with self.assertRaises(ValueError):
            check_receipt(r, now)
        r["previous_cloud_points"], r["previous_gpu_minutes"] = 0, 175
        with self.assertRaises(ValueError):
            check_receipt(r, now)

    def test_unknown_units_block(self):
        r, now = self.receipt()
        r["billing_unit"] = "minutes"
        with self.assertRaises(ValueError):
            check_receipt(r, now)

    def test_full_download_refused(self):
        class Response(io.BytesIO):
            status = 200
            headers = {}
        response = Response(b"large body")
        with patch("urllib.request.urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "refusing full download"):
                RangeFile("https://example.test/archive.zip", 100, ByteBudget(100)).read(10)


if __name__ == "__main__":
    unittest.main()
