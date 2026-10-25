"""Synthetic fixtures in temporary directories only; not model experiment results."""
import contextlib
import io
import re
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode
from datetime import datetime, timedelta, timezone
from adr.budget import check_receipt, utc
from adr.deliver import generate
from adr.io import read_json, write_json, digest_file
from adr.prepare import tile_count
from adr.review import classification, ideal_review
from adr.viewer import make_server
from adr.operator import main as operator_main


class OperatorTests(unittest.TestCase):
    def test_plan_reserves_open_session_from_balance(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_json(Path(tmp) / "sessions.json", [{"kind": "cpu", "status": "open", "reserved_points": 20, "session_minutes": 20}])
            with patch("sys.argv", ["operator", "--root", tmp, "plan", "--kind", "gpu", "--balance", "50", "--rate", "3", "--minutes", "20"]), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):
                    operator_main()

    def test_close_session_writes_actual_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_json(Path(tmp) / "sessions.json", [{"kind": "gpu", "status": "open", "reserved_points": 60, "session_minutes": 20}])
            with patch("sys.argv", ["operator", "--root", tmp, "end", "--points", "15", "--gpu-minutes", "5", "--stopped-confirmed"]), contextlib.redirect_stdout(io.StringIO()):
                operator_main()
            summary = read_json(Path(tmp) / "usage_final.json")
            self.assertEqual((summary["cloud_points"], summary["gpu_session_minutes"]), (15, 5))

    def test_expansion_keeps_test_fixed(self):
        cfg = {"tiles_per_event": 2, "tiles_per_event_by_split": {"train": 8, "val": 8, "test": 2}}
        self.assertEqual([tile_count(cfg, s) for s in ["train", "val", "test"]], [8, 8, 2])

    def test_timestamps_require_timezone(self):
        with self.assertRaises(ValueError):
            utc("2026-10-24T10:00:00")

    def test_cpu_does_not_spend_gpu_time_and_nan_blocked(self):
        now = datetime.now(timezone.utc)
        r = {k: True for k in ["balance_verified", "multiplier_verified", "auto_stop_verified", "persistent_storage_verified"]}
        r.update(kind="cpu", billing_unit="points", previous_cloud_points=0, previous_gpu_minutes=179,
                 points_per_minute=1, available_points=100, session_minutes=20,
                 started_at_utc=now.isoformat(), auto_stop_at_utc=(now+timedelta(minutes=20)).isoformat())
        self.assertGreater(check_receipt(r, now), 0)
        r["previous_gpu_minutes"] = 180
        with self.assertRaises(ValueError):
            check_receipt(r, now)
        r["previous_gpu_minutes"] = 179
        r["available_points"] = float("nan")
        with self.assertRaises(ValueError):
            check_receipt(r, now)


class ViewerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        row = {"id": "fixture", "optical": [.9, .05, .05], "sar": [.1, .8, .1], "fusion": [.5, .4, .1],
               "optical_path": "crops/a.png", "sar_path": "crops/b.png"}
        write_json(self.root / "run/test_predictions.json", [row])
        write_json(self.root / "run/policy.lock.json", {"alpha": .5})
        write_json(self.root / "sealed/test_labels.json", {"must_not_be_served": 2})
        self.server = make_server(self.root, 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.tmp.cleanup()

    def test_proxy_relative_links_and_hidden_labels(self):
        page = urllib.request.urlopen(self.base).read().decode()
        self.assertIn('src="./crops/a.png"', page)
        self.assertIn('action="./review"', page)
        self.assertNotIn('must_not_be_served', page)
        for path in ["/sealed/test_labels.json", "/crops/../sealed/test_labels.json"]:
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(self.base + path)
            self.assertEqual(caught.exception.code, 404)

    def test_saved_unknown_is_not_ground_truth(self):
        page = urllib.request.urlopen(self.base).read().decode()
        csrf = re.search(r'name="csrf" value="([^"]+)"', page).group(1)
        data = urlencode({"csrf": csrf, "id": "fixture", "decision": "unknown", "note": "纹理不清"}).encode()
        result = urllib.request.urlopen(urllib.request.Request(self.base + "/review", data=data)).read().decode()
        self.assertIn("已保存：无法判断", result)
        saved = read_json(self.root / "human_review.json")["fixture"]
        self.assertIsNone(saved["decision"])
        self.assertEqual(saved["status"], "unable_to_judge")
        self.assertEqual(read_json(self.root / "sealed/test_labels.json"), {"must_not_be_served": 2})

    def test_cross_site_form_without_token_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(urllib.request.Request(self.base + "/review", data=b'id=fixture&decision=0'))
        self.assertEqual(caught.exception.code, 403)
        self.assertFalse((self.root / "human_review.json").exists())


class DeliveryTests(unittest.TestCase):
    def test_missing_actual_outputs_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                generate(tmp, Path(tmp) / "missing.json")
            self.assertFalse((Path(tmp) / "delivery").exists())

    def test_real_file_contract_and_allowlist_using_synthetic_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            cfg = root / "config.json"
            write_json(cfg, {"fixture_only": True})
            write_json(root / "manifest.json", [{"id": "fixture"}])
            write_json(root / "features_complete.json", {"fixture_only": True})
            write_json(root / "environment.json", {"fixture_only": True})
            write_json(root / "sessions.json", [{"kind": "cpu", "status": "open", "private_note": "not for release"}])
            rows = [{"id": "fixture", "fusion": [.9,.05,.05], "optical": [.9,.05,.05], "sar": [.9,.05,.05]}]
            labels = {"fixture": 0}
            write_json(run / "test_predictions.json", rows)
            (run / "heads.pt").write_bytes(b"synthetic unit-test bytes, not weights")
            policy = {"heads_sha256": digest_file(run / "heads.pt"), "config_sha256": digest_file(cfg),
                      "manifest_sha256": digest_file(root / "manifest.json")}
            write_json(run / "policy.lock.json", policy)
            write_json(run / "prediction_receipt.json", {"policy_sha256": digest_file(run / "policy.lock.json"), "predictions_sha256": digest_file(run / "test_predictions.json")})
            metrics = {m: classification(rows, labels, m) for m in ["fusion", "optical", "sar"]}
            write_json(run / "final_evaluation.json", {"classification": metrics, "review": [{"method":"combined", "requested_fraction":.1, **ideal_review(rows,labels,["fixture"],0)}]})
            (run / "per_building.csv").write_text("fixture\n")
            write_json(run / "evaluation_receipt.json", {k: digest_file(run / n) for k,n in [
                ("policy_sha256","policy.lock.json"),("predictions_sha256","test_predictions.json"),
                ("report_sha256","final_evaluation.json"),("per_building_sha256","per_building.csv")]})
            write_json(root / "sealed/test_labels.json", labels)
            (root / "credentials.txt").write_text("private fixture")
            with contextlib.redirect_stdout(io.StringIO()):
                out = generate(root, cfg)
            self.assertTrue((out / "report.html").is_file())
            self.assertTrue((root / "course-results.zip").is_file())
            self.assertFalse((out / "sealed").exists())
            self.assertFalse((out / "credentials.txt").exists())
            self.assertNotIn("private_note", read_json(out / "sessions.json")[0])
            self.assertEqual(read_json(out / "failure_cases.json"), [])


if __name__ == "__main__":
    unittest.main()
