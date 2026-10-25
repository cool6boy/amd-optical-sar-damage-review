"""Continuation controls tested with synthetic temporary fixtures, not ML results."""
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from adr.budget import check_receipt
from adr.continue_run import validate
from adr.io import write_json, digest_file


class ContinuationTests(unittest.TestCase):
    def receipt(self):
        now = datetime.now(timezone.utc)
        r = {"control_mode":"runtime_only_no_cost_cap", "kind":"gpu", "session_minutes":20,
             "previous_gpu_minutes_reserved":0, "persistent_storage_verified":True, "auto_stop_verified":True,
             "started_at_utc":now.isoformat(), "auto_stop_at_utc":(now+timedelta(minutes=18)).isoformat()}
        return r, now

    def test_no_money_required_but_deadline_preserved(self):
        r, now = self.receipt()
        self.assertEqual(check_receipt(r,now), 990)
        with self.assertRaises(ValueError):
            check_receipt(r, now+timedelta(minutes=19))

    def test_auto_stop_and_gpu_cap_still_required(self):
        r, now = self.receipt()
        r["auto_stop_verified"] = False
        with self.assertRaises(ValueError):
            check_receipt(r,now)
        r["auto_stop_verified"] = True
        r["previous_gpu_minutes_reserved"] = 170
        with self.assertRaises(ValueError):
            check_receipt(r,now)

    def fixture(self, root):
        config = root / "config.json"
        write_json(config,{"fixture_only":True})
        rows=[]
        for split in ["train","val","test"]:
            for k in range(3 if split != "test" else 1):
                ident=split+str(k)
                row={"id":ident,"event":split+"-event","split":split,"label":None if split=="test" else k,"file_sha256":{}}
                for mode in ["optical","sar"]:
                    path=root / "crops" / (ident+mode+".png")
                    path.parent.mkdir(exist_ok=True)
                    path.write_bytes(b"synthetic crop fixture")
                    row[mode+"_path"]=path.relative_to(root).as_posix()
                    row["file_sha256"][mode]=digest_file(path)
                rows.append(row)
        write_json(root / "manifest.json",rows)
        write_json(root / "preparation_complete.json", {"config_sha256":digest_file(config),"manifest_sha256":digest_file(root / "manifest.json")})
        weight=root / "cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
        weight.parent.mkdir(parents=True)
        weight.write_bytes(b"0"*1_000_000)
        (root / "sealed").mkdir()
        (root / "sealed/test_labels.json").write_text("NOT READABLE JSON; validator must never open this")
        return config

    def test_prepared_validator_does_not_read_test_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            config=self.fixture(root)
            result=validate(root,config)
            self.assertEqual(result["total_buildings"],7)
            self.assertNotIn("class_counts",result["splits"]["test"])

    def test_validator_rejects_changed_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            config=self.fixture(root)
            write_json(config,{"changed":True})
            with self.assertRaises(ValueError):
                validate(root,config)


if __name__ == "__main__":
    unittest.main()
