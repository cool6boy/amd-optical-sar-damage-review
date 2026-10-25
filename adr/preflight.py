"""Metadata-only availability check, safe to run without any ML environment."""
import argparse
from collections import Counter
import json
import re
from datetime import datetime, timezone
import urllib.request
from .io import read_json, write_json, sha256
from .remotezip import RemoteZip, ByteBudget


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.json")
    parser.add_argument("--output", default="evidence/data_preflight.json")
    args = parser.parse_args()
    cfg = read_json(args.config)
    api = "https://huggingface.co/api/datasets/" + cfg["dataset"]
    with urllib.request.urlopen(api, timeout=15) as response:
        revision = json.load(response)["sha"]
    base = f'https://huggingface.co/datasets/{cfg["dataset"]}/resolve/{revision}/'
    budget = ByteBudget(4 * 1024 * 1024)
    archive = RemoteZip(base + "cvprw26_trainval_instance_labels.zip", cfg["archive_sizes"]["cvprw26_trainval_instance_labels.zip"], budget)
    names = [n for n in archive.names() if n.endswith("_instance_damage.json")]
    counts = Counter(re.sub(r"_\d+_instance_damage\.json$", "", n.rsplit("/", 1)[-1]) for n in names)
    # Only inspect a train event's annotation content. Never inspect test labels.
    train_event = cfg["events"]["train"][0]
    sample = sorted(n for n in names if n.rsplit("/", 1)[-1].startswith(train_event + "_"))[0]
    raw = archive.read(sample)
    annotation = json.loads(raw.decode("utf-8-sig"))
    record = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "dataset": cfg["dataset"],
              "revision": revision, "range_read_verified": True, "metadata_bytes_received": budget.received,
              "annotation_members": len(names), "events": dict(sorted(counts.items())), "train_sample_member": sample,
              "train_sample_sha256": sha256(raw), "sample_image": annotation["images"][0],
              "sample_categories": annotation.get("categories"), "train_sample_buildings": len(annotation.get("annotations", [])),
              "first_train_bbox": annotation.get("annotations", [{}])[0].get("bbox"),
              "gpu_started": False, "images_downloaded": False,
              "note": "Metadata/annotation validation only. No image inference, model training or test-label evaluation."}
    write_json(args.output, record)
    cfg["revision"] = revision
    write_json("configs/pilot.lock.json", cfg)
    archive.close()
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
