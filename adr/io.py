import hashlib
import json
import os
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def digest_file(path):
    return sha256(Path(path).read_bytes())


def ensure_disjoint(events):
    seen = set()
    if set(events) != {"train", "val", "test"}:
        raise ValueError("Exactly train, val, test are required")
    for split, names in events.items():
        if not names or len(names) != len(set(names)):
            raise ValueError("Empty or duplicated event list: " + split)
        if seen.intersection(names):
            raise ValueError("Event leakage across splits")
        seen.update(names)


def storage_used(root):
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())
