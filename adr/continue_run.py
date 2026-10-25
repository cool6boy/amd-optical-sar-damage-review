"""Resume actual prepared data with no monetary cap; retain bounded sessions and GPU time cap."""
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import math
import subprocess
import sys
from .budget import check_receipt
from .io import read_json, write_json, digest_file


def validate(root, config):
    prepared = read_json(root / "preparation_complete.json")
    if prepared["manifest_sha256"] != digest_file(root / "manifest.json") or prepared["config_sha256"] != digest_file(config):
        raise ValueError("Prepared manifest/config hash mismatch. Preserve files; resolve on CPU.")
    allow_missing_val_classes = bool(prepared.get("allow_missing_val_classes", False) or read_json(config).get("allow_missing_val_classes", False))
    rows = read_json(root / "manifest.json")
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Empty or duplicated building IDs")
    summary = {}
    events = {}
    for split in ["train", "val", "test"]:
        items = [r for r in rows if r["split"] == split]
        if not items:
            raise ValueError("Empty split: " + split)
        events[split] = {r["event"] for r in items}
        summary[split] = {"n": len(items), "events": sorted(events[split])}
        if split != "test":
            counts = {str(i): sum(r["label"] == i for r in items) for i in range(3)}
            summary[split]["class_counts"] = counts
            if split == "train" and any(v == 0 for v in counts.values()):
                raise ValueError("Train/validation class coverage failed")
            if split == "val" and any(v == 0 for v in counts.values()) and not allow_missing_val_classes:
                raise ValueError("Train/validation class coverage failed")
            if split == "val" and any(v == 0 for v in counts.values()) and allow_missing_val_classes:
                summary[split]["warning"] = "allowed_missing_val_classes_exploratory_v2"
        elif any(r["label"] is not None for r in items):
            raise ValueError("Test labels must be sealed, not exposed in manifest")
        for r in items:
            for mode in ["optical", "sar"]:
                path = (root / r[mode + "_path"]).resolve()
                if not path.is_relative_to(root / "crops") or not path.is_file():
                    raise ValueError("Missing/invalid crop: " + r["id"])
                if r.get("file_sha256", {}).get(mode) != digest_file(path):
                    raise ValueError("Crop hash mismatch: " + r["id"])
    if events["train"] & events["val"] or events["train"] & events["test"] or events["val"] & events["test"]:
        raise ValueError("Events leak across splits")
    weight = root / "cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
    if not weight.is_file() or weight.stat().st_size < 1_000_000:
        raise ValueError("Pretrained weight cache missing")
    status = "ready; test labels not inspected" if not allow_missing_val_classes else "ready with explicit exploratory val-class limitation; test labels not inspected"
    return {"total_buildings": len(rows), "total_image_inputs": len(rows)*2, "splits": summary,
            "weight_bytes": weight.stat().st_size, "allow_missing_val_classes": allow_missing_val_classes,
            "status": status}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["validate", "start", "pilot", "features", "fit", "predict", "evaluate", "deliver"])
    p.add_argument("--root", default="/home/jovyan/adr-work")
    p.add_argument("--config", default="configs/pilot.lock.json")
    p.add_argument("--kind", choices=["cpu", "gpu"])
    p.add_argument("--session-minutes", type=float)
    p.add_argument("--remaining-minutes", type=float)
    p.add_argument("--persistent-confirmed", action="store_true")
    p.add_argument("--auto-stop-confirmed", action="store_true")
    a = p.parse_args()
    root = Path(a.root).resolve()
    if a.action == "validate":
        result = validate(root, a.config)
        write_json(root / "continuation_validation.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if sys.platform == "win32":
        raise SystemExit("Compute/session actions belong on the AUP Linux server")
    if a.action == "start":
        if a.kind is None or a.session_minutes is None or a.remaining_minutes is None:
            raise ValueError("Provide actual kind, full session length, and current platform countdown")
        if not all(math.isfinite(v) for v in [a.session_minutes,a.remaining_minutes]) or not 0 < a.remaining_minutes <= a.session_minutes:
            raise ValueError("Invalid actual duration/countdown")
        history_path = root / "runtime_sessions.json"
        history = read_json(history_path) if history_path.exists() else []
        legacy = read_json(root / "sessions.json") if (root / "sessions.json").exists() else []
        previous = sum(float(r.get("actual_gpu_minutes", r.get("session_minutes", 0))) for r in legacy if r.get("kind") == "gpu")
        previous += sum(float(r["session_minutes"]) for r in history if r["kind"] == "gpu")
        now = datetime.now(timezone.utc)
        receipt = {"control_mode": "runtime_only_no_cost_cap", "kind": a.kind,
                   "session_minutes": a.session_minutes, "previous_gpu_minutes_reserved": previous,
                   "started_at_utc": now.isoformat(),
                   "auto_stop_at_utc": (now + timedelta(minutes=a.remaining_minutes)).isoformat(),
                   "persistent_storage_verified": a.persistent_confirmed, "auto_stop_verified": a.auto_stop_confirmed,
                   "note": "No monetary cap by user instruction. Deadline from manually observed countdown. No actual billing claimed; GPU history uses conservative full-session reservations."}
        check_receipt(receipt)
        if (root / "session.json").exists():
            old = read_json(root / "session.json")
            if old.get("kind") == a.kind and old.get("control_mode") == receipt["control_mode"]:
                raise ValueError("Existing same-kind continuation receipt. For a genuinely NEW platform session, archive session.json first; do not restart receipt in the same running session.")
            write_json(root / ("session_before_continuation_" + now.strftime("%Y%m%dT%H%M%S%fZ") + ".json"), old)
        write_json(root / "session.json", receipt)
        history.append({"kind": a.kind, "session_minutes": a.session_minutes, "receipt_recorded_at_utc": now.isoformat(),
                        "accounting": "reserved full duration, NOT actual platform usage or stopped confirmation"})
        write_json(history_path, history)
        print("Runtime receipt saved. No server was launched/stopped. Monetary cap disabled; 180 GPU-minute cap retained.")
        return
    receipt = read_json(root / "session.json")
    if receipt.get("control_mode") != "runtime_only_no_cost_cap":
        raise ValueError("Record current continuation session with start first")
    kind = "gpu" if a.action in {"pilot", "features"} else "cpu"
    if receipt["kind"] != kind:
        raise ValueError("Wrong resource for this stage")
    seconds = check_receipt(receipt)
    if seconds <= 5:
        raise ValueError("Session nearly expired; preserve results and Stop My Server")
    if kind == "gpu":
        import os
        os.environ["TORCH_HOME"] = str(root / "cache/torch")
        os.environ["XDG_CACHE_HOME"] = str(root / "cache")
    if a.action == "deliver":
        cmd = [sys.executable,"-m","adr.deliver","--root",str(root),"--config",a.config]
    else:
        stage = "features" if kind == "gpu" else a.action
        cmd = [sys.executable,"-m","adr.pipeline",stage,"--root",str(root),"--config",a.config,"--cloud-confirm"]
        if kind == "gpu":
            cmd += ["--max-seconds",str(min(int(seconds),120 if a.action == "pilot" else 300))]
            if a.action == "pilot":
                cmd += ["--limit","8"]
    try:
        subprocess.run(cmd, check=True, timeout=int(seconds))
    except subprocess.TimeoutExpired:
        raise SystemExit("Stage interrupted before session deadline. Cache preserved; this did not stop platform billing.")


if __name__ == "__main__":
    main()
