"""Human-operated cloud sessions. This does not launch or stop AUP servers."""
import argparse
import importlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone
from .io import read_json, write_json
from .budget import check_receipt, utc


def ledger(root):
    path = Path(root) / "sessions.json"
    return read_json(path) if path.exists() else []


def used(entries):
    if any(r.get("status") != "stopped_confirmed" for r in entries):
        raise ValueError("Previous session is still open in the ledger. Stop it in AUP and record end first.")
    return sum(r["actual_points"] for r in entries), sum(r["actual_gpu_minutes"] for r in entries)


def positive(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("Expected a finite nonnegative number")
    return result


def doctor(root, kind):
    info = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "root": str(Path(root).resolve()), "kind": kind, "modules": {}}
    required = ["numpy", "PIL", "torch", "torchvision"] + (["rasterio"] if kind == "cpu" else [])
    errors = []
    for name in required:
        try:
            module = importlib.import_module(name)
            info["modules"][name] = getattr(module, "__version__", "unknown")
        except Exception as e:
            errors.append(name + ": " + str(e))
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        info["hip"] = torch.version.hip
        info["gpu_available"] = torch.cuda.is_available()
        if kind == "gpu":
            if not info["hip"] or not info["gpu_available"]:
                errors.append("GPU stage requires the platform AMD ROCm PyTorch environment")
            else:
                info["device"] = torch.cuda.get_device_name(0)
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    info["free_bytes"] = __import__("shutil").disk_usage(directory).free
    info["errors"] = errors
    write_json(directory / ("doctor_" + kind + ".json"), info)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit("Environment check failed. Do not install/replace torch on a charged GPU. Stop the server first.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    sub = p.add_subparsers(dest="action", required=True)
    d = sub.add_parser("doctor")
    d.add_argument("--kind", choices=["cpu", "gpu"], required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--kind", choices=["cpu", "gpu"], required=True)
    plan.add_argument("--balance", type=positive, required=True)
    plan.add_argument("--rate", type=positive, required=True)
    plan.add_argument("--minutes", type=positive, required=True)
    sub.add_parser("begin")
    end = sub.add_parser("end")
    end.add_argument("--points", type=positive, required=True)
    end.add_argument("--gpu-minutes", type=positive, required=True)
    end.add_argument("--stopped-confirmed", action="store_true", required=True)
    end.add_argument("--note", default="Manually read actual usage after Stop My Server")
    run = sub.add_parser("run")
    run.add_argument("stage", choices=["prepare", "weights", "pilot", "features", "fit", "predict", "evaluate", "deliver"])
    run.add_argument("--config", default="configs/pilot.lock.json")
    expand = sub.add_parser("expand")
    expand.add_argument("--tiles", type=int, required=True)
    expand.add_argument("--config", default="configs/pilot.lock.json")
    a = p.parse_args()
    root = Path(a.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if a.action == "doctor":
        return doctor(root, a.kind)
    if a.action == "expand":
        if (root / "features_complete.json").exists() or (root / "run" / "policy.lock.json").exists():
            raise ValueError("Freeze data before GPU/experiments. Do not expand this experiment after feature completion.")
        cfg = read_json(a.config)
        if not 2 <= a.tiles <= 16:
            raise ValueError("Use 2..16 tiles per training/validation event, without broad searches")
        old = cfg.get("tiles_per_event_by_split", {})
        cfg["tiles_per_event_by_split"] = {"train": a.tiles, "val": a.tiles,
                                            "test": old.get("test", cfg["tiles_per_event"])}
        write_json(a.config, cfg)
        print("Expanded training/validation only; test tile count unchanged. Rerun CPU prepare.")
        return
    entries = ledger(root)
    if a.action == "plan":
        # An ongoing session counts at its entire reservation, never as zero.
        points = sum(r["actual_points"] if r["status"] == "stopped_confirmed" else r["reserved_points"] for r in entries)
        gpu = sum(r["actual_gpu_minutes"] if r["status"] == "stopped_confirmed" else
                  r["session_minutes"] if r["kind"] == "gpu" else 0 for r in entries)
        if a.rate <= 0 or a.minutes <= 0:
            raise ValueError("Rate and duration must be positive")
        unsettled = sum(r["reserved_points"] for r in entries if r["status"] == "open")
        maximum = min((540 - points) / a.rate, (a.balance - unsettled) / a.rate,
                      180 - gpu if a.kind == "gpu" else float("inf"))
        if points >= 540 or gpu >= 180:
            maximum = 0
        print(json.dumps({"previous_points": points, "previous_gpu_minutes": gpu,
                          "maximum_reservable_minutes": max(0, maximum),
                          "unsettled_points_reserved_from_balance": unsettled,
                          "requested_reserved_points": a.rate * a.minutes,
                          "fits": a.minutes <= maximum,
                          "note": "Plan only. No server launched, no points charged by this tool."}, indent=2))
        if a.minutes > maximum:
            raise SystemExit("Requested platform session exceeds remaining budget")
        return
    if a.action == "begin":
        points, gpu = used(entries)
        if not entries:
            print("First ledger: include any earlier manual CPU/GPU sessions in THIS project round.")
            points = positive(input("Earlier cloud points already consumed (0 only if none): "))
            gpu = positive(input("Earlier complete GPU session minutes (0 only if none): "))
            if points or gpu:
                entries.append({"kind": "baseline", "status": "stopped_confirmed", "actual_points": points,
                                "actual_gpu_minutes": gpu, "note": "User-reported earlier usage in this project round"})
        kind = input("Resource kind cpu/gpu: ").strip().lower()
        if kind not in {"cpu", "gpu"}:
            raise ValueError("Unknown kind")
        print("Enter actual platform values. Do not interpret an unexplained 'min remaining' as points.")
        unit = input("Verified billing unit (must be points): ").strip()
        balance = positive(input("Actual balance in points before this session: "))
        rate = positive(input("Actual points per minute for selected hardware: "))
        duration = positive(input("Full platform session limit in minutes (including startup/idle): "))
        start = utc(input("Observed launch time ISO, e.g. YYYY-MM-DDTHH:MM:SS+08:00: ").strip())
        stop = utc(input("Platform automatic stop deadline ISO with timezone: ").strip())
        verified = input("Platform automatic stop confirmed AND persistent storage confirmed? type YES: ").strip() == "YES"
        receipt = {"kind": kind, "billing_unit": unit, "available_points": balance,
                   "points_per_minute": rate, "session_minutes": duration,
                   "previous_cloud_points": points, "previous_gpu_minutes": gpu,
                   "started_at_utc": start.isoformat(), "auto_stop_at_utc": stop.isoformat(),
                   **{k: verified for k in ["balance_verified", "multiplier_verified", "auto_stop_verified", "persistent_storage_verified"]}}
        check_receipt(receipt)
        write_json(root / "session.json", receipt)
        entries.append({**receipt, "status": "open", "reserved_points": rate * duration})
        write_json(root / "sessions.json", entries)
        print("Receipt recorded. This tool did not launch a server and does not stop billing.")
        return
    if a.action == "end":
        if not entries or entries[-1]["status"] != "open":
            raise ValueError("No open session to close")
        if entries[-1]["kind"] == "cpu" and a.gpu_minutes != 0:
            raise ValueError("CPU session must record zero GPU minutes")
        entries[-1].update(status="stopped_confirmed", actual_points=a.points,
                           actual_gpu_minutes=a.gpu_minutes, ended_recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                           note=a.note)
        write_json(root / "sessions.json", entries)
        totals = used(entries)
        write_json(root / "usage_final.json", {"cloud_points": totals[0], "gpu_session_minutes": totals[1],
                   "closed_records": len(entries), "basis": "manually entered actual platform usage after stopped confirmation"})
        print(json.dumps({"cumulative_cloud_points": totals[0], "cumulative_gpu_minutes": totals[1]}))
        if totals[0] >= 540 or totals[1] >= 180:
            print("HARD LIMIT REACHED. Do not launch another session.")
        return
    if sys.platform == "win32":
        raise SystemExit("Run cloud stages on Linux, not the local Windows computer")
    receipt = read_json(root / "session.json")
    if not entries or entries[-1]["status"] != "open":
        raise ValueError("Record the actual active session with begin first")
    seconds = check_receipt(receipt)
    kind = "gpu" if a.stage in {"pilot", "features"} else "cpu"
    if receipt["kind"] != kind:
        raise ValueError("Wrong session type for this stage; stop and switch resources first")
    if seconds < 5:
        raise ValueError("No safe time remains. Save and stop in AUP.")
    os.environ["TORCH_HOME"] = str(root / "cache" / "torch")
    os.environ["XDG_CACHE_HOME"] = str(root / "cache")
    common = ["--root", str(root)]
    if a.stage == "prepare":
        cmd = [sys.executable, "-m", "adr.prepare", *common, "--config", a.config, "--cloud-confirm"]
    elif a.stage == "deliver":
        cmd = [sys.executable, "-m", "adr.deliver", *common, "--config", a.config]
    else:
        stage = "features" if kind == "gpu" else a.stage
        cmd = [sys.executable, "-m", "adr.pipeline", stage, *common, "--config", a.config, "--cloud-confirm"]
        if kind == "gpu":
            cmd += ["--max-seconds", str(min(int(seconds), 120 if a.stage == "pilot" else 300))]
            if a.stage == "pilot":
                cmd += ["--limit", "8"]
    try:
        subprocess.run(cmd, check=True, timeout=max(1, int(seconds)))
    except subprocess.TimeoutExpired:
        raise SystemExit("Computation stopped before deadline. Use Stop My Server and confirm Stopped; billing is not stopped by this script.")


if __name__ == "__main__":
    main()
