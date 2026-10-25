"""Fail-closed resource gate; estimates do not replace actual billing records."""
from datetime import datetime, timezone
import math


def utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timestamp must include Z or an explicit UTC offset")
    return parsed.astimezone(timezone.utc)


def check_receipt(receipt, now=None):
    now = now or datetime.now(timezone.utc)
    if receipt.get("control_mode") == "runtime_only_no_cost_cap":
        if receipt.get("persistent_storage_verified") is not True or receipt.get("auto_stop_verified") is not True:
            raise ValueError("Persistent storage and automatic server termination must be verified")
        duration = float(receipt["session_minutes"])
        previous = float(receipt["previous_gpu_minutes_reserved"])
        kind = receipt["kind"]
        if kind not in {"cpu", "gpu"} or not all(math.isfinite(x) for x in [duration, previous]) or duration <= 0 or previous < 0:
            raise ValueError("Invalid runtime receipt")
        # User lifted the monetary cap, not the original 180-minute GPU hard limit.
        if previous >= 180 or (kind == "gpu" and previous + duration > 180):
            raise ValueError("Original 180-minute GPU limit reached; preserve outputs before further work")
        started, deadline = utc(receipt["started_at_utc"]), utc(receipt["auto_stop_at_utc"])
        if now < started or now >= deadline or (deadline - started).total_seconds() > duration * 60 + 1:
            raise ValueError("Runtime receipt expired or exceeds the selected session")
        return max(0, (deadline - now).total_seconds() - 90)
    required = ["balance_verified", "multiplier_verified", "auto_stop_verified", "persistent_storage_verified"]
    if not all(receipt.get(key) is True for key in required):
        raise ValueError("Platform prerequisites not verified; GPU work is blocked")
    if receipt.get("billing_unit") != "points":
        raise ValueError("Billing units must be verified as points")
    used_points = float(receipt["previous_cloud_points"])
    gpu_minutes = float(receipt["previous_gpu_minutes"])
    multiplier = float(receipt["points_per_minute"])
    duration = float(receipt["session_minutes"])
    balance = float(receipt["available_points"])
    if not all(math.isfinite(v) for v in (used_points, gpu_minutes, multiplier, duration, balance)):
        raise ValueError("Nonfinite budget values")
    if min(used_points, gpu_minutes) < 0 or multiplier <= 0 or duration <= 0:
        raise ValueError("Invalid budget values")
    if used_points >= 540 or gpu_minutes >= 180:
        raise ValueError("A hard cumulative limit was already reached; no further cloud session")
    kind = receipt.get("kind", "gpu")
    if kind not in {"cpu", "gpu"}:
        raise ValueError("Unknown resource kind")
    if used_points + multiplier * duration > 540 or gpu_minutes + (duration if kind == "gpu" else 0) > 180:
        raise ValueError("Session reservation exceeds a hard budget limit")
    if balance < multiplier * duration:
        raise ValueError("Insufficient actual platform balance")
    started, deadline = utc(receipt["started_at_utc"]), utc(receipt["auto_stop_at_utc"])
    if now < started or now >= deadline:
        raise ValueError("No active verified bounded session")
    if (deadline - started).total_seconds() > duration * 60 + 1:
        raise ValueError("Auto-stop deadline exceeds reserved duration")
    return max(0, (deadline - now).total_seconds() - 90)
