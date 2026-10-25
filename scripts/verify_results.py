"""Offline verification of published experiment results; no image downloads or ML."""
import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-root", type=Path, help="Optional extracted Release results directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "results"
    read = lambda p: json.loads((root / p).read_text(encoding="utf-8-sig"))
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    passed = []
    def check(name, condition):
        if not condition:
            raise SystemExit("FAIL: " + name)
        passed.append(name)
    report = read("run/final_evaluation.json")
    policy = read("run/policy.lock.json")
    prediction = read("run/prediction_receipt.json")
    receipt = read("run/evaluation_receipt.json")
    rows = list(csv.DictReader((root / "run/per_building.csv").open(encoding="utf-8-sig", newline="")))
    ids = {r["id"] for r in rows}
    check("unique evaluated IDs", len(ids) == len(rows))
    for name, value, path in [
        ("report SHA", receipt["report_sha256"], "run/final_evaluation.json"),
        ("CSV SHA", receipt["per_building_sha256"], "run/per_building.csv"),
        ("evaluation policy SHA", receipt["policy_sha256"], "run/policy.lock.json"),
        ("prediction policy SHA", prediction["policy_sha256"], "run/policy.lock.json"),
        ("config SHA", policy["config_sha256"], "experiment.lock.json"),
    ]:
        check(name, digest(root / path) == value)
    for mode in ["optical", "sar", "fusion"]:
        cm = [[0] * 3 for _ in range(3)]
        for r in rows:
            cm[int(r["truth"])][int(r[mode + "_pred"])] += 1
        saved = report["classification"][mode]
        check(mode + " confusion matrix", cm == saved["confusion_matrix"])
        check(mode + " evaluated count", len(rows) == saved["n"])
        f1 = []
        for k in range(3):
            support, predicted = sum(cm[k]), sum(row[k] for row in cm)
            f1.append(2 * cm[k][k] / (support + predicted) if support + predicted else 0.0)
        check(mode + " macro F1", math.isclose(sum(f1) / 3, saved["macro_f1"], abs_tol=1e-12))
        check(mode + " damaged-to-intact count", cm[1][0] + cm[2][0] == saved["damaged_as_intact"])
    errors = {r["id"] for r in rows if int(r["truth"]) != int(r["fusion_pred"])}
    misses = {r["id"] for r in rows if int(r["truth"]) > 0 and int(r["fusion_pred"]) == 0}
    rankings = {m: read("run/ranking_" + m + ".json") for m in ["uncertainty", "combined", "random"]}
    for method, ranking in rankings.items():
        check(method + " complete unique ranking", len(ranking) == len(ids) and set(ranking) == ids)
    def outcome(selected, k):
        return {
            "errors_found": len(errors & selected),
            "remaining_errors": len(errors - selected),
            "remaining_unreviewed_error_rate": len(errors - selected) / (len(rows) - k) if len(rows) > k else None,
            "remaining_damaged_as_intact": len(misses - selected),
            "damaged_misses_found": len(misses & selected),
        }
    for entry in report["review"]:
        k = int(len(rows) * entry["requested_fraction"])
        check("equal budget " + str(entry["method"]) + str(k), k == entry["review_count"])
        if entry["method"] != "random":
            values = outcome(set(rankings[entry["method"]][:k]), k)
            for key, value in values.items():
                check(entry["method"] + " " + key + " " + str(k), value == entry[key] if value is None else math.isclose(value, entry[key], abs_tol=1e-12))
        else:
            repeated = []
            for seed in range(entry["repeats"]):
                order = sorted(ids)
                random.Random(seed).shuffle(order)
                repeated.append(outcome(set(order[:k]), k))
            for key in repeated[0]:
                values = [v[key] for v in repeated if v[key] is not None]
                check("random mean " + key + str(k), math.isclose(statistics.mean(values), entry[key + "_mean"], abs_tol=1e-12))
                check("random SD " + key + str(k), math.isclose(statistics.pstdev(values), entry[key + "_std"], abs_tol=1e-12))
    if args.full_root:
        full = args.full_root.resolve()
        for name, value in [
            ("manifest.json", policy["manifest_sha256"]),
            ("run/heads.pt", policy["heads_sha256"]),
            ("run/test_predictions.json", prediction["predictions_sha256"]),
        ]:
            check("Release " + name, digest(full / name) == value)
        manifest = json.loads((full / "manifest.json").read_text(encoding="utf-8-sig"))
        complete = json.loads((full / "features_complete.json").read_text(encoding="utf-8-sig"))
        check("Release feature count", len(manifest) * 2 == complete["feature_count"])
    print(json.dumps({"status": "passed", "checks": len(passed), "test_buildings": len(rows),
                      "fusion_errors": len(errors), "damaged_to_intact": len(misses),
                      "full_release_checked": bool(args.full_root),
                      "scope": "File, table and ranking consistency; not a training rerun or feature-content verification."}, indent=2))

if __name__ == "__main__":
    main()
