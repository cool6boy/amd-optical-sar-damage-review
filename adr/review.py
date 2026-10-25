"""Pure Python metrics. Review rankings never receive test labels."""
import math
import random


def uncertainty(probabilities):
    return -sum(p * math.log(max(p, 1e-12)) for p in probabilities if p > 0) / math.log(len(probabilities))


def disagreement(optical, sar):
    # Normalized Jensen-Shannon divergence, bounded by [0,1].
    middle = [(a + b) / 2 for a, b in zip(optical, sar)]
    def kl(a, b):
        return sum(x * math.log(x / max(y, 1e-12)) for x, y in zip(a, b) if x > 0)
    return (kl(optical, middle) + kl(sar, middle)) / (2 * math.log(2))


def order(rows, method, alpha=0.5, seed=11):
    rows = sorted(rows, key=lambda r: r["id"])
    if method == "random":
        random.Random(seed).shuffle(rows)
        return [r["id"] for r in rows]
    def score(r):
        u = uncertainty(r["fusion"])
        d = disagreement(r["optical"], r["sar"])
        return u if method == "uncertainty" else alpha * u + (1 - alpha) * d
    if method not in {"uncertainty", "combined"}:
        raise ValueError(method)
    return [r["id"] for r in sorted(rows, key=lambda r: (-score(r), r["id"]))]


def pred(p):
    return max(range(len(p)), key=lambda k: p[k])


def classification(rows, labels, mode):
    matrix = [[0] * 3 for _ in range(3)]
    for row in rows:
        matrix[labels[row["id"]]][pred(row[mode])] += 1
    n = len(rows)
    if not n:
        raise ValueError("Empty evaluation")
    recalls, f1 = [], []
    for k in range(3):
        tp, support, predicted = matrix[k][k], sum(matrix[k]), sum(r[k] for r in matrix)
        recalls.append(tp / support if support else None)
        f1.append(2 * tp / (support + predicted) if support + predicted else 0.0)
    damaged_total = sum(sum(matrix[k]) for k in (1, 2))
    missed = matrix[1][0] + matrix[2][0]
    return {"n": n, "confusion_matrix": matrix, "class_recall": recalls, "macro_f1": sum(f1) / 3,
            "damaged_as_intact": missed, "damaged_as_intact_rate": missed / damaged_total if damaged_total else None}


def ideal_review(rows, labels, ranking, count):
    if len(ranking) != len(rows) or set(ranking) != {r["id"] for r in rows}:
        raise ValueError("Ranking must contain each evaluated ID exactly once")
    count = max(0, min(int(count), len(rows)))
    selected = set(ranking[:count])
    errors = {r["id"] for r in rows if pred(r["fusion"]) != labels[r["id"]]}
    missed = {r["id"] for r in rows if labels[r["id"]] > 0 and pred(r["fusion"]) == 0}
    remaining = errors - selected
    return {"simulation": "ideal label-oracle review; not measured human performance", "review_count": count,
            "errors_found": len(errors & selected), "error_capture_rate": len(errors & selected) / len(errors) if errors else None,
            "damaged_misses_found": len(missed & selected), "remaining_errors": len(remaining),
            "remaining_unreviewed_n": len(rows) - count,
            "remaining_unreviewed_error_rate": len(remaining) / (len(rows) - count) if len(rows) > count else None,
            "remaining_damaged_as_intact": len(missed - selected)}
