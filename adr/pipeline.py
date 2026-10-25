"""Cloud-only frozen feature extraction and CPU head training; no local ML work."""
import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from .io import read_json, write_json, digest_file, sha256
from .review import order, ideal_review, classification, uncertainty, disagreement


def check_cloud(args):
    if not args.cloud_confirm or sys.platform == "win32":
        raise SystemExit("ML execution is restricted to the cloud Linux environment; pass --cloud-confirm there")


def environment(root):
    import torch
    import torchvision
    info = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "hip": torch.version.hip,
            "gpu_available": torch.cuda.is_available(), "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    write_json(Path(root) / "environment.json", info)
    return info


def weights(args):
    from torchvision.models import resnet18, ResNet18_Weights
    # Do this on the cloud CPU BEFORE starting a charged GPU session.
    resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    print(json.dumps(environment(args.root)), flush=True)


def features(args):
    import numpy as np
    import torch
    from PIL import Image
    from torchvision.models import resnet18, ResNet18_Weights
    from torchvision import transforms
    root = Path(args.root)
    if not (root / "preparation_complete.json").is_file():
        raise ValueError("CPU data preparation and train/val coverage must pass before GPU work")
    prepared = read_json(root / "preparation_complete.json")
    if prepared["manifest_sha256"] != digest_file(root / "manifest.json") or prepared["config_sha256"] != digest_file(args.config):
        raise ValueError("Prepared manifest/config changed; rerun CPU validation")
    from .budget import check_receipt
    seconds_remaining = check_receipt(read_json(root / "session.json"))
    if read_json(root / "session.json").get("kind", "gpu") != "gpu":
        raise ValueError("Feature extraction requires a GPU session receipt")
    if seconds_remaining <= 0:
        raise ValueError("No safe time remains in the reserved session")
    args.max_seconds = min(args.max_seconds, seconds_remaining)
    rows = read_json(root / "manifest.json")
    cache = root / "features"
    cache.mkdir(exist_ok=True)
    info = environment(root)
    if not info["hip"] or not info["gpu_available"]:
        raise RuntimeError("Verified AMD ROCm GPU is required for feature extraction")
    weight_path = Path(torch.hub.get_dir()) / "checkpoints" / "resnet18-f37072fd.pth"
    if not weight_path.is_file():
        raise RuntimeError("Weights must be downloaded on CPU before GPU startup")
    weight_digest = digest_file(weight_path)
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model = model.eval().to("cuda")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    limit = args.limit or len(rows)
    tasks = []
    for row in rows[:limit]:
        for mode in ("optical", "sar"):
            image = root / row[mode + "_path"]
            fingerprint = sha256((digest_file(image) + weight_digest + "rgb224_imagenet_v1").encode())
            dst = cache / (row["id"] + "_" + mode + ".npz")
            if dst.exists():
                with np.load(dst, allow_pickle=False) as old:
                    if str(old["fingerprint"]) == fingerprint:
                        continue
            tasks.append((image, dst, fingerprint))
    started = time.monotonic()
    completed = 0
    for start in range(0, len(tasks), args.batch_size):
        remaining_now = check_receipt(read_json(root / "session.json"))
        if remaining_now <= 0 or time.monotonic() - started > args.max_seconds:
            print("Feature time limit reached; per-batch cache saved", flush=True)
            break
        batch = tasks[start:start + args.batch_size]
        tensor = torch.stack([transform(Image.open(path).convert("RGB")) for path, _, _ in batch]).to("cuda")
        with torch.inference_mode():
            output = model(tensor).float().cpu().numpy()
        for (_, dst, fingerprint), vector in zip(batch, output):
            tmp = dst.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, feature=vector, fingerprint=fingerprint)
            os.replace(tmp, dst)
        completed += len(batch)
        seconds = time.monotonic() - started
        write_json(root / "feature_progress.json", {"completed_this_session": completed,
                   "pending_this_session": len(tasks) - completed, "seconds": seconds,
                   "images_per_second": completed / seconds if seconds else None, "weights_sha256": weight_digest})
        print(json.dumps({"computed": completed, "pending": len(tasks) - completed, "seconds": round(seconds, 3)}), flush=True)
    if limit >= len(rows) and completed == len(tasks):
        write_json(root / "features_complete.json", {"manifest_sha256": digest_file(root / "manifest.json"),
                   "config_sha256": digest_file(args.config), "weights_sha256": weight_digest,
                   "feature_count": len(rows) * 2})


def matrix(root, rows, mode):
    import numpy as np
    data = []
    for r in rows:
        modes = ("optical", "sar") if mode == "fusion" else (mode,)
        vectors = []
        for m in modes:
            with np.load(root / "features" / (r["id"] + "_" + m + ".npz"), allow_pickle=False) as item:
                vectors.append(item["feature"])
        data.append(np.concatenate(vectors))
    return np.stack(data).astype(np.float32)


def fit(args):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    root, cfg = Path(args.root), read_json(args.config)
    completed = read_json(root / "features_complete.json")
    if completed["manifest_sha256"] != digest_file(root / "manifest.json") or completed["config_sha256"] != digest_file(args.config):
        raise ValueError("Features are not validated for current manifest/config")
    rows = read_json(root / "manifest.json")
    train, val = [[r for r in rows if r["split"] == split] for split in ("train", "val")]
    if not train or not val:
        raise ValueError("Training and validation samples are required")
    yt = torch.tensor([r["label"] for r in train], dtype=torch.long)
    yv = torch.tensor([r["label"] for r in val], dtype=torch.long)
    counts = torch.bincount(yt, minlength=3)
    if (counts == 0).any():
        raise ValueError("Training subset misses a class; inspect training labels before expansion")
    class_weights = len(yt) / (3 * counts.float())
    run = root / "run"
    run.mkdir(exist_ok=True)
    histories, val_rows = {}, [{"id": r["id"]} for r in val]
    models = {}
    for mode in ("optical", "sar", "fusion"):
        xt, xv = matrix(root, train, mode), matrix(root, val, mode)
        mean, std = xt.mean(0), np.maximum(xt.std(0), 1e-5)
        xt, xv = torch.from_numpy((xt - mean) / std), torch.from_numpy((xv - mean) / std)
        probs, members = [], []
        for seed in cfg["ensemble_seeds"]:
            torch.manual_seed(seed)
            head = torch.nn.Linear(xt.shape[1], 3)
            opt = torch.optim.AdamW(head.parameters(), lr=cfg["head_lr"], weight_decay=cfg["head_weight_decay"])
            best, best_state, history = float("inf"), None, []
            for epoch in range(cfg["head_epochs"]):
                head.train()
                opt.zero_grad()
                loss = torch.nn.functional.cross_entropy(head(xt), yt, weight=class_weights)
                loss.backward()
                opt.step()
                head.eval()
                with torch.no_grad():
                    validation_loss = torch.nn.functional.cross_entropy(head(xv), yv).item()
                history.append({"epoch": epoch + 1, "train_loss": loss.item(), "val_loss": validation_loss})
                if validation_loss < best:
                    best, best_state = validation_loss, {k: v.detach().clone() for k, v in head.state_dict().items()}
            head.load_state_dict(best_state)
            with torch.no_grad():
                probs.append(head(xv).softmax(1).numpy())
            member = {"state": best_state, "seed": seed}
            members.append(member)
            histories[mode + "_" + str(seed)] = history
            # Checkpoint each completed small head.
            temp = run / (mode + "_" + str(seed) + ".tmp.pt")
            torch.save({"mean": torch.from_numpy(mean), "std": torch.from_numpy(std), **member}, temp)
            os.replace(temp, run / (mode + "_" + str(seed) + ".pt"))
        models[mode] = {"mean": torch.from_numpy(mean), "std": torch.from_numpy(std), "members": members}
        mean_probability = np.mean(probs, axis=0)
        for row, probability in zip(val_rows, mean_probability):
            row[mode] = probability.tolist()
    torch.save(models, run / "heads.pt")
    write_json(run / "history.json", histories)
    labels = {r["id"]: r["label"] for r in val}
    policy_scores = []
    for alpha in cfg["review_mix_candidates"]:
        ranking = order(val_rows, "combined", alpha=alpha)
        captured = sum(ideal_review(val_rows, labels, ranking, int(len(val_rows) * fraction))["errors_found"] for fraction in cfg["review_fractions"])
        policy_scores.append({"alpha": alpha, "validation_errors_found_sum": captured})
    alpha = max(policy_scores, key=lambda x: (x["validation_errors_found_sum"], -abs(x["alpha"] - 0.5)))["alpha"]
    write_json(run / "validation_predictions.json", val_rows)
    write_json(run / "policy.lock.json", {"alpha": alpha, "candidates": policy_scores,
               "review_fractions": cfg["review_fractions"], "random_repeats": cfg["random_repeats"],
               "class_decision": "argmax, fixed before test", "review_score": "alpha*normalized_entropy+(1-alpha)*modal_JSD",
               "manifest_sha256": digest_file(root / "manifest.json"), "config_sha256": digest_file(args.config),
               "heads_sha256": digest_file(run / "heads.pt"), "selected_using": "validation labels only"})
    print("Heads and validation-only policy saved; no test labels accessed", flush=True)


def predict(args):
    import numpy as np
    import torch
    root = Path(args.root)
    run = root / "run"
    policy = read_json(run / "policy.lock.json")
    if policy["manifest_sha256"] != digest_file(root / "manifest.json") or policy["heads_sha256"] != digest_file(run / "heads.pt"):
        raise ValueError("Locked experiment inputs changed")
    models = torch.load(run / "heads.pt", map_location="cpu", weights_only=True)
    rows = [r for r in read_json(root / "manifest.json") if r["split"] == "test"]
    output = [{"id": r["id"], "event": r["event"], "optical_path": r["optical_path"], "sar_path": r["sar_path"], "status": "predicted"} for r in rows]
    for mode, model in models.items():
        x = torch.from_numpy(matrix(root, rows, mode))
        x = (x - model["mean"]) / model["std"]
        probabilities = []
        for member in model["members"]:
            head = torch.nn.Linear(x.shape[1], 3)
            head.load_state_dict(member["state"])
            with torch.no_grad():
                probabilities.append(head(x).softmax(1).numpy())
        for row, probability in zip(output, np.mean(probabilities, axis=0)):
            row[mode] = probability.tolist()
    write_json(run / "test_predictions.json", output)
    for method in ("random", "uncertainty", "combined"):
        write_json(run / ("ranking_" + method + ".json"), order(output, method, policy["alpha"]))
    write_json(run / "prediction_receipt.json", {"policy_sha256": digest_file(run / "policy.lock.json"), "predictions_sha256": digest_file(run / "test_predictions.json"), "test_labels_accessed": False})


def evaluate(args):
    import statistics
    root = Path(args.root)
    run = root / "run"
    receipt = read_json(run / "prediction_receipt.json")
    if receipt["policy_sha256"] != digest_file(run / "policy.lock.json") or receipt["predictions_sha256"] != digest_file(run / "test_predictions.json"):
        raise ValueError("Predictions or policy changed after inference")
    policy, rows = read_json(run / "policy.lock.json"), read_json(run / "test_predictions.json")
    labels = read_json(root / "sealed" / "test_labels.json")
    if set(labels) != {r["id"] for r in rows}:
        raise ValueError("Held-out labels and predictions disagree")
    report = {"scope": "Known-location crop classification, pre-event optical/post-event SAR", "classification": {}, "review": []}
    for mode in ("optical", "sar", "fusion"):
        report["classification"][mode] = classification(rows, labels, mode)
    for fraction in policy["review_fractions"]:
        k = int(len(rows) * fraction)
        for method in ("uncertainty", "combined"):
            result = ideal_review(rows, labels, order(rows, method, policy["alpha"]), k)
            report["review"].append({"method": method, "requested_fraction": fraction, **result})
        repeats = [ideal_review(rows, labels, order(rows, "random", seed=seed), k) for seed in range(policy["random_repeats"])]
        summary = {"method": "random", "requested_fraction": fraction, "repeats": len(repeats), "review_count": k,
                   "simulation": "ideal label-oracle review; not measured human performance"}
        for key in ("errors_found", "remaining_errors", "remaining_unreviewed_error_rate", "remaining_damaged_as_intact", "damaged_misses_found"):
            vals = [r[key] for r in repeats if r[key] is not None]
            summary[key + "_mean"] = statistics.mean(vals) if vals else None
            summary[key + "_std"] = statistics.pstdev(vals) if vals else None
        report["review"].append(summary)
    write_json(run / "final_evaluation.json", report)
    with (run / "per_building.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=["id", "event", "truth", "optical_pred", "sar_pred", "fusion_pred", "uncertainty", "disagreement", "damaged_as_intact"])
        writer.writeheader()
        for r in rows:
            predictions = {mode: max(range(3), key=lambda k: r[mode][k]) for mode in ("optical", "sar", "fusion")}
            writer.writerow({"id": r["id"], "event": r["event"], "truth": labels[r["id"]],
                             **{k + "_pred": v for k, v in predictions.items()}, "uncertainty": uncertainty(r["fusion"]),
                             "disagreement": disagreement(r["optical"], r["sar"]),
                             "damaged_as_intact": int(labels[r["id"]] > 0 and predictions["fusion"] == 0)})
    print(json.dumps(report, indent=2), flush=True)
    write_json(run / "evaluation_receipt.json", {"policy_sha256": digest_file(run / "policy.lock.json"),
               "predictions_sha256": digest_file(run / "test_predictions.json"),
               "report_sha256": digest_file(run / "final_evaluation.json"),
               "per_building_sha256": digest_file(run / "per_building.csv")})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["weights", "features", "fit", "predict", "evaluate"])
    p.add_argument("--root", required=True)
    p.add_argument("--config", default="configs/pilot.lock.json")
    p.add_argument("--cloud-confirm", action="store_true")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-seconds", type=int, default=180)
    args = p.parse_args()
    check_cloud(args)
    {"weights": weights, "features": features, "fit": fit, "predict": predict, "evaluate": evaluate}[args.stage](args)


if __name__ == "__main__":
    main()
