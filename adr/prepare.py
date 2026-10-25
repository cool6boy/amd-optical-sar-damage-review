"""CPU/cloud-only: fetch selected ZIP members and create bbox-conditioned crops."""
import argparse
import io
import json
import random
import re
import sys
from pathlib import Path
from .io import read_json, write_json, sha256, ensure_disjoint, storage_used, digest_file
from .remotezip import RemoteZip, ByteBudget


def event_of(member):
    return re.sub(r"_\d+_instance_damage\.json$", "", member.rsplit("/", 1)[-1])


def crop_box(bbox, width, height, context):
    x, y, w, h = map(float, bbox)
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width + 1 or y + h > height + 1:
        raise ValueError("Invalid annotation bounding box")
    side = max(w, h) * context
    cx, cy = x + w / 2, y + h / 2
    import math
    return [max(0, math.floor(cx - side / 2)), max(0, math.floor(cy - side / 2)),
            min(width, math.ceil(cx + side / 2)), min(height, math.ceil(cy + side / 2))]


def tile_count(cfg, split):
    return int(cfg.get("tiles_per_event_by_split", {}).get(split, cfg["tiles_per_event"]))


def stratified_train_sample(candidates, seed=20261023, target_min_per_class=50, intact_cap=None):
    """Keep rare damaged/destroyed positives first, then cap intact counts under a fixed seed.

    The function is intentionally deterministic and used only for v2-style train selection.
    """
    if not candidates:
        return []
    groups = {0: [], 1: [], 2: []}
    for item in candidates:
        raw = item.get("label")
        if raw is None:
            raw = item.get("category_id")
            if raw is not None:
                raw = int(raw) - 1
        else:
            raw = int(raw)
        label = int(raw)
        if label not in groups:
            continue
        groups[label].append(item)
    rng = random.Random(str(seed))
    for label in groups:
        rng.shuffle(groups[label])
    selected = []
    for label in (1, 2):
        pool = groups[label]
        take = min(len(pool), max(0, target_min_per_class))
        selected.extend(pool[:take])
    intact = groups[0]
    if intact_cap is None:
        selected.extend(intact)
    else:
        selected.extend(intact[:max(0, intact_cap)])
    if not selected:
        return []
    rng.shuffle(selected)
    return selected


def validate_prepare_classes(rows, root, allow_missing_val_classes=False):
    """Ensure train is three-way complete; val must stay non-empty and may warn on missing classes for exploratory v2.

    The exploratory v2 run keeps the real class limit explicit in the log and JSON rather than hiding it or
    fabricating class labels.
    """
    for split in ("train", "val"):
        classes = {int(r["label"]) for r in rows if r.get("split") == split and r.get("label") is not None}
        observed = sorted(classes)
        if split == "train":
            if classes != {0, 1, 2}:
                payload = {"split": split, "observed_classes": observed, "required_classes": [0, 1, 2],
                           "allow_missing_val_classes": allow_missing_val_classes,
                           "action": "Expand only training tiles before any GPU work; do not inspect test labels."}
                write_json(root / "needs_more_data.json", payload)
                raise ValueError("Missing training classes: inspect needs_more_data.json before GPU startup")
            continue
        if not classes:
            payload = {"split": split, "observed_classes": observed, "required_classes": [0, 1, 2],
                       "allow_missing_val_classes": allow_missing_val_classes,
                       "action": "Validation split is empty; add valid rows before training."}
            write_json(root / "needs_more_data.json", payload)
            raise ValueError("Empty validation split: inspect needs_more_data.json before GPU startup")
        if classes != {0, 1, 2} and not allow_missing_val_classes:
            payload = {"split": split, "observed_classes": observed, "required_classes": [0, 1, 2],
                       "allow_missing_val_classes": allow_missing_val_classes,
                       "action": "Expand only validation tiles before any GPU work; do not inspect test labels."}
            write_json(root / "needs_more_data.json", payload)
            raise ValueError("Missing validation classes: inspect needs_more_data.json before GPU startup")
        if classes != {0, 1, 2} and allow_missing_val_classes:
            payload = {"split": split, "observed_classes": observed, "required_classes": [0, 1, 2],
                       "allow_missing_val_classes": allow_missing_val_classes,
                       "warning": "Exploratory v2 run permits missing validation classes and records the real limitation.",
                       "action": "Continue with the real validation split, but do not claim complete class coverage."}
            write_json(root / "needs_more_data.json", payload)
            print(json.dumps({"warning": "validation_missing_classes_allowed", "observed_classes": observed,
                              "split": split, "allow_missing_val_classes": True}, sort_keys=True), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pilot.lock.json")
    p.add_argument("--root", required=True)
    p.add_argument("--cloud-confirm", action="store_true")
    args = p.parse_args()
    if not args.cloud_confirm or sys.platform == "win32":
        raise SystemExit("Use a cloud CPU environment; --cloud-confirm is required")
    import numpy as np
    import rasterio
    from rasterio.io import MemoryFile
    from PIL import Image
    cfg, root = read_json(args.config), Path(args.root).resolve()
    ensure_disjoint(cfg["events"])
    if cfg["revision"] == "main":
        raise ValueError("Run metadata preflight to pin the dataset revision first")
    root.mkdir(parents=True, exist_ok=True)
    # An interrupted expansion must not leave an old completion marker usable.
    (root / "preparation_complete.json").unlink(missing_ok=True)
    (root / "crops").mkdir(exist_ok=True)
    budget = ByteBudget(cfg["network_byte_limit"])
    base = f'https://huggingface.co/datasets/{cfg["dataset"]}/resolve/{cfg["revision"]}/'
    archives = {name: RemoteZip(base + name, size, budget) for name, size in cfg["archive_sizes"].items()}
    labels = archives["cvprw26_trainval_instance_labels.zip"]
    names = sorted(n for n in labels.names() if n.endswith("_instance_damage.json"))
    # Tile selection is label-blind and locked by seed, including on held-out events.
    selected = []
    for split, events in cfg["events"].items():
        for event in events:
            candidates = [n for n in names if event_of(n) == event]
            random.Random(str(cfg["seed"]) + event).shuffle(candidates)
            count = tile_count(cfg, split)
            if count < 1 or len(candidates) < count:
                raise ValueError("Missing enough tiles: " + event)
            selected += [(split, event, n) for n in candidates[:count]]
    write_json(root / "selected_tiles.json", selected)
    rows, sealed, checks, hashes = [], {}, [], {}
    # Tile checkpoints exclude only the expansion size from their recipe hash.
    recipe = {k: v for k, v in cfg.items() if k not in {"tiles_per_event", "tiles_per_event_by_split"}}
    recipe_sha = sha256(json.dumps(recipe, sort_keys=True).encode())
    def register(h, split):
        if h in hashes and hashes[h] != split:
            raise ValueError("Duplicate source or crop crosses splits")
        hashes[h] = split
    def rgb(array):
        array = array[:3]
        if array.shape[0] == 1:
            array = np.repeat(array, 3, axis=0)
        if array.shape[0] != 3:
            raise ValueError("Unsupported band count")
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite crop; keep for manual investigation")
        out = np.zeros(array.shape, dtype=np.uint8)
        for band in range(3):
            lo, hi = np.percentile(array[band], [1, 99])
            if hi <= lo:
                raise ValueError("Uninformative crop; unable to judge")
            out[band] = (np.clip((array[band] - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(out.transpose(1, 2, 0))
    for split, event, member in selected:
        tile = member.rsplit("/", 1)[-1].removesuffix("_instance_damage.json")
        checkpoint = root / "tile_checkpoints" / (tile + ".json")
        if checkpoint.exists():
            saved = read_json(checkpoint)
            if saved.get("recipe_sha256") == recipe_sha and saved["split"] == split:
                valid = all((root / r[m + "_path"]).is_file() and
                            digest_file(root / r[m + "_path"]) == r["file_sha256"][m]
                            for r in saved["rows"] for m in ("optical", "sar"))
                if valid:
                    for r in saved["rows"]:
                        for h in list(r["source_hash"].values()) + list(r["crop_hash"].values()):
                            register(h, split)
                    rows.extend(saved["rows"])
                    sealed.update(saved["sealed"])
                    checks.extend(saved["checks"])
                    print(json.dumps({"tile": tile, "resumed_from_checkpoint": True}), flush=True)
                    continue
        qc_start, row_start = len(checks), len(rows)
        annotation_raw = labels.read(member)
        annotation = json.loads(annotation_raw.decode("utf-8-sig"))
        categories = {c["id"]: c["name"] for c in annotation["categories"]}
        if categories != {1: "intact", 2: "damaged", 3: "destroyed"}:
            raise ValueError("Unexpected category mapping")
        arrays, geo, source_hash = {}, {}, {}
        for mode, archive_name, suffix in [("optical", "pre-event.zip", "pre_disaster"), ("sar", "post-event.zip", "post_disaster")]:
            z = archives[archive_name]
            filename = z.find_basename(f"{tile}_{suffix}.tif")
            data = z.read(filename)
            h = sha256(data)
            if h in hashes and hashes[h] != split:
                raise ValueError("Duplicate source imagery crosses splits")
            hashes[h], source_hash[mode] = split, h
            with MemoryFile(data) as mf:
                with mf.open() as ds:
                    arrays[mode] = ds.read().astype(np.float32)
                    geo[mode] = {"crs": str(ds.crs), "transform": list(ds.transform), "shape": [ds.height, ds.width]}
        expected = annotation["images"][0]
        for array in arrays.values():
            if list(array.shape[1:]) != [expected["height"], expected["width"]]:
                raise ValueError("Image/annotation shape mismatch")
        if geo["optical"]["crs"] != geo["sar"]["crs"] or not np.allclose(geo["optical"]["transform"], geo["sar"]["transform"], rtol=0, atol=1e-7):
            raise ValueError("Georeferencing mismatch; do not silently assume alignment")
        annotations = [a for a in annotation["annotations"] if not a.get("iscrowd", 0)]
        if split == "train" and cfg.get("sample_policy") == "stratified_train":
            relabeled = [{**a, "label": int(a["category_id"]) - 1} for a in annotations]
            annotations = stratified_train_sample(
                relabeled,
                seed=str(cfg["seed"]) + "::" + tile,
                target_min_per_class=int(cfg.get("train_target_min_per_class", 50)),
                intact_cap=int(cfg.get("train_intact_cap", 200)),
            )
            if cfg.get("max_buildings_per_tile") is not None:
                annotations = annotations[:cfg["max_buildings_per_tile"]]
        else:
            # Selection never uses class labels; note this capped subset is not population prevalence.
            random.Random(tile + str(cfg["seed"])).shuffle(annotations)
            annotations = annotations[:cfg["max_buildings_per_tile"]]
        for a in annotations:
            ident = tile + "__" + str(a["id"])
            row = {"id": ident, "tile": tile, "event": event, "building_id": ident, "split": split,
                   "bbox": a["bbox"], "label": None if split == "test" else int(a["category_id"]) - 1,
                   "source_hash": source_hash, "annotation_sha256": sha256(annotation_raw)}
            row["crop_hash"], row["file_sha256"] = {}, {}
            try:
                if min(a["bbox"][2:]) < cfg["min_bbox_side"]:
                    raise ValueError("Building too small for this pilot")
                x0, y0, x1, y1 = crop_box(a["bbox"], expected["width"], expected["height"], cfg["crop_context"])
                row["crop_box"] = [x0, y0, x1, y1]
                for mode, array in arrays.items():
                    dst = root / "crops" / (ident + "_" + mode + ".png")
                    image = rgb(array[:, y0:y1, x0:x1])
                    crop_hash = sha256(image.tobytes())
                    if crop_hash in hashes and hashes[crop_hash] != split:
                        raise ValueError("Duplicate crop crosses splits")
                    hashes[crop_hash] = split
                    row["crop_hash"][mode] = crop_hash
                    image.resize((224, 224), Image.Resampling.BILINEAR).save(dst)
                    row[mode + "_path"] = str(dst.relative_to(root))
                    row["file_sha256"][mode] = digest_file(dst)
                row["status"] = "ready"
                rows.append(row)
                if split == "test":
                    sealed[ident] = int(a["category_id"]) - 1
            except ValueError as error:
                if "crosses splits" in str(error):
                    raise
                checks.append({"id": ident, "status": "unable_to_judge", "reason": str(error)})
        if storage_used(root) > cfg["storage_byte_limit"]:
            raise RuntimeError("Storage budget exceeded")
        # Per-tile checkpoint supports interruption and reproducible re-entry.
        write_json(checkpoint, {"recipe_sha256": recipe_sha, "split": split, "rows": rows[row_start:],
                               "sealed": {r["id"]: sealed[r["id"]] for r in rows[row_start:] if split == "test"},
                               "checks": checks[qc_start:]})
        write_json(root / "manifest.json", rows)
        write_json(root / "sealed" / "test_labels.json", sealed)
        write_json(root / "data_qc.json", checks)
        write_json(root / "preparation_progress.json", {"completed_tile": tile, "rows": len(rows), "network_bytes": budget.received, "geo": geo})
        print(json.dumps({"tile": tile, "ready_buildings": len(rows), "network_bytes": budget.received}), flush=True)
    for z in archives.values():
        z.close()
    write_json(root / "manifest.json", rows)
    write_json(root / "sealed" / "test_labels.json", sealed)
    write_json(root / "data_qc.json", checks)
    validate_prepare_classes(rows, root, allow_missing_val_classes=bool(cfg.get("allow_missing_val_classes", False)))
    completion = {"manifest_sha256": digest_file(root / "manifest.json"), "config_sha256": digest_file(args.config),
                  "buildings": len(rows), "network_bytes": budget.received, "scope": cfg["scope"],
                  "allow_missing_val_classes": bool(cfg.get("allow_missing_val_classes", False))}
    if bool(cfg.get("allow_missing_val_classes", False)):
        completion["warning"] = "Exploratory v2 validation keeps the real missing-class limitation explicit; no labels were fabricated."
    write_json(root / "preparation_complete.json", completion)


if __name__ == "__main__":
    main()
