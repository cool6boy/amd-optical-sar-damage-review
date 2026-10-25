"""One train-tile image/coordinate check; no ML dependencies, no full archives."""
import io
import ast
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from .io import read_json, write_json, sha256
from .remotezip import RemoteZip, ByteBudget
from .prepare import crop_box


def compatible_georef(images):
    a, b = images["optical"]["geo_tags"], images["sar"]["geo_tags"]
    if a["34735"] != b["34735"]:
        return False
    for tag in ("33550", "33922"):
        x, y = ast.literal_eval(a[tag]), ast.literal_eval(b[tag])
        if x is None or y is None or len(x) != len(y):
            return False
        if any(abs(float(i) - float(j)) > 1e-10 for i, j in zip(x, y)):
            return False
    return True


def main():
    from PIL import Image
    import numpy as np
    cfg = read_json("configs/pilot.lock.json")
    previous = read_json("evidence/data_preflight.json")
    member = previous["train_sample_member"]
    tile = member.rsplit("/", 1)[-1].removesuffix("_instance_damage.json")
    base = f'https://huggingface.co/datasets/{cfg["dataset"]}/resolve/{cfg["revision"]}/'
    budget = ByteBudget(12 * 1024 * 1024)
    annotations = RemoteZip(base + "cvprw26_trainval_instance_labels.zip", cfg["archive_sizes"]["cvprw26_trainval_instance_labels.zip"], budget)
    annotation = json.loads(annotations.read(member).decode("utf-8-sig"))
    a = max(annotation["annotations"], key=lambda x: x["bbox"][2] * x["bbox"][3])
    expected = annotation["images"][0]
    box = crop_box(a["bbox"], expected["width"], expected["height"], cfg["crop_context"])
    output = {"dataset_revision": cfg["revision"], "train_tile": tile, "bbox": a["bbox"], "crop_box": box,
              "train_sample_class_counts": dict(Counter(x["category_id"] for x in annotation["annotations"])),
              "images": {}, "gpu_used": False, "model_run": False}
    for mode, archive_name, suffix in [("optical", "pre-event.zip", "pre_disaster"), ("sar", "post-event.zip", "post_disaster")]:
        z = RemoteZip(base + archive_name, cfg["archive_sizes"][archive_name], budget)
        name = z.find_basename(f"{tile}_{suffix}.tif")
        data = z.read(name)
        image = Image.open(io.BytesIO(data))
        image.load()
        if image.size != (expected["width"], expected["height"]):
            raise ValueError("Annotation/image dimensions disagree")
        tags = {str(k): str(image.tag_v2.get(k)) for k in [33550, 33922, 34735]}
        array = np.asarray(image.crop(tuple(box))).astype(np.float32)
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        if array.shape[-1] != 3 or not np.isfinite(array).all():
            raise ValueError("Unsupported crop values")
        scaled = np.zeros_like(array, dtype=np.uint8)
        for band in range(3):
            lo, hi = np.percentile(array[:, :, band], [1, 99])
            if hi <= lo:
                raise ValueError("Uninformative training crop")
            scaled[:, :, band] = (np.clip((array[:, :, band] - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
        path = Path("evidence") / ("train_sample_" + mode + ".png")
        Image.fromarray(scaled).resize((224, 224), Image.Resampling.BILINEAR).save(path)
        output["images"][mode] = {"member": name, "source_sha256": sha256(data), "shape": image.size,
                                  "mode": image.mode, "source_bytes": len(data), "geo_tags": tags,
                                  "crop_file": str(path)}
        z.close()
    output["geo_tags_equal"] = output["images"]["optical"]["geo_tags"] == output["images"]["sar"]["geo_tags"]
    output["georef_compatible_within_tolerance"] = compatible_georef(output["images"])
    output["payload_bytes_received"] = budget.received
    output["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
    output["scope"] = "Single public training tile only. Source TIFFs read in RAM and not saved. Crop images are data QC, not predictions."
    write_json("evidence/image_preflight.json", output)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    annotations.close()


if __name__ == "__main__":
    main()
