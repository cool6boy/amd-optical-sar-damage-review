"""Generate tables, failures and an allowlisted downloadable package from actual outputs."""
import argparse
import csv
import html
import json
from pathlib import Path
import shutil
import zipfile
from .io import read_json, write_json, digest_file
from .review import pred, uncertainty


def table(path, rows):
    rows = list(rows)
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def generate(root, config):
    root, config = Path(root).resolve(), Path(config)
    run = root / "run"
    report = read_json(run / "final_evaluation.json")
    receipt = read_json(run / "prediction_receipt.json")
    final = read_json(run / "evaluation_receipt.json")
    for key, name in [("policy_sha256", "policy.lock.json"), ("predictions_sha256", "test_predictions.json"),
                      ("report_sha256", "final_evaluation.json"), ("per_building_sha256", "per_building.csv")]:
        if final[key] != digest_file(run / name):
            raise ValueError("Final evaluation changed: " + name)
    if receipt["predictions_sha256"] != digest_file(run / "test_predictions.json") or receipt["policy_sha256"] != digest_file(run / "policy.lock.json"):
        raise ValueError("Actual inference receipt mismatch")
    policy = read_json(run / "policy.lock.json")
    if policy["config_sha256"] != digest_file(config) or policy["manifest_sha256"] != digest_file(root / "manifest.json"):
        raise ValueError("Experiment config or manifest changed")
    if policy["heads_sha256"] != digest_file(run / "heads.pt"):
        raise ValueError("Trained heads do not match the locked policy")
    if not (root / "features_complete.json").is_file() or not (root / "environment.json").is_file():
        raise ValueError("Real feature/environment outputs are required")
    out = root / "delivery"
    if out.exists():
        raise ValueError("Delivery already exists. Preserve it and choose a new run root for another experiment.")
    out.mkdir()
    classification = []
    for mode, m in report["classification"].items():
        classification.append({"input": mode, "n": m["n"], "macro_f1": m["macro_f1"],
                               **{f"support_{i}": sum(row) for i, row in enumerate(m["confusion_matrix"])},
                               **{f"recall_{i}": v for i, v in enumerate(m["class_recall"])},
                               "damaged_as_intact": m["damaged_as_intact"],
                               "damaged_as_intact_rate": m["damaged_as_intact_rate"]})
    table(out / "classification_comparison.csv", classification)
    review = []
    for m in report["review"]:
        random = m["method"] == "random"
        row = {"method": m["method"], "requested_fraction": m["requested_fraction"], "review_count": m["review_count"]}
        for k in ["errors_found", "remaining_errors", "remaining_unreviewed_error_rate", "remaining_damaged_as_intact", "damaged_misses_found"]:
            row[k] = m.get(k + "_mean") if random else m.get(k)
            row[k + "_random_std"] = m.get(k + "_std") if random else None
        review.append(row)
    table(out / "review_comparison.csv", review)
    rows = read_json(run / "test_predictions.json")
    labels = read_json(root / "sealed" / "test_labels.json")
    # Final evaluation has already accessed held-out labels. These are case studies only.
    errors = [r for r in rows if pred(r["fusion"]) != labels[r["id"]]]
    errors.sort(key=lambda r: (not(labels[r["id"]] > 0 and pred(r["fusion"]) == 0), uncertainty(r["fusion"]), r["id"]))
    cases = []
    failures = out / "failures"
    failures.mkdir()
    for index, r in enumerate(errors[:6], 1):
        item = {"id": r["id"], "truth": labels[r["id"]], "fusion_prediction": pred(r["fusion"]),
                "fusion_probabilities": r["fusion"], "damaged_as_intact": labels[r["id"]] > 0 and pred(r["fusion"]) == 0}
        for mode in ["optical", "sar"]:
            src = (root / r[mode + "_path"]).resolve()
            if not src.is_relative_to(root / "crops") or src.suffix != ".png":
                raise ValueError("Invalid crop path")
            dst = failures / f"case_{index}_{mode}.png"
            shutil.copy2(src, dst)
            item[mode + "_image"] = dst.relative_to(out).as_posix()
        cases.append(item)
    write_json(out / "failure_cases.json", cases)
    files = ["environment.json", "doctor_cpu.json", "doctor_gpu.json", "manifest.json", "selected_tiles.json",
             "data_qc.json", "preparation_complete.json", "features_complete.json", "feature_progress.json", "sessions.json",
             "run/heads.pt", "run/policy.lock.json", "run/prediction_receipt.json", "run/evaluation_receipt.json", "run/test_predictions.json",
             "run/final_evaluation.json", "run/per_building.csv", "run/history.json", "run/validation_predictions.json"]
    for name in files:
        src = root / name
        if src.is_file():
            dst = out / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if name == "sessions.json":
                # No usernames, account balance, human free text or credentials in public delivery.
                fields = {"kind", "points_per_minute", "session_minutes", "started_at_utc", "auto_stop_at_utc",
                          "status", "actual_points", "actual_gpu_minutes", "reserved_points"}
                write_json(dst, [{k: v for k, v in r.items() if k in fields} for r in read_json(src)])
            else:
                shutil.copy2(src, dst)
    shutil.copy2(config, out / "experiment.lock.json")
    def htable(items):
        keys = list(items[0]) if items else []
        return '<table><tr>' + ''.join('<th>'+html.escape(k)+'</th>' for k in keys) + '</tr>' + ''.join(
            '<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in keys)+'</tr>' for r in items) + '</table>'
    sections = []
    for case in cases:
        sections.append('<h3>'+html.escape(case["id"])+ '</h3><p>Truth: '+str(case["truth"])+', fusion prediction: '+str(case["fusion_prediction"])+
                        '</p>'+''.join('<img width="224" src="'+case[m+"_image"]+'">' for m in ["optical", "sar"]))
    body = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>实验结果</title><style>body{font-family:system-ui;margin:36px;color:#183348}table{border-collapse:collapse;font-size:14px;display:block;overflow:auto}td,th{border:1px solid #aaa;padding:10px}img{margin-right:18px}</style><h1>建筑损毁分类与复核实验</h1><p>已知建筑位置条件下的分类。0=完整，1=受损，2=毁坏。未开展整幅检测或结构安全鉴定。</p><h2>分类对照</h2>'+htable(classification)+'<h2>理想复核模拟</h2><p>已有标签模拟纠错，不能视为真实人工审核效果。随机策略报告均值与标准差。</p>'+htable(review)+'<h2>失败案例</h2>'+(''.join(sections) if cases else '<p>当前测试子集未出现分类错误。不能据此断言模型可靠，请报告样本数与类别支持。</p>')+'<p>影像及标签遵守 BRIGHT 事件许可。数据来源：https://huggingface.co/datasets/Kullervo/BRIGHT 。光学/标签通常 CC BY-NC 4.0，SAR CC BY 4.0。课程非商业使用，保留来源归属。</p></html>'
    (out / "report.html").write_text(body, encoding="utf-8")
    (out / "README_RESULTS.md").write_text("# 实际结果包\n\n本目录由实际推理与最终评价生成，不代表已完成真实人工审查。\n\n打开 report.html 查看对照与失败案例。sessions.json 是生成时快照，最后会话停止后需补记实际消耗。\n\n训练/验证/测试按事件划分，小规模混合灾种探索不能代表全数据集。原始数据不随包分发。少量失败案例影像保留 BRIGHT 的原始事件许可，不能按代码 MIT 许可重新授权。权重仅为自有轻量分类头，不包含第三方主干权重。\n", encoding="utf-8")
    write_json(out / "SHA256SUMS.json", {p.relative_to(out).as_posix(): digest_file(p) for p in sorted(out.rglob("*")) if p.is_file()})
    archive = root / "course-results.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(out).as_posix())
    print(json.dumps({"report": str(out / "report.html"), "results_zip": str(archive), "bytes": archive.stat().st_size,
                      "failure_cases": len(cases), "note": "Before publishing, stop the server and update final actual usage."}, ensure_ascii=False, indent=2))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--config", default="configs/pilot.lock.json")
    a = p.parse_args()
    generate(a.root, a.config)


if __name__ == "__main__":
    main()
