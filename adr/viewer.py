"""Private crop review UI. Human judgments remain separate from oracle simulation."""
import argparse
from datetime import datetime, timezone
import html
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode
from .io import read_json, write_json
from .review import order, pred


def make_server(root, port=8765):
    root = Path(root).resolve()
    rows = read_json(root / "run" / "test_predictions.json")
    policy = read_json(root / "run" / "policy.lock.json")
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Nonempty unique prediction IDs required")
    by_id = {r["id"]: r for r in rows}
    names = ["完整", "受损", "毁坏"]
    csrf, lock = secrets.token_urlsafe(24), threading.Lock()
    judgments = root / "human_review.json"

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if urlparse(self.path).path != "/review":
                return self.send_error(404)
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 1 <= size <= 4096:
                    raise ValueError("Body too large")
                form = parse_qs(self.rfile.read(size).decode("utf-8"), keep_blank_values=True)
                if not secrets.compare_digest(form.get("csrf", [""])[0], csrf):
                    return self.send_error(403)
                ident, decision = form["id"][0], form["decision"][0]
                if ident not in by_id or decision not in {"0", "1", "2", "unknown"}:
                    raise ValueError("Invalid judgment")
                note = form.get("note", [""])[0]
                if len(note) > 500:
                    raise ValueError("Note too long")
                with lock:
                    saved = read_json(judgments) if judgments.exists() else {}
                    saved[ident] = {"decision": None if decision == "unknown" else int(decision),
                                    "status": "unable_to_judge" if decision == "unknown" else "human_judgment",
                                    "note": note, "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                                    "scope": "Crops inspected with model predictions visible; not ground truth or structural safety"}
                    write_json(judgments, saved)
                method = form.get("method", ["combined"])[0]
                if method not in {"random", "uncertainty", "combined"}:
                    method = "combined"
                index = max(0, int(form.get("index", [0])[0]))
                self.send_response(303)
                self.send_header("Location", "./?" + urlencode({"method": method, "index": index, "saved": 1}))
                self.end_headers()
            except (ValueError, KeyError, UnicodeError):
                self.send_error(400)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path.startswith("/crops/"):
                path = (root / parsed.path.lstrip("/")).resolve()
                if not path.is_relative_to(root / "crops") or path.suffix != ".png" or not path.is_file():
                    return self.send_error(404)
                body, mime = path.read_bytes(), "image/png"
            elif parsed.path == "/":
                query = parse_qs(parsed.query)
                method = query.get("method", ["combined"])[0]
                if method not in {"combined", "random", "uncertainty"}:
                    method = "combined"
                ranking = order(rows, method, policy["alpha"])
                try:
                    index = max(0, min(len(rows) - 1, int(query.get("index", [0])[0])))
                except ValueError:
                    index = 0
                row = by_id[ranking[index]]
                with lock:
                    saved = read_json(judgments) if judgments.exists() else {}
                current = saved.get(row["id"])
                description = "尚未保存判断" if current is None else "已保存：" + ("无法判断" if current["decision"] is None else names[current["decision"]])
                fields = "".join(f'<p>{html.escape(mode)}：{names[pred(row[mode])]}，分数 {max(row[mode]):.3f}</p>' for mode in ["optical", "sar", "fusion"])
                images = "".join('<figure><img width="280" src="./'+html.escape(row[m+"_path"], quote=True)+'"><figcaption>'+label+'</figcaption></figure>' for m, label in [("optical", "灾前光学"), ("sar", "灾后 SAR")])
                body = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>建筑复核查看器</title>
<style>body{{font-family:system-ui;max-width:1000px;margin:36px auto;color:#183348;background:#f7f9fb}}figure{{display:inline-block;margin:10px}}a{{margin-right:22px}}button,select,input{{font-size:18px;padding:8px}}input{{width:480px}}details{{margin-top:22px}}</style>
<h1>建筑复核查看器</h1><p>已知建筑位置分类研究，不代表整幅检测或现场安全鉴定。人工判断与理想复核模拟分别保存。</p>
<nav><a href="./?method=random">随机</a><a href="./?method=uncertainty">不确定性</a><a href="./?method=combined">不确定性与分歧</a></nav>
<h2>第 {index+1} / {len(rows)} 项</h2><p>{html.escape(row['id'])}</p>{images}{fields}
<p><a href="./?method={method}&index={max(0,index-1)}">上一项</a><a href="./?method={method}&index={min(len(rows)-1,index+1)}">下一项</a></p>
<form action="./review" method="post"><input type="hidden" name="csrf" value="{csrf}"><input type="hidden" name="id" value="{html.escape(row['id'], quote=True)}"><input type="hidden" name="method" value="{method}"><input type="hidden" name="index" value="{index}">
<label>人工判断 <select name="decision"><option value="unknown">无法判断</option><option value="0">完整</option><option value="1">受损</option><option value="2">毁坏</option></select></label><p><input name="note" maxlength="500" placeholder="观察依据，可留空"></p><button type="submit">保存判断</button></form><p>{description}。记录写入 human_review.json。</p>
<details><summary>解释边界</summary>分数未经概率校准。查看者能看到预测，判断可能受其影响。不展示隐藏测试标签。人工判断不自动覆盖数据集真值，也不计入理想复核模拟收益。</details></html>'''.encode("utf-8")
                mime = "text/html; charset=utf-8"
            else:
                return self.send_error(404)
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args()
    server = make_server(a.root, a.port)
    print(f"Private review viewer: http://127.0.0.1:{a.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
