"""Download curated BioModels ODE models (SBML) as untrusted data, for the option-(i) sourcing rule.

Files are written only into --out (a dedicated directory); nothing downloaded is ever executed. A manifest records each
model's identifier, name, file name, byte count and SHA-256.

  python biomodels_fetch.py --out <scratch>/dl/biomodels
"""

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://www.biomodels.org"
QUERY = '*:* AND curationstatus:"Manually curated" AND modellingapproach:"ordinary differential equation model"'


def get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "research-script (contact: repository owner)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def list_models():
    out, offset = [], 0
    while True:
        q = urllib.parse.quote(QUERY)
        d = json.loads(get(f"{BASE}/search?query={q}&offset={offset}&numResults=100&format=json"))
        batch = d.get("models", [])
        out += [{"id": m["id"], "name": m.get("name", "")} for m in batch]
        offset += len(batch)
        if not batch or offset >= d.get("matches", 0):
            return out, d.get("matches", 0)


def retry_failed(out, delay):
    """For entries whose guessed file name failed, look up the main SBML file name via the model's file listing."""
    mf = json.loads((out / "manifest.json").read_text())
    for entry in mf["models"]:
        if "error" not in entry:
            continue
        try:
            files = json.loads(get(f"{BASE}/{entry['id']}?format=json")).get("files", {}).get("main", [])
            fn = files[0]["name"]
            if not fn.endswith(".xml"):
                raise ValueError(f"main file {fn} is not XML")
            data = get(f"{BASE}/model/download/{entry['id']}?filename={urllib.parse.quote(fn)}")
            if b"<sbml" not in data[:5000]:
                raise ValueError("not SBML")
            (out / fn).write_bytes(data)
            entry.pop("error")
            entry.update(file=fn, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        except Exception as e:
            entry["error"] = f"retry {type(e).__name__}: {e}"[:200]
        time.sleep(delay)
    (out / "manifest.json").write_text(json.dumps(mf, indent=1))
    print("after retry:", sum("sha256" in x for x in mf["models"]), "files,", sum("error" in x for x in mf["models"]), "errors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--delay", type=float, default=0.3)
    ap.add_argument("--retry-failed", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.retry_failed:
        return retry_failed(out, a.delay)
    models, total = list_models()
    print("listed", len(models), "of", total, flush=True)
    manifest = []
    for i, m in enumerate(models):
        fn = f"{m['id']}_url.xml"
        entry = {**m, "file": fn}
        try:
            data = get(f"{BASE}/model/download/{m['id']}?filename={urllib.parse.quote(fn)}")
            if not data.lstrip().startswith(b"<?xml") and b"<sbml" not in data[:2000]:
                raise ValueError("not SBML")
            (out / fn).write_bytes(data)
            entry.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        except Exception as e:  # recorded, not retried silently
            entry["error"] = f"{type(e).__name__}: {e}"[:200]
        manifest.append(entry)
        if (i + 1) % 100 == 0:
            print(i + 1, "downloaded;", sum("error" in x for x in manifest), "errors", flush=True)
        time.sleep(a.delay)
    (out / "manifest.json").write_text(json.dumps({"query": QUERY, "listed": total, "models": manifest}, indent=1))
    print("done:", sum("sha256" in x for x in manifest), "files,", sum("error" in x for x in manifest), "errors")


if __name__ == "__main__":
    main()
