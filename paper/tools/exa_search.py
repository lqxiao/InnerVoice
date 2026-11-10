"""Minimal Exa search client with a hard usage cap (billing: $7 per 1,000 requests; stop at $10 => 1,428 requests; we cap at 120).
    python paper/tools/exa_search.py "query" [--n 5] [--contents]   -> prints title / url / published / snippet
Counter file: paper/tools/exa_usage.json (requests made so far)."""
import argparse, json, os, sys, requests
KEY_FILE = os.path.expanduser("~/exa/exa.key"); USAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exa_usage.json"); CAP = 120
ap = argparse.ArgumentParser(); ap.add_argument("query"); ap.add_argument("--n", type=int, default=5); ap.add_argument("--contents", action="store_true"); ap.add_argument("--category", default=None)
a = ap.parse_args()
u = json.load(open(USAGE)) if os.path.exists(USAGE) else {"requests": 0}
if u["requests"] >= CAP: sys.exit(f"Exa cap reached ({CAP} requests, ~${CAP*7/1000:.2f}); refusing to query")
body = {"query": a.query, "numResults": a.n, "type": "auto"}
if a.contents: body["contents"] = {"text": {"maxCharacters": 600}}
if a.category: body["category"] = a.category
r = requests.post("https://api.exa.ai/search", headers={"x-api-key": open(KEY_FILE).read().strip(), "Content-Type": "application/json"}, json=body, timeout=60)
u["requests"] += 1; json.dump(u, open(USAGE, "w"))
if r.status_code != 200: sys.exit(f"HTTP {r.status_code}: {r.text[:300]}")
for i, res in enumerate(r.json().get("results", [])):
    print(f"[{i}] {res.get('title')}\n    {res.get('url')}  {res.get('publishedDate', '')[:10]}  {res.get('author', '') or ''}")
    if a.contents and res.get("text"): print("    " + res["text"][:500].replace("\n", " "))
print(f"(exa requests used: {u['requests']}/{CAP}, ~${u['requests']*7/1000:.2f})")
