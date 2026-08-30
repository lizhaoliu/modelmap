#!/usr/bin/env python3
"""Trending-coverage canary (design doc §29): can modelmap map what the Hub's
front page is showing people *today*?

Pulls the Hub's trending list, asks the deployed server for each model's
summary, and buckets the outcome:

  full / structural  the real map (config instantiated, maybe traced)
  weights            header-only fallback — degraded but honest
  gated              needs a license click + token; expected, not a failure
  unsupported        no readable format (pickle-only, ONNX…); expected
  FAIL               anything else — a 5xx, a timeout, a masked error

Exits non-zero only on FAIL rows, so a scheduled run stays green while the
expected buckets drift. Stdlib only; needs no token (and sends none).

  python scripts/canary.py [--base https://modelmap.cc] [--limit 20] [--md out.md]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

UA = {"User-Agent": "modelmap-canary"}


def _get(url: str, timeout: float = 240.0) -> tuple[int, dict | list | None]:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, None
    except Exception as e:
        print(f"  !! {url}: {e}", file=sys.stderr)
        return 0, None


def trending(limit: int) -> list[str]:
    st, body = _get(
        "https://huggingface.co/api/models?sort=trendingScore&limit="
        + str(limit + 10) + "&expand[]=trendingScore&expand[]=library_name"
    )
    if st != 200 or not isinstance(body, list):
        sys.exit(f"could not fetch the trending list (HTTP {st})")
    # keep repos of every library — the canary's whole point is seeing how the
    # site handles what's popular, not what it already supports
    return [m["id"] for m in body[:limit]]


def probe(base: str, mid: str) -> tuple[str, str]:
    """(bucket, detail) for one model against the deployed API."""
    for attempt in (0, 1):
        st, body = _get(f"{base}/api/summary/{urllib.parse.quote(mid)}")
        detail = (body or {}).get("detail", "") if isinstance(body, dict) else ""
        if st == 200 and isinstance(body, dict):
            fid = body.get("fidelity", "?")
            n = body.get("params_total")
            p = f"{n / 1e9:.1f}B" if isinstance(n, (int, float)) and n else "?"
            return fid, f"{p} params"
        if st == 403 and "gated" in detail:
            return "gated", detail[:80]
        if st == 422 and ("pickle checkpoints" in detail or "nothing modelmap can read" in detail):
            return "unsupported", detail[:80]
        # the Hub throttling the server's shared IP is the site being honest,
        # not broken: wait it out once, then report without failing the run
        if st == 503 and "rate-limiting" in detail:
            if attempt == 0:
                import re
                import time
                m = re.search(r"about (\d+) s", detail)
                time.sleep(min(int(m.group(1)) if m else 60, 240))
                continue
            return "hub-limited", detail[:80]
        break
    return "FAIL", f"HTTP {st}: {detail[:120]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="https://modelmap.cc")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--md", help="also write a markdown report (e.g. $GITHUB_STEP_SUMMARY)")
    args = ap.parse_args()

    rows: list[tuple[str, str, str]] = []
    for mid in trending(args.limit):
        bucket, detail = probe(args.base.rstrip("/"), mid)
        rows.append((mid, bucket, detail))
        print(f"{bucket:11s} {mid:55s} {detail}")

    counts: dict[str, int] = {}
    for _, b, _ in rows:
        counts[b] = counts.get(b, 0) + 1
    summary = "  ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    print(f"\n{len(rows)} trending models — {summary}")

    if args.md:
        icon = {"full": "🟢", "structural": "🟢", "weights": "🟡", "gated": "🔒", "unsupported": "⚪", "hub-limited": "⏳", "FAIL": "🔴"}
        with open(args.md, "a") as f:
            f.write(f"## modelmap trending canary — {args.base}\n\n{summary}\n\n")
            f.write("| model | outcome | detail |\n|---|---|---|\n")
            for mid, b, d in rows:
                f.write(f"| [{mid}]({args.base}/m/{urllib.parse.quote(mid)}) | {icon.get(b, '')} {b} | {d} |\n")

    fails = [(m, d) for m, b, d in rows if b == "FAIL"]
    if fails:
        print("\nFAILures:", file=sys.stderr)
        for m, d in fails:
            print(f"  {m}: {d}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
