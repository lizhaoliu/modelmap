#!/usr/bin/env python3
"""What are people actually loading? Aggregates Cloud Run request logs for the
deployed service into the report that drives prioritization: top models by
distinct clients, failure buckets, referers.

Needs gcloud auth with access to the project (operator-only; not part of CI).

  python scripts/traffic.py [--days 7] [--project modelmap-260817] [--service modelmap]
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import urllib.parse


def _read(project: str, filt: str, days: int, fmt: str, limit: int = 4000) -> list[str]:
    out = subprocess.run(
        ["gcloud", "logging", "read", filt, f"--project={project}",
         f"--freshness={days}d", f"--limit={limit}", f"--format=value({fmt})"],
        capture_output=True, text=True, check=True,
    ).stdout
    return [l for l in out.splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--project", default="modelmap-260817")
    ap.add_argument("--service", default="modelmap")
    args = ap.parse_args()

    base = f'resource.type="cloud_run_revision" AND resource.labels.service_name="{args.service}"'
    rows = _read(
        args.project, base + ' AND httpRequest.requestUrl:"/api/graph/"', args.days,
        "httpRequest.requestUrl,httpRequest.status,httpRequest.remoteIp",
    )

    by_model_ips: dict[str, set] = collections.defaultdict(set)
    status = collections.Counter()
    fails = collections.Counter()
    all_ips = set()
    for row in rows:
        parts = row.split("\t")
        if len(parts) < 2:
            continue
        url, st = parts[0], parts[1].strip()
        ip = parts[2] if len(parts) > 2 else "?"
        mid = urllib.parse.unquote(urllib.parse.urlparse(url).path).split("/api/graph/", 1)[-1]
        by_model_ips[mid].add(ip)
        all_ips.add(ip)
        status[st] += 1
        if st not in ("200", "304", "429"):
            fails[(st, mid)] += 1

    print(f"== /api/graph, last {args.days}d: {len(rows)} requests, "
          f"{len(all_ips)} distinct clients, {len(by_model_ips)} unique models")
    print("status:", dict(status.most_common()))
    print("\n-- top models by distinct clients")
    for mid, ips in sorted(by_model_ips.items(), key=lambda kv: -len(kv[1]))[:25]:
        print(f"  {len(ips):4d}  {mid}")
    print("\n-- failures (status, model, count)")
    for (st, mid), c in fails.most_common(25):
        print(f"  {c:3d}  {st}  {mid}")

    print("\n-- referers on /m/ pages (where visitors come from)")
    refs = _read(args.project, base + ' AND httpRequest.requestUrl:"/m/"', args.days, "httpRequest.referer")
    ref_counts = collections.Counter(
        urllib.parse.urlparse(r).netloc or r for r in refs if r.strip()
    )
    for ref, c in ref_counts.most_common(15):
        print(f"  {c:4d}  {ref}")

    print("\njson:", json.dumps({"requests": len(rows), "clients": len(all_ips),
                                 "models": len(by_model_ips), "status": dict(status)}))


if __name__ == "__main__":
    main()
