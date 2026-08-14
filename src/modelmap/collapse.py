"""Structural signatures and repeat collapsing (design doc §06).

Signatures hash a subtree's kinds, classes, and weight shapes — never the
numeric index in a module's name — so a "×36" stack is a structural fact, not
a naming convention. Runs of ≥ MIN_RUN consecutive identical siblings ship as
one representative subtree plus a Repeat record.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

from modelmap.schema import Node, Repeat

MIN_RUN = 3


def collapse_repeats(nodes: list[Node]) -> tuple[list[Node], list[Repeat]]:
    children: dict[str, list[Node]] = defaultdict(list)
    for n in nodes:
        if n.parent is not None:
            children[n.parent].append(n)
    for ks in children.values():
        ks.sort(key=lambda n: n.order)

    sigs: dict[str, str] = {}

    def sig(node: Node) -> str:
        parts = [node.kind, node.cls, json.dumps(node.weight_shapes or {}, sort_keys=True)]
        parts += [sig(c) for c in children.get(node.id, [])]
        h = hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]
        sigs[node.id] = h
        return h

    for root in (n for n in nodes if n.parent is None):
        sig(root)

    repeats: list[Repeat] = []
    doomed: set[str] = set()
    for parent, ks in children.items():
        i = 0
        while i < len(ks):
            j = i
            while j + 1 < len(ks) and sigs[ks[j + 1].id] == sigs[ks[i].id]:
                j += 1
            run = ks[i : j + 1]
            if len(run) >= MIN_RUN:
                repeats.append(Repeat(
                    parent=parent,
                    representative=run[0].id,
                    count=len(run),
                    signature=sigs[run[0].id],
                    members=[n.id.rsplit(".", 1)[-1] for n in run],
                ))
                for n in run[1:]:
                    doomed.add(n.id)
                    doomed.update(_descendants(n.id, children))
            i = j + 1

    return [n for n in nodes if n.id not in doomed], repeats


def _descendants(nid: str, children: dict[str, list[Node]]) -> set[str]:
    out: set[str] = set()
    stack = [nid]
    while stack:
        for c in children.get(stack.pop(), []):
            out.add(c.id)
            stack.append(c.id)
    return out
