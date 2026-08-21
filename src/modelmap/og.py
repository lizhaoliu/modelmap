"""Social preview cards (design doc §25): the picture a link unfurls into.

Every shareable page gets a 1200×630 PNG drawn from the cached graph — the
model id, its headline numbers, the structural tags the zoo derives, and a
miniature of the architecture itself (top-level modules, the repeated block
opened to show attention / MLP / MoE proportions). Crawlers never run JS, so
the server injects the matching <meta property="og:*"> tags into index.html
per URL (server.py) and serves the PNGs from /og/….

Pillow only — no browser. Fonts ship with the package (OFL: IBM Plex Sans,
Bricolage Grotesque) so the card looks like the site on a fontless host.
"""

from __future__ import annotations

import io
import math
from functools import lru_cache
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from modelmap.analytics import Assumptions, _num, build_index, compute_costs, fmt_bytes, fmt_params

W, H = 1200, 630
_FONTS = Path(__file__).parent / "assets" / "fonts"

# dark theme palette (web/src/theme.css)
PAPER = (18, 22, 29)
INK = (232, 236, 242)
DIM = (147, 160, 178)
LINE = (42, 51, 64)
FLOW = (255, 180, 84)
KIND_RGB: dict[str, tuple[int, int, int]] = {
    "attention": (226, 126, 162),
    "mlp": (79, 179, 167),
    "moe": (79, 179, 167),
    "embedding": (129, 148, 232),
    "norm": (126, 140, 160),
    "head": (129, 148, 232),
    "linear": (147, 160, 178),
    "conv": (147, 160, 178),
    "container": (147, 160, 178),
    "module": (147, 160, 178),
}


@lru_cache(maxsize=64)
def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(_FONTS / name), size)


def display(size: int) -> ImageFont.FreeTypeFont:
    return _font("BricolageGrotesque-Bold.ttf", size)


def body(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return _font("IBMPlexSans-SemiBold.ttf" if bold else "IBMPlexSans-Regular.ttf", size)


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))  # type: ignore[return-value]


def _fit(draw: ImageDraw.ImageDraw, text: str, font_of, max_w: int, start: int, floor: int = 18) -> ImageFont.FreeTypeFont:
    """Largest size ≤ start at which `text` fits in max_w."""
    size = start
    while size > floor and draw.textlength(text, font=font_of(size)) > max_w:
        size -= 2
    return font_of(size)


def _ellipsize(draw: ImageDraw.ImageDraw, text: str, font, max_w: int) -> str:
    if draw.textlength(text, font=font) <= max_w:
        return text
    while text and draw.textlength(text + "…", font=font) > max_w:
        text = text[:-1]
    return text + "…"


def _canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(im)
    # the site's dot grid
    for y in range(26, H, 26):
        for x in range(26, W, 26):
            d.point((x, y), fill=(30, 36, 46))
    return im, d


def _chrome(d: ImageDraw.ImageDraw, foot: str = "modelmap.cc") -> None:
    d.text((56, 40), "modelmap", font=display(34), fill=INK)
    wl = d.textlength("modelmap", font=display(34))
    d.ellipse((56 + wl + 10, 56, 56 + wl + 20, 66), fill=FLOW)
    d.text((W - 56 - d.textlength(foot, font=body(22)), 48), foot, font=body(22), fill=DIM)


def _chips(d: ImageDraw.ImageDraw, tags: list[str], x: int, y: int, max_x: int, size: int = 22) -> int:
    f = body(size, bold=True)
    for t in tags:
        tw = d.textlength(t, font=f) + 26
        if x + tw > max_x:
            break
        d.rounded_rectangle((x, y, x + tw, y + size + 16), radius=(size + 16) // 2, outline=LINE, width=2)
        d.text((x + 13, y + 7), t, font=f, fill=FLOW if t.startswith(("moe", "mla")) else INK)
        x += tw + 10
    return x


# ---------------------------------------------------------------- model card


def _facts(doc: dict, rep) -> tuple[str, list[str]]:
    c = dict(doc.get("config") or {})
    tc = c.get("text_config")
    if isinstance(tc, dict):
        for k, v in tc.items():
            c.setdefault(k, v)
    parts = []
    total = doc.get("params_total") or 0
    active = rep.root.active_params
    if total:
        parts.append(f"{fmt_params(total)} params" + (f" · {fmt_params(active)} active" if active and active < total * 0.98 else ""))
    layers = _num(c, "num_hidden_layers", "n_layer", "num_layers", "encoder_layers")
    dec_layers = _num(c, "num_decoder_layers", "decoder_layers")
    if layers:
        parts.append(f"{int(layers)} layers" if not c.get("is_encoder_decoder") else f"{int(layers)} + {int(dec_layers or layers)} layers")
    hidden = _num(c, "hidden_size", "n_embd", "d_model")
    if hidden:
        parts.append(f"hidden {int(hidden):,}")
    heads = _num(c, "num_attention_heads", "n_head", "num_heads", "encoder_attention_heads")
    kv = _num(c, "num_key_value_heads")
    if heads:
        parts.append(f"{int(heads)} heads" + (f" / {int(kv)} KV" if kv and kv != heads else ""))
    ctx = _num(c, "max_position_embeddings", "n_positions")
    if ctx:
        parts.append(f"ctx {int(ctx):,}" if ctx < 8192 else f"ctx {int(ctx) // 1024}k" if ctx < 1 << 20 else f"ctx {ctx / (1 << 20):g}M")
    second = []
    if rep.root.param_bytes:
        second.append(f"weights {fmt_bytes(rep.root.param_bytes)}")
    if rep.root.kv_per_token:
        second.append(f"KV {fmt_bytes(rep.root.kv_per_token)}/token")
    if rep.root.macs:
        second.append(f"{fmt_params(rep.root.macs / 4096)} MACs/token")
    return " · ".join(parts), second


def _strip_prefix(cls: str) -> str:
    """T5LayerFF → LayerFF: drop the model-family prefix from a class name."""
    import re

    m = re.match(r"^[A-Z][a-z0-9]*(?:[A-Z][a-z0-9]*)*?(?=Layer|Block|Attention|MLP|Dense|Feed)", cls)
    return cls[m.end():] if m and m.end() < len(cls) else cls


def _strip(d: ImageDraw.ImageDraw, doc: dict, index, box: tuple[int, int, int, int]) -> None:
    """A miniature of the architecture: top-level modules in execution order,
    the largest repeated stack opened one level to show its block recipe."""
    x0, y0, x1, y1 = box
    top = [n for n in index.children.get("", []) if n.get("kind") != "module" or n.get("params")]
    if not top:
        top = index.children.get("", [])
    # the biggest stack gets the room: its block recipe is the interesting part
    def rep_of(n):
        # a repeat is recorded on the *parent* container (model.layers) with a
        # representative child (model.layers.0): the container is the stack
        reps = index.repeats_by_parent.get(n["id"])
        if reps:
            return max(reps, key=lambda r: r["count"])
        return index.repeat_by_rep.get(n["id"])

    def nested_stack(n):
        """A stack inside a tower (visual → blocks ×32): the largest repeat
        within two levels, with the container that holds it."""
        frontier = [(n, 0)]
        best = None
        while frontier:
            cur, depth = frontier.pop(0)
            for r in index.repeats_by_parent.get(cur["id"], []):
                if best is None or r["count"] * index.by_id[r["representative"]]["params"] > best[0]:
                    best = (r["count"] * index.by_id[r["representative"]]["params"], r, cur)
            if depth < 2:
                frontier.extend((k, depth + 1) for k in index.children.get(cur["id"], []))
        return best[1:] if best else None

    items: list[dict] = []
    total_params = doc.get("params_total") or 1
    # a bare container holding the real stack (`model` in *ForCausalLM, or
    # `language_model` inside it) is opened so the chain reads embed → blocks → head
    frontier = list(top)
    depth = 0
    while frontier and depth < 3:
        nxt: list[dict] = []
        opened = False
        for n in frontier:
            kids = index.children.get(n["id"], [])
            if n["kind"] in ("container", "module") and kids and not rep_of(n) and n["params"] > 0.5 * total_params and len(frontier) <= 3:
                nxt.extend(kids)
                opened = True
            else:
                nxt.append(n)
        frontier = nxt
        depth += 1
        if not opened:
            break
    items = [n for n in frontier if n.get("params", 0) > 0 or rep_of(n)][:12]
    if not items:
        return
    gap = 22
    avail = (x1 - x0) - gap * (len(items) - 1)
    # every module gets a legible base width; repeated stacks share the rest
    # in proportion to sqrt(params) so a 671B stack doesn't crush its neighbours
    base = max(70, min(140, avail // max(1, len(items))))
    stacks = [i for i, n in enumerate(items) if rep_of(n) or nested_stack(n)]
    widths = [base] * len(items)
    if stacks:
        spare = avail - base * len(items)
        raw = [math.sqrt(max(1, items[i].get("params", 0))) for i in stacks]
        for i, r in zip(stacks, raw):
            widths[i] = base + int(spare * r / sum(raw))
    else:
        total = sum(max(1, n.get("params", 0)) for n in items)
        raw = [math.sqrt(max(1, n.get("params", 0)) / total) for n in items]
        widths = [max(base, int(avail * r / sum(raw))) for r in raw]
        scale = avail / sum(widths)
        widths = [int(w * scale) for w in widths]
    x = x0
    hbox = y1 - y0
    for n, w in zip(items, widths):
        r = rep_of(n)
        nested = None if r else nested_stack(n)
        if nested:
            r, holder = nested
        kind = n.get("kind", "module")
        col = KIND_RGB.get(kind, KIND_RGB["module"])
        if r:
            # stacked cards: the ×N repeat
            for k in (2, 1):
                d.rounded_rectangle((x + 6 * k, y0 - 6 * k, x + w + 6 * k, y1 - 6 * k), radius=12, fill=_mix(PAPER, LINE, 0.5), outline=LINE, width=2)
        d.rounded_rectangle((x, y0, x + w, y1), radius=12, fill=_mix(PAPER, col, 0.16), outline=col, width=3)
        label = n["id"].split(".")[-1] or "model"
        if nested:
            label = f"{label} · {holder['id'].split('.')[-1]} ×{r['count']}"
        elif r:
            total_count = sum(q["count"] for q in index.repeats_by_parent.get(n["id"], [r]))
            label = f"{label.rstrip('0123456789.') or label} ×{total_count}"
        f = _fit(d, label, lambda sz: body(sz, bold=True), w - 20, 24, 14)
        d.text((x + 12, y0 + 12), _ellipsize(d, label, f, w - 22), font=f, fill=INK)
        # inner recipe for the stack: the representative block's children
        inner = index.children.get(r["representative"], []) if r else []
        # a block that is just a wrapper (T5: block → layer → [attn, ff]) opens once more
        for _ in range(2):
            if len(inner) == 1 and index.children.get(inner[0]["id"]):
                inner = index.children.get(inner[0]["id"], [])
        inner = [m for m in inner if m.get("params", 0) > 0][:6]
        if inner and w > 140:
            iy0, iy1 = y0 + 52, y1 - 16
            ix = x + 12
            iw_avail = w - 24 - 8 * (len(inner) - 1)
            # norms are slivers; the real blocks share the rest by sqrt(params)
            big = [m for m in inner if m.get("kind") != "norm"]
            iws = [24 if m.get("kind") == "norm" else 96 for m in inner]
            spare = iw_avail - sum(iws)
            if big and spare > 0:
                wts = {m["id"]: math.sqrt(max(1, m["params"])) for m in big}
                tot = sum(wts.values())
                iws = [iw + (int(spare * wts[m["id"]] / tot) if m["id"] in wts else 0) for m, iw in zip(inner, iws)]
            isc = min(1.0, iw_avail / max(1, sum(iws)))
            for m, iw in zip(inner, iws):
                iw = int(iw * isc)
                mc = KIND_RGB.get(m.get("kind", "module"), KIND_RGB["module"])
                d.rounded_rectangle((ix, iy0, ix + iw, iy1), radius=8, fill=_mix(PAPER, mc, 0.45), outline=mc, width=2)
                lab = m["id"].split(".")[-1]
                if lab.isdigit():  # T5-style numbered sublayers: say what they are
                    k = m.get("kind")
                    lab = {"attention": "attn", "mlp": "mlp", "moe": "moe", "norm": "norm"}.get(k) or _strip_prefix(m.get("cls") or lab)
                if m.get("kind") == "moe" and "moe" not in lab:
                    lab += " · moe"
                fi = _fit(d, lab, lambda sz: body(sz), iw - 10, 18, 11)
                if iw > 30:
                    d.text((ix + 6, iy0 + 6), _ellipsize(d, lab, fi, iw - 12), font=fi, fill=INK)
                    sub = fmt_params(m["params"])
                    if hbox > 120:
                        d.text((ix + 6, iy1 - 26), sub, font=body(14), fill=DIM)
                ix += iw + 8
        elif n.get("params"):
            d.text((x + 12, y1 - 34), fmt_params(n["params"]), font=body(18), fill=DIM)
        # arrow to the next
        if n is not items[-1]:
            ax = x + w + (18 if r else 0)
            my = (y0 + y1) // 2
            d.line((x + w + (12 if r else 0), my, ax + gap - 6, my), fill=DIM, width=3)
            d.polygon([(ax + gap - 2, my), (ax + gap - 10, my - 6), (ax + gap - 10, my + 6)], fill=DIM)
        x += w + gap


def render_model_card(doc: dict) -> bytes:
    from modelmap.zoo import structural_tags

    im, d = _canvas()
    _chrome(d)
    index = build_index(doc)
    rep = compute_costs(doc, index, Assumptions(T=4096, B=1, dtype="bf16"))
    mid = str(doc.get("model_id") or "model")
    if doc.get("variant"):
        mid += f" · {doc['variant']}"
    owner, _, name = mid.rpartition("/")
    y = 104
    if owner:
        d.text((56, y), owner + " /", font=body(30), fill=DIM)
        y += 40
    f = _fit(d, name, display, W - 112, 72, 30)
    d.text((56, y), name, font=f, fill=INK)
    y += f.size + 22
    facts, second = _facts(doc, rep)
    ff = _fit(d, facts, lambda s: body(s, bold=True), W - 112, 28, 18)
    d.text((56, y), facts, font=ff, fill=INK)
    y += ff.size + 14
    tags = structural_tags(doc)
    arch = doc.get("architecture") or ""
    fid = doc.get("fidelity")
    chips = ([arch] if arch else []) + [t for t in tags if not t.startswith("ctx ")] + ([f"{fid} fidelity"] if fid and fid != "full" else [])
    _chips(d, chips, 56, y, W - 56)
    y += 56
    # architecture strip
    strip_top = max(y + 10, 330)
    _strip(d, doc, index, (56, strip_top, W - 56, 540))
    foot = " · ".join(second) if second else "interactive map · forward-pass replay · GPU fit planner"
    d.text((56, 572), foot, font=body(22), fill=DIM)
    tail = "interactive map · flow replay · fits on my GPU?"
    d.text((W - 56 - d.textlength(tail, font=body(22)), 572), tail, font=body(22), fill=DIM)
    return _png(im)


# ---------------------------------------------------------------- other cards


def render_default_card() -> bytes:
    im, d = _canvas()
    _chrome(d)
    d.text((56, 150), "Every model, mapped.", font=display(78), fill=INK)
    line = "Paste a Hugging Face id → an interactive, animated architecture map —"
    d.text((56, 250), line, font=body(30), fill=DIM)
    d.text((56, 292), "and the answer to “will it fit on my GPU?”", font=body(30), fill=DIM)
    # decorative decoder chain
    seq = [("embed", "embedding", 130), ("block ×32", "attention", 420), ("norm", "norm", 110), ("lm_head", "head", 180)]
    x, y0, y1 = 56, 390, 520
    for i, (lab, kind, w) in enumerate(seq):
        col = KIND_RGB[kind]
        if "×" in lab:
            for k in (2, 1):
                d.rounded_rectangle((x + 6 * k, y0 - 6 * k, x + w + 6 * k, y1 - 6 * k), radius=12, fill=_mix(PAPER, LINE, 0.5), outline=LINE, width=2)
            d.rounded_rectangle((x, y0, x + w, y1), radius=12, fill=_mix(PAPER, DIM, 0.12), outline=DIM, width=3)
            d.text((x + 12, y0 + 12), lab, font=body(24, bold=True), fill=INK)
            ix = x + 12
            for il, ik, iw in (("attention", "attention", 150), ("mlp", "mlp", 230)):
                ic = KIND_RGB[ik]
                d.rounded_rectangle((ix, y0 + 52, ix + iw, y1 - 16), radius=8, fill=_mix(PAPER, ic, 0.45), outline=ic, width=2)
                d.text((ix + 8, y0 + 60), il, font=body(18), fill=INK)
                ix += iw + 8
        else:
            d.rounded_rectangle((x, y0, x + w, y1), radius=12, fill=_mix(PAPER, col, 0.16), outline=col, width=3)
            d.text((x + 12, y0 + 12), lab, font=body(24, bold=True), fill=INK)
        if i < len(seq) - 1:
            my = (y0 + y1) // 2
            d.line((x + w + 12, my, x + w + 36, my), fill=DIM, width=3)
            d.polygon([(x + w + 40, my), (x + w + 32, my - 6), (x + w + 32, my + 6)], fill=DIM)
        x += w + 42
    d.text((56, 572), "no weights downloaded · open source (MIT)", font=body(22), fill=DIM)
    return _png(im)


def render_family_card(family: dict, entries: dict[str, dict | None]) -> bytes:
    im, d = _canvas()
    _chrome(d)
    d.text((56, 110), family["title"], font=display(64), fill=INK)
    d.text((56, 196), "architecture lineage · every arrow is a live structural diff", font=body(26), fill=DIM)
    y = 262
    members = family["members"]
    f = body(24, bold=True)
    chips = []
    for m in members:
        e = entries.get(m)
        name = m.rpartition("/")[2]
        sub = ""
        if e and e.get("params_total"):
            sub = fmt_params(e["params_total"])
            act = e.get("active_params") or 0
            if act and act < 0.98 * e["params_total"]:
                sub += f" · {fmt_params(act)} active"
        tw = max(d.textlength(name, font=f), d.textlength(sub, font=body(18))) + 28
        chips.append((name, sub, tw))
    x = 56
    for i, (name, sub, tw) in enumerate(chips):
        if x + tw > W - 56:
            x = 56
            y += 96
        d.rounded_rectangle((x, y, x + tw, y + 72), radius=12, fill=_mix(PAPER, KIND_RGB["attention"], 0.10), outline=LINE, width=2)
        d.text((x + 14, y + 10), name, font=f, fill=INK)
        if sub:
            d.text((x + 14, y + 42), sub, font=body(18), fill=DIM)
        # arrow only when the next chip stays on this line
        if i < len(chips) - 1 and x + tw + 44 + chips[i + 1][2] <= W - 56:
            d.line((x + tw + 6, y + 36, x + tw + 30, y + 36), fill=DIM, width=3)
            d.polygon([(x + tw + 34, y + 36), (x + tw + 26, y + 30), (x + tw + 26, y + 42)], fill=DIM)
        x += tw + 44
    blurb = family.get("blurb") or ""
    # wrap the blurb into ≤3 lines
    words, lines, cur = blurb.split(), [], ""
    fb = body(22)
    for w_ in words:
        t = (cur + " " + w_).strip()
        if d.textlength(t, font=fb) > W - 112:
            lines.append(cur)
            cur = w_
        else:
            cur = t
    if cur:
        lines.append(cur)
    yy = max(y + 110, 430)
    for ln in lines[:3]:
        d.text((56, yy), ln, font=fb, fill=DIM)
        yy += 32
    return _png(im)


def render_compare_card(da: dict, db: dict) -> bytes:
    im, d = _canvas()
    _chrome(d)
    ia, ib = build_index(da), build_index(db)
    ra = compute_costs(da, ia, Assumptions(T=4096, B=1, dtype="bf16"))
    rb = compute_costs(db, ib, Assumptions(T=4096, B=1, dtype="bf16"))
    col_w = (W - 112 - 60) // 2
    for i, (doc, rep) in enumerate(((da, ra), (db, rb))):
        x = 56 + i * (col_w + 60)
        mid = str(doc.get("model_id") or "")
        owner, _, name = mid.rpartition("/")
        d.text((x, 110), owner + " /" if owner else "", font=body(22), fill=DIM)
        f = _fit(d, name, display, col_w, 44, 22)
        d.text((x, 140), name, font=f, fill=INK)
        facts, _ = _facts(doc, rep)
        yy = 200
        for part in facts.split(" · ")[:5]:
            d.text((x, yy), part, font=body(24), fill=INK)
            yy += 32
    d.text((W // 2 - 18, 150), "vs", font=display(40), fill=FLOW)
    # the takeaways are the shareable part: up to three, wrapped
    from modelmap.insights import insights

    y = 378
    d.line((56, y - 12, W - 56, y - 12), fill=LINE, width=2)
    fb = body(22)
    # the structural story first; raw size last
    order = ["moe", "attention", "kv", "mlp", "positions", "rope_scaling", "norm", "qk_norm", "vision", "seq2seq", "shape", "context", "params"]
    rank = {t: i for i, t in enumerate(order)}
    items = sorted(insights(da, db), key=lambda it: rank.get(it["topic"], len(order)))
    for it in items[:4]:
        lines = _wrap(d, it["text"], fb, W - 112 - 24)[:2]
        if y + 30 * len(lines) > 560:
            break
        d.polygon([(58, y + 8), (58, y + 22), (68, y + 15)], fill=FLOW)
        for ln in lines:
            d.text((80, y), ln, font=fb, fill=INK)
            y += 30
        y += 6
    d.text((56, 572), "module-by-module diff · config changes · takeaways", font=body(22), fill=DIM)
    return _png(im)


def _wrap(d: ImageDraw.ImageDraw, text: str, font, max_w: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w_ in words:
        t = (cur + " " + w_).strip()
        if d.textlength(t, font=font) > max_w and cur:
            lines.append(cur)
            cur = w_
        else:
            cur = t
    if cur:
        lines.append(cur)
    return lines


def _png(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ---------------------------------------------------------------- meta tags

SITE = "https://modelmap.cc"


def meta_for(path: str, query: dict[str, str], *, site: str = SITE) -> dict[str, str]:
    """og:* / twitter:* values for a page URL; `image` is relative to `site`.
    Pure function of the URL — what the crawler should see, cache or not."""
    from urllib.parse import quote

    q = lambda v: quote(v, safe="/:@.-_~")  # noqa: E731
    title, desc, image, canonical = "modelmap", "Interactive, animated architecture maps for any Hugging Face model — and whether it fits on your GPU.", "/og/default.png", "/"
    if path.startswith("/m/"):
        mid = path[3:].strip("/")
        if mid:
            title = f"{mid} — architecture map"
            desc = f"Interactive map of {mid}: every module, traced shapes, forward-pass replay, memory and GPU fit."
            image = f"/og/m/{q(mid)}.png"
            canonical = f"/m/{q(mid)}"
            s = _summary_line(mid)
            if s:
                desc = f"{mid}: {s}. Interactive map with forward-pass replay, memory lenses and a GPU fit planner."
    elif path.startswith("/arch/"):
        key = path[6:].strip("/")
        from modelmap.zoo import FAMILIES

        fam = next((f for f in FAMILIES if f["key"] == key), None)
        if fam:
            title = f"{fam['title']} — architecture lineage"
            desc = fam["blurb"]
            image = f"/og/arch/{key}.png"
            canonical = f"/arch/{key}"
    elif path == "/models":
        title = "The architecture zoo — every mapped model"
        desc = "Structural facts for every mapped model: params, active params, attention type, MoE, MLA, KV cache per token."
        canonical = "/models"
    elif path.startswith("/compare/") and "..." in path[9:]:
        a, b = path[9:].split("...", 1)
        title = f"{a} vs {b}"
        desc = f"Module-by-module structural diff of {a} and {b}: what changed, was added, removed — and what it means."
        image = f"/og/compare.png?a={q(a)}&b={q(b)}"
        canonical = f"/compare/{q(a)}...{q(b)}"
    return {"title": title, "description": desc, "image": site + image, "url": site + canonical}


def _summary_line(model_id: str) -> str | None:
    from modelmap import cache

    s = cache.summary(model_id, "main") if cache.has(model_id, "main") else None
    if not s:
        return None
    parts = [f"{fmt_params(s['params_total'])} params"] if s.get("params_total") else []
    if s.get("architecture"):
        parts.append(s["architecture"])
    return " · ".join(parts) if parts else None


def inject_meta(html: str, meta: dict[str, str]) -> str:
    """Put the og/twitter tags (and the page title) into index.html."""
    from html import escape

    e = {k: escape(v, quote=True) for k, v in meta.items()}
    tags = (
        f'<meta property="og:type" content="website" />'
        f'<meta property="og:site_name" content="modelmap" />'
        f'<meta property="og:title" content="{e["title"]}" />'
        f'<meta property="og:description" content="{e["description"]}" />'
        f'<meta property="og:image" content="{e["image"]}" />'
        f'<meta property="og:image:width" content="{W}" />'
        f'<meta property="og:image:height" content="{H}" />'
        f'<meta property="og:url" content="{e["url"]}" />'
        f'<meta name="twitter:card" content="summary_large_image" />'
        f'<meta name="twitter:title" content="{e["title"]}" />'
        f'<meta name="twitter:description" content="{e["description"]}" />'
        f'<meta name="twitter:image" content="{e["image"]}" />'
        f'<link rel="canonical" href="{e["url"]}" />'
    )
    html = html.replace("<title>modelmap</title>", f"<title>{e['title']}</title>", 1)
    html = html.replace('<meta name="description" content="Interactive, animated architecture maps for any Hugging Face model" />',
                        f'<meta name="description" content="{e["description"]}" />', 1)
    return html.replace("</head>", tags + "</head>", 1)


def card_cache_key(kind: str, ident: str, schema_version: int) -> str:
    import hashlib

    return f"og-{kind}-{hashlib.sha1(ident.encode()).hexdigest()[:16]}-s{schema_version}.png"


__all__ = [
    "render_model_card", "render_default_card", "render_family_card", "render_compare_card",
    "meta_for", "inject_meta", "card_cache_key",
]


# ---------------------------------------------------------------- README badge

def badge_svg(label: str, value: str, color: str = "#E08A00") -> str:
    """A shields-style flat badge (pure SVG, no fonts fetched) for model
    cards and READMEs: [ modelmap | 8.19B · GQA 4× · 36 layers ]."""
    from html import escape

    def w(t: str) -> int:  # Verdana-11 average advance, the shields convention
        return int(round(sum(7.2 if ch.isupper() or ch in "mw" else 4.0 if ch in " .·×il'" else 6.3 for ch in t))) + 12

    lw, vw = w(label), w(value)
    total = lw + vw
    el, ev = escape(label), escape(value)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total}" height="20" role="img" aria-label="{el}: {ev}">'
        f'<title>{el}: {ev}</title>'
        f'<linearGradient id="s" x2="0" y2="100%"><stop offset="0" stop-color="#bbb" stop-opacity=".1"/><stop offset="1" stop-opacity=".1"/></linearGradient>'
        f'<clipPath id="r"><rect width="{total}" height="20" rx="3" fill="#fff"/></clipPath>'
        f'<g clip-path="url(#r)"><rect width="{lw}" height="20" fill="#1B2430"/><rect x="{lw}" width="{vw}" height="20" fill="{color}"/>'
        f'<rect width="{total}" height="20" fill="url(#s)"/></g>'
        f'<g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="11">'
        f'<text x="{lw / 2:.1f}" y="15" fill="#010101" fill-opacity=".3">{el}</text><text x="{lw / 2:.1f}" y="14">{el}</text>'
        f'<text x="{lw + vw / 2:.1f}" y="15" fill="#010101" fill-opacity=".3">{ev}</text><text x="{lw + vw / 2:.1f}" y="14">{ev}</text>'
        f'</g></svg>'
    )


def badge_value(model_id: str) -> str:
    """What the badge says for a mapped model: params · attention · layers."""
    from modelmap import cache

    doc = cache.get(model_id, "main") if cache.has(model_id, "main") else None
    if not doc:
        return "view architecture"
    from modelmap.insights import profile

    p = profile(doc)
    bits = []
    if p["params"]:
        bits.append(fmt_params(p["params"]) + (f" ({fmt_params(p['active'])} active)" if p["experts"] and p["active"] < 0.9 * p["params"] else ""))
    if p["experts"]:
        bits.append(f"MoE {int(p['top_k'] or 0)}/{int(p['experts'])}")
    if p["attention"]:
        bits.append(p["attention"] + (f" {int(p['heads'] / p['kv_heads'])}×" if p["attention"] == "GQA" and p["kv_heads"] else ""))
    if p["layers"]:
        bits.append(f"{int(p['layers'])} layers")
    return " · ".join(bits) or "view architecture"
