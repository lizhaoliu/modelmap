"""Edges follow execution order; side computations (rotary) attach as dashed aux inputs."""
from playwright.sync_api import sync_playwright

from _common import BASE, SHOTS, check, finish

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    _ctx = b.new_context(viewport={"width": 1440, "height": 900})
    _ctx.add_init_script("localStorage.setItem('mm-autoflow-done','1')")
    page = _ctx.new_page()
    page.goto(f"{BASE}/m/Qwen/Qwen3-8B", wait_until="networkidle")
    page.wait_for_selector(".react-flow__node", timeout=90000); page.wait_for_timeout(600)
    ids = page.evaluate("[...document.querySelectorAll('.react-flow__edge')].map(e => e.getAttribute('data-id'))")
    check("main chain embed → layers → norm", "model.embed_tokens→model.layers" in ids and "model.layers→model.norm" in ids, str(ids))
    check("rotary is a side input into layers, not chained after norm", "model.rotary_emb→model.layers" in ids and "model.norm→model.rotary_emb" not in ids)
    dashed = page.evaluate("[...document.querySelectorAll('.react-flow__edge')].filter(e => e.getAttribute('data-id') === 'model.rotary_emb→model.layers').map(e => getComputedStyle(e.querySelector('path.react-flow__edge-path')).strokeDasharray)")
    check("aux edge is dashed", bool(dashed) and dashed[0] not in ("none", ""), str(dashed))
    page.screenshot(path=f"{SHOTS}/edges-qwen3.png")
    b.close()

finish()
