"""Phone layout: bottom-sheet inspector, lean top bar, touch expand, flow, compare stacked."""
from playwright.sync_api import sync_playwright

from _common import BASE, SHOTS, check, finish

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    iphone = p.devices["iPhone 13"]
    ctx = b.new_context(**iphone)
    ctx.add_init_script("localStorage.setItem('mm-autoflow-done','1')")
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))

    page.goto(BASE, wait_until="networkidle")
    page.wait_for_selector(".mm-card", timeout=30000)
    check("landing renders on phone without horizontal scroll", page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"))
    page.screenshot(path=f"{SHOTS}/mobile-1-landing.png")

    page.locator(".mm-card-main", has_text="Qwen/Qwen3-8B").first.tap()
    page.wait_for_selector(".react-flow__node", timeout=60000); page.wait_for_timeout(700)
    check("sheet present and collapsed", page.locator(".mm-sheet").count() == 1 and page.locator(".mm-sheet.is-open").count() == 0)
    canvas_w = page.evaluate("document.querySelector('.mm-canvas').getBoundingClientRect().width")
    check("canvas uses the full width", canvas_w >= iphone["viewport"]["width"] - 2, f"{canvas_w}px")
    check("minimap hidden", page.locator(".mm-minimap").count() == 0)
    check("secondary top-bar chrome hidden", page.locator(".mm-btn-share").is_hidden() and page.locator(".mm-lens").count() == 0 or page.locator(".mm-lens").first.is_hidden())
    page.screenshot(path=f"{SHOTS}/mobile-2-graph.png")

    # tap a node → sheet opens with details; tap handle → closes
    page.locator(".react-flow__node", has_text="embed_tokens").first.tap()
    page.wait_for_timeout(500)
    check("selecting a node opens the sheet", page.locator(".mm-sheet.is-open").count() == 1)
    check("sheet shows inspector content", "embed_tokens" in page.locator(".mm-sheet .mm-inspector").inner_text())
    page.screenshot(path=f"{SHOTS}/mobile-3-sheet.png")
    page.locator(".mm-sheet-handle").tap()
    page.wait_for_timeout(400)
    check("handle collapses the sheet", page.locator(".mm-sheet.is-open").count() == 0)

    # expand via badge (touch), flow bar fits
    page.locator(".mm-badge", has_text="×36").tap()
    page.wait_for_timeout(900)
    check("stack expands on tap", page.locator(".mm-badge", has_text="1 of 36").count() == 1)
    page.locator(".mm-btn-flow").tap()
    page.wait_for_selector(".mm-hud", timeout=5000); page.wait_for_timeout(800)
    fb = page.evaluate("(() => { const r = document.querySelector('.mm-flowbar-wrap').getBoundingClientRect(); return [r.left, r.right, window.innerWidth] })()")
    check("flow bar fits the phone width", fb[0] >= 0 and fb[1] <= fb[2] + 1, str(fb))
    page.screenshot(path=f"{SHOTS}/mobile-4-flow.png")

    # compare stacks vertically
    page.goto(f"{BASE}/compare/openai-community/gpt2...openai-community/gpt2-medium", wait_until="networkidle")
    page.wait_for_selector(".mm-cmp-summary", timeout=120000); page.wait_for_timeout(800)
    tops = page.evaluate("[...document.querySelectorAll('.mm-cmp-canvases .mm-canvas')].map(c => Math.round(c.getBoundingClientRect().top))")
    check("compare canvases stacked", len(tops) == 2 and tops[1] > tops[0] + 100, str(tops))
    page.screenshot(path=f"{SHOTS}/mobile-5-compare.png")

    check("no page JS errors", not errors, "; ".join(errors[:2]))
    b.close()

finish()
