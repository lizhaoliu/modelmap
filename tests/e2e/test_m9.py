"""M9 acceptance: the map feels alive — landing hero replay, first-visit
autoplay, edge motion, camera follow, reduced-motion respect."""
from playwright.sync_api import sync_playwright

from _common import BASE, SHOTS, check, finish

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)

    # ---- landing hero animates
    ctx = b.new_context(viewport={"width": 1440, "height": 900})
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"{BASE}/", wait_until="networkidle")
    page.wait_for_selector(".mm-hero-canvas")
    check("hero mini-replay on the landing", page.locator(".mm-hero-pulse").count() == 1 and page.locator(".mm-hero-node").count() == 5)
    anim = page.evaluate("getComputedStyle(document.querySelector('.mm-hero-pulse')).animationName")
    check("hero pulse is animating", anim == "mm-hero-travel", anim)
    cap1 = page.locator(".mm-hero-caption").inner_text()
    page.wait_for_timeout(3500)
    cap2 = page.locator(".mm-hero-caption").inner_text()
    check("hero caption narrates along", cap1 != cap2, f"{cap1!r} → {cap2!r}")
    page.screenshot(path=f"{SHOTS}/m9-1-hero.png")

    # ---- first visit: the replay starts itself
    page.goto(f"{BASE}/m/openai-community/gpt2", wait_until="networkidle")
    page.wait_for_selector(".react-flow__node", timeout=60000)
    page.wait_for_selector(".mm-hud", timeout=8000)
    check("first-visit autoplay starts the replay", page.locator(".mm-hud").count() == 1)
    check("edges drift while the replay runs", page.locator(".mm-canvas.mm-flow-on").count() == 1)
    check("follow toggle on and available", page.locator(".mm-flow-follow.is-on").count() == 1)
    page.screenshot(path=f"{SHOTS}/m9-2-autoplay.png")
    # a manual pan hands the camera back
    page.mouse.move(600, 400)
    page.mouse.down()
    page.mouse.move(700, 450, steps=4)
    page.mouse.up()
    page.wait_for_timeout(200)
    check("manual pan turns follow off", page.locator(".mm-flow-follow.is-on").count() == 0)
    # the autoplay bows out at the end (gpt2's replay is short at 4×)
    page.locator(".mm-flowbar .mm-btn", has_text="1×").click()
    page.locator(".mm-flowbar .mm-btn", has_text="2×").click()
    page.wait_for_selector(".mm-hud", state="detached", timeout=120000)
    check("autoplay exits by itself at the end", True)
    check("exit toast invites a replay", "replay" in (page.locator(".mm-toast").inner_text() if page.locator(".mm-toast").count() else ""))

    # ---- second open: no autoplay, but the flow glow is gone (it was used)
    page.goto(f"{BASE}/m/Qwen/Qwen3-8B", wait_until="networkidle")
    page.wait_for_selector(".react-flow__node", timeout=60000)
    page.wait_for_timeout(1600)
    check("no autoplay on later visits", page.locator(".mm-hud").count() == 0)
    check("flow button glow gone once flow has run", page.locator(".mm-btn-flow.mm-glow").count() == 0)
    ctx.close()

    # ---- fresh context, flow never used: the button glows
    ctx2 = b.new_context(viewport={"width": 1440, "height": 900}, reduced_motion="reduce")
    page2 = ctx2.new_page()
    page2.goto(f"{BASE}/m/openai-community/gpt2", wait_until="networkidle")
    page2.wait_for_selector(".react-flow__node", timeout=60000)
    page2.wait_for_timeout(1600)
    check("reduced motion: no autoplay", page2.locator(".mm-hud").count() == 0)
    check("reduced motion: flow button still advertises itself", page2.locator(".mm-btn-flow.mm-glow").count() == 1)
    anim2 = page2.evaluate("getComputedStyle(document.querySelector('.mm-btn-flow')).animationName")
    check("reduced motion: glow does not animate", anim2 == "none", anim2)
    page2.goto(f"{BASE}/", wait_until="networkidle")
    page2.wait_for_selector(".mm-hero-canvas")
    anim3 = page2.evaluate("getComputedStyle(document.querySelector('.mm-hero-pulse')).animationName")
    check("reduced motion: hero is a static diagram", anim3 == "none", anim3)
    ctx2.close()

    # ---- edge widths encode tensor size
    ctx3 = b.new_context(viewport={"width": 1440, "height": 900})
    ctx3.add_init_script("localStorage.setItem('mm-autoflow-done','1')")
    page3 = ctx3.new_page()
    page3.goto(f"{BASE}/m/Qwen/Qwen3-8B", wait_until="networkidle")
    page3.wait_for_selector(".react-flow__edge-path", timeout=60000)
    widths = page3.eval_on_selector_all(
        ".react-flow__edge-path", "els => els.map(e => getComputedStyle(e).strokeWidth)"
    )
    check("edge widths vary with traced tensor size", len(set(widths)) >= 2, str(set(widths)))
    ctx3.close()

    check("no page errors", not errors, str(errors)[:300])
    b.close()
finish()
