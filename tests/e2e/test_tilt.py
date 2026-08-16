"""2.5D tilt prototype: toggle, click-through under 3D transform, flow under tilt."""
from playwright.sync_api import sync_playwright

from _common import BASE, SHOTS, check, finish

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    page = b.new_page(viewport={"width": 1440, "height": 900})
    errs = []
    page.on("pageerror", lambda e: errs.append(str(e)))
    page.goto(f"{BASE}/m/Qwen/Qwen3-8B", wait_until="networkidle")
    page.wait_for_selector(".react-flow__node", timeout=60000)
    page.wait_for_timeout(500)
    page.keyboard.press("t")
    page.wait_for_timeout(900)
    check("T toggles tilt on", page.locator(".mm-canvas.is-tilt").count() == 1)
    z = page.evaluate("getComputedStyle(document.querySelector('.mm-canvas')).getPropertyValue('--mm-zoom')")
    check("zoom var tracks viewport", z.strip() not in ("", "1"), z)

    page.locator(".mm-badge", has_text="×36").click()
    page.wait_for_timeout(900)
    page.locator(".react-flow__node", has_text="self_attn").locator(".mm-chevron").first.click()
    page.wait_for_timeout(900)
    page.locator(".react-flow__node", has_text="q_proj").first.click()
    page.wait_for_timeout(400)
    check("click-through selects under 3D", "q_proj" in page.locator(".mm-inspector h2").inner_text())
    page.keyboard.press("0")
    page.wait_for_timeout(600)
    page.screenshot(path=f"{SHOTS}/tilt-1-deep.png")

    page.keyboard.press("f")
    page.wait_for_selector(".mm-hud", timeout=5000)
    page.wait_for_timeout(1500)
    check("flow runs under tilt", page.locator(".mm-flow-pulse").count() == 1)
    page.screenshot(path=f"{SHOTS}/tilt-2-flow.png")
    page.keyboard.press("Escape")
    page.keyboard.press("t")
    page.wait_for_timeout(700)
    check("T toggles tilt off", page.locator(".mm-canvas.is-tilt").count() == 0)
    check("no page JS errors", not errs, "; ".join(errs[:2]))
    b.close()

finish()
