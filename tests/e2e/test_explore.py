"""M2 acceptance: landing → load gpt2 → expand stack → inspect → deep link → search."""
from playwright.sync_api import sync_playwright

from _common import BASE, SHOTS, check, finish

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))

    page.goto(BASE, wait_until="networkidle")
    check("landing renders wordmark", page.locator(".mm-wordmark").count() == 1)
    check("landing has example chips", page.locator(".mm-example").count() >= 5)

    page.locator(".mm-example", has_text="openai-community/gpt2").click()
    page.wait_for_selector(".react-flow__node", timeout=60000)
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(800)
    n_nodes = page.locator(".react-flow__node").count()
    check("graph renders nodes", n_nodes >= 6, f"{n_nodes} nodes")
    check("url updated", "/m/openai-community/gpt2" in page.url, page.url)
    check("stack badge ×12 visible", page.locator(".mm-badge", has_text="×12").count() == 1)
    check("fidelity chip full", page.locator(".mm-fidelity.is-full").count() == 1)
    page.screenshot(path=f"{SHOTS}/explore-1-default.png")

    page.locator(".mm-badge", has_text="×12").click()
    page.wait_for_timeout(900)
    n_after = page.locator(".react-flow__node").count()
    check("stack expands", n_after > n_nodes, f"{n_nodes} → {n_after}")
    check("'1 of 12' marker shown", page.locator(".mm-badge", has_text="1 of 12").count() >= 1)
    page.keyboard.press("0")
    page.wait_for_timeout(600)
    page.screenshot(path=f"{SHOTS}/explore-2-expanded.png")

    page.locator(".react-flow__node", has_text="ln_1").first.click()
    page.wait_for_timeout(300)
    insp = page.locator(".mm-inspector").inner_text()
    check("inspector shows path", "transformer.h.0.ln_1" in insp)
    check("inspector shows params share", "of model" in insp)
    check("inspector shows labeled I/O shapes", "input" in insp and "768 hidden" in insp)
    check("url carries selection", "sel=" in page.url, page.url)
    check("breadcrumb shows path", "ln_1" in page.locator(".mm-crumbs").inner_text())

    page.goto(f"{BASE}/m/openai-community/gpt2?sel=transformer.wte", wait_until="networkidle")
    page.wait_for_selector(".react-flow__node", timeout=60000)
    page.wait_for_timeout(600)
    insp = page.locator(".mm-inspector").inner_text()
    check("deep link restores selection", "wte" in insp and "embedding" in insp)

    page.locator(".mm-search input").first.fill("qwen3")
    page.wait_for_selector(".mm-search-results li", timeout=15000)
    check("search returns hits", page.locator(".mm-search-results li").count() >= 3)
    page.keyboard.press("Escape")

    page.evaluate("document.documentElement.dataset.theme = 'dark'")
    page.wait_for_timeout(300)
    page.screenshot(path=f"{SHOTS}/explore-3-dark.png")

    check("no page JS errors", not errors, "; ".join(errors[:3]))
    browser.close()

finish()
