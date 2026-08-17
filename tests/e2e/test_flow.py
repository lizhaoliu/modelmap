"""M3 acceptance: flow-mode replay on gpt2 (controls, HUD, stack counter, reduced motion)."""
from playwright.sync_api import sync_playwright

from _common import BASE, SHOTS, check, finish

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    page = b.new_page(viewport={"width": 1440, "height": 900})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))

    page.goto(f"{BASE}/m/openai-community/gpt2", wait_until="networkidle")
    page.wait_for_selector(".react-flow__node", timeout=60000)
    page.wait_for_timeout(600)

    page.locator(".mm-btn-flow").click()
    page.wait_for_selector(".mm-hud", timeout=5000)
    hud = page.locator(".mm-hud").inner_text()
    check("flow starts at wte", "wte" in hud, hud.replace("\n", " · ")[:90])
    check("HUD shows labeled trace shapes", "[1 batch × 7 seq]" in hud and "768 hidden" in hud)
    check("HUD shows caption", "look up each token" in hud)
    check("url carries mode=flow", "mode=flow" in page.url)
    page.screenshot(path=f"{SHOTS}/flow-1-start.png")

    t1 = page.evaluate("document.querySelector('.mm-flow-pulse')?.style.transform")
    page.wait_for_timeout(1100)
    t2 = page.evaluate("document.querySelector('.mm-flow-pulse')?.style.transform")
    check("pulse moves while playing", t1 is not None and t1 != t2, f"{t1} → {t2}")

    total = float(page.locator(".mm-flow-scrub").get_attribute("max"))
    check("replay duration ≈ design target", 8 <= total <= 25, f"{total:.1f}s")

    saw_counter = False
    for _ in range(40):
        m = page.locator(".mm-hud-member")
        if m.count() and "/ 12" in m.inner_text():
            saw_counter = True
            break
        page.wait_for_timeout(400)
    check("layer counter during ×12 stack", saw_counter)
    page.screenshot(path=f"{SHOTS}/flow-2-stack.png")

    page.keyboard.press(" ")
    page.wait_for_timeout(200)
    f1 = page.evaluate("document.querySelector('.mm-flow-pulse')?.style.transform")
    page.wait_for_timeout(400)
    f2 = page.evaluate("document.querySelector('.mm-flow-pulse')?.style.transform")
    check("space pauses the pulse", f1 == f2)

    b1 = page.locator(".mm-flow-step").inner_text()
    page.keyboard.press("ArrowRight")
    page.wait_for_timeout(150)
    b2 = page.locator(".mm-flow-step").inner_text()
    check("arrow steps a beat", b1 != b2, f"{b1} → {b2}")

    page.locator(".mm-flow-scrub").fill(str(total))
    page.wait_for_timeout(250)
    check("scrub to end reaches lm_head", "lm_head" in page.locator(".mm-hud").inner_text())
    spent = page.locator(".mm-flow-spent").count()
    check("spent nodes marked", spent >= 4, f"{spent} spent")

    page.keyboard.press("Escape")
    page.wait_for_timeout(200)
    check("esc exits flow mode", page.locator(".mm-hud").count() == 0)
    check("mode param cleared", "mode=flow" not in page.url)

    # Qwen3-8B: design target ~15s, hidden-size narrative, /36 counter, reduced motion
    page.goto(f"{BASE}/m/Qwen/Qwen3-8B?mode=flow", wait_until="networkidle")
    page.wait_for_selector(".mm-hud", timeout=60000)
    total = float(page.locator(".mm-flow-scrub").get_attribute("max"))
    check("Qwen3-8B replay ≈ 15s design target", 9 <= total <= 20, f"{total:.1f}s")
    saw_4096 = saw_36 = False
    for _ in range(45):
        hud = page.locator(".mm-hud").inner_text()
        saw_4096 = saw_4096 or "4096" in hud
        saw_36 = saw_36 or "/ 36" in hud
        if saw_4096 and saw_36:
            break
        page.wait_for_timeout(350)
    check("shape narrative shows 4096 hidden", saw_4096)
    check("layer counter / 36", saw_36)
    page.emulate_media(reduced_motion="reduce")
    page.wait_for_timeout(200)
    vis = page.evaluate("getComputedStyle(document.querySelector('.mm-flow-pulse')).display")
    check("reduced motion hides the pulse", vis == "none", vis)
    h1 = page.locator(".mm-hud").inner_text()
    page.wait_for_timeout(1500)
    check("stepped narration continues without pulse", h1 != page.locator(".mm-hud").inner_text())

    # vision-language model: the vision tower is traced and replayed first
    page.goto(f"{BASE}/m/Qwen/Qwen2.5-VL-3B-Instruct?sel=model.visual.blocks.0.norm1&mode=flow", wait_until="networkidle")
    page.wait_for_selector(".mm-hud", timeout=120000)
    insp = page.locator(".mm-inspector").inner_text()
    check("VLM vision module has traced I/O", "input" in insp and "vision hidden" in insp)
    check("VLM replay starts in the vision tower", "patch_embed" in page.locator(".mm-hud-name").inner_text())
    check("patch count labeled", "patches" in insp)
    page.screenshot(path=f"{SHOTS}/flow-3-vlm.png")

    check("no page JS errors", not errors, "; ".join(errors[:2]))
    b.close()

finish()
