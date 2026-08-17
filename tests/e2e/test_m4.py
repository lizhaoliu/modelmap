"""M4 acceptance: gallery landing, treemap, micro-views, help, token, toast, render budget."""
from playwright.sync_api import sync_playwright

from _common import BASE, SHOTS, check, finish

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    ctx = b.new_context(viewport={"width": 1440, "height": 900}, permissions=["clipboard-read", "clipboard-write"])
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))

    # gallery landing
    page.goto(BASE, wait_until="networkidle")
    page.wait_for_selector(".mm-card", timeout=20000)
    n = page.locator(".mm-card").count()
    check("gallery renders cards", n >= 6, f"{n} cards")
    check("cards show cached summaries", page.locator(".mm-card-meta b").count() >= 4)
    page.screenshot(path=f"{SHOTS}/m4-1-landing.png")

    # help overlay via ?
    page.keyboard.press("?")
    check("help overlay opens", page.locator(".mm-help").count() == 1)
    page.keyboard.press("Escape")
    page.mouse.click(10, 10)  # click backdrop
    page.wait_for_timeout(150)

    # card → model
    page.locator(".mm-card-main", has_text="Qwen/Qwen3-8B").click()
    page.wait_for_selector(".react-flow__node", timeout=60000)
    page.wait_for_timeout(700)
    check("card loads model", "/m/Qwen/Qwen3-8B" in page.url)

    # treemap on root, then on a selected container
    check("root treemap shown", page.locator(".mm-treemap .mm-tm-cell").count() >= 3)
    page.locator(".mm-tm-cell", has_text="layers").first.click()
    page.wait_for_timeout(300)
    check("treemap cell click selects", "layers" in page.locator(".mm-inspector h2").inner_text())
    page.screenshot(path=f"{SHOTS}/m4-2-treemap.png")

    # token popover
    page.locator(".mm-topbar .mm-btn", has_text="token").click()
    check("token popover opens", page.locator(".mm-pop input").count() == 1)
    page.keyboard.press("Escape")

    # share toast
    page.locator(".mm-topbar .mm-btn", has_text="share").click()
    page.wait_for_timeout(200)
    check("share shows toast", page.locator(".mm-toast").count() == 1)

    # micro-view: default (collapsed) replay → block choreography with real children
    page.keyboard.press("Escape")
    page.keyboard.press("f")
    page.wait_for_selector(".mm-hud", timeout=5000)
    saw_block = False
    for _ in range(30):
        if page.locator(".mm-micro").count() and "inside one block" in page.locator(".mm-micro-title").inner_text().lower():
            saw_block = True
            break
        page.wait_for_timeout(300)
    check("block micro-view during stack", saw_block)
    if saw_block:
        labels = page.locator(".mm-micro-label").all_inner_texts()
        check("block stages from real children", "self_attn" in labels and "mlp" in labels and "⊕" in labels, str(labels))
        check("micro dot animating", page.locator(".mm-micro-dot").count() == 1)
    page.screenshot(path=f"{SHOTS}/m4-3-micro-block.png")
    page.keyboard.press(" ")  # pause
    page.wait_for_timeout(200)
    check("micro pauses with replay", page.locator(".mm-micro.is-paused").count() == 1)
    page.keyboard.press("Escape")

    # attention micro-view: expand into the block, replay, scrub to self_attn beat
    page.locator(".mm-badge", has_text="×36").click()
    page.wait_for_timeout(900)
    page.keyboard.press("f")
    page.wait_for_selector(".mm-hud", timeout=5000)
    page.keyboard.press(" ")
    saw_attn = False
    for _ in range(40):
        page.keyboard.press("ArrowRight")
        page.wait_for_timeout(120)
        if page.locator(".mm-micro").count() and "attention" in page.locator(".mm-micro-title").inner_text().lower():
            saw_attn = True
            break
    check("attention micro-view", saw_attn)
    if saw_attn:
        txt = page.locator(".mm-micro").inner_text()
        check("attention shows heads/head-dim shapes", "32" in txt and "128" in txt and "softmax" in txt, txt[:120].replace("\n", " "))
    page.screenshot(path=f"{SHOTS}/m4-4-micro-attn.png")
    page.keyboard.press("Escape")

    # render budget: open experts on the 235B MoE (128 experts) — no toast expected under 300;
    # then verify the mechanism by asserting visible nodes never exceed the budget on a deep open
    page.goto(f"{BASE}/m/Qwen/Qwen3-235B-A22B", wait_until="networkidle")
    page.wait_for_selector(".react-flow__node", timeout=60000)
    page.locator(".mm-badge", has_text="×94").click()
    page.wait_for_timeout(700)
    page.locator(".react-flow__node", has_text="mlp").locator(".mm-chevron").first.click()
    page.wait_for_timeout(700)
    exp = page.locator(".mm-badge", has_text="×128")
    if exp.count():
        exp.first.click()
        page.wait_for_timeout(900)
    n = page.locator(".react-flow__node").count()
    check("visible nodes stay under budget", n <= 300, f"{n} nodes")

    check("no page JS errors", not errors, "; ".join(errors[:2]))
    b.close()

finish()
