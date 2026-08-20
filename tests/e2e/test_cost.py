"""M5 acceptance: cost lens — active params on the MoE, heat + badges, what-if T scaling, URL state."""
from playwright.sync_api import sync_playwright

from _common import BASE, SHOTS, check, finish

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    _ctx = b.new_context(viewport={"width": 1440, "height": 900})
    _ctx.add_init_script("localStorage.setItem('mm-autoflow-done','1')")
    page = _ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))

    page.goto(f"{BASE}/m/Qwen/Qwen3-235B-A22B", wait_until="networkidle")
    page.wait_for_selector(".react-flow__node", timeout=90000)
    check("lens off by default: badges are params", "6.95B" in page.locator(".react-flow__node", has_text="layers").inner_text() or "B" in page.locator(".mm-params").first.inner_text())

    page.locator(".mm-lens-btn", has_text="compute").click()
    page.wait_for_timeout(400)
    check("lens=compute in URL", "lens=compute" in page.url)
    insp = page.locator(".mm-inspector").inner_text()
    check("root cost rows shown", "compute" in insp and "/tok" in insp)
    import re
    m = re.search(r"active params\s+([\d.]+)B/tok", insp)
    check("235B MoE reports ≈22B active params/token", bool(m) and 20 <= float(m.group(1)) <= 24, m.group(0) if m else insp[:200])
    check("MoE note (8 of 128 experts)", "8 of 128" in insp)
    check("nodes carry heat", page.locator(".react-flow__node .has-heat").count() >= 3)
    badge = page.locator(".react-flow__node", has_text="layers").locator(".mm-params").inner_text()
    check("stack badge shows MACs", "MAC" in badge, badge)
    page.screenshot(path=f"{SHOTS}/cost-1-compute.png")

    # kv lens + what-if T
    page.locator(".mm-lens-btn", has_text="kv").click()
    page.wait_for_timeout(300)
    chip = page.locator(".mm-lens-assume").inner_text()
    check("assumptions chip shows KV at T", "KV" in chip, chip)
    page.locator(".mm-lens-assume").click()
    page.wait_for_selector(".mm-whatif")
    slider = page.locator(".mm-whatif input[type=range]")
    slider.fill("12")  # 131072
    page.wait_for_timeout(300)
    check("T=131072 in URL", "T=131072" in page.url, page.url)
    chip2 = page.locator(".mm-lens-assume").inner_text()
    check("KV grows with T", chip2 != chip and "GB" in chip2, chip2)
    page.keyboard.press("Escape")

    # compute lens: attention share grows with T (Qwen3-8B, expand block)
    page.goto(f"{BASE}/m/Qwen/Qwen3-8B?lens=compute&T=128", wait_until="networkidle")
    page.wait_for_selector(".react-flow__node", timeout=60000)
    page.locator(".mm-badge", has_text="×36").click(); page.wait_for_timeout(700)
    page.locator(".react-flow__node", has_text="self_attn").first.click(); page.wait_for_timeout(300)
    a1 = page.locator(".mm-inspector").inner_text()
    page.goto(f"{BASE}/m/Qwen/Qwen3-8B?lens=compute&T=32768", wait_until="networkidle")
    page.wait_for_selector(".react-flow__node", timeout=60000)
    page.locator(".mm-badge", has_text="×36").click(); page.wait_for_timeout(700)
    page.locator(".react-flow__node", has_text="self_attn").first.click(); page.wait_for_timeout(300)
    a2 = page.locator(".mm-inspector").inner_text()
    def share(txt):
        mm = re.search(r"forward · ([\d.<]+)%", txt)
        return float(mm.group(1).replace("<", "")) if mm else None
    s1, s2 = share(a1), share(a2)
    check("attention share of compute grows with T", s1 is not None and s2 is not None and s2 > s1, f"{s1}% → {s2}%")
    check("formula on hover (title)", page.locator(".mm-cost dd[title*='attention core']").count() == 1)
    page.screenshot(path=f"{SHOTS}/cost-2-attention-longT.png")

    check("no page JS errors", not errors, "; ".join(errors[:2]))
    b.close()

finish()
