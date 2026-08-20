"""M10 acceptance: Live mode — real inference in the browser. Downloads the
9 MB TinyLLama-v0 checkpoint from the Hub, so it needs network access."""
from playwright.sync_api import sync_playwright

from _common import BASE, SHOTS, check, finish

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    ctx = b.new_context(viewport={"width": 1440, "height": 900})
    ctx.add_init_script("localStorage.setItem('mm-autoflow-done','1')")
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))

    # ---- unsupported model: the panel explains and offers picks
    page.goto(f"{BASE}/m/Qwen/Qwen3-8B", wait_until="networkidle")
    page.wait_for_selector(".react-flow__node", timeout=60000)
    page.locator(".mm-btn-live").click()
    page.wait_for_selector(".mm-livebar")
    note = page.locator(".mm-live-note").inner_text()
    check("unsupported model explains why", "supports llama-family and gpt2" in note, note)
    check("panel offers runnable picks", page.locator(".mm-live-picks button").count() >= 2)

    # ---- a pick navigates to a live-capable model
    page.locator(".mm-live-picks button", has_text="TinyLLama-v0").click()
    page.wait_for_selector(".react-flow__node", timeout=120000)
    check("pick navigates", "/m/Maykeye/TinyLLama-v0" in page.url, page.url)

    # ---- download + run
    page.locator(".mm-btn-live").click()
    page.wait_for_selector(".mm-live-cta .mm-btn-primary", timeout=10000)
    cta = page.locator(".mm-live-cta .mm-btn-primary").inner_text()
    check("download CTA shows the size", "9 MB" in cta, cta)
    page.locator(".mm-live-cta .mm-btn-primary").click()
    page.wait_for_selector(".mm-live-promptrow", timeout=180000)
    check("weights load and the prompt row appears", True)
    page.locator(".mm-live-promptrow input").fill("Once upon a time there was a little")
    page.locator(".mm-live-promptrow button[type=submit]").click()
    page.wait_for_selector(".mm-live-cand", timeout=60000)
    cands = page.locator(".mm-live-cand").all_inner_texts()
    check("real next-token candidates with probabilities", len(cands) == 5 and all("%" in c for c in cands), str(cands))
    check("the model knows its TinyStories ('girl' tops the list)", "girl" in cands[0], cands[0])

    # logit lens: one cell per layer
    page.wait_for_selector(".mm-live-lenscell", timeout=30000)
    check("logit lens has a cell per layer", page.locator(".mm-live-lenscell").count() == 8)

    # attention heatmap: real values on canvas, slider changes layers
    page.wait_for_selector(".mm-live-attn canvas", timeout=30000)
    page.wait_for_timeout(400)
    amber = page.evaluate(
        """() => { const cv = document.querySelector('.mm-live-attn canvas');
             const d = cv.getContext('2d').getImageData(0,0,cv.width,cv.height).data;
             let n = 0; for (let i = 0; i < d.length; i += 4) if (d[i] > 150 && d[i+2] < 100) n++;
             return n; }"""
    )
    check("attention heatmap carries real weights", amber > 50, f"{amber} amber pixels")
    page.locator(".mm-live-attn-head input[type=range]").fill("5")
    page.wait_for_timeout(300)
    check("layer slider retargets the heatmap", "layer 5" in page.locator(".mm-live-attn-head label").inner_text())
    page.locator(".mm-live-attn-head select").select_option("3")
    page.wait_for_timeout(200)
    check("head selector works", page.locator(".mm-live-attn-head select").input_value() == "3")

    # clicking an attention node on the map retargets the heatmap layer
    page.locator(".mm-badge", has_text="×8").click()
    page.wait_for_timeout(700)
    page.locator(".react-flow__node", has_text="self_attn").first.click()
    page.wait_for_timeout(400)
    check("clicking the map's attention block sets the heatmap layer", "layer 0" in page.locator(".mm-live-attn-head label").inner_text())
    page.screenshot(path=f"{SHOTS}/live-1-run.png")

    # ---- generation streams tokens and stays coherent-ish
    before = page.locator(".mm-live-toks i").count()
    page.locator(".mm-live-promptrow button", has_text="generate").click()
    page.wait_for_timeout(2500)
    after = page.locator(".mm-live-toks i").count()
    check("generation streams tokens", after > before + 4, f"{before} → {after}")
    page.wait_for_selector(".mm-live-promptrow button:has-text('generate')", timeout=60000)
    text = page.evaluate("() => document.querySelector('.mm-live-toks').innerText").replace("·", " ")
    check("generated text is words, not noise", sum(c.isalpha() or c.isspace() for c in text) / max(len(text), 1) > 0.8, text[:120])
    page.screenshot(path=f"{SHOTS}/live-2-generate.png")

    # ---- flow and live are exclusive
    page.keyboard.press("f")
    page.wait_for_timeout(400)
    check("starting a replay closes live", page.locator(".mm-livebar").count() == 0 and page.locator(".mm-hud").count() == 1)
    page.keyboard.press("Escape")

    check("no page errors", not errors, str(errors)[:300])
    b.close()
finish()
