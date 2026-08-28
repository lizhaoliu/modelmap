"""M19 acceptance (design doc §29): the failure message is the product.
Gated repos say the fix, pasted URLs and ollama-style ids load, pickle-only
repos explain themselves, vendor-only architectures get an honest banner."""
from playwright.sync_api import sync_playwright

from _common import BASE, SHOTS, check, finish

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    ctx = b.new_context(viewport={"width": 1440, "height": 900})
    ctx.add_init_script("localStorage.setItem('mm-autoflow-done','1')")
    page = ctx.new_page()

    # ---- gated repo: the message names the fix, not a header failure
    page.goto(f"{BASE}/m/google/gemma-3-270m", wait_until="networkidle")
    page.wait_for_selector(".mm-error", timeout=120000)
    err = page.inner_text(".mm-error")
    check("gated repo says it is gated", "gated repo" in err, err[:160])
    check("gated repo names the fix", "accept its terms" in err and "token" in err, err[:160])
    check("no masked header error", "safetensors headers" not in err)

    # ---- pickle-only repo: what it holds and why that's unreadable
    page.goto(f"{BASE}/m/Ultralytics/YOLO26", wait_until="networkidle")
    page.wait_for_selector(".mm-error", timeout=120000)
    err = page.inner_text(".mm-error")
    check("pickle-only repo explains itself", "pickle checkpoints" in err and "executing" in err, err[:160])
    page.screenshot(path=f"{SHOTS}/m19-1-messages.png")

    # ---- pasted shapes: an hf.co URL in the address bar just works
    page.goto(f"{BASE}/m/hf.co/openai-community/gpt2", wait_until="networkidle")
    page.wait_for_selector(".react-flow__node", timeout=120000)
    check("hf.co-prefixed path loads the model", page.locator(".react-flow__node").count() > 3)

    # ---- pasting a full URL into the search box lands on the canonical URL
    page.goto(f"{BASE}/", wait_until="networkidle")
    page.wait_for_selector(".mm-search input", timeout=30000)
    page.locator(".mm-search input").first.fill("https://huggingface.co/openai-community/gpt2/tree/main")
    page.keyboard.press("Enter")
    page.wait_for_selector(".react-flow__node", timeout=120000)
    check("pasted URL canonicalizes in the URL bar", "/m/openai-community/gpt2" in page.url and "tree" not in page.url, page.url)

    # ---- vendor-only architecture: honest banner, and the subfolder
    #      component is on the map (Breeze's audio tokenizer)
    page.goto(f"{BASE}/m/BreezeBlue/Breeze-TTS-2", wait_until="networkidle")
    page.wait_for_selector(".react-flow__node", timeout=180000)
    fid = page.locator(".mm-fidelity").get_attribute("title") or ""
    check("vendor-arch banner is honest", "isn't in transformers" in fid, fid[:200])
    check("no pip-upgrade parroting", "pip install" not in fid)
    check("audio tokenizer joined the map from its subfolder",
          page.locator(".react-flow__node[data-id='audio_tokenizer']").count() == 1)
    page.screenshot(path=f"{SHOTS}/m19-2-breeze.png")

    b.close()

finish()
