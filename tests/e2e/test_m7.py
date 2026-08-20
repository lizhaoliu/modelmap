"""M7/M8 acceptance: export menu (SVG/PNG/embed/link), embed mode, planner,
GGUF variants with real quant dtypes, interleaved repeat stacks, node deep links,
local checkpoints."""
import json
import os
import re
import tempfile

from playwright.sync_api import sync_playwright

from _common import BASE, SHOTS, check, finish

# a local checkpoint: gpt2's config.json alone is enough for a full-fidelity map
LOCAL = tempfile.mkdtemp(prefix="mm-local-")
from huggingface_hub import hf_hub_download  # noqa: E402
import shutil  # noqa: E402
shutil.copy(hf_hub_download("openai-community/gpt2", "config.json"), os.path.join(LOCAL, "config.json"))

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    ctx = b.new_context(viewport={"width": 1440, "height": 900}, permissions=["clipboard-read", "clipboard-write"], accept_downloads=True)
    ctx.add_init_script("localStorage.setItem('mm-autoflow-done','1')")
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))

    # ---- export menu
    page.goto(f"{BASE}/m/Qwen/Qwen3-8B", wait_until="networkidle")
    page.wait_for_selector(".react-flow__node", timeout=120000)
    page.locator(".mm-btn-export").click()
    page.wait_for_selector(".mm-export-pop")
    items = page.locator(".mm-export-pop button[role=menuitem]").all_inner_texts()
    check("export menu lists images, data and share actions", any("PNG" in i for i in items) and any("CSV" in i for i in items) and any("embed" in i.lower() for i in items), str(items))
    with page.expect_download(timeout=30000) as dl:
        page.locator(".mm-export-pop button", has_text="SVG").click()
    path = dl.value.path()
    svg = open(path, encoding="utf-8").read()
    check("SVG download is an SVG of the view", svg.startswith("<svg") and "embed_tokens" in svg and "×36" in svg and "modelmap.cc/m/Qwen/Qwen3-8B" in svg, svg[:80])
    page.locator(".mm-btn-export").click()
    with page.expect_download(timeout=60000) as dl:
        page.locator(".mm-export-pop button", has_text="PNG").click()
    png = open(dl.value.path(), "rb").read()
    check("PNG download is a PNG", png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 10000, str(len(png)))
    page.locator(".mm-btn-export").click()
    with page.expect_download(timeout=60000) as dl:
        page.locator(".mm-export-pop button", has_text="CSV").click()
    csv_text = open(dl.value.path(), encoding="utf-8").read()
    check("CSV export has module rows", csv_text.startswith("module,kind,class") and "model.layers.0.self_attn.q_proj" in csv_text)
    page.locator(".mm-btn-export").click()
    with page.expect_download(timeout=60000) as dl:
        page.locator(".mm-export-pop button", has_text="Markdown").click()
    md = open(dl.value.path(), encoding="utf-8").read()
    check("Markdown export", md.startswith("# Qwen/Qwen3-8B") and "KV cache" in md)
    page.locator(".mm-btn-export").click()
    page.locator(".mm-export-pop button", has_text="Copy embed code").click()
    page.wait_for_timeout(300)
    clip = page.evaluate("navigator.clipboard.readText()")
    check("embed code is an iframe with ?embed=1", clip.startswith("<iframe") and "embed=1" in clip, clip[:120])
    page.screenshot(path=f"{SHOTS}/m7-1-export.png")

    # ---- node deep link
    page.locator(".mm-badge", has_text="×36").click(); page.wait_for_timeout(700)
    page.locator(".react-flow__node", has_text="self_attn").first.click(); page.wait_for_timeout(300)
    page.locator(".mm-insp-copy").click(); page.wait_for_timeout(300)
    clip = page.evaluate("navigator.clipboard.readText()")
    check("inspector copy link carries ?sel=", "sel=model.layers.0.self_attn" in clip, clip)

    # ---- planner
    page.locator(".mm-btn-plan").click()
    page.wait_for_selector(".mm-planner")
    verdict = page.locator(".mm-plan-verdict").inner_text()
    check("planner default (1× 80 GB) fits Qwen3-8B", verdict.lower() == "fits", verdict)
    page.locator(".mm-plan-grid input[type=number]").first.fill("8")  # memory GB
    page.wait_for_timeout(200)
    check("planner: 8 GB does not fit", "not" in page.locator(".mm-plan-verdict").inner_text().lower())
    page.locator(".mm-plan-grid input[type=number]").first.fill("24")
    page.locator(".mm-plan-grid input[type=number]").nth(1).fill("2")  # gpus
    page.wait_for_timeout(200)
    rows = page.locator(".mm-plan-table tbody tr").count()
    check("planner: 2 GPUs → 2 pipeline stages by default", rows == 2, str(rows))
    summary = page.locator(".mm-plan-summary").first.inner_text()
    check("planner reports max context and boundary traffic", "max context" in summary and "stage boundary" in summary, summary)
    check("planner state in URL", "gpus=2" in page.url and "gmem=24" in page.url, page.url)
    page.screenshot(path=f"{SHOTS}/m7-2-planner.png")
    page.keyboard.press("Escape")

    # ---- embed mode
    page.goto(f"{BASE}/m/Qwen/Qwen3-8B?embed=1&lens=params", wait_until="networkidle")
    page.wait_for_selector(".react-flow__node", timeout=60000)
    check("embed: no top bar, no inspector", page.locator(".mm-topbar").count() == 0 and page.locator(".mm-inspector").count() == 0)
    badge = page.locator(".mm-embed-badge")
    check("embed: attribution badge links to the full page", badge.count() == 1 and "embed=1" not in badge.get_attribute("href") and "/m/Qwen/Qwen3-8B" in badge.get_attribute("href"))
    page.screenshot(path=f"{SHOTS}/m7-3-embed.png")

    # ---- GGUF variants
    page.goto(f"{BASE}/m/Qwen/Qwen3-8B-GGUF", wait_until="networkidle")
    page.wait_for_selector(".react-flow__node", timeout=180000)
    sel = page.locator(".mm-variant select")
    check("GGUF: variant picker shows Q4_K_M by default", sel.count() == 1 and sel.input_value() == "Q4_K_M", sel.input_value() if sel.count() else "none")
    insp = page.locator(".mm-inspector").inner_text()
    check("GGUF: inspector says checkpoint gguf · Q4_K_M", "gguf" in insp and "Q4_K_M" in insp, insp[:200])
    page.locator(".mm-badge", has_text="×36").click(); page.wait_for_timeout(700)
    page.locator(".react-flow__node", has_text="self_attn").first.click(); page.wait_for_timeout(200)
    page.keyboard.press("e"); page.wait_for_timeout(900)
    page.locator(".react-flow__node", has_text="q_proj").first.click(); page.wait_for_timeout(300)
    insp = page.locator(".mm-inspector").inner_text()
    check("GGUF: q_proj dtype shows the quant type and bpw", "Q4_K" in insp and "bpw" in insp, insp[:300])
    page.locator(".mm-lens-btn", has_text="memory").click(); page.wait_for_timeout(300)
    page.locator(".mm-lens-btn", has_text="memory").click()
    page.locator(".react-flow__pane").click(position={"x": 5, "y": 5}); page.wait_for_timeout(200)
    sel.select_option("Q8_0")
    page.wait_for_timeout(500)
    page.wait_for_selector(".react-flow__node", timeout=180000)
    check("GGUF: switching variant navigates to :Q8_0", page.url.endswith("Qwen3-8B-GGUF:Q8_0") or "Qwen3-8B-GGUF:Q8_0" in page.url, page.url)
    page.wait_for_function("document.querySelector('.mm-variant select') && document.querySelector('.mm-variant select').value === 'Q8_0'", timeout=180000)
    page.screenshot(path=f"{SHOTS}/m7-4-gguf.png")

    # ---- interleaved stacks (DeepSeek-V4)
    page.goto(f"{BASE}/m/deepseek-ai/DeepSeek-V4-Flash", wait_until="networkidle")
    page.wait_for_selector(".react-flow__node", timeout=240000)
    badge = page.locator(".react-flow__node", has_text="layers").locator(".mm-badge").first
    check("DeepSeek-V4: layers collapse into a stack", badge.count() == 1 and "×43" in badge.inner_text(), badge.inner_text() if badge.count() else "no badge")
    check("DeepSeek-V4: stack title explains the designs", "2 repeated designs (×20, ×20) + 3 unique" in (badge.get_attribute("title") or ""), badge.get_attribute("title"))
    badge.click(); page.wait_for_timeout(900)
    reps = page.locator(".react-flow__node .mm-badge.is-static", has_text="1 of")
    check("DeepSeek-V4: opening the stack shows one block of each variant", reps.count() >= 2, str(reps.count()))
    page.locator(".mm-badge.is-static", has_text="1 of 20").first.click(force=True); page.wait_for_timeout(400)
    insp = page.locator(".mm-inspector").inner_text()
    check("DeepSeek-V4: inspector lists interleaved members", "interleaved" in insp, insp[:300])
    page.screenshot(path=f"{SHOTS}/m7-5-dsv4.png")

    # ---- local checkpoint
    page.goto(f"{BASE}/", wait_until="networkidle")
    page.wait_for_selector(".mm-landing")
    le = page.locator(".mm-local-entry input")
    check("landing offers a local checkpoint field when served locally", le.count() == 1)
    le.fill(LOCAL)
    page.locator(".mm-local-entry button").click()
    page.wait_for_selector(".react-flow__node", timeout=120000)
    check("local checkpoint renders (gpt2 config only)", page.locator(".mm-model-chip").inner_text().startswith("local:") and page.locator(".react-flow__node", has_text="wte").count() == 1)
    page.screenshot(path=f"{SHOTS}/m7-6-local.png")

    check("no page errors", not errors, str(errors)[:300])
    b.close()
finish()
