"""M11 + M13 acceptance: dropped graph files, trust_remote_code messaging,
the /models catalog and /arch family pages."""
import json

from playwright.sync_api import sync_playwright

from _common import BASE, SHOTS, check, finish

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    ctx = b.new_context(viewport={"width": 1440, "height": 900})
    ctx.add_init_script("localStorage.setItem('mm-autoflow-done','1')")
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))

    # ---- M13: catalog
    page.goto(f"{BASE}/models", wait_until="networkidle")
    page.wait_for_selector(".mm-zoo-table tbody tr", timeout=30000)
    rows = page.locator(".mm-zoo-table tbody tr").count()
    check("catalog lists the cached models", rows >= 15, f"{rows} rows")
    page.locator(".mm-zoo-tagbtn", has_text="moe").first.click()
    page.wait_for_timeout(300)
    moe_rows = page.locator(".mm-zoo-table tbody tr").count()
    check("tag filter narrows to MoE models", 0 < moe_rows < rows, f"{moe_rows}")
    page.locator(".mm-zoo-tagbtn", has_text="moe").first.click()
    page.locator(".mm-zoo-filters input").fill("whisper")
    page.wait_for_timeout(300)
    check("text filter works", page.locator(".mm-zoo-table tbody tr").count() >= 1)
    page.locator(".mm-zoo-filters input").fill("")
    page.locator(".mm-zoo-table tbody tr").first.click()
    page.wait_for_selector(".react-flow__node", timeout=60000)
    check("catalog row opens the map", "/m/" in page.url, page.url)
    page.screenshot(path=f"{SHOTS}/zoo-1-catalog.png")

    # ---- M13: family page with live lineage diffs
    page.goto(f"{BASE}/arch/qwen", wait_until="networkidle")
    page.wait_for_selector(".mm-zoo-member", timeout=30000)
    check("qwen family lists its lineage", page.locator(".mm-zoo-member").count() >= 5)
    page.wait_for_function("[...document.querySelectorAll('.mm-zoo-step')].some(s => /changed|identical/.test(s.innerText))", timeout=60000)
    steps = page.locator(".mm-zoo-step").all_inner_texts()
    check("lineage arrows carry live diffs", any("changed" in s for s in steps), str(steps)[:200])
    check("identical pair says so plainly", any("identical structure" in s for s in steps), str(steps)[:200])
    page.locator(".mm-zoo-step .mm-link", has_text="changed").first.click()
    page.wait_for_selector(".mm-cmp-canvases, .mm-compare", timeout=90000)
    check("a lineage arrow opens the full compare", "/compare/" in page.url, page.url)
    page.screenshot(path=f"{SHOTS}/zoo-2-family.png")

    # ---- landing family chips
    page.goto(f"{BASE}/", wait_until="networkidle")
    page.wait_for_selector(".mm-card", timeout=30000)
    check("landing offers family chips", page.locator(".mm-zoo-tagbtn", has_text="deepseek").count() == 1)

    # ---- M11: drop a graph file
    doc = json.loads(page.evaluate("fetch('/api/graph/openai-community/gpt2').then(r => r.text())" if False else "''") or "{}")
    import urllib.request
    doc = json.load(urllib.request.urlopen(f"{BASE}/api/graph/openai-community/gpt2"))
    doc["model_id"] = "acme/secret-model"  # pretend it came from a private dump
    payload = json.dumps(doc)
    page.evaluate(
        """(payload) => {
          const file = new File([payload], 'secret-model.graph.json', { type: 'application/json' })
          const dt = new DataTransfer()
          dt.items.add(file)
          window.dispatchEvent(new DragEvent('drop', { dataTransfer: dt, bubbles: true, cancelable: true }))
        }""",
        payload,
    )
    page.wait_for_selector(".react-flow__node", timeout=30000)
    check("dropped graph renders without any server call", page.locator(".mm-model-chip", has_text="acme/secret-model").count() == 1)
    check("file tag marks the provenance", page.locator(".mm-file-tag").count() == 1)
    page.keyboard.press("f")
    page.wait_for_selector(".mm-hud", timeout=8000)
    check("flow replay works on a dropped graph", True)
    page.keyboard.press("Escape")
    page.locator(".mm-btn-export").click()
    page.wait_for_selector(".mm-export-pop")
    items = page.locator(".mm-export-pop button[role=menuitem]").all_inner_texts()
    check("file docs keep client exports, drop server ones", any("JSON" in i for i in items) and not any("CSV" in i for i in items), str(items))
    page.keyboard.press("Escape")

    # ---- M11: trust_remote_code tooltip on the disabled flow button
    page.goto(f"{BASE}/m/moonshotai/Kimi-K3", wait_until="domcontentloaded")
    page.wait_for_selector(".react-flow__node", timeout=300000)
    title = page.locator(".mm-topbar .mm-btn[disabled]", has_text="flow").get_attribute("title") or ""
    check("disabled flow explains trust_remote_code and the drop workflow", "trust_remote_code" in title and "drop" in title, title[:160])

    check("no page errors", not errors, str(errors)[:300])
    b.close()
finish()
