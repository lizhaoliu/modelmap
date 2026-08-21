"""M15–M18 acceptance (design doc §25–§28): social cards + meta tags, the
vram lens with the planner painted onto the graph, the landing's fit entry,
compare takeaways, the README badge, the node finder, deep links and
double-click-to-open."""
import io
import urllib.request

from playwright.sync_api import sync_playwright

from _common import BASE, SHOTS, check, finish

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    ctx = b.new_context(viewport={"width": 1440, "height": 900})
    ctx.add_init_script("localStorage.setItem('mm-autoflow-done','1')")
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))

    # ---- §25 social cards: meta tags are server-injected, the PNG is real
    html = urllib.request.urlopen(f"{BASE}/m/Qwen/Qwen3-8B?lens=kv").read().decode()
    check("model page carries og:image", '<meta property="og:image" content="https://modelmap.cc/og/m/Qwen/Qwen3-8B.png" />' in html)
    check("model page title is the model", "<title>Qwen/Qwen3-8B — architecture map</title>" in html)
    check("canonical drops view state", '<link rel="canonical" href="https://modelmap.cc/m/Qwen/Qwen3-8B" />' in html)
    png = urllib.request.urlopen(f"{BASE}/og/m/Qwen/Qwen3-8B.png").read()
    from PIL import Image

    im = Image.open(io.BytesIO(png))
    check("model card is a 1200×630 PNG", im.size == (1200, 630) and png[:4] == b"\x89PNG", str(im.size))
    cmp_png = urllib.request.urlopen(f"{BASE}/og/compare.png?a=Qwen/Qwen2.5-7B&b=Qwen/Qwen3-8B").read()
    check("compare card renders", Image.open(io.BytesIO(cmp_png)).size == (1200, 630))
    fam_png = urllib.request.urlopen(f"{BASE}/og/arch/qwen.png").read()
    check("family card renders", Image.open(io.BytesIO(fam_png)).size == (1200, 630))
    cmp_html = urllib.request.urlopen(f"{BASE}/compare/Qwen/Qwen2.5-7B...Qwen/Qwen3-8B").read().decode()
    check("compare page title is A vs B", "<title>Qwen/Qwen2.5-7B vs Qwen/Qwen3-8B</title>" in cmp_html)
    svg = urllib.request.urlopen(f"{BASE}/badge/Qwen/Qwen3-8B.svg").read().decode()
    check("README badge says params · attention · layers", "8.19B" in svg and "GQA 4×" in svg and "36 layers" in svg and svg.startswith("<svg"))

    # ---- §26 vram lens: the planner on the graph
    page.goto(f"{BASE}/m/Qwen/Qwen3-8B?lens=vram&gpu=RTX%204090%2024GB", wait_until="networkidle")
    page.wait_for_selector(".mm-vrambar", timeout=60000)
    page.wait_for_timeout(400)
    fit = page.inner_text(".mm-vram-fit")
    check("vram strip gives a verdict for the chosen GPU", "fits on 1× RTX 4090" in fit, fit)
    legend = page.inner_text(".mm-vram-legend")
    check("legend splits weights / KV / activations", "weights" in legend and "KV" in legend and "act" in legend, legend)
    # slide context to 128k: the cache outweighs the model, the verdict flips
    page.locator(".mm-vram-ctx input[type=range]").fill("8")  # 131072
    page.wait_for_timeout(400)
    legend = page.inner_text(".mm-vram-legend")
    check("at 128k the KV cache outweighs the weights", "outweighs" in legend, legend)
    check("context travels in the URL", "T=131072" in page.url, page.url)
    fit = page.inner_text(".mm-vram-fit")
    check("verdict escalates to tensor-parallel", "needs 2×" in fit, fit)
    badge_before = page.locator(".react-flow__node[data-id='model.layers'] .mm-params").inner_text()
    page.select_option(".mm-vram-knobs select[aria-label='weight precision']", "int4")
    page.wait_for_timeout(400)
    badge_after = page.locator(".react-flow__node[data-id='model.layers'] .mm-params").inner_text()
    check("weights toggle re-prices the nodes live", badge_before != badge_after, f"{badge_before} → {badge_after}")
    check("precision travels in the URL", "w=int4" in page.url, page.url)
    check("assumptions chip shows the precision", "int4 weights" in page.inner_text(".mm-lens-assume"))
    page.screenshot(path=f"{SHOTS}/m15-1-vram.png")
    # attention nodes carry the KV share in their hover detail
    page.locator(".react-flow__node[data-id='model.layers'] .mm-node").first.dblclick()
    page.wait_for_selector(".react-flow__node[data-id='model.layers.0.self_attn']", timeout=20000)
    page.wait_for_timeout(300)
    title = page.locator(".react-flow__node[data-id='model.layers.0.self_attn'] .mm-params").get_attribute("title") or ""
    check("attention badge explains weights + KV", "KV cache" in title and "weights" in title, title)
    # double-click-to-open actually works (it was dead under d3-zoom's dblclick handler)
    check("double-click opened the stack", page.locator(".react-flow__node[data-id='model.layers.0.mlp']").count() == 1)
    page.locator(".react-flow__node[data-id='lm_head'] .mm-node").first.dblclick()
    page.wait_for_selector(".mm-toast", timeout=3000)
    check("double-clicking a leaf explains itself", "leaf module" in page.inner_text(".mm-toast"))

    # ---- §26 landing: the fit question is the front door
    page.goto(f"{BASE}/", wait_until="networkidle")
    page.wait_for_selector(".mm-fit-entry", timeout=30000)
    page.wait_for_selector(".mm-card-fit", timeout=30000)
    check("card fit? buttons exist", page.locator(".mm-card-fit").count() >= 5)
    page.select_option(".mm-fit-entry select", "RTX 4090 24GB")
    page.locator(".mm-fit-entry .mm-link", has_text="Qwen3-8B").click()
    page.wait_for_selector(".mm-vrambar", timeout=60000)
    check("landing fit entry opens the vram lens", "lens=vram" in page.url and "gpu=RTX" in page.url, page.url)
    page.goto(f"{BASE}/", wait_until="networkidle")
    page.wait_for_selector(".mm-card-fit", timeout=30000)
    page.locator(".mm-card .mm-card-fit").first.click()
    page.wait_for_selector(".mm-vrambar", timeout=90000)
    check("a card's fit? lands on the vram lens", "lens=vram" in page.url)

    # ---- §27 takeaways
    page.goto(f"{BASE}/compare/Qwen/Qwen2.5-7B...Qwen/Qwen3-8B", wait_until="networkidle")
    page.wait_for_selector(".mm-cmp-takeaways li", timeout=90000)
    items = page.locator(".mm-cmp-takeaways li").all_inner_texts()
    check("compare page leads with derived takeaways", len(items) == 4, str(len(items)))
    check("takeaways quantify the KV-cache consequence", any("KV cache per token" in t and "2.6× larger" in t for t in items), str(items)[:300])
    page.locator(".mm-cmp-takeaways .mm-link").click()
    page.wait_for_timeout(200)
    check("'more' reveals the full list", page.locator(".mm-cmp-takeaways li").count() > 4)
    page.screenshot(path=f"{SHOTS}/m15-2-takeaways.png")
    page.goto(f"{BASE}/arch/qwen", wait_until="networkidle")
    page.wait_for_selector(".mm-zoo-takeaways li", timeout=90000)
    check("lineage arrows speak in takeaways", page.locator(".mm-zoo-takeaways li").count() >= 6)
    recipe = page.locator(".mm-zoo-recipe").first.inner_text()
    check("member cards show the recipe", "GQA" in recipe and "RoPE" in recipe, recipe)

    # ---- §28 node finder + deep links
    page.goto(f"{BASE}/m/Qwen/Qwen3-8B", wait_until="networkidle")
    page.wait_for_selector(".mm-finder-btn", timeout=60000)
    page.keyboard.press("/")
    page.wait_for_selector(".mm-finder-box input")
    page.keyboard.type("k_norm")
    page.wait_for_selector(".mm-finder-list li[role=option]")
    check("finder matches by leaf name", "self_attn.k_norm" in page.inner_text(".mm-finder-list li >> nth=0"))
    page.keyboard.press("Enter")
    page.wait_for_selector(".mm-node.is-selected", timeout=10000)
    page.wait_for_timeout(500)
    check("Enter reveals the module and selects it", "sel=model.layers.0.self_attn.k_norm" in page.url, page.url)
    check("breadcrumb shows the path", "k_norm" in page.inner_text(".mm-crumbs"))
    page.keyboard.press("/")
    page.keyboard.type("RMSNorm")
    page.wait_for_selector(".mm-finder-list li[role=option]")
    n = page.locator(".mm-finder-list li[role=option]").count()
    check("finder matches by class too", n >= 3, str(n))
    page.keyboard.press("Escape")
    page.goto(f"{BASE}/m/Qwen/Qwen3-8B?sel=model.layers.0.mlp.down_proj", wait_until="networkidle")
    page.wait_for_selector(".mm-node.is-selected", timeout=60000)
    check("a deep link opens the way to its module", "down_proj" in page.inner_text(".mm-crumbs"))
    page.screenshot(path=f"{SHOTS}/m15-3-finder.png")

    # ---- export menu: README badge
    page.locator(".mm-btn-export").click()
    page.wait_for_selector("[role=menuitem]")
    check("export menu offers the README badge", page.locator("[role=menuitem]", has_text="README badge").count() == 1)

    check("no page errors", not errors, str(errors)[:300])
    b.close()

finish()
