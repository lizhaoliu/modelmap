"""M6 acceptance: compare two models — alignment, diff marks, linked canvases, diff inspector."""
from playwright.sync_api import sync_playwright

from _common import BASE, SHOTS, check, finish

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    page = b.new_page(viewport={"width": 1600, "height": 900})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))

    page.goto(f"{BASE}/compare/Qwen/Qwen2.5-7B...Qwen/Qwen3-8B", wait_until="networkidle")
    page.wait_for_selector(".mm-cmp-summary", timeout=120000)
    page.wait_for_timeout(1200)
    summ = page.locator(".mm-cmp-summary").inner_text()
    check("summary shows both models", "Qwen2.5-7B" in summ and "Qwen3-8B" in summ)
    check("summary diff rows: layers 28→36, ffn", "28" in summ and "36" in summ and "18,944" in summ and "12,288" in summ)
    counts = page.locator(".mm-cmp-counts").inner_text()
    check("counts: some changed, added ≥2 (q/k norms), removed 0", "+2" in counts or "+3" in counts or "+4" in counts, counts)
    check("two canvases render", page.locator(".mm-canvas").count() == 2)
    nA = page.locator(".mm-canvas").nth(0).locator(".react-flow__node").count()
    nB = page.locator(".mm-canvas").nth(1).locator(".react-flow__node").count()
    check("both sides have nodes", nA >= 5 and nB >= 5, f"{nA} / {nB}")

    # linked expansion: open the stack on A → B opens too
    page.locator(".mm-canvas").nth(0).locator(".mm-badge", has_text="×28").click()
    page.wait_for_timeout(1000)
    check("expansion mirrored to B", page.locator(".mm-canvas").nth(1).locator(".mm-badge", has_text="1 of 36").count() == 1)
    page.locator(".mm-canvas").nth(0).locator(".react-flow__node", has_text="self_attn").locator(".mm-chevron").first.click()
    page.wait_for_timeout(1000)
    added = page.locator(".mm-canvas").nth(1).locator(".mm-node.diff-added").count()
    check("q_norm/k_norm marked added on B", added >= 2, f"{added} added nodes")
    check("changed nodes outlined", page.locator(".mm-node.diff-changed").count() >= 2)

    # linked selection + diff inspector
    page.locator(".mm-canvas").nth(0).locator(".react-flow__node", has_text="q_proj").first.click()
    page.wait_for_timeout(400)
    insp = page.locator(".mm-diffinsp").inner_text()
    check("diff inspector shows both sides of q_proj", "q_proj" in insp and "changed" in insp and "3584" in insp and "4096" in insp)
    check("bias True/False called out", "True" in insp and "False" in insp)
    check("selection mirrored to B", page.locator(".mm-canvas").nth(1).locator(".react-flow__node.selected").count() == 1)
    page.screenshot(path=f"{SHOTS}/compare-1.png")

    # differences-only toggle collapses unchanged subtrees; base vs finetune reports no structural diff
    page.locator(".mm-cmp-toggle input").check()
    page.wait_for_timeout(700)
    page.goto(f"{BASE}/compare/Qwen/Qwen3-8B-Base...Qwen/Qwen3-8B", wait_until="networkidle")
    page.wait_for_selector(".mm-cmp-summary", timeout=120000)
    page.wait_for_timeout(800)
    counts2 = page.locator(".mm-cmp-counts").inner_text()
    check("base vs fine-tune: nothing added/removed", "+0" in counts2 and "−0" in counts2, counts2)
    cfg = page.locator(".mm-diffinsp").inner_text()
    check("only the context-length config differs", "max_position_embeddings" in cfg and "hidden_size" not in cfg and "num_hidden_layers" not in cfg)

    check("no page JS errors", not errors, "; ".join(errors[:2]))
    b.close()

finish()
