# Browser integration

Two ways to get a **view in modelmap ↗** button on every Hugging Face model page:

## Chrome / Edge / Brave extension (unpacked)

1. `chrome://extensions` → enable *Developer mode* → *Load unpacked* → pick `extensions/chrome/`.
2. Open any model page, e.g. https://huggingface.co/Qwen/Qwen3-8B — the button sits next to the model name.

The extension is pure DOM (no permissions, no network, no storage); it links to `https://modelmap.cc/m/<owner>/<name>`.
Self-hosting? Edit `MODELMAP_BASE` in `content.js`.

## Userscript (Tampermonkey / Violentmonkey / Firefox)

Install `extensions/modelmap.user.js` — same behaviour, any browser with a userscript manager.

## Embedding modelmap elsewhere

Any model page can be embedded chrome-less with `?embed=1`:

```html
<iframe src="https://modelmap.cc/m/Qwen/Qwen3-8B?embed=1" width="100%" height="520"
        style="border:1px solid #d9dee6;border-radius:10px" loading="lazy"></iframe>
```

The export menu (`export ▾`) on any model page copies this snippet for the exact current view
(expansion state, lens, selection all travel in the URL).
