// ==UserScript==
// @name         modelmap — view on Hugging Face
// @namespace    https://modelmap.cc
// @version      0.2.0
// @description  Adds a "view in modelmap" button to Hugging Face model pages (interactive architecture map, no weights downloaded)
// @match        https://huggingface.co/*
// @grant        none
// @run-at       document-idle
// ==/UserScript==
// modelmap — Hugging Face model pages get a "view in modelmap" button.
// Pure DOM: no network, no storage, no permissions. The button links to
// https://modelmap.cc/m/<owner>/<name> (set MODELMAP_BASE to self-host).
(() => {
  const MODELMAP_BASE = 'https://modelmap.cc'
  const NOT_MODELS = new Set(['datasets', 'spaces', 'docs', 'blog', 'papers', 'collections', 'models', 'tasks', 'pricing', 'settings', 'organizations', 'join', 'login', 'new', 'search', 'api', 'posts', 'chat', 'enterprise', 'learn', 'terms', 'privacy', 'huggingface', 'changelog', 'brand', 'inference-endpoints'])

  function modelId() {
    const parts = location.pathname.split('/').filter(Boolean)
    if (parts.length < 1 || NOT_MODELS.has(parts[0])) return null
    // /owner/name or /owner/name/tree/main…; legacy top-level ids (/gpt2) too
    if (parts.length === 1) return /^[\w.-]+$/.test(parts[0]) ? parts[0] : null
    if (!/^[\w.-]+$/.test(parts[0]) || !/^[\w.-]+$/.test(parts[1])) return null
    return `${parts[0]}/${parts[1]}`
  }

  function inject() {
    const id = modelId()
    document.querySelectorAll('.mm-ext-btn').forEach((b) => b.remove())
    if (!id) return
    // the header with the repo name (h1 inside the model page header)
    const h1 = document.querySelector('header h1, h1')
    if (!h1) return
    const a = document.createElement('a')
    a.className = 'mm-ext-btn'
    a.href = `${MODELMAP_BASE}/m/${id}`
    a.target = '_blank'
    a.rel = 'noopener'
    a.title = 'Interactive architecture map: modules, tensor shapes, cost estimates — no weights downloaded'
    a.textContent = 'view in modelmap ↗'
    Object.assign(a.style, {
      display: 'inline-flex', alignItems: 'center', gap: '6px', marginLeft: '10px', padding: '3px 10px',
      borderRadius: '999px', border: '1px solid #E08A00', color: '#9A5E00', background: 'rgba(224,138,0,0.08)',
      font: '600 12px/1.6 system-ui, sans-serif', textDecoration: 'none', verticalAlign: 'middle', whiteSpace: 'nowrap',
    })
    h1.appendChild(a)
  }

  inject()
  // HF is a SPA: re-inject on navigation
  let last = location.pathname
  new MutationObserver(() => {
    if (location.pathname !== last || !document.querySelector('.mm-ext-btn')) {
      last = location.pathname
      inject()
    }
  }).observe(document.documentElement, { childList: true, subtree: true })
})()
