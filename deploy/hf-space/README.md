---
title: modelmap
emoji: 🕸️
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: Interactive, animated architecture maps for any HF model
---

# modelmap

Paste a Hugging Face model id. Get a living map of the network — explorable down to
every projection, animated so you can watch a token flow from embedding to logits.
No weights are downloaded: the model is instantiated on the meta device and a fake
forward pass is traced for real execution order and tensor shapes.

Source and docs: see the `README.md` of the code repository bundled in this Space
(`DEPLOY.md`, `EXTENDING.md`, `docs/design.html`).
