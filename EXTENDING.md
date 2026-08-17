# Extending modelmap

Everything the extractor records is generic: forward hooks on every `nn.Module`
capture input/output shapes for **any module that runs**, and annotators
attach metadata to **any module that exists**. Two registries decide the
family-specific parts. Import a module that registers into them and point
`MODELMAP_PLUGINS=your.module` at it (comma-separated; the server's worker
processes inherit the variable), or import it before calling `extract_graph`.

## 1. Input builders — how to make a model run

`modelmap.inputs` holds an ordered list of `(predicate, builder)` pairs; the
first predicate that matches `(model, config)` decides the dummy kwargs fed
to the traced forward on the meta device. Built-ins cover: text (`input_ids`),
encoder-decoder text (`+ decoder_input_ids`), images (`pixel_values`),
waveforms (`input_values`), log-mel features and speech seq2seq
(`input_features [+ decoder_input_ids]`), and multimodal models (text path
here, encoder towers separately — see §3).

```python
# my_plugin.py
import torch
from modelmap.inputs import register_input_builder

@register_input_builder(
    "protein-lm",
    lambda model, config: getattr(config, "model_type", "") == "esm",
    first=True,                              # ahead of the built-ins
    notes=["dummy input: 7 residues"],       # surfaced in the graph's notes
)
def esm_inputs(model, config, seq_len):
    return {"input_ids": torch.zeros((1, seq_len), dtype=torch.long, device="meta")}
```

Rules of thumb: tensors must live on `device="meta"` (no memory, shape-only);
if an op inside the model is data-dependent (`.item()`, `.tolist()`), the
forward faults at that point and the graph is marked *structural* with the
steps recorded so far — that is expected and non-fatal.

## 2. Annotators — metadata per module

`modelmap.annotate` runs each registered function over every module and
merges the returned dict into the node's `attrs` (shown in the inspector).
Built-ins parse `extra_repr()` (so `nn.Linear` gets `in_features` /
`out_features` / `bias`, convs get kernel/stride, norms get `eps`) and add
`_src` / `_src_url` — the defining file:line, linked to GitHub for
transformers and torch classes at the installed versions.

```python
from modelmap.annotate import register_annotator

@register_annotator
def flops_estimate(name, module):
    import torch.nn as nn
    if isinstance(module, nn.Linear):
        return {"macs_per_token": module.in_features * module.out_features}
```

Keys starting with `_` are treated as internal by the UI (not listed as
attributes). Values are stringified; keep them short.

## 3. Encoder towers in multimodal models

For any config that declares `vision_config` or `audio_config`, the text
path is traced normally and the encoder tower is traced separately
(`modelmap/towers.py`): standalone on the meta device first, then a
*shallow twin* — the tower's own class instantiated with depth 1 on the CPU
and run on a real tiny input, with block-0 steps replicated per block. To
support a new tower input convention, extend `_pixel_input` there; to help
detection, towers are found by class name (`Vision|Visual|Siglip|Clip|…`,
`Audio|Whisper|Speech|…`) and by *not* owning a token vocabulary.

## 4. Kinds, colors, captions

Node `kind` (embedding / attention / mlp / moe / norm / linear / conv / head /
container / module) is assigned in `extract._classify` from class and leaf
names and drives color, collapse defaults, and Flow-mode captions
(`web/src/flow/captions.ts`) and micro-views (`web/src/flow/micro.ts`). Adding
a family usually means one regex there and one caption.
