"""Generate fixtures that pin the Live-mode JS engine to real transformers.

Writes to web/tests/fixtures/live/:
  llama-tiny.{safetensors,bf16.safetensors,config.json,expected.json}
  gpt2-tiny.{safetensors,config.json,expected.json}
  tok-{tinyllama,smollm2,gpt2}.json.gz  + tok.expected.json

The tiny models are random-weight (seeded) but run through the *real*
LlamaForCausalLM / GPT2LMHeadModel, so every architectural convention the JS
engine must reproduce — rope rotate_half, GQA grouping, rmsnorm eps, gelu_new,
Conv1D transposition, attention scaling, tied heads — is captured in the
expected logits / attentions / greedy continuations.

Run once, commit the outputs:  uv run python scripts/gen_live_fixtures.py
"""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

OUT = Path(__file__).parent.parent / "web" / "tests" / "fixtures" / "live"
OUT.mkdir(parents=True, exist_ok=True)

PROMPT = [1, 5, 9, 2, 77, 33]
GREEDY_STEPS = 6


def dump(name: str, model, config, prompt: list[int]) -> None:
    model.eval()
    sd = {k: v.contiguous() for k, v in model.state_dict().items()}
    # drop duplicated tied weights the way checkpoints do
    if getattr(config, "tie_word_embeddings", False):
        sd = {k: v for k, v in sd.items() if k != "lm_head.weight"}
    save_file(sd, str(OUT / f"{name}.safetensors"))
    save_file({k: v.to(torch.bfloat16) for k, v in sd.items()}, str(OUT / f"{name}.bf16.safetensors"))
    (OUT / f"{name}.config.json").write_text(json.dumps(config.to_dict()))

    ids = torch.tensor([prompt])
    with torch.no_grad():
        out = model(ids, output_attentions=True, output_hidden_states=True)
    logits = out.logits[0, -1].tolist()
    attn0 = out.attentions[0][0].tolist()  # layer 0: [heads, S, S]
    attn_last_row = out.attentions[-1][0, :, -1, :].tolist()  # last layer, all heads, last row
    # logit lens: each layer's output hidden state → final norm → head
    final_norm = model.model.norm if hasattr(model, "model") else model.transformer.ln_f
    head = model.lm_head
    lens_top1 = []
    for h in out.hidden_states[1:]:  # skip the embedding layer output
        with torch.no_grad():
            lg = head(final_norm(h[0, -1:]))[0]
        lens_top1.append(int(lg.argmax()))
    with torch.no_grad():
        gen = model.generate(
            ids, max_new_tokens=GREEDY_STEPS, do_sample=False,
            pad_token_id=0, use_cache=True,
        )[0, len(prompt):].tolist()
    # top-2 margin — the JS greedy comparison skips steps that are a coin flip
    with torch.no_grad():
        top2 = torch.topk(out.logits[0, -1], 2).values
    (OUT / f"{name}.expected.json").write_text(json.dumps({
        "prompt": prompt,
        "logits_last": logits,
        "attn_layer0": attn0,
        "attn_lastlayer_lastrow": attn_last_row,
        "lens_top1": lens_top1,
        "greedy": gen,
        "top1_margin": float(top2[0] - top2[1]),
    }))
    print(f"{name}: params={sum(p.numel() for p in model.parameters()):,} greedy={gen}")


def main() -> None:
    from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    lcfg = LlamaConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        intermediate_size=64, vocab_size=128, rope_theta=10000.0, rms_norm_eps=1e-6,
        tie_word_embeddings=False, max_position_embeddings=64, attn_implementation="eager",
    )
    dump("llama-tiny", LlamaForCausalLM(lcfg), lcfg, PROMPT)

    torch.manual_seed(1)
    gcfg = GPT2Config(
        n_embd=32, n_layer=2, n_head=4, n_positions=64, vocab_size=120,
        attn_implementation="eager",
    )
    dump("gpt2-tiny", GPT2LMHeadModel(gcfg), gcfg, PROMPT)

    # ---- tokenizers: real tokenizer.json files + expected encodings
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    strings = [
        "Once upon a time there was a little",
        "Hello, world!",
        " leading space",
        "tabs\tand\nnewlines",
        "numbers 12345 and 3.14",
        "emoji 🦙 and ünïcode",
        "  double  spaces  ",
        "don't stop, y'all've",
        "CamelCase XMLHttpRequest",
        "价格是42元",
        "a",
    ]
    expected: dict[str, dict] = {}
    for key, repo in [
        ("tinyllama", "Maykeye/TinyLLama-v0"),
        ("smollm2", "HuggingFaceTB/SmolLM2-135M"),
        ("gpt2", "openai-community/gpt2"),
    ]:
        path = hf_hub_download(repo, "tokenizer.json")
        with open(path, "rb") as f, gzip.open(OUT / f"tok-{key}.json.gz", "wb") as g:
            shutil.copyfileobj(f, g)
        tok = Tokenizer.from_file(path)
        expected[key] = {
            "repo": repo,
            "cases": [
                {"text": s, "ids": tok.encode(s, add_special_tokens=False).ids}
                for s in strings
            ],
        }
        # decode round-trip reference
        for c in expected[key]["cases"]:
            c["decoded"] = tok.decode(c["ids"])
    (OUT / "tok.expected.json").write_text(json.dumps(expected, ensure_ascii=False))
    print("tokenizer fixtures written")


if __name__ == "__main__":
    main()
