"""Recurrent-depth SmolLM2 retrofit probe.

This tests the prerequisite for JumpRec-on-LLM: can a small pretrained LM be
turned into a useful looped model when recurrence lives inside the model path?

Local shape sanity:
    python run_recurrent_smol.py --local --mode dry

H100 runs:
    modal run run_recurrent_smol.py --mode retrofit_probe
    modal run run_recurrent_smol.py --mode jumprec_probe
    modal run run_recurrent_smol.py --mode retrofit_8n4h
    modal run run_recurrent_smol.py --mode retrofit_unfreeze
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import modal


app = modal.App("recurrent-smol")
out_vol = modal.Volume.from_name("recurrent-smol-results", create_if_missing=True)
cache_vol = modal.Volume.from_name("jumprec-smol-cache", create_if_missing=True)
GPU_TYPE = "H100"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "transformers", "accelerate", "safetensors")
    .env({"HF_HOME": "/cache/huggingface", "TRANSFORMERS_CACHE": "/cache/huggingface"})
)


@dataclass
class Config:
    mode: str = "retrofit_probe"
    seed: int = 42
    model_id: str = "HuggingFaceTB/SmolLM2-135M"
    use_fake_model: bool = False
    n_nodes: int = 6
    max_hops: int = 3
    preserve_steps: int = 2
    d_model: int = 576
    n_heads: int = 8
    d_ff: int = 1536
    prelude_layers: int = 4
    core_start: int = 4
    core_layers: int = 2
    coda_start: int = 28
    coda_layers: int = 2
    train_prelude: bool = False
    train_core: bool = True
    train_coda: bool = True
    train_embeddings: bool = False
    final_steps: int = 2500
    recurrent_steps: int = 3500
    batch_size: int = 64
    eval_batches: int = 64
    max_length: int = 160
    lr_blocks: float = 5e-5
    lr_head: float = 3e-4
    weight_decay: float = 0.01
    log_every: int = 250
    timing_batches: int = 16
    reinject_init: float = 0.20
    jump_steps: int = 0
    max_correct: int = 3
    jump_layers: int = 1
    jump_lr: float = 4e-4
    verifier_loss_weight: float = 0.2
    use_temp_adapter: bool = True
    adapter_rank: int = 8

    @property
    def loop_steps(self) -> int:
        return self.max_hops + self.preserve_steps


def config_for_mode(mode: str) -> Config:
    cfg = Config(mode=mode)
    if mode == "dry":
        cfg.use_fake_model = True
        cfg.d_model = 64
        cfg.n_heads = 4
        cfg.d_ff = 192
        cfg.prelude_layers = 1
        cfg.core_start = 1
        cfg.core_layers = 1
        cfg.coda_start = 2
        cfg.coda_layers = 1
        cfg.final_steps = 4
        cfg.recurrent_steps = 6
        cfg.batch_size = 12
        cfg.eval_batches = 2
        cfg.max_length = 96
        cfg.log_every = 2
        cfg.timing_batches = 2
        cfg.jump_steps = 4
        cfg.max_correct = 2
        cfg.jump_layers = 1
    elif mode == "retrofit_probe":
        pass
    elif mode == "jumprec_probe":
        cfg.jump_steps = 3500
        cfg.max_correct = 3
        cfg.jump_layers = 1
        cfg.eval_batches = 64
        cfg.log_every = 250
    elif mode == "retrofit_8n4h":
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.final_steps = 3500
        cfg.recurrent_steps = 5000
        cfg.eval_batches = 64
        cfg.log_every = 250
    elif mode == "retrofit_8n4h_unfreeze":
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.prelude_layers = 2
        cfg.core_start = 2
        cfg.core_layers = 4
        cfg.coda_start = 26
        cfg.coda_layers = 4
        cfg.train_prelude = True
        cfg.final_steps = 4000
        cfg.recurrent_steps = 6000
        cfg.lr_blocks = 3e-5
        cfg.lr_head = 3e-4
        cfg.eval_batches = 64
        cfg.log_every = 500
    elif mode == "retrofit_12n6h":
        cfg.n_nodes = 12
        cfg.max_hops = 6
        cfg.preserve_steps = 2
        cfg.final_steps = 5000
        cfg.recurrent_steps = 7000
        cfg.eval_batches = 64
        cfg.log_every = 500
    elif mode == "retrofit_unfreeze":
        cfg.prelude_layers = 2
        cfg.core_start = 2
        cfg.core_layers = 4
        cfg.coda_start = 26
        cfg.coda_layers = 4
        cfg.train_prelude = True
        cfg.final_steps = 4000
        cfg.recurrent_steps = 5000
        cfg.lr_blocks = 3e-5
        cfg.lr_head = 3e-4
        cfg.batch_size = 64
        cfg.eval_batches = 64
        cfg.log_every = 250
    elif mode == "retrofit_long":
        cfg.final_steps = 5000
        cfg.recurrent_steps = 7000
        cfg.batch_size = 64
        cfg.eval_batches = 96
        cfg.log_every = 500
    else:
        raise ValueError(f"unknown mode: {mode}")
    return cfg


def run_experiment(cfg: Config, device_name: str = "cuda") -> Dict[str, object]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    print(f"[device] {device}")
    if device.type == "cuda":
        print(f"[gpu] {torch.cuda.get_device_name()}")
    print(f"[config] {json.dumps(asdict(cfg), indent=2)}")

    letters = [chr(ord("A") + i) for i in range(cfg.n_nodes)]

    def make_examples(batch_size: int):
        texts = []
        targets_l = []
        step_targets_l = []
        hops_l = []
        for _ in range(batch_size):
            perm = list(range(cfg.n_nodes))
            random.shuffle(perm)
            start = random.randrange(cfg.n_nodes)
            hops = random.randint(1, cfg.max_hops)
            cur = start
            step_targets = []
            for step in range(cfg.loop_steps):
                if hops > step:
                    cur = perm[cur]
                step_targets.append(cur)
            mapping = " ".join(f"{letters[i]}->{letters[perm[i]]}" for i in range(cfg.n_nodes))
            text = f"Task: forward. Map: {mapping}. Start: {letters[start]}. Hops: {hops}. Answer:"
            texts.append(text)
            targets_l.append(step_targets[-1])
            step_targets_l.append(step_targets)
            hops_l.append(hops)
        target = torch.tensor(targets_l, dtype=torch.long, device=device)
        step_targets = torch.tensor(step_targets_l, dtype=torch.long, device=device)
        hops = torch.tensor(hops_l, dtype=torch.long, device=device)
        return texts, target, step_targets, hops

    class FakeDecoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.d_ff,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )

        def forward(self, hidden_states, attention_mask=None, **_kwargs):
            key_padding_mask = None
            if attention_mask is not None and attention_mask.dim() == 2:
                key_padding_mask = ~attention_mask.bool()
            return self.layer(hidden_states, src_key_padding_mask=key_padding_mask)

    class RecurrentSmol(nn.Module):
        def __init__(self):
            super().__init__()
            self.fake = cfg.use_fake_model
            if cfg.use_fake_model:
                self.vocab_size = 4096
                self.embed_tokens = nn.Embedding(self.vocab_size, cfg.d_model)
                self.pos_tokens = nn.Embedding(cfg.max_length, cfg.d_model)
                self.prelude = nn.ModuleList([FakeDecoderLayer() for _ in range(cfg.prelude_layers)])
                self.core = nn.ModuleList([FakeDecoderLayer() for _ in range(cfg.core_layers)])
                self.coda = nn.ModuleList([FakeDecoderLayer() for _ in range(cfg.coda_layers)])
                self.norm = nn.LayerNorm(cfg.d_model)
                self.rotary_emb = None
                print("[model] using fake recurrent decoder")
            else:
                from transformers import AutoModelForCausalLM

                base = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=torch.float32)
                hidden_size = int(base.config.hidden_size)
                if hidden_size != cfg.d_model:
                    print(f"[config] overriding d_model {cfg.d_model} -> model hidden {hidden_size}")
                    cfg.d_model = hidden_size
                layers = base.model.layers
                self.embed_tokens = base.model.embed_tokens
                self.prelude = nn.ModuleList(layers[: cfg.prelude_layers])
                self.core = nn.ModuleList(layers[cfg.core_start : cfg.core_start + cfg.core_layers])
                self.coda = nn.ModuleList(layers[cfg.coda_start : cfg.coda_start + cfg.coda_layers])
                self.norm = base.model.norm
                self.rotary_emb = base.model.rotary_emb
                self.pos_tokens = None
                print(
                    "[model] "
                    f"{cfg.model_id} prelude={cfg.prelude_layers} "
                    f"core={cfg.core_start}:{cfg.core_start + cfg.core_layers} "
                    f"coda={cfg.coda_start}:{cfg.coda_start + cfg.coda_layers}"
                )
                del base

            self.loop_emb = nn.Parameter(torch.zeros(cfg.loop_steps, cfg.d_model))
            init = min(max(cfg.reinject_init, 1e-4), 1.0 - 1e-4)
            self.reinject_logit = nn.Parameter(torch.tensor(math.log(init / (1.0 - init))))
            self.head = nn.Linear(cfg.d_model, cfg.n_nodes)
            nn.init.normal_(self.loop_emb, std=0.02)

            for p in self.parameters():
                p.requires_grad_(False)
            if cfg.train_embeddings:
                for p in self.embed_tokens.parameters():
                    p.requires_grad_(True)
            if cfg.use_fake_model or cfg.train_prelude:
                for p in self.prelude.parameters():
                    p.requires_grad_(True)
            if cfg.use_fake_model or cfg.train_core:
                for p in self.core.parameters():
                    p.requires_grad_(True)
            if cfg.use_fake_model or cfg.train_coda:
                for p in self.coda.parameters():
                    p.requires_grad_(True)
            for p in self.norm.parameters():
                p.requires_grad_(cfg.use_fake_model)
            self.loop_emb.requires_grad_(True)
            self.reinject_logit.requires_grad_(True)
            for p in self.head.parameters():
                p.requires_grad_(True)

        def make_layer_inputs(self, input_ids, attention_mask, hidden):
            bsz, seq_len = input_ids.shape
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
            if self.fake:
                return attention_mask.bool(), position_ids, None
            min_val = torch.finfo(hidden.dtype).min
            causal = torch.full((seq_len, seq_len), min_val, dtype=hidden.dtype, device=hidden.device)
            causal = torch.triu(causal, diagonal=1)
            pad = torch.zeros((bsz, 1, 1, seq_len), dtype=hidden.dtype, device=hidden.device)
            pad = pad.masked_fill(~attention_mask[:, None, None, :].bool(), min_val)
            layer_mask = causal.view(1, 1, seq_len, seq_len) + pad
            position_embeddings = self.rotary_emb(hidden, position_ids)
            return layer_mask, position_ids, position_embeddings

        def run_layers(self, layers, hidden, layer_mask, position_ids, position_embeddings):
            for layer in layers:
                if self.fake:
                    hidden = layer(hidden, attention_mask=layer_mask)
                else:
                    hidden = layer(
                        hidden,
                        attention_mask=layer_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                    )
            return hidden

        def encode(self, input_ids, attention_mask):
            hidden = self.embed_tokens(input_ids)
            if self.fake:
                pos = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand_as(input_ids)
                hidden = hidden + self.pos_tokens(pos)
            layer_mask, position_ids, position_embeddings = self.make_layer_inputs(input_ids, attention_mask, hidden)
            hidden = self.run_layers(self.prelude, hidden, layer_mask, position_ids, position_embeddings)
            return hidden, layer_mask, position_ids, position_embeddings

        def loop_once(self, hidden, input_state, layer_mask, position_ids, position_embeddings, step_idx: int):
            idx = min(step_idx, cfg.loop_steps - 1)
            gate = torch.sigmoid(self.reinject_logit).to(hidden.dtype)
            hidden = hidden + self.loop_emb[idx].view(1, 1, -1).to(hidden.dtype) + gate * input_state
            return self.run_layers(self.core, hidden, layer_mask, position_ids, position_embeddings)

        def run_steps_from(
            self,
            hidden,
            input_state,
            layer_mask,
            position_ids,
            position_embeddings,
            start_step: int,
            n_steps: int,
        ):
            for i in range(n_steps):
                hidden = self.loop_once(
                    hidden,
                    input_state,
                    layer_mask,
                    position_ids,
                    position_embeddings,
                    start_step + i,
                )
            return hidden

        def read_last(self, hidden, lengths):
            idx = lengths.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
            return hidden.gather(1, idx).squeeze(1)

        def classify_state(self, hidden, lengths, layer_mask, position_ids, position_embeddings):
            hidden = self.run_layers(self.coda, hidden, layer_mask, position_ids, position_embeddings)
            hidden = self.norm(hidden)
            read = self.read_last(hidden, lengths)
            return self.head(read)

        def collect_logits(self, input_ids, attention_mask, lengths, loops: int, include_zero: bool = False):
            input_state, layer_mask, position_ids, position_embeddings = self.encode(input_ids, attention_mask)
            hidden = input_state
            logits = []
            if include_zero:
                logits.append(self.classify_state(hidden, lengths, layer_mask, position_ids, position_embeddings))
            for step in range(loops):
                hidden = self.loop_once(hidden, input_state, layer_mask, position_ids, position_embeddings, step)
                logits.append(self.classify_state(hidden, lengths, layer_mask, position_ids, position_embeddings))
            return logits

        def forward(self, input_ids, attention_mask, lengths, loops: int):
            return self.collect_logits(input_ids, attention_mask, lengths, loops, include_zero=False)[-1]

    if cfg.use_fake_model:
        tokenizer = None

        def encode_texts(texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            ids = torch.zeros((len(texts), cfg.max_length), dtype=torch.long, device=device)
            attention = torch.zeros_like(ids)
            for i, text in enumerate(texts):
                vals = [ord(ch) % 4096 for ch in text[: cfg.max_length]]
                ids[i, : len(vals)] = torch.tensor(vals, dtype=torch.long, device=device)
                attention[i, : len(vals)] = 1
            lengths = attention.sum(dim=1) - 1
            return ids, attention.bool(), lengths

    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def encode_texts(texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            batch = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.max_length,
            ).to(device)
            lengths = batch["attention_mask"].sum(dim=1) - 1
            return batch["input_ids"], batch["attention_mask"].bool(), lengths

    def batch_encoded(batch_size: int):
        texts, target, step_targets, hops = make_examples(batch_size)
        input_ids, attention_mask, lengths = encode_texts(texts)
        return input_ids, attention_mask, lengths, target, step_targets, hops

    def accuracy(logits, target) -> float:
        return (logits.argmax(dim=-1) == target).float().mean().item()

    model = RecurrentSmol().to(device)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    print(f"[trainable] params={sum(p.numel() for _, p in trainable)/1e6:.3f}M")
    head_names = ("head.", "loop_emb", "reinject_logit")
    block_params = [p for n, p in trainable if not n.startswith(head_names)]
    head_params = [p for n, p in trainable if n.startswith(head_names)]
    optim_groups = []
    if block_params:
        optim_groups.append({"params": block_params, "lr": cfg.lr_blocks, "weight_decay": cfg.weight_decay})
    optim_groups.append({"params": head_params, "lr": cfg.lr_head, "weight_decay": 0.0})
    opt = torch.optim.AdamW(optim_groups)

    t0 = time.time()
    model.train()
    for step in range(1, cfg.final_steps + 1):
        input_ids, attention_mask, lengths, target, _, _ = batch_encoded(cfg.batch_size)
        logits = model(input_ids, attention_mask, lengths, cfg.loop_steps)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
        opt.step()
        if step % cfg.log_every == 0 or step == cfg.final_steps:
            print(
                f"  final step {step:5d}/{cfg.final_steps} "
                f"loss {loss.item():.4f} acc {accuracy(logits, target)*100:.1f}% "
                f"gate {torch.sigmoid(model.reinject_logit).item():.3f} "
                f"elapsed {time.time()-t0:.1f}s",
                flush=True,
            )

    for step in range(1, cfg.recurrent_steps + 1):
        input_ids, attention_mask, lengths, target, step_targets, _ = batch_encoded(cfg.batch_size)
        logits_by_loop = model.collect_logits(input_ids, attention_mask, lengths, cfg.loop_steps, include_zero=False)
        losses = []
        for i, logits_i in enumerate(logits_by_loop):
            weight = 2.0 if i == cfg.loop_steps - 1 else 1.0
            losses.append(weight * F.cross_entropy(logits_i, step_targets[:, i]))
        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
        opt.step()
        if step % cfg.log_every == 0 or step == cfg.recurrent_steps:
            full_logits = logits_by_loop[-1]
            one_logits = logits_by_loop[0]
            print(
                f"  recur step {step:5d}/{cfg.recurrent_steps} "
                f"loss {loss.item():.4f} one {accuracy(one_logits, target)*100:.1f}% "
                f"full {accuracy(full_logits, target)*100:.1f}% "
                f"elapsed {time.time()-t0:.1f}s",
                flush=True,
            )

    def eval_model():
        model.eval()
        final_acc = [0.0 for _ in range(cfg.loop_steps + 1)]
        step_acc = [None] + [0.0 for _ in range(cfg.loop_steps)]
        by_hop = {str(i): [] for i in range(1, cfg.max_hops + 1)}
        with torch.no_grad():
            for _ in range(cfg.eval_batches):
                input_ids, attention_mask, lengths, target, step_targets, hops = batch_encoded(cfg.batch_size)
                logits_by_loop = model.collect_logits(
                    input_ids, attention_mask, lengths, cfg.loop_steps, include_zero=True
                )
                preds = [logits.argmax(dim=-1) for logits in logits_by_loop]
                for i, pred in enumerate(preds):
                    final_acc[i] += (pred == target).float().mean().item()
                    if i > 0:
                        step_acc[i] += (pred == step_targets[:, i - 1]).float().mean().item()
                full_pred = preds[-1]
                for hop in range(1, cfg.max_hops + 1):
                    mask = hops == hop
                    if mask.any():
                        by_hop[str(hop)].append((full_pred[mask] == target[mask]).float().mean().item())
        model.train()
        final_acc = [x / cfg.eval_batches for x in final_acc]
        step_acc_out = {
            str(i): (step_acc[i] / cfg.eval_batches if i > 0 else None)
            for i in range(cfg.loop_steps + 1)
        }
        return {
            "final_acc_by_loops": {str(i): final_acc[i] for i in range(cfg.loop_steps + 1)},
            "step_acc_by_loops": step_acc_out,
            "zero_loop_acc": final_acc[0],
            "one_loop_acc": final_acc[1],
            "full_loop_acc": final_acc[-1],
            "loop_gain_vs_zero": final_acc[-1] - final_acc[0],
            "loop_gain_vs_one": final_acc[-1] - final_acc[1],
            "full_by_hop": {k: (sum(v) / len(v) if v else None) for k, v in by_hop.items()},
            "reinject_gate": torch.sigmoid(model.reinject_logit).item(),
        }

    eval_summary = eval_model()
    print(f"[eval] {json.dumps(eval_summary, indent=2)}")

    jumprec = None
    jump_summary = None
    if cfg.jump_steps > 0:
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()

        class WindowAdapter(nn.Module):
            def __init__(self):
                super().__init__()
                out_dim = 2 * cfg.d_model * cfg.adapter_rank + 1
                self.hyper = nn.Sequential(
                    nn.LayerNorm(cfg.d_model),
                    nn.Linear(cfg.d_model, cfg.d_model),
                    nn.GELU(),
                    nn.Linear(cfg.d_model, out_dim),
                )
                self.scale = nn.Parameter(torch.tensor(0.05))
                nn.init.normal_(self.hyper[-1].weight, std=1e-3)
                nn.init.zeros_(self.hyper[-1].bias)

            def forward(self, hidden, ctx):
                bsz, _, d_model = hidden.shape
                out = self.hyper(ctx)
                n = d_model * cfg.adapter_rank
                u = out[:, :n].view(bsz, d_model, cfg.adapter_rank)
                v = out[:, n : 2 * n].view(bsz, cfg.adapter_rank, d_model)
                gate = torch.sigmoid(out[:, -1]).view(bsz, 1, 1)
                delta = torch.bmm(torch.bmm(hidden, u), v)
                return self.scale * gate * torch.tanh(delta)

        class JumpRec(nn.Module):
            def __init__(self, teacher: RecurrentSmol):
                super().__init__()
                self.teacher = teacher
                source_layers = [teacher.core[i % len(teacher.core)] for i in range(cfg.jump_layers)]
                self.jump_blocks = nn.ModuleList([copy.deepcopy(layer) for layer in source_layers])
                self.landing_emb = nn.Parameter(torch.zeros(cfg.max_correct + 1, cfg.d_model))
                self.adapter = WindowAdapter() if cfg.use_temp_adapter else None
                self.norm = nn.LayerNorm(cfg.d_model)
                verifier_in = cfg.d_model + 3
                self.verifiers = nn.ModuleList(
                    [
                        nn.Sequential(nn.Linear(verifier_in, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, 1))
                        for _ in range(cfg.max_correct + 1)
                    ]
                )
                nn.init.normal_(self.landing_emb, std=0.02)

            def jump(self, state0, layer_mask, position_ids, position_embeddings, lengths, corrections: int):
                hidden = state0 + self.landing_emb[corrections].view(1, 1, -1).to(state0.dtype)
                hidden = self.teacher.run_layers(
                    self.jump_blocks,
                    hidden,
                    layer_mask,
                    position_ids,
                    position_embeddings,
                )
                if self.adapter is not None:
                    ctx = self.teacher.read_last(state0, lengths) + self.landing_emb[corrections].view(1, -1)
                    hidden = hidden + self.adapter(hidden, ctx)
                return hidden

            def verifier_features(self, state, logits, lengths):
                probs = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1, keepdim=True) / math.log(cfg.n_nodes)
                top2 = probs.topk(2, dim=-1).values
                margin = top2[:, 0:1] - top2[:, 1:2]
                max_prob = top2[:, 0:1]
                raw = self.norm(self.teacher.read_last(state, lengths))
                return torch.cat([raw, entropy, margin, max_prob], dim=-1)

            def forward_encoded(self, state0, layer_mask, position_ids, position_embeddings, lengths):
                landing_states, final_states, logits, verify = [], [], [], []
                for corrections in range(cfg.max_correct + 1):
                    landing = self.jump(state0, layer_mask, position_ids, position_embeddings, lengths, corrections)
                    final = self.teacher.run_steps_from(
                        landing,
                        state0,
                        layer_mask,
                        position_ids,
                        position_embeddings,
                        cfg.loop_steps - corrections,
                        corrections,
                    )
                    final_logits = self.teacher.classify_state(
                        final,
                        lengths,
                        layer_mask,
                        position_ids,
                        position_embeddings,
                    )
                    verifier_logit = self.verifiers[corrections](
                        self.verifier_features(final, final_logits, lengths)
                    ).squeeze(-1)
                    landing_states.append(landing)
                    final_states.append(final)
                    logits.append(final_logits)
                    verify.append(verifier_logit)
                return {
                    "landing_states": landing_states,
                    "final_states": final_states,
                    "logits": logits,
                    "verify": verify,
                }

        def teacher_collect(input_ids, attention_mask, lengths):
            state0, layer_mask, position_ids, position_embeddings = model.encode(input_ids, attention_mask)
            hidden = state0
            states = []
            for i in range(cfg.loop_steps):
                hidden = model.loop_once(hidden, state0, layer_mask, position_ids, position_embeddings, i)
                states.append(hidden)
            logits = model.classify_state(hidden, lengths, layer_mask, position_ids, position_embeddings)
            return state0, layer_mask, position_ids, position_embeddings, states, logits

        jumprec = JumpRec(model).to(device)
        print(f"[jumprec] trainable params={sum(p.numel() for p in jumprec.parameters() if p.requires_grad)/1e6:.3f}M")
        opt_j = torch.optim.AdamW([p for p in jumprec.parameters() if p.requires_grad], lr=cfg.jump_lr)
        t1 = time.time()
        jumprec.train()
        for step in range(1, cfg.jump_steps + 1):
            input_ids, attention_mask, lengths, target, _, _ = batch_encoded(cfg.batch_size)
            with torch.no_grad():
                state0, layer_mask, position_ids, position_embeddings, teacher_states, teacher_logits = teacher_collect(
                    input_ids,
                    attention_mask,
                    lengths,
                )
                soft_teacher = F.softmax(teacher_logits, dim=-1)
            out = jumprec.forward_encoded(state0.detach(), layer_mask, position_ids, position_embeddings, lengths)
            ce_losses, distill_losses, verifier_losses = [], [], []
            for corrections in range(cfg.max_correct + 1):
                final_logits = out["logits"][corrections]
                ce_losses.append(F.cross_entropy(final_logits, target))
                distill_losses.append(
                    F.kl_div(F.log_softmax(final_logits, dim=-1), soft_teacher, reduction="batchmean")
                )
                good = (final_logits.detach().argmax(dim=-1) == target).float()
                verifier_losses.append(F.binary_cross_entropy_with_logits(out["verify"][corrections], good))
            loss = (
                torch.stack(ce_losses).mean()
                + 0.2 * torch.stack(distill_losses).mean()
                + cfg.verifier_loss_weight * torch.stack(verifier_losses).mean()
            )
            opt_j.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(jumprec.parameters(), 1.0)
            opt_j.step()
            if step % cfg.log_every == 0 or step == cfg.jump_steps:
                print(
                    f"  jump step {step:5d}/{cfg.jump_steps} loss {loss.item():.4f} "
                    f"ce0 {ce_losses[0].item():.3f} ce{cfg.max_correct} {ce_losses[-1].item():.3f} "
                    f"elapsed {time.time()-t1:.1f}s",
                    flush=True,
                )

        def eval_jumprec():
            jumprec.eval()
            thresholds = [(0.80, "080"), (0.90, "090"), (0.95, "095")]
            metrics = {f"jump_c{c}_acc": 0.0 for c in range(cfg.max_correct + 1)}
            for _, suffix in thresholds:
                metrics[f"strict_{suffix}_fallback_acc"] = 0.0
                metrics[f"strict_{suffix}_fallback_full_loop_rate"] = 0.0
                metrics[f"strict_{suffix}_fallback_avg_tail_loops"] = 0.0
                metrics[f"strict_{suffix}_fallback_avg_core_layers"] = 0.0
            with torch.no_grad():
                for _ in range(cfg.eval_batches):
                    input_ids, attention_mask, lengths, target, _, _ = batch_encoded(cfg.batch_size)
                    state0, layer_mask, position_ids, position_embeddings, _, teacher_logits = teacher_collect(
                        input_ids,
                        attention_mask,
                        lengths,
                    )
                    full_pred = teacher_logits.argmax(dim=-1)
                    out = jumprec.forward_encoded(state0, layer_mask, position_ids, position_embeddings, lengths)
                    preds = [logits_i.argmax(dim=-1) for logits_i in out["logits"]]
                    pred_stack = torch.stack(preds, dim=0)
                    verify_stack = torch.stack([torch.sigmoid(v) for v in out["verify"]], dim=0)
                    logits_stack = torch.stack(out["logits"], dim=0)
                    probs = F.softmax(logits_stack, dim=-1)
                    top2 = probs.topk(2, dim=-1).values
                    margin_stack = top2[:, :, 0] - top2[:, :, 1]
                    max_prob_stack = top2[:, :, 0]
                    agree_stack = torch.ones_like(verify_stack, dtype=torch.bool)
                    agree_stack[:-1] = pred_stack[:-1] == pred_stack[1:]
                    agree_stack[-1] = False
                    for c, pred in enumerate(preds):
                        metrics[f"jump_c{c}_acc"] += (pred == target).float().mean().item()
                    for threshold, suffix in thresholds:
                        trusted = torch.zeros_like(target, dtype=torch.bool)
                        chosen = torch.full_like(target, cfg.loop_steps)
                        pred_fb = full_pred
                        for c in range(cfg.max_correct + 1):
                            accept = (
                                (~trusted)
                                & (verify_stack[c] >= threshold)
                                & (margin_stack[c] >= 0.05)
                                & (max_prob_stack[c] >= 0.45)
                                & agree_stack[c]
                            )
                            chosen = torch.where(accept, torch.full_like(chosen, c), chosen)
                            pred_fb = torch.where(accept, pred_stack[c], pred_fb)
                            trusted = trusted | accept
                        core_layers = torch.where(
                            trusted,
                            torch.full_like(chosen.float(), float(cfg.jump_layers))
                            + chosen.float() * float(cfg.core_layers),
                            torch.full_like(chosen.float(), float(cfg.loop_steps * cfg.core_layers)),
                        )
                        metrics[f"strict_{suffix}_fallback_acc"] += (pred_fb == target).float().mean().item()
                        metrics[f"strict_{suffix}_fallback_full_loop_rate"] += (~trusted).float().mean().item()
                        metrics[f"strict_{suffix}_fallback_avg_tail_loops"] += torch.where(
                            trusted,
                            chosen.float(),
                            torch.full_like(chosen.float(), float(cfg.loop_steps)),
                        ).mean().item()
                        metrics[f"strict_{suffix}_fallback_avg_core_layers"] += core_layers.mean().item()
            for k in list(metrics):
                metrics[k] /= cfg.eval_batches
            full_core_layers = float(cfg.loop_steps * cfg.core_layers)
            metrics["full_core_layers"] = full_core_layers
            for _, suffix in thresholds:
                metrics[f"strict_{suffix}_fallback_core_savings_vs_full_pct"] = (
                    1.0 - metrics[f"strict_{suffix}_fallback_avg_core_layers"] / full_core_layers
                ) * 100.0
            jumprec.train()
            return metrics

        jump_summary = eval_jumprec()
        print(f"[jumprec eval] {json.dumps(jump_summary, indent=2)}")

    def benchmark():
        if cfg.timing_batches <= 0:
            return {}

        def sync():
            if device.type == "cuda":
                torch.cuda.synchronize()

        def time_fn(fn):
            with torch.no_grad():
                for _ in range(2):
                    input_ids, attention_mask, lengths, _, _, _ = batch_encoded(cfg.batch_size)
                    fn(input_ids, attention_mask, lengths)
                sync()
                start = time.perf_counter()
                for _ in range(cfg.timing_batches):
                    input_ids, attention_mask, lengths, _, _, _ = batch_encoded(cfg.batch_size)
                    fn(input_ids, attention_mask, lengths)
                sync()
                return 1000.0 * (time.perf_counter() - start) / cfg.timing_batches

        model.eval()
        out = {
            "batch_size": float(cfg.batch_size),
            "one_loop_ms_per_batch": time_fn(lambda ids, mask, lens: model(ids, mask, lens, 1)),
            "full_loop_ms_per_batch": time_fn(lambda ids, mask, lens: model(ids, mask, lens, cfg.loop_steps)),
        }
        if jumprec is not None:
            out["jumprec_all_budgets_ms_per_batch"] = time_fn(
                lambda ids, mask, lens: jumprec.forward_encoded(*model.encode(ids, mask), lens)
            )
        model.train()
        return out

    timing_summary = benchmark()
    print(f"[timing] {json.dumps(timing_summary, indent=2)}")

    return {
        "config": asdict(cfg),
        "eval": eval_summary,
        "jumprec_eval": jump_summary,
        "timing": timing_summary,
    }


@app.function(
    gpu=GPU_TYPE,
    timeout=72000,
    image=image,
    volumes={"/results": out_vol, "/cache": cache_vol},
)
def run_remote(mode: str = "retrofit_probe", seed: int | None = None) -> Dict[str, object]:
    import torch

    cfg = config_for_mode(mode)
    if seed is not None:
        cfg.seed = seed
    summary = run_experiment(cfg, "cuda")
    seed_suffix = f"_seed{cfg.seed}"
    torch.save(summary, f"/results/recurrent_smol_{mode}{seed_suffix}_summary.pt")
    with open(f"/results/recurrent_smol_{mode}{seed_suffix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    out_vol.commit()
    cache_vol.commit()
    return summary


@app.local_entrypoint()
def main(mode: str = "retrofit_probe", seed: int | None = None):
    summary = run_remote.remote(mode, seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="dry",
        choices=[
            "dry",
            "retrofit_probe",
            "jumprec_probe",
            "retrofit_8n4h",
            "retrofit_8n4h_unfreeze",
            "retrofit_12n6h",
            "retrofit_unfreeze",
            "retrofit_long",
        ],
    )
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if not args.local:
        raise SystemExit("Use `modal run run_recurrent_smol.py --mode retrofit_probe`, or add `--local`.")
    if args.mode != "dry":
        raise SystemExit("Local CPU guard: only --mode dry is allowed locally.")
    cfg = config_for_mode(args.mode)
    if args.seed is not None:
        cfg.seed = args.seed
    result = run_experiment(cfg, "cpu")
    print(json.dumps(result, indent=2))
