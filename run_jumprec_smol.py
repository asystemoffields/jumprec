"""JumpRec + SmolLM2-135M crash test.

This wraps a frozen pretrained LM as the text encoder, trains a looped
refinement teacher over its hidden states, then trains JumpRec to skip most of
that recurrence with verifier-controlled fallback.

Local shape sanity:
    python run_jumprec_smol.py --local --mode dry

H100 run:
    modal run run_jumprec_smol.py --mode smol_pointer
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import modal


app = modal.App("jumprec-smol")
out_vol = modal.Volume.from_name("jumprec-smol-results", create_if_missing=True)
cache_vol = modal.Volume.from_name("jumprec-smol-cache", create_if_missing=True)
GPU_TYPE = "H100"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "transformers", "accelerate", "safetensors")
    .env({"HF_HOME": "/cache/huggingface", "TRANSFORMERS_CACHE": "/cache/huggingface"})
)


@dataclass
class Config:
    mode: str = "smol_pointer"
    seed: int = 42
    model_id: str = "HuggingFaceTB/SmolLM2-135M"
    use_fake_encoder: bool = False
    n_nodes: int = 8
    max_hops: int = 4
    preserve_steps: int = 2
    max_correct: int = 4
    d_model: int = 576
    n_heads: int = 8
    d_ff: int = 1536
    adapter_rank: int = 8
    teacher_steps: int = 800
    jump_steps: int = 1000
    direct_steps: int = 1000
    batch_size: int = 48
    eval_batches: int = 32
    lr_teacher: float = 3e-4
    lr_jump: float = 4e-4
    hidden_loss_weight: float = 0.0
    verifier_loss_weight: float = 0.2
    log_every: int = 100
    use_temp_adapter: bool = True
    mixed_tasks: bool = False
    run_direct_baseline: bool = True
    timing_batches: int = 16
    max_length: int = 192

    @property
    def loop_steps(self) -> int:
        return self.max_hops + self.preserve_steps


def config_for_mode(mode: str) -> Config:
    cfg = Config(mode=mode)
    if mode == "dry":
        cfg.use_fake_encoder = True
        cfg.n_nodes = 6
        cfg.max_hops = 3
        cfg.max_correct = 2
        cfg.d_model = 64
        cfg.n_heads = 4
        cfg.d_ff = 192
        cfg.adapter_rank = 4
        cfg.teacher_steps = 10
        cfg.jump_steps = 10
        cfg.direct_steps = 10
        cfg.batch_size = 16
        cfg.eval_batches = 2
        cfg.timing_batches = 2
        cfg.log_every = 5
        cfg.max_length = 96
    elif mode == "smol_pointer":
        pass
    elif mode == "smol_pointer_easy":
        cfg.n_nodes = 6
        cfg.max_hops = 3
        cfg.max_correct = 3
        cfg.teacher_steps = 3500
        cfg.jump_steps = 2500
        cfg.direct_steps = 2500
        cfg.batch_size = 64
        cfg.eval_batches = 48
        cfg.log_every = 250
    elif mode == "smol_mix":
        cfg.mixed_tasks = True
        cfg.teacher_steps = 1200
        cfg.jump_steps = 1400
        cfg.direct_steps = 1400
    elif mode == "smol_pointer_long":
        cfg.teacher_steps = 1600
        cfg.jump_steps = 2000
        cfg.direct_steps = 2000
        cfg.eval_batches = 64
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
    task_names = ["forward", "inverse", "alternate", "square"] if cfg.mixed_tasks else ["forward"]

    if cfg.use_fake_encoder:
        tokenizer = None
        vocab_size = 4096
        fake_tok = nn.Embedding(vocab_size, cfg.d_model).to(device)
        fake_pos = nn.Embedding(cfg.max_length, cfg.d_model).to(device)
        for p in fake_tok.parameters():
            p.requires_grad_(False)
        for p in fake_pos.parameters():
            p.requires_grad_(False)
        print("[encoder] using fake frozen encoder")
    else:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        base_model = AutoModel.from_pretrained(cfg.model_id, torch_dtype=dtype).to(device)
        base_model.eval()
        for p in base_model.parameters():
            p.requires_grad_(False)
        hidden_size = int(base_model.config.hidden_size)
        if hidden_size != cfg.d_model:
            print(f"[config] overriding d_model {cfg.d_model} -> encoder hidden {hidden_size}")
            cfg.d_model = hidden_size
        print(f"[encoder] {cfg.model_id} hidden={cfg.d_model} params={sum(p.numel() for p in base_model.parameters())/1e6:.1f}M")

    def transition(perms, inv_perms, cur, task_ids, step, ar):
        fwd = perms[ar, cur]
        inv = inv_perms[ar, cur]
        alt = torch.where(torch.full_like(cur, step % 2 == 0, dtype=torch.bool), fwd, inv)
        sq = perms[ar, fwd]
        nxt = fwd
        if cfg.mixed_tasks:
            nxt = torch.where(task_ids == 1, inv, nxt)
            nxt = torch.where(task_ids == 2, alt, nxt)
            nxt = torch.where(task_ids == 3, sq, nxt)
        return nxt

    def make_examples(batch_size: int):
        perms_l = []
        texts = []
        starts_l = []
        hops_l = []
        tasks_l = []
        targets_l = []
        step_targets_l = []
        for _ in range(batch_size):
            perm = list(range(cfg.n_nodes))
            random.shuffle(perm)
            inv = [0 for _ in range(cfg.n_nodes)]
            for i, j in enumerate(perm):
                inv[j] = i
            start = random.randrange(cfg.n_nodes)
            hops = random.randint(1, cfg.max_hops)
            task_id = random.randrange(len(task_names))
            cur = start
            step_targets = []
            for step in range(cfg.loop_steps):
                fwd = perm[cur]
                invn = inv[cur]
                if task_id == 0:
                    nxt = fwd
                elif task_id == 1:
                    nxt = invn
                elif task_id == 2:
                    nxt = fwd if step % 2 == 0 else invn
                else:
                    nxt = perm[fwd]
                if hops > step:
                    cur = nxt
                step_targets.append(cur)
            mapping = " ".join(f"{letters[i]}->{letters[perm[i]]}" for i in range(cfg.n_nodes))
            task = task_names[task_id]
            text = (
                f"Task: {task}. Map: {mapping}. "
                f"Start: {letters[start]}. Hops: {hops}. Answer:"
            )
            texts.append(text)
            perms_l.append(perm)
            starts_l.append(start)
            hops_l.append(hops)
            tasks_l.append(task_id)
            targets_l.append(step_targets[-1])
            step_targets_l.append(step_targets)
        target = torch.tensor(targets_l, dtype=torch.long, device=device)
        step_targets = torch.tensor(step_targets_l, dtype=torch.long, device=device)
        hops = torch.tensor(hops_l, dtype=torch.long, device=device)
        task_ids = torch.tensor(tasks_l, dtype=torch.long, device=device)
        return texts, target, step_targets, hops, task_ids

    def encode_texts(texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if cfg.use_fake_encoder:
            ids = torch.zeros((len(texts), cfg.max_length), dtype=torch.long, device=device)
            attention = torch.zeros_like(ids)
            for i, text in enumerate(texts):
                vals = [ord(ch) % 4096 for ch in text[: cfg.max_length]]
                ids[i, : len(vals)] = torch.tensor(vals, dtype=torch.long, device=device)
                attention[i, : len(vals)] = 1
            pos = torch.arange(cfg.max_length, device=device).unsqueeze(0).expand_as(ids)
            with torch.no_grad():
                hidden = fake_tok(ids) + fake_pos(pos)
            lengths = attention.sum(dim=1) - 1
            return hidden.float(), attention.bool(), lengths

        batch = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_length,
        ).to(device)
        with torch.no_grad():
            out = base_model(**batch)
            hidden = out.last_hidden_state.float()
        lengths = batch["attention_mask"].sum(dim=1) - 1
        return hidden, batch["attention_mask"].bool(), lengths

    class TransformerBlock(nn.Module):
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

        def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            return self.layer(x, src_key_padding_mask=~attention_mask)

    def gather_last(state: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        idx = lengths.view(-1, 1, 1).expand(-1, 1, state.size(-1))
        return state.gather(1, idx).squeeze(1)

    class LoopedRefiner(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.LayerNorm(cfg.d_model),
                nn.Linear(cfg.d_model, cfg.d_model),
            )
            self.loop_block = TransformerBlock()
            self.step_emb = nn.Parameter(torch.zeros(cfg.loop_steps, cfg.d_model))
            self.norm = nn.LayerNorm(cfg.d_model)
            self.head = nn.Linear(cfg.d_model, cfg.n_nodes)
            nn.init.normal_(self.step_emb, std=0.02)

        def encode(self, hidden: torch.Tensor) -> torch.Tensor:
            return self.input_proj(hidden)

        def loop_once(self, state: torch.Tensor, attention_mask: torch.Tensor, step_idx: int) -> torch.Tensor:
            idx = min(step_idx, self.step_emb.size(0) - 1)
            return self.loop_block(state + self.step_emb[idx].view(1, 1, -1), attention_mask)

        def run_steps_from(self, state, attention_mask, start_step: int, n_steps: int):
            for i in range(n_steps):
                state = self.loop_once(state, attention_mask, start_step + i)
            return state

        def classify(self, state: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
            return self.head(self.norm(gather_last(state, lengths)))

        def collect_from_state(self, state, attention_mask, lengths, start_step: int, n_steps: int):
            logits = []
            states = []
            for step in range(n_steps):
                state = self.loop_once(state, attention_mask, start_step + step)
                states.append(state)
                logits.append(self.classify(state, lengths))
            return logits, states

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

        def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
            bsz, _, d_model = x.shape
            out = self.hyper(ctx)
            n = d_model * cfg.adapter_rank
            u = out[:, :n].view(bsz, d_model, cfg.adapter_rank)
            v = out[:, n:2 * n].view(bsz, cfg.adapter_rank, d_model)
            gate = torch.sigmoid(out[:, -1]).view(bsz, 1, 1)
            delta = torch.bmm(torch.bmm(x, u), v)
            return self.scale * gate * torch.tanh(delta)

    class JumpRec(nn.Module):
        def __init__(self, teacher: LoopedRefiner):
            super().__init__()
            self.teacher = teacher
            self.jump_block = TransformerBlock()
            self.landing_emb = nn.Parameter(torch.zeros(cfg.max_correct + 1, cfg.d_model))
            self.adapter = WindowAdapter() if cfg.use_temp_adapter else None
            verifier_in = cfg.d_model + 3
            self.verifiers = nn.ModuleList([
                nn.Sequential(nn.Linear(verifier_in, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, 1))
                for _ in range(cfg.max_correct + 1)
            ])
            nn.init.normal_(self.landing_emb, std=0.02)

        def jump(self, state0, attention_mask, lengths, corrections: int):
            state = self.jump_block(state0 + self.landing_emb[corrections].view(1, 1, -1), attention_mask)
            if self.adapter is not None:
                ctx = gather_last(state0, lengths) + self.landing_emb[corrections].view(1, -1)
                state = state + self.adapter(state, ctx)
            return state

        def verifier_features(self, state, logits, lengths):
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1, keepdim=True) / math.log(cfg.n_nodes)
            top2 = probs.topk(2, dim=-1).values
            margin = top2[:, 0:1] - top2[:, 1:2]
            max_prob = top2[:, 0:1]
            return torch.cat([self.teacher.norm(gather_last(state, lengths)), entropy, margin, max_prob], dim=-1)

        def forward_from_state(self, state0, attention_mask, lengths):
            landing_states, final_states, logits, verify = [], [], [], []
            for corrections in range(cfg.max_correct + 1):
                landing_state = self.jump(state0, attention_mask, lengths, corrections)
                final_state = self.teacher.run_steps_from(
                    landing_state,
                    attention_mask,
                    cfg.loop_steps - corrections,
                    corrections,
                )
                final_logits = self.teacher.classify(final_state, lengths)
                verifier_logit = self.verifiers[corrections](
                    self.verifier_features(final_state, final_logits, lengths)
                ).squeeze(-1)
                landing_states.append(landing_state)
                final_states.append(final_state)
                logits.append(final_logits)
                verify.append(verifier_logit)
            return {"landing_states": landing_states, "final_states": final_states, "logits": logits, "verify": verify}

    class DirectControl(nn.Module):
        def __init__(self, teacher: LoopedRefiner):
            super().__init__()
            self.teacher = teacher
            self.blocks = nn.ModuleList([TransformerBlock() for _ in range(3)])

        def forward_from_state(self, state, attention_mask, lengths):
            for block in self.blocks:
                state = block(state, attention_mask)
            return self.teacher.classify(state, lengths)

    def batch_encoded(batch_size: int):
        texts, target, step_targets, hops, task_ids = make_examples(batch_size)
        hidden, attention_mask, lengths = encode_texts(texts)
        return hidden, attention_mask, lengths, target, step_targets, hops, task_ids

    def accuracy(logits, target) -> float:
        return (logits.argmax(dim=-1) == target).float().mean().item()

    teacher = LoopedRefiner().to(device)
    print(f"[teacher] trainable params={sum(p.numel() for p in teacher.parameters())/1e6:.3f}M")
    opt_t = torch.optim.AdamW(teacher.parameters(), lr=cfg.lr_teacher)
    t0 = time.time()
    for step in range(1, cfg.teacher_steps + 1):
        hidden, attention_mask, lengths, target, step_targets, _, _ = batch_encoded(cfg.batch_size)
        state0 = teacher.encode(hidden)
        logits_by_step, _ = teacher.collect_from_state(state0, attention_mask, lengths, 0, cfg.loop_steps)
        losses = []
        for i, logits_i in enumerate(logits_by_step):
            losses.append((2.0 if i == cfg.loop_steps - 1 else 1.0) * F.cross_entropy(logits_i, step_targets[:, i]))
        loss = torch.stack(losses).mean()
        opt_t.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
        opt_t.step()
        if step % cfg.log_every == 0 or step == cfg.teacher_steps:
            print(f"  teacher step {step:5d}/{cfg.teacher_steps} loss {loss.item():.4f} elapsed {time.time()-t0:.1f}s", flush=True)

    def eval_teacher():
        teacher.eval()
        acc_by_step = [0.0 for _ in range(cfg.loop_steps)]
        by_hop = {str(i): [] for i in range(1, cfg.max_hops + 1)}
        by_task = {name: [] for name in task_names}
        with torch.no_grad():
            for _ in range(cfg.eval_batches):
                hidden, attention_mask, lengths, target, _, hops, task_ids = batch_encoded(cfg.batch_size)
                state0 = teacher.encode(hidden)
                logits_by_step, _ = teacher.collect_from_state(state0, attention_mask, lengths, 0, cfg.loop_steps)
                for i, logits_i in enumerate(logits_by_step):
                    acc_by_step[i] += accuracy(logits_i, target)
                pred = logits_by_step[-1].argmax(dim=-1)
                for hop in range(1, cfg.max_hops + 1):
                    mask = hops == hop
                    if mask.any():
                        by_hop[str(hop)].append((pred[mask] == target[mask]).float().mean().item())
                for task_id, name in enumerate(task_names):
                    mask = task_ids == task_id
                    if mask.any():
                        by_task[name].append((pred[mask] == target[mask]).float().mean().item())
        teacher.train()
        return {
            "one_loop_acc": acc_by_step[0] / cfg.eval_batches,
            "full_loop_acc": acc_by_step[-1] / cfg.eval_batches,
            "fixed_loop_acc_by_blocks": {str(i + 1): acc_by_step[i] / cfg.eval_batches for i in range(cfg.loop_steps)},
            "full_by_hop": {k: (sum(v) / len(v) if v else None) for k, v in by_hop.items()},
            "full_by_task": {k: (sum(v) / len(v) if v else None) for k, v in by_task.items()},
        }

    teacher_eval = eval_teacher()
    print(f"[teacher eval] {json.dumps(teacher_eval, indent=2)}")

    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    jumprec = JumpRec(teacher).to(device)
    print(f"[jumprec] trainable params={sum(p.numel() for p in jumprec.parameters() if p.requires_grad)/1e6:.3f}M")
    opt_j = torch.optim.AdamW([p for p in jumprec.parameters() if p.requires_grad], lr=cfg.lr_jump)
    t1 = time.time()
    for step in range(1, cfg.jump_steps + 1):
        hidden, attention_mask, lengths, target, _, _, _ = batch_encoded(cfg.batch_size)
        with torch.no_grad():
            state0 = teacher.encode(hidden)
            teacher_logits_by_step, teacher_states = teacher.collect_from_state(state0, attention_mask, lengths, 0, cfg.loop_steps)
            teacher_logits = teacher_logits_by_step[-1]
            soft_teacher = F.softmax(teacher_logits, dim=-1)
        out = jumprec.forward_from_state(state0.detach(), attention_mask, lengths)
        ce_losses, distill_losses, verifier_losses, hidden_losses = [], [], [], []
        for corrections in range(cfg.max_correct + 1):
            final_logits = out["logits"][corrections]
            ce_losses.append(F.cross_entropy(final_logits, target))
            distill_losses.append(F.kl_div(F.log_softmax(final_logits, dim=-1), soft_teacher, reduction="batchmean"))
            good = (final_logits.detach().argmax(dim=-1) == target).float()
            verifier_losses.append(F.binary_cross_entropy_with_logits(out["verify"][corrections], good))
            landing_step_count = cfg.loop_steps - corrections
            landing_target = state0 if landing_step_count == 0 else teacher_states[landing_step_count - 1]
            hidden_losses.append(F.mse_loss(out["landing_states"][corrections], landing_target))
        loss = (
            torch.stack(ce_losses).mean()
            + 0.2 * torch.stack(distill_losses).mean()
            + cfg.verifier_loss_weight * torch.stack(verifier_losses).mean()
            + cfg.hidden_loss_weight * torch.stack(hidden_losses).mean()
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
            metrics[f"strict_{suffix}_acc"] = 0.0
            metrics[f"strict_{suffix}_avg_correct_loops"] = 0.0
            metrics[f"strict_{suffix}_fallback_acc"] = 0.0
            metrics[f"strict_{suffix}_fallback_full_loop_rate"] = 0.0
            metrics[f"strict_{suffix}_fallback_avg_correct_loops"] = 0.0
        with torch.no_grad():
            for _ in range(cfg.eval_batches):
                hidden, attention_mask, lengths, target, _, _, _ = batch_encoded(cfg.batch_size)
                state0 = teacher.encode(hidden)
                out = jumprec.forward_from_state(state0, attention_mask, lengths)
                full_logits = teacher.classify(teacher.run_steps_from(state0, attention_mask, 0, cfg.loop_steps), lengths)
                full_pred = full_logits.argmax(dim=-1)
                preds = [logits_i.argmax(dim=-1) for logits_i in out["logits"]]
                pred_stack = torch.stack(preds, dim=0)
                prob_stack = torch.stack([torch.sigmoid(v) for v in out["verify"]], dim=0)
                logits_stack = torch.stack(out["logits"], dim=0)
                probs = F.softmax(logits_stack, dim=-1)
                top2 = probs.topk(2, dim=-1).values
                margin_stack = top2[:, :, 0] - top2[:, :, 1]
                max_prob_stack = top2[:, :, 0]
                agree_stack = torch.ones_like(prob_stack, dtype=torch.bool)
                agree_stack[:-1] = pred_stack[:-1] == pred_stack[1:]
                agree_stack[-1] = False
                for c, pred in enumerate(preds):
                    metrics[f"jump_c{c}_acc"] += (pred == target).float().mean().item()
                for threshold, suffix in thresholds:
                    chosen = torch.full_like(target, cfg.max_correct)
                    open_mask = torch.ones_like(target, dtype=torch.bool)
                    trusted = torch.zeros_like(target, dtype=torch.bool)
                    chosen_fb = torch.full_like(target, cfg.loop_steps)
                    pred_fb = full_pred
                    for c in range(cfg.max_correct + 1):
                        accept = (
                            open_mask
                            & (prob_stack[c] >= threshold)
                            & (margin_stack[c] >= 0.05)
                            & (max_prob_stack[c] >= 0.45)
                            & agree_stack[c]
                        )
                        chosen = torch.where(accept, torch.full_like(chosen, c), chosen)
                        open_mask = open_mask & ~accept
                        accept_fb = (
                            (~trusted)
                            & (prob_stack[c] >= threshold)
                            & (margin_stack[c] >= 0.05)
                            & (max_prob_stack[c] >= 0.45)
                            & agree_stack[c]
                        )
                        chosen_fb = torch.where(accept_fb, torch.full_like(chosen_fb, c), chosen_fb)
                        pred_fb = torch.where(accept_fb, pred_stack[c], pred_fb)
                        trusted = trusted | accept_fb
                    pred_adapt = pred_stack.gather(0, chosen.unsqueeze(0)).squeeze(0)
                    fb_blocks = torch.where(
                        trusted,
                        1.0 + chosen_fb.float(),
                        torch.full_like(chosen.float(), float(cfg.loop_steps)),
                    )
                    metrics[f"strict_{suffix}_acc"] += (pred_adapt == target).float().mean().item()
                    metrics[f"strict_{suffix}_avg_correct_loops"] += chosen.float().mean().item()
                    metrics[f"strict_{suffix}_fallback_acc"] += (pred_fb == target).float().mean().item()
                    metrics[f"strict_{suffix}_fallback_full_loop_rate"] += (~trusted).float().mean().item()
                    metrics[f"strict_{suffix}_fallback_avg_correct_loops"] += (fb_blocks - 1.0).mean().item()
        for k in list(metrics):
            metrics[k] /= cfg.eval_batches
        metrics["full_loop_blocks"] = float(cfg.loop_steps)
        for _, suffix in thresholds:
            metrics[f"strict_{suffix}_loop_equiv_blocks"] = 1.0 + metrics[f"strict_{suffix}_avg_correct_loops"]
            metrics[f"strict_{suffix}_compute_savings_vs_full_pct"] = (
                1.0 - metrics[f"strict_{suffix}_loop_equiv_blocks"] / cfg.loop_steps
            ) * 100.0
            metrics[f"strict_{suffix}_fallback_loop_equiv_blocks"] = (
                1.0 + metrics[f"strict_{suffix}_fallback_avg_correct_loops"]
            )
            metrics[f"strict_{suffix}_fallback_compute_savings_vs_full_pct"] = (
                1.0 - metrics[f"strict_{suffix}_fallback_loop_equiv_blocks"] / cfg.loop_steps
            ) * 100.0
        jumprec.train()
        return metrics

    jump_eval = eval_jumprec()
    print(f"[jumprec eval] {json.dumps(jump_eval, indent=2)}")

    direct_eval = None
    if cfg.run_direct_baseline:
        direct = DirectControl(teacher).to(device)
        print(f"[direct] trainable params={sum(p.numel() for p in direct.parameters() if p.requires_grad)/1e6:.3f}M")
        opt_d = torch.optim.AdamW(direct.parameters(), lr=cfg.lr_jump)
        t2 = time.time()
        for step in range(1, cfg.direct_steps + 1):
            hidden, attention_mask, lengths, target, _, _, _ = batch_encoded(cfg.batch_size)
            with torch.no_grad():
                state0 = teacher.encode(hidden)
            logits = direct.forward_from_state(state0.detach(), attention_mask, lengths)
            loss = F.cross_entropy(logits, target)
            opt_d.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(direct.parameters(), 1.0)
            opt_d.step()
            if step % cfg.log_every == 0 or step == cfg.direct_steps:
                print(f"  direct step {step:5d}/{cfg.direct_steps} loss {loss.item():.4f} elapsed {time.time()-t2:.1f}s", flush=True)
        direct.eval()
        acc = 0.0
        with torch.no_grad():
            for _ in range(cfg.eval_batches):
                hidden, attention_mask, lengths, target, _, _, _ = batch_encoded(cfg.batch_size)
                state0 = teacher.encode(hidden)
                acc += accuracy(direct.forward_from_state(state0, attention_mask, lengths), target)
        direct_eval = {
            "direct_acc": acc / cfg.eval_batches,
            "direct_block_equiv": 3.0,
            "direct_compute_savings_vs_full_pct": (1.0 - 3.0 / cfg.loop_steps) * 100.0,
        }
        print(f"[direct eval] {json.dumps(direct_eval, indent=2)}")

    def benchmark():
        if cfg.timing_batches <= 0:
            return {}
        def sync():
            if device.type == "cuda":
                torch.cuda.synchronize()
        def time_fn(fn):
            with torch.no_grad():
                for _ in range(2):
                    hidden, attention_mask, lengths, _, _, _, _ = batch_encoded(cfg.batch_size)
                    fn(hidden, attention_mask, lengths)
                sync()
                start = time.perf_counter()
                for _ in range(cfg.timing_batches):
                    hidden, attention_mask, lengths, _, _, _, _ = batch_encoded(cfg.batch_size)
                    fn(hidden, attention_mask, lengths)
                sync()
                return 1000.0 * (time.perf_counter() - start) / cfg.timing_batches
        return {
            "batch_size": float(cfg.batch_size),
            "teacher_full_ms_per_batch": time_fn(
                lambda h, m, l: teacher.classify(teacher.run_steps_from(teacher.encode(h), m, 0, cfg.loop_steps), l)
            ),
            "jumprec_all_budgets_ms_per_batch": time_fn(
                lambda h, m, l: jumprec.forward_from_state(teacher.encode(h), m, l)
            ),
        }

    timing_eval = benchmark()
    print(f"[timing eval] {json.dumps(timing_eval, indent=2)}")

    return {
        "config": asdict(cfg),
        "teacher_eval": teacher_eval,
        "jumprec_eval": jump_eval,
        "direct_eval": direct_eval,
        "timing_eval": timing_eval,
    }


@app.function(
    gpu=GPU_TYPE,
    timeout=72000,
    image=image,
    volumes={"/results": out_vol, "/cache": cache_vol},
)
def run_remote(mode: str = "smol_pointer", seed: int | None = None) -> Dict[str, object]:
    import torch

    cfg = config_for_mode(mode)
    if seed is not None:
        cfg.seed = seed
    summary = run_experiment(cfg, "cuda")
    seed_suffix = f"_seed{cfg.seed}"
    torch.save(summary, f"/results/jumprec_smol_{mode}{seed_suffix}_summary.pt")
    with open(f"/results/jumprec_smol_{mode}{seed_suffix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    out_vol.commit()
    cache_vol.commit()
    return summary


@app.local_entrypoint()
def main(mode: str = "smol_pointer", seed: int | None = None):
    summary = run_remote.remote(mode, seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="dry",
        choices=["dry", "smol_pointer", "smol_pointer_easy", "smol_mix", "smol_pointer_long"],
    )
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if not args.local:
        raise SystemExit("Use `modal run run_jumprec_smol.py --mode smol_pointer`, or add `--local`.")
    if args.mode != "dry":
        raise SystemExit("Local CPU guard: only --mode dry is allowed locally.")
    cfg = config_for_mode(args.mode)
    if args.seed is not None:
        cfg.seed = args.seed
    result = run_experiment(cfg, "cpu")
    print(json.dumps(result, indent=2))
