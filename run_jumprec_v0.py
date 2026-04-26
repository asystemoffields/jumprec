"""JumpRec v0: speculative recursive-state refinement.

This experiment tests whether a learned jump module can land on the teacher's
recurrent trajectory with some number of loop steps remaining, then use the
actual trained loop to finish only when the verifier is uncertain.

Task: synthetic pointer chasing over a random mapping table.

Run locally for a shape/sanity test:
    python run_jumprec_v0.py --local --mode dry

Run on Modal:
    modal run run_jumprec_v0.py --mode quick
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


app = modal.App("jumprec-v0")
out_vol = modal.Volume.from_name("jumprec-v0-results", create_if_missing=True)
GPU_TYPE = "H100"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy")
)


@dataclass
class Config:
    mode: str = "quick"
    seed: int = 42
    n_nodes: int = 16
    max_hops: int = 8
    preserve_steps: int = 2
    max_correct: int = 2
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    adapter_rank: int = 8
    teacher_steps: int = 2500
    jump_steps: int = 2500
    batch_size: int = 256
    eval_batches: int = 64
    lr_teacher: float = 3e-4
    lr_jump: float = 4e-4
    hidden_loss_weight: float = 0.2
    verifier_loss_weight: float = 0.2
    log_every: int = 100
    use_temp_adapter: bool = True
    train_max_hops: int | None = None
    direct_blocks: int = 3
    mixed_tasks: bool = False
    accept_min_margin: float = 0.0
    accept_min_prob: float = 0.0
    accept_require_next_agreement: bool = False
    teacher_min_full_acc: float = 0.0
    teacher_min_task_acc: float = 0.0
    teacher_extra_steps: int = 0
    teacher_extra_lr_scale: float = 0.5
    run_direct_baseline: bool = False
    timing_batches: int = 0

    @property
    def loop_steps(self) -> int:
        return self.max_hops + self.preserve_steps

    @property
    def total_step_slots(self) -> int:
        return self.loop_steps

    @property
    def seq_len(self) -> int:
        # [BOS] [MAP] plus key/value pairs plus [Q] start [H] hop [OUT]
        return 2 + 2 * self.n_nodes + 5


def config_for_mode(mode: str) -> Config:
    cfg = Config(mode=mode)
    if mode == "dry":
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.d_model = 64
        cfg.d_ff = 192
        cfg.adapter_rank = 4
        cfg.teacher_steps = 30
        cfg.jump_steps = 30
        cfg.batch_size = 48
        cfg.eval_batches = 4
        cfg.log_every = 10
    elif mode == "dry_mixed":
        cfg = config_for_mode("dry")
        cfg.mode = mode
        cfg.mixed_tasks = True
        cfg.accept_min_margin = 0.05
        cfg.accept_min_prob = 0.45
        cfg.accept_require_next_agreement = True
    elif mode == "dry_round":
        cfg = config_for_mode("dry_mixed")
        cfg.mode = mode
        cfg.teacher_min_full_acc = 0.0
        cfg.teacher_min_task_acc = 0.0
        cfg.run_direct_baseline = True
        cfg.timing_batches = 2
    elif mode == "smoke":
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.max_correct = 2
        cfg.d_model = 96
        cfg.d_ff = 384
        cfg.adapter_rank = 4
        cfg.teacher_steps = 900
        cfg.jump_steps = 900
        cfg.batch_size = 256
        cfg.eval_batches = 32
        cfg.log_every = 50
    elif mode == "probe":
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.max_correct = 4
        cfg.d_model = 128
        cfg.d_ff = 512
        cfg.adapter_rank = 8
        cfg.teacher_steps = 1400
        cfg.jump_steps = 5000
        cfg.batch_size = 512
        cfg.eval_batches = 128
        cfg.log_every = 250
    elif mode == "quick":
        cfg.n_nodes = 12
        cfg.max_hops = 6
        cfg.max_correct = 4
        cfg.teacher_steps = 3500
        cfg.jump_steps = 3500
        cfg.batch_size = 384
        cfg.eval_batches = 96
        cfg.log_every = 250
        pass
    elif mode == "quick_no_adapter":
        cfg = config_for_mode("quick")
        cfg.mode = mode
        cfg.use_temp_adapter = False
    elif mode == "quick_c6":
        cfg = config_for_mode("quick")
        cfg.mode = mode
        cfg.max_correct = 6
        cfg.jump_steps = 5000
    elif mode == "quick_c6_no_adapter":
        cfg = config_for_mode("quick_c6")
        cfg.mode = mode
        cfg.use_temp_adapter = False
    elif mode == "quick_c6_no_hidden":
        cfg = config_for_mode("quick_c6")
        cfg.mode = mode
        cfg.hidden_loss_weight = 0.0
    elif mode == "quick_direct3":
        cfg = config_for_mode("quick_c6")
        cfg.mode = mode
        cfg.direct_blocks = 3
        cfg.jump_steps = 5000
    elif mode == "quick_ood_hops":
        cfg = config_for_mode("quick_c6")
        cfg.mode = mode
        cfg.train_max_hops = 4
        cfg.teacher_steps = 4500
        cfg.jump_steps = 6000
    elif mode == "quick_mix":
        cfg = config_for_mode("quick_c6_no_hidden")
        cfg.mode = mode
        cfg.mixed_tasks = True
        cfg.teacher_steps = 6000
        cfg.jump_steps = 7000
        cfg.eval_batches = 128
    elif mode == "quick_mix_strict":
        cfg = config_for_mode("quick_mix")
        cfg.mode = mode
        cfg.accept_min_margin = 0.05
        cfg.accept_min_prob = 0.45
        cfg.accept_require_next_agreement = True
    elif mode == "quick_mix_direct3":
        cfg = config_for_mode("quick_mix")
        cfg.mode = mode
        cfg.direct_blocks = 3
        cfg.jump_steps = 7000
    elif mode == "quick_mix_round":
        cfg = config_for_mode("quick_mix_strict")
        cfg.mode = mode
        cfg.teacher_min_full_acc = 0.995
        cfg.teacher_min_task_acc = 0.980
        cfg.teacher_extra_steps = 4000
        cfg.run_direct_baseline = True
        cfg.timing_batches = 32
    elif mode == "full":
        cfg.d_model = 192
        cfg.d_ff = 768
        cfg.adapter_rank = 12
        cfg.teacher_steps = 8000
        cfg.jump_steps = 8000
        cfg.batch_size = 512
        cfg.eval_batches = 128
        cfg.log_every = 250
    else:
        raise ValueError(f"unknown mode: {mode}")
    return cfg


def run_experiment(cfg: Config, device_name: str = "cuda") -> Dict[str, object]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.max_correct > cfg.loop_steps:
        raise ValueError("max_correct cannot exceed loop_steps")
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")

    print(f"[device] {device}")
    if device.type == "cuda":
        print(f"[gpu] {torch.cuda.get_device_name()}")
    print(f"[config] {json.dumps(asdict(cfg), indent=2)}")

    BOS = 0
    MAP = 1
    Q = 2
    H = 3
    OUT = 4
    PAD = 5
    NODE_BASE = 6
    HOP_BASE = NODE_BASE + cfg.n_nodes
    TASK_BASE = HOP_BASE + cfg.max_hops
    task_names = ["forward", "inverse", "alternate", "square"] if cfg.mixed_tasks else ["forward"]
    vocab_size = TASK_BASE + len(task_names)
    out_pos = cfg.seq_len - 1

    def make_batch(
        batch_size: int,
        hop_limit: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if hop_limit is None:
            hop_limit = cfg.max_hops
        perms = torch.stack([torch.randperm(cfg.n_nodes) for _ in range(batch_size)]).to(device)
        inv_perms = torch.empty_like(perms)
        inv_perms.scatter_(
            1,
            perms,
            torch.arange(cfg.n_nodes, device=device).view(1, -1).expand(batch_size, -1),
        )
        starts = torch.randint(0, cfg.n_nodes, (batch_size,), device=device)
        hops = torch.randint(1, hop_limit + 1, (batch_size,), device=device)
        task_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        if cfg.mixed_tasks:
            task_ids = torch.randint(0, len(task_names), (batch_size,), device=device)

        ids = torch.full((batch_size, cfg.seq_len), PAD, dtype=torch.long, device=device)
        ids[:, 0] = BOS
        ids[:, 1] = MAP
        offset = 2
        for node in range(cfg.n_nodes):
            ids[:, offset + 2 * node] = NODE_BASE + node
            ids[:, offset + 2 * node + 1] = NODE_BASE + perms[:, node]
        q_pos = offset + 2 * cfg.n_nodes
        ids[:, q_pos] = TASK_BASE + task_ids if cfg.mixed_tasks else Q
        ids[:, q_pos + 1] = NODE_BASE + starts
        ids[:, q_pos + 2] = H
        ids[:, q_pos + 3] = HOP_BASE + hops - 1
        ids[:, q_pos + 4] = OUT

        ar = torch.arange(batch_size, device=device)
        cur = starts
        step_targets = []
        for step in range(cfg.total_step_slots):
            fwd = perms[ar, cur]
            inv = inv_perms[ar, cur]
            alt = torch.where(torch.full_like(cur, step % 2 == 0, dtype=torch.bool), fwd, inv)
            sq = perms[ar, fwd]
            nxt = fwd
            if cfg.mixed_tasks:
                nxt = torch.where(task_ids == 1, inv, nxt)
                nxt = torch.where(task_ids == 2, alt, nxt)
                nxt = torch.where(task_ids == 3, sq, nxt)
            cur = torch.where(hops > step, nxt, cur)
            step_targets.append(cur)
        step_targets_t = torch.stack(step_targets, dim=1)
        target = step_targets_t[:, cfg.loop_steps - 1]
        return ids, target, step_targets_t, hops, task_ids

    class SelfAttention(nn.Module):
        def __init__(self, d_model: int, n_heads: int):
            super().__init__()
            assert d_model % n_heads == 0
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            self.out = nn.Linear(d_model, d_model, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            bsz, seq_len, d_model = x.shape
            qkv = self.qkv(x).view(bsz, seq_len, 3, self.n_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            y = y.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
            return self.out(y)

    class TransformerBlock(nn.Module):
        def __init__(self, d_model: int, n_heads: int, d_ff: int):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = SelfAttention(d_model, n_heads)
            self.ln2 = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
            return x

    class LoopedPointerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(vocab_size, cfg.d_model)
            self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)
            self.input_block = TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff)
            self.loop_block = TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff)
            self.step_emb = nn.Parameter(torch.zeros(cfg.total_step_slots, cfg.d_model))
            self.norm = nn.LayerNorm(cfg.d_model)
            self.head = nn.Linear(cfg.d_model, cfg.n_nodes)
            nn.init.normal_(self.step_emb, std=0.02)

        def encode(self, ids: torch.Tensor) -> torch.Tensor:
            pos = torch.arange(ids.size(1), device=ids.device).unsqueeze(0).expand_as(ids)
            x = self.tok(ids) + self.pos(pos)
            return self.input_block(x)

        def loop_once(self, state: torch.Tensor, step_idx: int) -> torch.Tensor:
            idx = min(step_idx, self.step_emb.size(0) - 1)
            return self.loop_block(state + self.step_emb[idx].view(1, 1, -1))

        def run_steps_from(self, state: torch.Tensor, start_step: int, n_steps: int) -> torch.Tensor:
            for i in range(n_steps):
                state = self.loop_once(state, start_step + i)
            return state

        def classify(self, state: torch.Tensor) -> torch.Tensor:
            return self.head(self.norm(state[:, out_pos]))

        def collect_from_state(self, state: torch.Tensor, start_step: int, n_steps: int):
            logits = []
            states = []
            for step in range(n_steps):
                state = self.loop_once(state, start_step + step)
                states.append(state)
                logits.append(self.classify(state))
            return logits, states

        def collect(self, ids: torch.Tensor, n_steps: int):
            state = self.encode(ids)
            return self.collect_from_state(state, 0, n_steps)

        def forward(self, ids: torch.Tensor, n_steps: int, collect_logits: bool = False):
            logits, states = self.collect(ids, n_steps)
            if collect_logits:
                return logits, states[-1]
            return logits[-1], states[-1]

    class WindowAdapter(nn.Module):
        """Per-example low-rank adapter generated from the query/window state."""

        def __init__(self, d_model: int, rank: int):
            super().__init__()
            self.d_model = d_model
            self.rank = rank
            out_dim = 2 * d_model * rank + 1
            self.hyper = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, out_dim),
            )
            nn.init.zeros_(self.hyper[-1].bias)
            nn.init.normal_(self.hyper[-1].weight, std=1e-3)
            self.scale = nn.Parameter(torch.tensor(0.05))

        def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
            bsz, seq_len, d_model = x.shape
            out = self.hyper(ctx)
            n = d_model * self.rank
            u = out[:, :n].view(bsz, d_model, self.rank)
            v = out[:, n:2 * n].view(bsz, self.rank, d_model)
            gate = torch.sigmoid(out[:, -1]).view(bsz, 1, 1)
            delta = torch.bmm(torch.bmm(x, u), v)
            return self.scale * gate * torch.tanh(delta)

    class JumpRecModel(nn.Module):
        def __init__(self, teacher: LoopedPointerModel):
            super().__init__()
            self.teacher = teacher
            self.jump_block = TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff)
            self.landing_emb = nn.Parameter(torch.zeros(cfg.max_correct + 1, cfg.d_model))
            self.adapter = WindowAdapter(cfg.d_model, cfg.adapter_rank) if cfg.use_temp_adapter else None
            verifier_in = cfg.d_model + 3
            self.verifiers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(verifier_in, cfg.d_model),
                    nn.GELU(),
                    nn.Linear(cfg.d_model, 1),
                )
                for _ in range(cfg.max_correct + 1)
            ])
            nn.init.normal_(self.landing_emb, std=0.02)

        def jump(self, state0: torch.Tensor, corrections_remaining: int) -> torch.Tensor:
            landing = self.landing_emb[corrections_remaining].view(1, 1, -1)
            state = self.jump_block(state0 + landing)
            if self.adapter is not None:
                ctx = state0[:, out_pos] + self.landing_emb[corrections_remaining].view(1, -1)
                state = state + self.adapter(state, ctx)
            return state

        def verifier_features(self, state: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1, keepdim=True) / math.log(cfg.n_nodes)
            top2 = probs.topk(2, dim=-1).values
            margin = (top2[:, 0:1] - top2[:, 1:2])
            max_prob = top2[:, 0:1]
            return torch.cat([self.teacher.norm(state[:, out_pos]), entropy, margin, max_prob], dim=-1)

        def forward_from_state(self, state0: torch.Tensor):
            landing_states = []
            final_states = []
            logits = []
            verify = []
            for corrections in range(cfg.max_correct + 1):
                landing_state = self.jump(state0, corrections)
                final_state = self.teacher.run_steps_from(
                    landing_state,
                    cfg.loop_steps - corrections,
                    corrections,
                )
                final_logits = self.teacher.classify(final_state)
                verifier_logit = self.verifiers[corrections](
                    self.verifier_features(final_state, final_logits)
                ).squeeze(-1)
                landing_states.append(landing_state)
                final_states.append(final_state)
                logits.append(final_logits)
                verify.append(verifier_logit)
            return {
                "landing_states": landing_states,
                "final_states": final_states,
                "logits": logits,
                "verify": verify,
            }

    class DirectControlModel(nn.Module):
        """Equal-compute-ish non-jump baseline: trainable feed-forward depth from
        encoded state directly to answer, with no teacher trajectory landing and
        no recurrent tail loops."""

        def __init__(self, teacher: LoopedPointerModel):
            super().__init__()
            self.teacher = teacher
            self.blocks = nn.ModuleList([
                TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff)
                for _ in range(cfg.direct_blocks)
            ])

        def forward_from_state(self, state0: torch.Tensor) -> torch.Tensor:
            state = state0
            for block in self.blocks:
                state = block(state)
            return self.teacher.classify(state)

    def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
        return (logits.argmax(dim=-1) == target).float().mean().item()

    def grouped_accuracy(pred: torch.Tensor, target: torch.Tensor, hops: torch.Tensor) -> Dict[str, float]:
        out = {}
        for hop in range(1, cfg.max_hops + 1):
            mask = hops == hop
            if mask.any():
                out[str(hop)] = (pred[mask] == target[mask]).float().mean().item()
        return out

    def grouped_task_accuracy(pred: torch.Tensor, target: torch.Tensor, task_ids: torch.Tensor) -> Dict[str, float]:
        out = {}
        for task_id, name in enumerate(task_names):
            mask = task_ids == task_id
            if mask.any():
                out[name] = (pred[mask] == target[mask]).float().mean().item()
        return out

    teacher = LoopedPointerModel().to(device)
    n_teacher = sum(p.numel() for p in teacher.parameters())
    print(f"[teacher] params={n_teacher/1e6:.3f}M")

    opt = torch.optim.AdamW(teacher.parameters(), lr=cfg.lr_teacher)
    t0 = time.time()
    print("[teacher] training")
    train_hop_limit = cfg.train_max_hops or cfg.max_hops

    def train_teacher_steps(n_steps: int, start_step: int = 0, label: str = "teacher") -> None:
        for local_step in range(1, n_steps + 1):
            step = start_step + local_step
            ids, target, step_targets, _, _ = make_batch(cfg.batch_size, hop_limit=train_hop_limit)
            logits_by_step, _ = teacher(ids, cfg.loop_steps, collect_logits=True)
            losses = []
            for i, logits_i in enumerate(logits_by_step):
                weight = 2.0 if i == cfg.loop_steps - 1 else 1.0
                losses.append(weight * F.cross_entropy(logits_i, step_targets[:, i]))
            loss = torch.stack(losses).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            opt.step()
            if local_step % cfg.log_every == 0 or local_step == n_steps:
                print(
                    f"  {label} step {step:5d}/{start_step + n_steps} "
                    f"loss {loss.item():.4f} elapsed {time.time()-t0:.1f}s",
                    flush=True,
                )

    train_teacher_steps(cfg.teacher_steps)

    def eval_teacher() -> Dict[str, object]:
        teacher.eval()
        acc_by_step = [0.0 for _ in range(cfg.loop_steps)]
        by_hop_full_acc: Dict[str, List[float]] = {str(i): [] for i in range(1, cfg.max_hops + 1)}
        by_task_full_acc: Dict[str, List[float]] = {name: [] for name in task_names}
        with torch.no_grad():
            for _ in range(cfg.eval_batches):
                ids, target, _, hops, task_ids = make_batch(cfg.batch_size)
                logits_by_step, _ = teacher(ids, cfg.loop_steps, collect_logits=True)
                for i, logits_i in enumerate(logits_by_step):
                    acc_by_step[i] += accuracy(logits_i, target)
                pred_full = logits_by_step[-1].argmax(dim=-1)
                grouped = grouped_accuracy(pred_full, target, hops)
                for k, v in grouped.items():
                    by_hop_full_acc[k].append(v)
                grouped_task = grouped_task_accuracy(pred_full, target, task_ids)
                for k, v in grouped_task.items():
                    by_task_full_acc[k].append(v)
        teacher.train()
        by_hop_full = {
            k: (sum(v) / len(v) if v else None)
            for k, v in by_hop_full_acc.items()
        }
        by_task_full = {
            k: (sum(v) / len(v) if v else None)
            for k, v in by_task_full_acc.items()
        }
        fixed_loop_acc = {
            str(i + 1): acc_by_step[i] / cfg.eval_batches
            for i in range(cfg.loop_steps)
        }
        return {
            "one_loop_acc": fixed_loop_acc["1"],
            "full_loop_acc": fixed_loop_acc[str(cfg.loop_steps)],
            "fixed_loop_acc_by_blocks": fixed_loop_acc,
            "full_by_hop": by_hop_full,
            "full_by_task": by_task_full,
        }

    teacher_eval = eval_teacher()
    print(f"[teacher eval] {json.dumps(teacher_eval, indent=2)}")

    def teacher_quality_failures(eval_result: Dict[str, object]) -> List[str]:
        failures = []
        full_acc = float(eval_result["full_loop_acc"])
        if cfg.teacher_min_full_acc and full_acc < cfg.teacher_min_full_acc:
            failures.append(f"full_loop_acc {full_acc:.4f} < {cfg.teacher_min_full_acc:.4f}")
        if cfg.teacher_min_task_acc:
            by_task = eval_result.get("full_by_task", {})
            for task_name, task_acc in by_task.items():
                if task_acc is not None and float(task_acc) < cfg.teacher_min_task_acc:
                    failures.append(f"{task_name} task acc {float(task_acc):.4f} < {cfg.teacher_min_task_acc:.4f}")
        return failures

    quality_failures = teacher_quality_failures(teacher_eval)
    teacher_extra_used = 0
    if quality_failures and cfg.teacher_extra_steps > 0:
        print(f"[teacher gate] retrying: {quality_failures}", flush=True)
        for group in opt.param_groups:
            group["lr"] = cfg.lr_teacher * cfg.teacher_extra_lr_scale
        teacher_extra_used = cfg.teacher_extra_steps
        train_teacher_steps(cfg.teacher_extra_steps, start_step=cfg.teacher_steps, label="teacher-extra")
        teacher_eval = eval_teacher()
        quality_failures = teacher_quality_failures(teacher_eval)
        print(f"[teacher eval after gate] {json.dumps(teacher_eval, indent=2)}")
    if quality_failures:
        print(f"[teacher gate] failed: {quality_failures}", flush=True)
    else:
        print("[teacher gate] passed", flush=True)

    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    if cfg.mode in {"quick_direct3", "quick_mix_direct3"}:
        direct = DirectControlModel(teacher).to(device)
        n_direct_trainable = sum(p.numel() for p in direct.parameters() if p.requires_grad)
        print(f"[direct] trainable params={n_direct_trainable/1e6:.3f}M")

        opt_d = torch.optim.AdamW([p for p in direct.parameters() if p.requires_grad], lr=cfg.lr_jump)
        t1 = time.time()
        print("[direct] training")
        for step in range(1, cfg.jump_steps + 1):
            ids, target, _, _, _ = make_batch(cfg.batch_size, hop_limit=train_hop_limit)
            with torch.no_grad():
                state0 = teacher.encode(ids)
            logits = direct.forward_from_state(state0.detach())
            loss = F.cross_entropy(logits, target)
            opt_d.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(direct.parameters(), 1.0)
            opt_d.step()
            if step % cfg.log_every == 0 or step == cfg.jump_steps:
                print(
                    f"  direct step {step:5d}/{cfg.jump_steps} loss {loss.item():.4f} "
                    f"elapsed {time.time()-t1:.1f}s",
                    flush=True,
                )

        def eval_direct() -> Dict[str, object]:
            direct.eval()
            acc = 0.0
            by_hop_acc: Dict[str, List[float]] = {str(i): [] for i in range(1, cfg.max_hops + 1)}
            by_task_acc: Dict[str, List[float]] = {name: [] for name in task_names}
            with torch.no_grad():
                for _ in range(cfg.eval_batches):
                    ids, target, _, hops, task_ids = make_batch(cfg.batch_size)
                    state0 = teacher.encode(ids)
                    logits = direct.forward_from_state(state0)
                    pred = logits.argmax(dim=-1)
                    acc += (pred == target).float().mean().item()
                    grouped = grouped_accuracy(pred, target, hops)
                    for k, v in grouped.items():
                        by_hop_acc[k].append(v)
                    grouped_task = grouped_task_accuracy(pred, target, task_ids)
                    for k, v in grouped_task.items():
                        by_task_acc[k].append(v)
            direct.train()
            by_hop = {
                k: (sum(v) / len(v) if v else None)
                for k, v in by_hop_acc.items()
            }
            by_task = {
                k: (sum(v) / len(v) if v else None)
                for k, v in by_task_acc.items()
            }
            return {
                "direct_acc": acc / cfg.eval_batches,
                "direct_block_equiv": float(cfg.direct_blocks),
                "direct_compute_savings_vs_full_pct": (1.0 - cfg.direct_blocks / cfg.loop_steps) * 100.0,
                "direct_by_hop": by_hop,
                "direct_by_task": by_task,
            }

        direct_eval = eval_direct()
        print(f"[direct eval] {json.dumps(direct_eval, indent=2)}")
        return {
            "config": asdict(cfg),
            "teacher_params": n_teacher,
            "direct_trainable_params": n_direct_trainable,
            "teacher_eval": teacher_eval,
            "teacher_gate": {
                "passed": not quality_failures,
                "failures": quality_failures,
                "extra_steps_used": teacher_extra_used,
            },
            "direct_eval": direct_eval,
        }

    jumprec = JumpRecModel(teacher).to(device)
    n_jump_trainable = sum(p.numel() for p in jumprec.parameters() if p.requires_grad)
    print(f"[jumprec] trainable params={n_jump_trainable/1e6:.3f}M")

    opt_j = torch.optim.AdamW([p for p in jumprec.parameters() if p.requires_grad], lr=cfg.lr_jump)
    t1 = time.time()
    print("[jumprec] training")
    for step in range(1, cfg.jump_steps + 1):
        ids, target, _, _, _ = make_batch(cfg.batch_size, hop_limit=train_hop_limit)
        with torch.no_grad():
            state0 = teacher.encode(ids)
            teacher_logits_by_step, teacher_states = teacher.collect_from_state(state0, 0, cfg.loop_steps)
            teacher_final = teacher_states[-1]
            teacher_logits = teacher_logits_by_step[-1]

        out = jumprec.forward_from_state(state0.detach())
        soft_teacher = F.softmax(teacher_logits, dim=-1)
        ce_losses = []
        hidden_losses = []
        distill_losses = []
        verifier_losses = []
        for corrections in range(cfg.max_correct + 1):
            final_logits = out["logits"][corrections]
            ce_losses.append(F.cross_entropy(final_logits, target))
            landing_step_count = cfg.loop_steps - corrections
            landing_target = state0 if landing_step_count == 0 else teacher_states[landing_step_count - 1]
            hidden_losses.append(F.mse_loss(out["landing_states"][corrections], landing_target))
            logp = F.log_softmax(final_logits, dim=-1)
            distill_losses.append(F.kl_div(logp, soft_teacher, reduction="batchmean"))
            good = (final_logits.detach().argmax(dim=-1) == target).float()
            verifier_losses.append(F.binary_cross_entropy_with_logits(out["verify"][corrections], good))

        ce_loss = torch.stack(ce_losses).mean()
        hidden_loss = torch.stack(hidden_losses).mean()
        distill = torch.stack(distill_losses).mean()
        verifier_loss = torch.stack(verifier_losses).mean()

        loss = (
            ce_loss
            + 0.2 * distill
            + cfg.hidden_loss_weight * hidden_loss
            + cfg.verifier_loss_weight * verifier_loss
        )
        opt_j.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(jumprec.parameters(), 1.0)
        opt_j.step()

        if step % cfg.log_every == 0 or step == cfg.jump_steps:
            print(
                "  jump step "
                f"{step:5d}/{cfg.jump_steps} loss {loss.item():.4f} "
                f"ce0 {ce_losses[0].item():.3f} "
                f"ce{cfg.max_correct} {ce_losses[-1].item():.3f} "
                f"elapsed {time.time()-t1:.1f}s",
                flush=True,
            )

    def eval_jumprec() -> Dict[str, object]:
        jumprec.eval()
        thresholds = [(0.70, "070"), (0.80, "080"), (0.90, "090"), (0.95, "095")]
        policy_defs = {
            "verifier_only": {"min_margin": 0.0, "min_prob": 0.0, "require_next_agreement": False},
            "margin_conf": {"min_margin": 0.05, "min_prob": 0.45, "require_next_agreement": False},
            "strict": {"min_margin": 0.05, "min_prob": 0.45, "require_next_agreement": True},
        }
        metrics = {}
        for corrections in range(cfg.max_correct + 1):
            metrics[f"jump_c{corrections}_acc"] = 0.0
        for _, suffix in thresholds:
            metrics[f"adaptive_{suffix}_acc"] = 0.0
            metrics[f"adaptive_{suffix}_avg_correct_loops"] = 0.0
            metrics[f"adaptive_{suffix}_accept_c0_rate"] = 0.0
            metrics[f"fallback_{suffix}_acc"] = 0.0
            metrics[f"fallback_{suffix}_avg_correct_loops"] = 0.0
            metrics[f"fallback_{suffix}_full_loop_rate"] = 0.0
        policy_sweep = {}
        for policy_name in policy_defs:
            policy_sweep[policy_name] = {}
            for _, suffix in thresholds:
                policy_sweep[policy_name][suffix] = {
                    "adaptive_acc": 0.0,
                    "adaptive_avg_correct_loops": 0.0,
                    "fallback_acc": 0.0,
                    "fallback_avg_correct_loops": 0.0,
                    "fallback_full_loop_rate": 0.0,
                }
        hop_compute_080: Dict[str, List[float]] = {str(i): [] for i in range(1, cfg.max_hops + 1)}
        hop_acc_080: Dict[str, List[float]] = {str(i): [] for i in range(1, cfg.max_hops + 1)}
        hop_fb_compute_080: Dict[str, List[float]] = {str(i): [] for i in range(1, cfg.max_hops + 1)}
        hop_fb_acc_080: Dict[str, List[float]] = {str(i): [] for i in range(1, cfg.max_hops + 1)}
        task_acc_080: Dict[str, List[float]] = {name: [] for name in task_names}
        task_fb_acc_080: Dict[str, List[float]] = {name: [] for name in task_names}
        task_fb_blocks_080: Dict[str, List[float]] = {name: [] for name in task_names}
        task_fb_full_rate_080: Dict[str, List[float]] = {name: [] for name in task_names}
        with torch.no_grad():
            for _ in range(cfg.eval_batches):
                ids, target, _, hops, task_ids = make_batch(cfg.batch_size)
                state0 = teacher.encode(ids)
                out = jumprec.forward_from_state(state0)
                full_logits = teacher.classify(teacher.run_steps_from(state0, 0, cfg.loop_steps))
                full_pred = full_logits.argmax(dim=-1)
                preds = [logits_i.argmax(dim=-1) for logits_i in out["logits"]]
                pred_stack = torch.stack(preds, dim=0)  # (C+1, B)
                verifier_probs = [torch.sigmoid(v) for v in out["verify"]]
                prob_stack = torch.stack(verifier_probs, dim=0)
                logits_stack = torch.stack(out["logits"], dim=0)
                choice_probs = F.softmax(logits_stack, dim=-1)
                top2 = choice_probs.topk(2, dim=-1).values
                margin_stack = top2[:, :, 0] - top2[:, :, 1]
                max_prob_stack = top2[:, :, 0]
                next_agreement_stack = torch.ones_like(prob_stack, dtype=torch.bool)
                next_agreement_stack[:-1] = pred_stack[:-1] == pred_stack[1:]
                next_agreement_stack[-1] = False
                agreement_stack = torch.ones_like(prob_stack, dtype=torch.bool)
                if cfg.accept_require_next_agreement:
                    agreement_stack = next_agreement_stack

                for corrections, pred_i in enumerate(preds):
                    metrics[f"jump_c{corrections}_acc"] += (pred_i == target).float().mean().item()

                for policy_name, policy in policy_defs.items():
                    policy_agreement_stack = (
                        next_agreement_stack
                        if policy["require_next_agreement"]
                        else torch.ones_like(prob_stack, dtype=torch.bool)
                    )
                    for threshold, suffix in thresholds:
                        chosen_policy = torch.full_like(target, cfg.max_correct)
                        open_policy = torch.ones_like(target, dtype=torch.bool)
                        for corrections in range(cfg.max_correct + 1):
                            accept_policy = (
                                open_policy
                                & (prob_stack[corrections] >= threshold)
                                & (margin_stack[corrections] >= policy["min_margin"])
                                & (max_prob_stack[corrections] >= policy["min_prob"])
                                & policy_agreement_stack[corrections]
                            )
                            chosen_policy = torch.where(
                                accept_policy,
                                torch.full_like(chosen_policy, corrections),
                                chosen_policy,
                            )
                            open_policy = open_policy & ~accept_policy
                        pred_policy = pred_stack.gather(0, chosen_policy.unsqueeze(0)).squeeze(0)
                        policy_sweep[policy_name][suffix]["adaptive_acc"] += (
                            pred_policy == target
                        ).float().mean().item()
                        policy_sweep[policy_name][suffix]["adaptive_avg_correct_loops"] += (
                            chosen_policy.float().mean().item()
                        )

                        trusted_policy = torch.zeros_like(target, dtype=torch.bool)
                        chosen_policy_fb = torch.full_like(target, cfg.loop_steps)
                        pred_policy_fb = full_pred
                        for corrections in range(cfg.max_correct + 1):
                            accept_policy = (
                                (~trusted_policy)
                                & (prob_stack[corrections] >= threshold)
                                & (margin_stack[corrections] >= policy["min_margin"])
                                & (max_prob_stack[corrections] >= policy["min_prob"])
                                & policy_agreement_stack[corrections]
                            )
                            chosen_policy_fb = torch.where(
                                accept_policy,
                                torch.full_like(chosen_policy_fb, corrections),
                                chosen_policy_fb,
                            )
                            pred_policy_fb = torch.where(accept_policy, pred_stack[corrections], pred_policy_fb)
                            trusted_policy = trusted_policy | accept_policy
                        fb_policy_blocks = torch.where(
                            trusted_policy,
                            1.0 + chosen_policy_fb.float(),
                            torch.full_like(chosen_policy.float(), float(cfg.loop_steps)),
                        )
                        policy_sweep[policy_name][suffix]["fallback_acc"] += (
                            pred_policy_fb == target
                        ).float().mean().item()
                        policy_sweep[policy_name][suffix]["fallback_avg_correct_loops"] += (
                            fb_policy_blocks - 1.0
                        ).mean().item()
                        policy_sweep[policy_name][suffix]["fallback_full_loop_rate"] += (
                            ~trusted_policy
                        ).float().mean().item()

                for threshold, suffix in thresholds:
                    chosen = torch.full_like(target, cfg.max_correct)
                    open_mask = torch.ones_like(target, dtype=torch.bool)
                    for corrections in range(cfg.max_correct + 1):
                        accept = (
                            open_mask
                            & (prob_stack[corrections] >= threshold)
                            & (margin_stack[corrections] >= cfg.accept_min_margin)
                            & (max_prob_stack[corrections] >= cfg.accept_min_prob)
                            & agreement_stack[corrections]
                        )
                        chosen = torch.where(accept, torch.full_like(chosen, corrections), chosen)
                        open_mask = open_mask & ~accept
                    pred_adapt = pred_stack.gather(0, chosen.unsqueeze(0)).squeeze(0)
                    correct_loops = chosen.float()
                    metrics[f"adaptive_{suffix}_acc"] += (pred_adapt == target).float().mean().item()
                    metrics[f"adaptive_{suffix}_avg_correct_loops"] += correct_loops.mean().item()
                    metrics[f"adaptive_{suffix}_accept_c0_rate"] += (chosen == 0).float().mean().item()

                    trusted = torch.zeros_like(target, dtype=torch.bool)
                    chosen_fb = torch.full_like(target, cfg.loop_steps)
                    pred_fb = full_pred
                    for corrections in range(cfg.max_correct + 1):
                        accept = (
                            (~trusted)
                            & (prob_stack[corrections] >= threshold)
                            & (margin_stack[corrections] >= cfg.accept_min_margin)
                            & (max_prob_stack[corrections] >= cfg.accept_min_prob)
                            & agreement_stack[corrections]
                        )
                        chosen_fb = torch.where(accept, torch.full_like(chosen_fb, corrections), chosen_fb)
                        pred_fb = torch.where(accept, pred_stack[corrections], pred_fb)
                        trusted = trusted | accept
                    fb_loop_blocks = torch.where(
                        trusted,
                        1.0 + chosen_fb.float(),
                        torch.full_like(correct_loops, float(cfg.loop_steps)),
                    )
                    fb_correct_loops = fb_loop_blocks - 1.0
                    metrics[f"fallback_{suffix}_acc"] += (pred_fb == target).float().mean().item()
                    metrics[f"fallback_{suffix}_avg_correct_loops"] += fb_correct_loops.mean().item()
                    metrics[f"fallback_{suffix}_full_loop_rate"] += (~trusted).float().mean().item()
                    if suffix == "080":
                        for hop in range(1, cfg.max_hops + 1):
                            mask = hops == hop
                            if mask.any():
                                hop_compute_080[str(hop)].append(correct_loops[mask].float().mean().item())
                                hop_acc_080[str(hop)].append((pred_adapt[mask] == target[mask]).float().mean().item())
                                hop_fb_compute_080[str(hop)].append(fb_loop_blocks[mask].float().mean().item())
                                hop_fb_acc_080[str(hop)].append((pred_fb[mask] == target[mask]).float().mean().item())
                        for task_id, name in enumerate(task_names):
                            mask = task_ids == task_id
                            if mask.any():
                                task_acc_080[name].append((pred_adapt[mask] == target[mask]).float().mean().item())
                                task_fb_acc_080[name].append((pred_fb[mask] == target[mask]).float().mean().item())
                                task_fb_blocks_080[name].append(fb_loop_blocks[mask].float().mean().item())
                                task_fb_full_rate_080[name].append((~trusted[mask]).float().mean().item())

        for k in list(metrics):
            metrics[k] /= cfg.eval_batches
        for policy_name in policy_sweep:
            for suffix in policy_sweep[policy_name]:
                policy_metrics = policy_sweep[policy_name][suffix]
                for key in list(policy_metrics):
                    policy_metrics[key] /= cfg.eval_batches
                policy_metrics["adaptive_loop_equiv_blocks"] = (
                    1.0 + policy_metrics["adaptive_avg_correct_loops"]
                )
                policy_metrics["adaptive_compute_savings_vs_full_pct"] = (
                    1.0 - policy_metrics["adaptive_loop_equiv_blocks"] / cfg.loop_steps
                ) * 100.0
                policy_metrics["fallback_loop_equiv_blocks"] = (
                    1.0 + policy_metrics["fallback_avg_correct_loops"]
                )
                policy_metrics["fallback_compute_savings_vs_full_pct"] = (
                    1.0 - policy_metrics["fallback_loop_equiv_blocks"] / cfg.loop_steps
                ) * 100.0

        by_hop = {}
        for hop in range(1, cfg.max_hops + 1):
            key = str(hop)
            by_hop[key] = {
                "adaptive_080_acc": sum(hop_acc_080[key]) / len(hop_acc_080[key]) if hop_acc_080[key] else None,
                "adaptive_080_avg_correct_loops": sum(hop_compute_080[key]) / len(hop_compute_080[key]) if hop_compute_080[key] else None,
                "fallback_080_acc": sum(hop_fb_acc_080[key]) / len(hop_fb_acc_080[key]) if hop_fb_acc_080[key] else None,
                "fallback_080_avg_block_equiv": sum(hop_fb_compute_080[key]) / len(hop_fb_compute_080[key]) if hop_fb_compute_080[key] else None,
            }
        by_task = {}
        for name in task_names:
            by_task[name] = {
                "adaptive_080_acc": sum(task_acc_080[name]) / len(task_acc_080[name]) if task_acc_080[name] else None,
                "fallback_080_acc": sum(task_fb_acc_080[name]) / len(task_fb_acc_080[name]) if task_fb_acc_080[name] else None,
                "fallback_080_avg_block_equiv": sum(task_fb_blocks_080[name]) / len(task_fb_blocks_080[name]) if task_fb_blocks_080[name] else None,
                "fallback_080_full_loop_rate": sum(task_fb_full_rate_080[name]) / len(task_fb_full_rate_080[name]) if task_fb_full_rate_080[name] else None,
            }

        # One jump block is roughly one recurrent block worth of compute.
        metrics["full_loop_blocks"] = float(cfg.loop_steps)
        for _, suffix in thresholds:
            metrics[f"adaptive_{suffix}_loop_equiv_blocks"] = (
                1.0 + metrics[f"adaptive_{suffix}_avg_correct_loops"]
            )
            metrics[f"adaptive_{suffix}_compute_savings_vs_full_pct"] = (
                1.0 - metrics[f"adaptive_{suffix}_loop_equiv_blocks"] / cfg.loop_steps
            ) * 100.0
            metrics[f"fallback_{suffix}_loop_equiv_blocks"] = (
                1.0 + metrics[f"fallback_{suffix}_avg_correct_loops"]
            )
            metrics[f"fallback_{suffix}_compute_savings_vs_full_pct"] = (
                1.0 - metrics[f"fallback_{suffix}_loop_equiv_blocks"] / cfg.loop_steps
            ) * 100.0
        jumprec.train()
        return {"overall": metrics, "by_hop": by_hop, "by_task": by_task, "policy_sweep": policy_sweep}

    jump_eval = eval_jumprec()
    print(f"[jumprec eval] {json.dumps(jump_eval, indent=2)}")

    direct_eval = None
    n_direct_trainable = 0
    direct_model = None
    if cfg.run_direct_baseline:
        direct_model = DirectControlModel(teacher).to(device)
        n_direct_trainable = sum(p.numel() for p in direct_model.parameters() if p.requires_grad)
        print(f"[direct mixed] trainable params={n_direct_trainable/1e6:.3f}M")
        opt_d = torch.optim.AdamW([p for p in direct_model.parameters() if p.requires_grad], lr=cfg.lr_jump)
        t2 = time.time()
        print("[direct mixed] training")
        for step in range(1, cfg.jump_steps + 1):
            ids, target, _, _, _ = make_batch(cfg.batch_size, hop_limit=train_hop_limit)
            with torch.no_grad():
                state0 = teacher.encode(ids)
            logits = direct_model.forward_from_state(state0.detach())
            loss = F.cross_entropy(logits, target)
            opt_d.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(direct_model.parameters(), 1.0)
            opt_d.step()
            if step % cfg.log_every == 0 or step == cfg.jump_steps:
                print(
                    f"  direct mixed step {step:5d}/{cfg.jump_steps} loss {loss.item():.4f} "
                    f"elapsed {time.time()-t2:.1f}s",
                    flush=True,
                )

        direct_model.eval()
        acc = 0.0
        by_hop_acc: Dict[str, List[float]] = {str(i): [] for i in range(1, cfg.max_hops + 1)}
        by_task_acc: Dict[str, List[float]] = {name: [] for name in task_names}
        with torch.no_grad():
            for _ in range(cfg.eval_batches):
                ids, target, _, hops, task_ids = make_batch(cfg.batch_size)
                state0 = teacher.encode(ids)
                logits = direct_model.forward_from_state(state0)
                pred = logits.argmax(dim=-1)
                acc += (pred == target).float().mean().item()
                for k, v in grouped_accuracy(pred, target, hops).items():
                    by_hop_acc[k].append(v)
                for k, v in grouped_task_accuracy(pred, target, task_ids).items():
                    by_task_acc[k].append(v)
        direct_eval = {
            "direct_acc": acc / cfg.eval_batches,
            "direct_block_equiv": float(cfg.direct_blocks),
            "direct_compute_savings_vs_full_pct": (1.0 - cfg.direct_blocks / cfg.loop_steps) * 100.0,
            "direct_by_hop": {
                k: (sum(v) / len(v) if v else None)
                for k, v in by_hop_acc.items()
            },
            "direct_by_task": {
                k: (sum(v) / len(v) if v else None)
                for k, v in by_task_acc.items()
            },
        }
        print(f"[direct mixed eval] {json.dumps(direct_eval, indent=2)}")

    def benchmark_runtime(direct_for_timing=None) -> Dict[str, float]:
        if cfg.timing_batches <= 0:
            return {}
        teacher.eval()
        jumprec.eval()
        if direct_for_timing is not None:
            direct_for_timing.eval()

        def sync() -> None:
            if device.type == "cuda":
                torch.cuda.synchronize()

        def time_batches(fn) -> float:
            with torch.no_grad():
                for _ in range(3):
                    ids, _, _, _, _ = make_batch(cfg.batch_size)
                    fn(ids)
                sync()
                start = time.perf_counter()
                for _ in range(cfg.timing_batches):
                    ids, _, _, _, _ = make_batch(cfg.batch_size)
                    fn(ids)
                sync()
                elapsed = time.perf_counter() - start
            return 1000.0 * elapsed / cfg.timing_batches

        def teacher_full(ids):
            state0 = teacher.encode(ids)
            return teacher.classify(teacher.run_steps_from(state0, 0, cfg.loop_steps))

        def jumprec_all(ids):
            state0 = teacher.encode(ids)
            return jumprec.forward_from_state(state0)

        def jumprec_budget(ids, corrections: int):
            state0 = teacher.encode(ids)
            landing = jumprec.jump(state0, corrections)
            final = teacher.run_steps_from(landing, cfg.loop_steps - corrections, corrections)
            return teacher.classify(final)

        timings = {
            "batch_size": float(cfg.batch_size),
            "timing_batches": float(cfg.timing_batches),
            "teacher_full_ms_per_batch": time_batches(teacher_full),
            "jumprec_all_budgets_ms_per_batch": time_batches(jumprec_all),
            "jumprec_c0_ms_per_batch": time_batches(lambda ids: jumprec_budget(ids, 0)),
            f"jumprec_c{cfg.max_correct}_ms_per_batch": time_batches(
                lambda ids: jumprec_budget(ids, cfg.max_correct)
            ),
        }
        if direct_for_timing is not None:
            timings["direct_ms_per_batch"] = time_batches(
                lambda ids: direct_for_timing.forward_from_state(teacher.encode(ids))
            )
        return timings

    timing_eval = benchmark_runtime(direct_model)
    if timing_eval:
        print(f"[timing eval] {json.dumps(timing_eval, indent=2)}")

    summary = {
        "config": asdict(cfg),
        "teacher_params": n_teacher,
        "jumprec_trainable_params": n_jump_trainable,
        "direct_trainable_params": n_direct_trainable,
        "teacher_eval": teacher_eval,
        "teacher_gate": {
            "passed": not quality_failures,
            "failures": quality_failures,
            "extra_steps_used": teacher_extra_used,
        },
        "jumprec_eval": jump_eval,
        "direct_eval": direct_eval,
        "timing_eval": timing_eval,
    }
    return summary


@app.function(
    gpu=GPU_TYPE,
    timeout=72000,
    image=image,
    volumes={"/results": out_vol},
)
def run_remote(mode: str = "quick", seed: int | None = None) -> Dict[str, object]:
    import json
    import torch

    cfg = config_for_mode(mode)
    if seed is not None:
        cfg.seed = seed
    summary = run_experiment(cfg, "cuda")
    seed_suffix = f"_seed{cfg.seed}"
    torch.save(summary, f"/results/jumprec_v0_{mode}{seed_suffix}_summary.pt")
    with open(f"/results/jumprec_v0_{mode}{seed_suffix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    out_vol.commit()
    return summary


@app.local_entrypoint()
def main(mode: str = "quick", seed: int | None = None):
    summary = run_remote.remote(mode, seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="dry",
        choices=[
            "dry",
            "dry_mixed",
            "dry_round",
            "smoke",
            "probe",
            "quick",
            "quick_no_adapter",
            "quick_c6",
            "quick_c6_no_adapter",
            "quick_c6_no_hidden",
            "quick_direct3",
            "quick_ood_hops",
            "quick_mix",
            "quick_mix_strict",
            "quick_mix_direct3",
            "quick_mix_round",
            "full",
        ],
    )
    parser.add_argument("--local", action="store_true", help="run in this Python process instead of Modal")
    parser.add_argument("--allow-slow-local", action="store_true", help="permit local modes heavier than dry")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if not args.local:
        raise SystemExit("Use `modal run run_jumprec_v0.py --mode quick` for Modal, or add `--local`.")
    if args.mode not in {"dry", "dry_mixed", "dry_round"} and not args.allow_slow_local:
        raise SystemExit("Local CPU guard: use --mode dry, or add --allow-slow-local if you really mean it.")
    cfg = config_for_mode(args.mode)
    if args.seed is not None:
        cfg.seed = args.seed
    result = run_experiment(cfg, "cpu")
    print(json.dumps(result, indent=2))
