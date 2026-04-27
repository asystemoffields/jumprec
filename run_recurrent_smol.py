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
import os
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
    timing_batch_size: int = 0
    timing_batch_sizes: str = ""
    save_checkpoints: bool = False
    load_checkpoints: bool = False
    checkpoint_tag: str = ""
    load_checkpoint_tag: str = ""
    load_jumprec_state: bool = True
    resume_teacher_training: bool = False
    teacher_val_every: int = 0
    teacher_val_batches: int = 0
    teacher_gate_min_full: float = 0.0
    teacher_gate_min_worst_hop: float = 0.0
    reinject_init: float = 0.20
    use_reinject: bool = True
    mixed_tasks: bool = False
    curriculum_hops: bool = False
    hard_hop_fraction: float = 0.0
    hop_sample_weights: str = ""
    hard_hop_loss_weight: float = 1.0
    hop_loss_weights: str = ""
    final_loop_loss_weight: float = 2.0
    jump_steps: int = 0
    max_correct: int = 3
    jump_layers: int = 1
    jump_lr: float = 4e-4
    distill_loss_weight: float = 0.2
    verifier_loss_weight: float = 0.2
    use_temp_adapter: bool = True
    strict_need_agreement: bool = True
    adapter_rank: int = 8
    direct_steps: int = 0
    direct_layers: int = 3
    direct_lr: float = 4e-4
    audit_prompt_variants: str = ""
    router_val_batches: int = 0
    router_val_max_drop: float = 0.0025
    router_threshold_candidates: str = "0.50,0.60,0.70,0.80,0.85,0.90,0.95,0.97,0.99"

    @property
    def loop_steps(self) -> int:
        return self.max_hops + self.preserve_steps


def config_for_mode(mode: str) -> Config:
    cfg = Config(mode=mode)
    if mode in (
        "dry",
        "dry_sweep",
        "dry_sweep_reuse",
        "dry_hardhop",
        "dry_strathop",
        "dry_strathop_gate",
        "dry_strathop_polish",
        "dry_strathop_polish2",
        "dry_strathop_eval",
        "dry_strathop_polish2_eval",
        "dry_strathop_audit",
        "dry_strathop_polish2_audit",
        "dry_strathop_polish2_verifier_audit",
    ):
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
        cfg.direct_steps = 4
        cfg.direct_layers = 2
        if mode == "dry_hardhop":
            cfg.n_nodes = 8
            cfg.max_hops = 4
            cfg.max_correct = 3
            cfg.hard_hop_fraction = 0.70
            cfg.hard_hop_loss_weight = 2.5
            cfg.final_loop_loss_weight = 4.0
            cfg.timing_batch_sizes = "1,2"
        elif mode == "dry_strathop":
            cfg.n_nodes = 8
            cfg.max_hops = 4
            cfg.max_correct = 3
            cfg.hop_sample_weights = "0.10,0.20,0.35,0.35"
            cfg.hop_loss_weights = "1.0,1.2,2.0,2.0"
            cfg.final_loop_loss_weight = 4.0
            cfg.timing_batch_sizes = "1,2"
        elif mode == "dry_strathop_gate":
            cfg.n_nodes = 8
            cfg.max_hops = 4
            cfg.max_correct = 3
            cfg.hop_sample_weights = "0.10,0.20,0.35,0.35"
            cfg.hop_loss_weights = "1.0,1.2,2.0,2.0"
            cfg.final_loop_loss_weight = 4.0
            cfg.teacher_val_every = 2
            cfg.teacher_val_batches = 1
            cfg.teacher_gate_min_full = 0.5
            cfg.teacher_gate_min_worst_hop = 0.25
            cfg.save_checkpoints = True
            cfg.checkpoint_tag = "dry_strathop_gate_seed{seed}"
            cfg.timing_batch_sizes = "1,2"
        elif mode == "dry_strathop_polish":
            cfg.n_nodes = 8
            cfg.max_hops = 4
            cfg.max_correct = 3
            cfg.final_steps = 0
            cfg.recurrent_steps = 4
            cfg.jump_steps = 0
            cfg.direct_steps = 0
            cfg.hard_hop_fraction = 0.70
            cfg.hard_hop_loss_weight = 2.5
            cfg.final_loop_loss_weight = 4.0
            cfg.teacher_val_every = 2
            cfg.teacher_val_batches = 1
            cfg.teacher_gate_min_full = 0.5
            cfg.teacher_gate_min_worst_hop = 0.25
            cfg.load_checkpoints = True
            cfg.resume_teacher_training = True
            cfg.load_checkpoint_tag = "dry_strathop_gate_seed{seed}"
            cfg.save_checkpoints = True
            cfg.checkpoint_tag = "dry_strathop_polish_seed{seed}"
            cfg.timing_batch_sizes = "1,2"
        elif mode == "dry_strathop_polish2":
            cfg.n_nodes = 8
            cfg.max_hops = 4
            cfg.max_correct = 3
            cfg.final_steps = 0
            cfg.recurrent_steps = 4
            cfg.jump_steps = 0
            cfg.direct_steps = 0
            cfg.lr_blocks = 2e-5
            cfg.lr_head = 1.5e-4
            cfg.hard_hop_fraction = 0.70
            cfg.hard_hop_loss_weight = 2.5
            cfg.final_loop_loss_weight = 8.0
            cfg.teacher_val_every = 2
            cfg.teacher_val_batches = 1
            cfg.teacher_gate_min_full = 0.5
            cfg.teacher_gate_min_worst_hop = 0.25
            cfg.load_checkpoints = True
            cfg.resume_teacher_training = True
            cfg.load_checkpoint_tag = "dry_strathop_polish_seed{seed}"
            cfg.save_checkpoints = True
            cfg.checkpoint_tag = "dry_strathop_polish2_seed{seed}"
            cfg.timing_batch_sizes = "1,2"
        elif mode == "dry_strathop_eval":
            cfg.n_nodes = 8
            cfg.max_hops = 4
            cfg.max_correct = 3
            cfg.final_steps = 0
            cfg.recurrent_steps = 0
            cfg.jump_steps = 0
            cfg.direct_steps = 0
            cfg.hop_sample_weights = "0.10,0.20,0.35,0.35"
            cfg.hop_loss_weights = "1.0,1.2,2.0,2.0"
            cfg.final_loop_loss_weight = 4.0
            cfg.load_checkpoints = True
            cfg.checkpoint_tag = "dry_strathop_gate_seed{seed}"
            cfg.timing_batch_sizes = "1,2"
        elif mode == "dry_strathop_polish2_eval":
            cfg.n_nodes = 8
            cfg.max_hops = 4
            cfg.max_correct = 3
            cfg.final_steps = 0
            cfg.recurrent_steps = 0
            cfg.jump_steps = 0
            cfg.direct_steps = 0
            cfg.hard_hop_fraction = 0.70
            cfg.hard_hop_loss_weight = 2.5
            cfg.final_loop_loss_weight = 8.0
            cfg.load_checkpoints = True
            cfg.checkpoint_tag = "dry_strathop_polish2_seed{seed}"
            cfg.timing_batch_sizes = "1,2"
        elif mode == "dry_strathop_audit":
            cfg.n_nodes = 8
            cfg.max_hops = 4
            cfg.max_correct = 3
            cfg.final_steps = 0
            cfg.recurrent_steps = 0
            cfg.jump_steps = 0
            cfg.direct_steps = 0
            cfg.hop_sample_weights = "0.10,0.20,0.35,0.35"
            cfg.hop_loss_weights = "1.0,1.2,2.0,2.0"
            cfg.final_loop_loss_weight = 4.0
            cfg.load_checkpoints = True
            cfg.checkpoint_tag = "dry_strathop_gate_seed{seed}"
            cfg.audit_prompt_variants = "normal,relabel,map_scramble,hop_random"
            cfg.timing_batch_sizes = "1,2"
        elif mode == "dry_strathop_polish2_audit":
            cfg.n_nodes = 8
            cfg.max_hops = 4
            cfg.max_correct = 3
            cfg.final_steps = 0
            cfg.recurrent_steps = 0
            cfg.jump_steps = 0
            cfg.direct_steps = 0
            cfg.hard_hop_fraction = 0.70
            cfg.hard_hop_loss_weight = 2.5
            cfg.final_loop_loss_weight = 8.0
            cfg.load_checkpoints = True
            cfg.checkpoint_tag = "dry_strathop_polish2_seed{seed}"
            cfg.audit_prompt_variants = "normal,relabel,map_scramble,hop_random"
            cfg.timing_batch_sizes = "1,2"
        elif mode == "dry_strathop_polish2_verifier_audit":
            cfg.n_nodes = 8
            cfg.max_hops = 4
            cfg.max_correct = 3
            cfg.final_steps = 0
            cfg.recurrent_steps = 0
            cfg.jump_steps = 4
            cfg.direct_steps = 0
            cfg.hard_hop_fraction = 0.70
            cfg.hard_hop_loss_weight = 2.5
            cfg.final_loop_loss_weight = 8.0
            cfg.load_checkpoints = True
            cfg.checkpoint_tag = "dry_strathop_polish2_seed{seed}"
            cfg.audit_prompt_variants = "normal,relabel,map_scramble,hop_random"
            cfg.router_val_batches = 1
            cfg.timing_batch_sizes = "1,2"
        elif mode == "dry_sweep":
            cfg.timing_batch_sizes = "1,2,4"
            cfg.save_checkpoints = True
            cfg.checkpoint_tag = "dry_router_seed{seed}"
        elif mode == "dry_sweep_reuse":
            cfg.final_steps = 0
            cfg.recurrent_steps = 0
            cfg.jump_steps = 0
            cfg.direct_steps = 0
            cfg.timing_batch_sizes = "1,2,4"
            cfg.load_checkpoints = True
            cfg.checkpoint_tag = "dry_router_seed{seed}"
    elif mode == "retrofit_probe":
        pass
    elif mode == "jumprec_probe":
        cfg.jump_steps = 3500
        cfg.max_correct = 3
        cfg.jump_layers = 1
        cfg.eval_batches = 64
        cfg.log_every = 250
    elif mode == "jumprec_no_adapter":
        cfg.jump_steps = 3500
        cfg.use_temp_adapter = False
        cfg.eval_batches = 64
        cfg.log_every = 250
    elif mode == "jumprec_no_agree":
        cfg.jump_steps = 3500
        cfg.strict_need_agreement = False
        cfg.eval_batches = 64
        cfg.log_every = 250
    elif mode == "direct_probe":
        cfg.direct_steps = 3500
        cfg.direct_layers = 3
        cfg.eval_batches = 64
        cfg.log_every = 250
    elif mode == "retrofit_no_reinject":
        cfg.use_reinject = False
        cfg.reinject_init = 1e-4
        cfg.eval_batches = 64
    elif mode == "retrofit_core1":
        cfg.core_layers = 1
        cfg.coda_start = 28
        cfg.coda_layers = 2
        cfg.eval_batches = 64
    elif mode == "retrofit_core3":
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.eval_batches = 64
    elif mode == "mixed_probe":
        cfg.mixed_tasks = True
        cfg.final_steps = 3500
        cfg.recurrent_steps = 5000
        cfg.eval_batches = 64
        cfg.log_every = 250
    elif mode == "mixed_jumprec_direct":
        cfg.mixed_tasks = True
        cfg.final_steps = 3500
        cfg.recurrent_steps = 5000
        cfg.jump_steps = 3500
        cfg.direct_steps = 3500
        cfg.direct_layers = 3
        cfg.eval_batches = 64
        cfg.log_every = 250
    elif mode == "mixed_core3_jumprec_direct":
        cfg.mixed_tasks = True
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 5000
        cfg.recurrent_steps = 7000
        cfg.jump_steps = 4500
        cfg.direct_steps = 4500
        cfg.direct_layers = 3
        cfg.eval_batches = 64
        cfg.log_every = 500
    elif mode == "mixed_core3_router_no_agree":
        cfg.mixed_tasks = True
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 5000
        cfg.recurrent_steps = 7000
        cfg.jump_steps = 4500
        cfg.direct_steps = 0
        cfg.strict_need_agreement = False
        cfg.eval_batches = 64
        cfg.log_every = 500
    elif mode == "mixed_core3_router_no_agree_b1":
        cfg.mixed_tasks = True
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 5000
        cfg.recurrent_steps = 7000
        cfg.jump_steps = 4500
        cfg.direct_steps = 0
        cfg.strict_need_agreement = False
        cfg.timing_batch_size = 1
        cfg.timing_batches = 64
        cfg.eval_batches = 64
        cfg.log_every = 500
    elif mode == "mixed_core3_router_bsize_sweep":
        cfg.mixed_tasks = True
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 5000
        cfg.recurrent_steps = 7000
        cfg.jump_steps = 4500
        cfg.direct_steps = 0
        cfg.strict_need_agreement = False
        cfg.timing_batches = 64
        cfg.timing_batch_sizes = "1,2,4,8,16,32,64"
        cfg.save_checkpoints = True
        cfg.checkpoint_tag = "mixed_core3_router_seed{seed}"
        cfg.eval_batches = 64
        cfg.log_every = 500
    elif mode == "mixed_core3_router_bsize_sweep_reuse":
        cfg.mixed_tasks = True
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 0
        cfg.recurrent_steps = 0
        cfg.jump_steps = 0
        cfg.direct_steps = 0
        cfg.strict_need_agreement = False
        cfg.timing_batches = 64
        cfg.timing_batch_sizes = "1,2,4,8,16,32,64"
        cfg.load_checkpoints = True
        cfg.checkpoint_tag = "mixed_core3_router_seed{seed}"
        cfg.eval_batches = 64
        cfg.log_every = 500
    elif mode == "mixed_core3_router_verifier1":
        cfg.mixed_tasks = True
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 5000
        cfg.recurrent_steps = 7000
        cfg.jump_steps = 4500
        cfg.direct_steps = 0
        cfg.strict_need_agreement = False
        cfg.verifier_loss_weight = 1.0
        cfg.eval_batches = 64
        cfg.log_every = 500
    elif mode == "retrofit_8n4h":
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.final_steps = 3500
        cfg.recurrent_steps = 5000
        cfg.eval_batches = 64
        cfg.log_every = 250
    elif mode == "retrofit_8n4h_curriculum":
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.curriculum_hops = True
        cfg.final_steps = 4500
        cfg.recurrent_steps = 6500
        cfg.eval_batches = 64
        cfg.log_every = 500
    elif mode == "jumprec_8n4h_direct":
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.final_steps = 3500
        cfg.recurrent_steps = 5000
        cfg.jump_steps = 4500
        cfg.direct_steps = 4500
        cfg.direct_layers = 3
        cfg.eval_batches = 64
        cfg.log_every = 500
    elif mode == "core3_8n4h_jumprec_direct":
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 5000
        cfg.recurrent_steps = 7000
        cfg.jump_steps = 4500
        cfg.direct_steps = 4500
        cfg.direct_layers = 3
        cfg.eval_batches = 64
        cfg.log_every = 500
    elif mode == "core3_8n4h_hardhop_teacher":
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 6500
        cfg.recurrent_steps = 10000
        cfg.hard_hop_fraction = 0.70
        cfg.hard_hop_loss_weight = 2.5
        cfg.final_loop_loss_weight = 4.0
        cfg.jump_steps = 0
        cfg.direct_steps = 4500
        cfg.direct_layers = 3
        cfg.save_checkpoints = True
        cfg.checkpoint_tag = "core3_8n4h_hardhop_seed{seed}"
        cfg.eval_batches = 96
        cfg.log_every = 500
    elif mode == "core3_8n4h_hardhop_jumprec":
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 0
        cfg.recurrent_steps = 0
        cfg.hard_hop_fraction = 0.70
        cfg.hard_hop_loss_weight = 2.5
        cfg.final_loop_loss_weight = 4.0
        cfg.jump_steps = 4500
        cfg.direct_steps = 4500
        cfg.direct_layers = 3
        cfg.strict_need_agreement = False
        cfg.load_checkpoints = True
        cfg.save_checkpoints = True
        cfg.checkpoint_tag = "core3_8n4h_hardhop_seed{seed}"
        cfg.timing_batches = 64
        cfg.timing_batch_sizes = "1,2,4,8,16,32,64"
        cfg.eval_batches = 96
        cfg.log_every = 500
    elif mode == "core3_8n4h_strathop_teacher":
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 6500
        cfg.recurrent_steps = 10000
        cfg.hop_sample_weights = "0.10,0.20,0.35,0.35"
        cfg.hop_loss_weights = "1.0,1.2,2.0,2.0"
        cfg.final_loop_loss_weight = 4.0
        cfg.jump_steps = 0
        cfg.direct_steps = 4500
        cfg.direct_layers = 3
        cfg.save_checkpoints = True
        cfg.checkpoint_tag = "core3_8n4h_strathop_seed{seed}"
        cfg.eval_batches = 96
        cfg.log_every = 500
    elif mode == "core3_8n4h_strathop_gate_teacher":
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 6500
        cfg.recurrent_steps = 10000
        cfg.hop_sample_weights = "0.10,0.20,0.35,0.35"
        cfg.hop_loss_weights = "1.0,1.2,2.0,2.0"
        cfg.final_loop_loss_weight = 4.0
        cfg.jump_steps = 0
        cfg.direct_steps = 4500
        cfg.direct_layers = 3
        cfg.save_checkpoints = True
        cfg.checkpoint_tag = "core3_8n4h_strathop_gate_seed{seed}"
        cfg.teacher_val_every = 500
        cfg.teacher_val_batches = 16
        cfg.teacher_gate_min_full = 0.995
        cfg.teacher_gate_min_worst_hop = 0.98
        cfg.eval_batches = 96
        cfg.log_every = 500
    elif mode == "core3_8n4h_strathop_polish_teacher":
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 0
        cfg.recurrent_steps = 3000
        cfg.hard_hop_fraction = 0.70
        cfg.hard_hop_loss_weight = 2.5
        cfg.final_loop_loss_weight = 4.0
        cfg.jump_steps = 0
        cfg.direct_steps = 0
        cfg.load_checkpoints = True
        cfg.resume_teacher_training = True
        cfg.load_checkpoint_tag = "core3_8n4h_strathop_gate_seed{seed}"
        cfg.save_checkpoints = True
        cfg.checkpoint_tag = "core3_8n4h_strathop_polish_seed{seed}"
        cfg.teacher_val_every = 250
        cfg.teacher_val_batches = 16
        cfg.teacher_gate_min_full = 0.995
        cfg.teacher_gate_min_worst_hop = 0.98
        cfg.eval_batches = 96
        cfg.log_every = 250
    elif mode == "core3_8n4h_strathop_polish2_teacher":
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 0
        cfg.recurrent_steps = 4000
        cfg.lr_blocks = 2e-5
        cfg.lr_head = 1.5e-4
        cfg.hard_hop_fraction = 0.70
        cfg.hard_hop_loss_weight = 2.5
        cfg.final_loop_loss_weight = 8.0
        cfg.jump_steps = 0
        cfg.direct_steps = 0
        cfg.load_checkpoints = True
        cfg.resume_teacher_training = True
        cfg.load_checkpoint_tag = "core3_8n4h_strathop_polish_seed{seed}"
        cfg.save_checkpoints = True
        cfg.checkpoint_tag = "core3_8n4h_strathop_polish2_seed{seed}"
        cfg.teacher_val_every = 250
        cfg.teacher_val_batches = 16
        cfg.teacher_gate_min_full = 0.995
        cfg.teacher_gate_min_worst_hop = 0.98
        cfg.eval_batches = 96
        cfg.log_every = 250
    elif mode == "core3_8n4h_strathop_eval_teacher":
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 0
        cfg.recurrent_steps = 0
        cfg.hop_sample_weights = "0.10,0.20,0.35,0.35"
        cfg.hop_loss_weights = "1.0,1.2,2.0,2.0"
        cfg.final_loop_loss_weight = 4.0
        cfg.jump_steps = 0
        cfg.direct_steps = 0
        cfg.load_checkpoints = True
        cfg.checkpoint_tag = "core3_8n4h_strathop_seed{seed}"
        cfg.eval_batches = 256
        cfg.timing_batches = 16
        cfg.log_every = 500
    elif mode == "core3_8n4h_strathop_polish2_eval_teacher":
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 0
        cfg.recurrent_steps = 0
        cfg.hard_hop_fraction = 0.70
        cfg.hard_hop_loss_weight = 2.5
        cfg.final_loop_loss_weight = 8.0
        cfg.jump_steps = 0
        cfg.direct_steps = 0
        cfg.load_checkpoints = True
        cfg.checkpoint_tag = "core3_8n4h_strathop_polish2_seed{seed}"
        cfg.eval_batches = 256
        cfg.timing_batches = 16
        cfg.log_every = 500
    elif mode == "core3_8n4h_strathop_audit_teacher":
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 0
        cfg.recurrent_steps = 0
        cfg.hop_sample_weights = "0.10,0.20,0.35,0.35"
        cfg.hop_loss_weights = "1.0,1.2,2.0,2.0"
        cfg.final_loop_loss_weight = 4.0
        cfg.jump_steps = 0
        cfg.direct_steps = 0
        cfg.load_checkpoints = True
        cfg.checkpoint_tag = "core3_8n4h_strathop_seed{seed}"
        cfg.audit_prompt_variants = "normal,relabel,map_scramble,hop_random"
        cfg.eval_batches = 128
        cfg.timing_batches = 8
        cfg.log_every = 500
    elif mode == "core3_8n4h_strathop_polish2_audit_teacher":
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 0
        cfg.recurrent_steps = 0
        cfg.hard_hop_fraction = 0.70
        cfg.hard_hop_loss_weight = 2.5
        cfg.final_loop_loss_weight = 8.0
        cfg.jump_steps = 0
        cfg.direct_steps = 0
        cfg.load_checkpoints = True
        cfg.checkpoint_tag = "core3_8n4h_strathop_polish2_seed{seed}"
        cfg.audit_prompt_variants = "normal,relabel,map_scramble,hop_random"
        cfg.eval_batches = 128
        cfg.timing_batches = 8
        cfg.log_every = 500
    elif mode == "core3_8n4h_strathop_verifier_audit":
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 0
        cfg.recurrent_steps = 0
        cfg.hop_sample_weights = "0.10,0.20,0.35,0.35"
        cfg.hop_loss_weights = "1.0,1.2,2.0,2.0"
        cfg.final_loop_loss_weight = 4.0
        cfg.jump_steps = 0
        cfg.direct_steps = 0
        cfg.direct_layers = 3
        cfg.strict_need_agreement = False
        cfg.load_checkpoints = True
        cfg.checkpoint_tag = "core3_8n4h_strathop_seed{seed}"
        cfg.eval_batches = 128
        cfg.router_val_batches = 64
        cfg.timing_batches = 8
        cfg.log_every = 500
    elif mode == "core3_8n4h_strathop_polish2_verifier_audit":
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 0
        cfg.recurrent_steps = 0
        cfg.hard_hop_fraction = 0.70
        cfg.hard_hop_loss_weight = 2.5
        cfg.final_loop_loss_weight = 8.0
        cfg.jump_steps = 0
        cfg.direct_steps = 0
        cfg.direct_layers = 3
        cfg.strict_need_agreement = False
        cfg.load_checkpoints = True
        cfg.checkpoint_tag = "core3_8n4h_strathop_polish2_seed{seed}"
        cfg.eval_batches = 128
        cfg.router_val_batches = 64
        cfg.timing_batches = 8
        cfg.log_every = 500
    elif mode == "core3_8n4h_strathop_jumprec":
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 0
        cfg.recurrent_steps = 0
        cfg.hop_sample_weights = "0.10,0.20,0.35,0.35"
        cfg.hop_loss_weights = "1.0,1.2,2.0,2.0"
        cfg.final_loop_loss_weight = 4.0
        cfg.jump_steps = 4500
        cfg.direct_steps = 4500
        cfg.direct_layers = 3
        cfg.strict_need_agreement = False
        cfg.load_checkpoints = True
        cfg.save_checkpoints = True
        cfg.checkpoint_tag = "core3_8n4h_strathop_seed{seed}"
        cfg.timing_batches = 64
        cfg.timing_batch_sizes = "1,2,4,8,16,32,64"
        cfg.eval_batches = 96
        cfg.log_every = 500
    elif mode == "core3_8n4h_strathop_gate_jumprec":
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 0
        cfg.recurrent_steps = 0
        cfg.hop_sample_weights = "0.10,0.20,0.35,0.35"
        cfg.hop_loss_weights = "1.0,1.2,2.0,2.0"
        cfg.final_loop_loss_weight = 4.0
        cfg.jump_steps = 4500
        cfg.direct_steps = 4500
        cfg.direct_layers = 3
        cfg.strict_need_agreement = False
        cfg.load_checkpoints = True
        cfg.save_checkpoints = True
        cfg.checkpoint_tag = "core3_8n4h_strathop_gate_seed{seed}"
        cfg.timing_batches = 64
        cfg.timing_batch_sizes = "1,2,4,8,16,32,64"
        cfg.eval_batches = 96
        cfg.log_every = 500
    elif mode == "core3_8n4h_strathop_polish_jumprec":
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 0
        cfg.recurrent_steps = 0
        cfg.hard_hop_fraction = 0.70
        cfg.hard_hop_loss_weight = 2.5
        cfg.final_loop_loss_weight = 4.0
        cfg.jump_steps = 4500
        cfg.direct_steps = 4500
        cfg.direct_layers = 3
        cfg.strict_need_agreement = False
        cfg.load_checkpoints = True
        cfg.save_checkpoints = True
        cfg.checkpoint_tag = "core3_8n4h_strathop_polish_seed{seed}"
        cfg.timing_batches = 64
        cfg.timing_batch_sizes = "1,2,4,8,16,32,64"
        cfg.eval_batches = 96
        cfg.log_every = 500
    elif mode == "core3_8n4h_strathop_polish2_jumprec":
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 0
        cfg.recurrent_steps = 0
        cfg.hard_hop_fraction = 0.70
        cfg.hard_hop_loss_weight = 2.5
        cfg.final_loop_loss_weight = 8.0
        cfg.jump_steps = 4500
        cfg.direct_steps = 4500
        cfg.direct_layers = 3
        cfg.strict_need_agreement = False
        cfg.load_checkpoints = True
        cfg.save_checkpoints = True
        cfg.checkpoint_tag = "core3_8n4h_strathop_polish2_seed{seed}"
        cfg.timing_batches = 64
        cfg.timing_batch_sizes = "1,2,4,8,16,32,64"
        cfg.eval_batches = 96
        cfg.log_every = 500
    elif mode in (
        "core3_8n4h_strathop_ablate_no_adapter",
        "core3_8n4h_strathop_ablate_no_distill",
        "core3_8n4h_strathop_ablate_no_verifier",
    ):
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 0
        cfg.recurrent_steps = 0
        cfg.hop_sample_weights = "0.10,0.20,0.35,0.35"
        cfg.hop_loss_weights = "1.0,1.2,2.0,2.0"
        cfg.final_loop_loss_weight = 4.0
        cfg.jump_steps = 4500
        cfg.direct_steps = 0
        cfg.direct_layers = 3
        cfg.strict_need_agreement = False
        cfg.load_checkpoints = True
        cfg.load_jumprec_state = False
        cfg.load_checkpoint_tag = "core3_8n4h_strathop_seed{seed}"
        cfg.save_checkpoints = True
        cfg.checkpoint_tag = f"{mode}_seed{{seed}}"
        cfg.timing_batches = 16
        cfg.eval_batches = 96
        cfg.log_every = 500
        if mode == "core3_8n4h_strathop_ablate_no_adapter":
            cfg.use_temp_adapter = False
        elif mode == "core3_8n4h_strathop_ablate_no_distill":
            cfg.distill_loss_weight = 0.0
        elif mode == "core3_8n4h_strathop_ablate_no_verifier":
            cfg.verifier_loss_weight = 0.0
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

    checkpoint_tag = (cfg.checkpoint_tag or f"{cfg.mode}_seed{{seed}}").format(seed=cfg.seed, mode=cfg.mode)
    load_checkpoint_tag = (cfg.load_checkpoint_tag or cfg.checkpoint_tag or f"{cfg.mode}_seed{{seed}}").format(
        seed=cfg.seed,
        mode=cfg.mode,
    )
    checkpoint_root = "/results/checkpoints" if os.path.isdir("/results") else "checkpoints"
    checkpoint_path = os.path.join(checkpoint_root, f"{checkpoint_tag}.pt")
    load_checkpoint_path = os.path.join(checkpoint_root, f"{load_checkpoint_tag}.pt")

    def save_checkpoint(model_obj, jumprec_obj=None, extra=None):
        if not cfg.save_checkpoints:
            return
        os.makedirs(checkpoint_root, exist_ok=True)
        payload = {
            "config": asdict(cfg),
            "checkpoint_tag": checkpoint_tag,
            "model_state": model_obj.state_dict(),
            "jumprec_state": jumprec_obj.state_dict() if jumprec_obj is not None else None,
        }
        if extra is not None:
            payload["extra"] = extra
        torch.save(payload, checkpoint_path)
        print(f"[checkpoint] saved {checkpoint_path}", flush=True)

    letters = [chr(ord("A") + i) for i in range(cfg.n_nodes)]
    task_names = ["forward", "inverse", "alternate", "square"] if cfg.mixed_tasks else ["forward"]

    def parse_float_list(value: str) -> List[float]:
        if not value.strip():
            return []
        return [float(x.strip()) for x in value.split(",") if x.strip()]

    def parse_str_list(value: str) -> List[str]:
        return [x.strip() for x in value.split(",") if x.strip()]

    hop_sample_weights = parse_float_list(cfg.hop_sample_weights)
    hop_loss_weights = parse_float_list(cfg.hop_loss_weights)
    audit_prompt_variants = parse_str_list(cfg.audit_prompt_variants)
    if hop_sample_weights and len(hop_sample_weights) < cfg.max_hops:
        raise ValueError("hop_sample_weights must provide at least max_hops entries")
    if hop_loss_weights and len(hop_loss_weights) < cfg.max_hops:
        raise ValueError("hop_loss_weights must provide at least max_hops entries")
    allowed_audit_variants = {"normal", "relabel", "map_scramble", "hop_random", "task_random"}
    for variant in audit_prompt_variants:
        if variant not in allowed_audit_variants:
            raise ValueError(f"unknown audit prompt variant: {variant}")

    def make_examples(
        batch_size: int,
        active_max_hops: int | None = None,
        hard_hop_sampling: bool = True,
        audit_variant: str = "normal",
    ):
        if audit_variant not in allowed_audit_variants:
            raise ValueError(f"unknown audit prompt variant: {audit_variant}")
        texts = []
        targets_l = []
        step_targets_l = []
        hops_l = []
        tasks_l = []
        max_hops = active_max_hops or cfg.max_hops
        for _ in range(batch_size):
            perm = list(range(cfg.n_nodes))
            random.shuffle(perm)
            inv = [0 for _ in range(cfg.n_nodes)]
            for i, j in enumerate(perm):
                inv[j] = i
            start = random.randrange(cfg.n_nodes)
            display_perm = list(perm)
            if audit_variant == "map_scramble":
                display_perm = list(range(cfg.n_nodes))
                random.shuffle(display_perm)
                if display_perm == perm:
                    display_perm = display_perm[1:] + display_perm[:1]
            label_map = list(range(cfg.n_nodes))
            if audit_variant == "relabel":
                random.shuffle(label_map)
            if hard_hop_sampling and hop_sample_weights:
                weights = hop_sample_weights[:max_hops]
                total_weight = sum(weights)
                if total_weight <= 0:
                    raise ValueError("hop_sample_weights must have positive mass")
                r = random.random() * total_weight
                acc = 0.0
                hops = max_hops
                for i, weight in enumerate(weights, start=1):
                    acc += weight
                    if r <= acc:
                        hops = i
                        break
            elif hard_hop_sampling and cfg.hard_hop_fraction > 0.0 and random.random() < cfg.hard_hop_fraction:
                hops = max_hops
            else:
                hops = random.randint(1, max_hops)
            task_id = random.randrange(len(task_names))
            display_task_id = task_id
            if audit_variant == "task_random" and len(task_names) > 1:
                choices = [i for i in range(len(task_names)) if i != task_id]
                display_task_id = random.choice(choices)
            display_hops = hops
            if audit_variant == "hop_random" and max_hops > 1:
                choices = [i for i in range(1, max_hops + 1) if i != hops]
                display_hops = random.choice(choices)
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
                step_targets.append(label_map[cur])
            mapping_pairs = [
                (label_map[i], label_map[display_perm[i]])
                for i in range(cfg.n_nodes)
            ]
            mapping_pairs.sort(key=lambda pair: pair[0])
            mapping = " ".join(f"{letters[src]}->{letters[dst]}" for src, dst in mapping_pairs)
            task = task_names[display_task_id]
            text = f"Task: {task}. Map: {mapping}. Start: {letters[label_map[start]]}. Hops: {display_hops}. Answer:"
            after_answer = text.split("Answer:", 1)[1]
            if after_answer.strip():
                raise AssertionError("generated prompt leaked text after Answer:")
            texts.append(text)
            targets_l.append(step_targets[-1])
            step_targets_l.append(step_targets)
            hops_l.append(hops)
            tasks_l.append(task_id)
        target = torch.tensor(targets_l, dtype=torch.long, device=device)
        step_targets = torch.tensor(step_targets_l, dtype=torch.long, device=device)
        hops = torch.tensor(hops_l, dtype=torch.long, device=device)
        task_ids = torch.tensor(tasks_l, dtype=torch.long, device=device)
        return texts, target, step_targets, hops, task_ids

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
            reinject = gate * input_state if cfg.use_reinject else 0.0
            hidden = hidden + self.loop_emb[idx].view(1, 1, -1).to(hidden.dtype) + reinject
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

    def batch_encoded(
        batch_size: int,
        active_max_hops: int | None = None,
        hard_hop_sampling: bool = True,
        audit_variant: str = "normal",
    ):
        texts, target, step_targets, hops, task_ids = make_examples(
            batch_size,
            active_max_hops,
            hard_hop_sampling,
            audit_variant,
        )
        input_ids, attention_mask, lengths = encode_texts(texts)
        return input_ids, attention_mask, lengths, target, step_targets, hops, task_ids

    def accuracy(logits, target) -> float:
        return (logits.argmax(dim=-1) == target).float().mean().item()

    def eval_teacher_quality(num_batches: int) -> Dict[str, object]:
        if num_batches <= 0:
            raise ValueError("teacher_val_batches must be positive when teacher validation is enabled")
        was_training = model.training
        model.eval()
        py_random_state = random.getstate()
        total_correct = 0.0
        total_count = 0
        hop_correct = {hop: 0.0 for hop in range(1, cfg.max_hops + 1)}
        hop_count = {hop: 0 for hop in range(1, cfg.max_hops + 1)}
        try:
            with torch.no_grad():
                for _ in range(num_batches):
                    input_ids, attention_mask, lengths, target, _, hops, _ = batch_encoded(
                        cfg.batch_size,
                        hard_hop_sampling=False,
                    )
                    logits = model(input_ids, attention_mask, lengths, cfg.loop_steps)
                    correct = logits.argmax(dim=-1) == target
                    total_correct += correct.float().sum().item()
                    total_count += int(target.numel())
                    for hop in range(1, cfg.max_hops + 1):
                        mask = hops == hop
                        if mask.any():
                            hop_correct[hop] += correct[mask].float().sum().item()
                            hop_count[hop] += int(mask.sum().item())
        finally:
            random.setstate(py_random_state)
        if was_training:
            model.train()
        by_hop = {
            str(hop): (hop_correct[hop] / hop_count[hop] if hop_count[hop] else None)
            for hop in range(1, cfg.max_hops + 1)
        }
        observed_hops = [value for value in by_hop.values() if value is not None]
        return {
            "full_acc": total_correct / max(1, total_count),
            "by_hop": by_hop,
            "worst_hop_acc": min(observed_hops) if observed_hops else None,
            "batches": num_batches,
        }

    def example_weights(hops):
        if hop_loss_weights:
            weight_table = torch.tensor(
                hop_loss_weights[: cfg.max_hops],
                dtype=torch.float32,
                device=hops.device,
            )
            return weight_table.index_select(0, (hops - 1).clamp(0, cfg.max_hops - 1))
        if cfg.hard_hop_loss_weight <= 1.0:
            return torch.ones_like(hops, dtype=torch.float32)
        hard = (hops == cfg.max_hops).float()
        return 1.0 + (cfg.hard_hop_loss_weight - 1.0) * hard

    def weighted_ce(logits, target, hops):
        weights = example_weights(hops)
        losses = F.cross_entropy(logits, target, reduction="none")
        return (losses * weights).sum() / weights.sum().clamp_min(1e-6)

    def weighted_kl(logits, soft_teacher, hops):
        weights = example_weights(hops)
        losses = F.kl_div(F.log_softmax(logits, dim=-1), soft_teacher, reduction="none").sum(dim=-1)
        return (losses * weights).sum() / weights.sum().clamp_min(1e-6)

    def weighted_bce_with_logits(logits, target, hops):
        weights = example_weights(hops)
        losses = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return (losses * weights).sum() / weights.sum().clamp_min(1e-6)

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
    loaded_checkpoint = None
    if cfg.load_checkpoints:
        if not os.path.exists(load_checkpoint_path):
            raise FileNotFoundError(f"checkpoint not found: {load_checkpoint_path}")
        loaded_checkpoint = torch.load(load_checkpoint_path, map_location=device)
        model.load_state_dict(loaded_checkpoint["model_state"])
        print(f"[checkpoint] loaded teacher {load_checkpoint_path}", flush=True)

    def active_hops_for_step(step: int, total_steps: int) -> int | None:
        if not cfg.curriculum_hops:
            return None
        phase = max(1, total_steps // cfg.max_hops)
        return min(cfg.max_hops, 1 + (step - 1) // phase)

    best_teacher_state = None
    best_teacher_summary = None
    best_teacher_score = -1.0
    best_teacher_full = -1.0

    def maybe_validate_teacher(step: int):
        nonlocal best_teacher_state, best_teacher_summary, best_teacher_score, best_teacher_full
        if cfg.teacher_val_every <= 0:
            return
        if step % cfg.teacher_val_every != 0 and step != cfg.recurrent_steps:
            return
        val_batches = cfg.teacher_val_batches or max(1, min(cfg.eval_batches, 16))
        summary = eval_teacher_quality(val_batches)
        worst = float(summary["worst_hop_acc"] or 0.0)
        full = float(summary["full_acc"])
        passed = (
            full >= cfg.teacher_gate_min_full
            and worst >= cfg.teacher_gate_min_worst_hop
        )
        print(
            "[teacher val] "
            f"step {step:5d}/{cfg.recurrent_steps} "
            f"full {full*100:.2f}% worst_hop {worst*100:.2f}% "
            f"passed {passed} by_hop {json.dumps(summary['by_hop'])}",
            flush=True,
        )
        if worst > best_teacher_score or (math.isclose(worst, best_teacher_score) and full > best_teacher_full):
            best_teacher_score = worst
            best_teacher_full = full
            best_teacher_summary = {
                "step": step,
                "passed": passed,
                **summary,
            }
            best_teacher_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            save_checkpoint(
                model,
                extra={
                    "teacher_val_best": best_teacher_summary,
                    "teacher_val_restored": False,
                },
            )
            print(
                "[teacher val] "
                f"new best step {step} full {full*100:.2f}% worst_hop {worst*100:.2f}%",
                flush=True,
            )

    t0 = time.time()
    model.train()
    if cfg.load_checkpoints and not cfg.resume_teacher_training:
        print("[train] skipped teacher training from loaded checkpoint", flush=True)
    else:
        for step in range(1, cfg.final_steps + 1):
            active_hops = active_hops_for_step(step, cfg.final_steps)
            input_ids, attention_mask, lengths, target, _, hops, _ = batch_encoded(cfg.batch_size, active_hops)
            logits = model(input_ids, attention_mask, lengths, cfg.loop_steps)
            loss = weighted_ce(logits, target, hops)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
            opt.step()
            if step % cfg.log_every == 0 or step == cfg.final_steps:
                print(
                    f"  final step {step:5d}/{cfg.final_steps} "
                    f"loss {loss.item():.4f} acc {accuracy(logits, target)*100:.1f}% "
                    f"hops {active_hops or cfg.max_hops} "
                    f"gate {torch.sigmoid(model.reinject_logit).item():.3f} "
                    f"elapsed {time.time()-t0:.1f}s",
                    flush=True,
                )

        for step in range(1, cfg.recurrent_steps + 1):
            active_hops = active_hops_for_step(step, cfg.recurrent_steps)
            input_ids, attention_mask, lengths, target, step_targets, hops, _ = batch_encoded(
                cfg.batch_size,
                active_hops,
            )
            logits_by_loop = model.collect_logits(input_ids, attention_mask, lengths, cfg.loop_steps, include_zero=False)
            losses = []
            for i, logits_i in enumerate(logits_by_loop):
                weight = cfg.final_loop_loss_weight if i == cfg.loop_steps - 1 else 1.0
                losses.append(weight * weighted_ce(logits_i, step_targets[:, i], hops))
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
                    f"hops {active_hops or cfg.max_hops} "
                    f"elapsed {time.time()-t0:.1f}s",
                    flush=True,
                )
            maybe_validate_teacher(step)
        if best_teacher_state is not None:
            model.load_state_dict(best_teacher_state)
            print(
                "[teacher val] "
                f"restored best step {best_teacher_summary['step']} "
                f"full {best_teacher_summary['full_acc']*100:.2f}% "
                f"worst_hop {best_teacher_summary['worst_hop_acc']*100:.2f}%",
                flush=True,
            )
            save_checkpoint(
                model,
                extra={
                    "teacher_val_best": best_teacher_summary,
                    "teacher_val_restored": True,
                },
            )
        else:
            save_checkpoint(model)

    def eval_model(audit_variant: str = "normal"):
        model.eval()
        final_acc = [0.0 for _ in range(cfg.loop_steps + 1)]
        step_acc = [None] + [0.0 for _ in range(cfg.loop_steps)]
        by_hop = {str(i): [] for i in range(1, cfg.max_hops + 1)}
        by_task = {name: [] for name in task_names}
        thresholds = [(0.80, "080"), (0.90, "090"), (0.95, "095")]
        exit_metrics = {}
        for _, suffix in thresholds:
            exit_metrics[f"early_{suffix}_acc"] = 0.0
            exit_metrics[f"early_{suffix}_avg_loops"] = 0.0
            exit_metrics[f"early_{suffix}_full_loop_rate"] = 0.0
            exit_metrics[f"early_{suffix}_core_savings_vs_full_pct"] = 0.0
        with torch.no_grad():
            for _ in range(cfg.eval_batches):
                input_ids, attention_mask, lengths, target, step_targets, hops, task_ids = batch_encoded(
                    cfg.batch_size,
                    hard_hop_sampling=False,
                    audit_variant=audit_variant,
                )
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
                for task_id, name in enumerate(task_names):
                    mask = task_ids == task_id
                    if mask.any():
                        by_task[name].append((full_pred[mask] == target[mask]).float().mean().item())
                logits_stack = torch.stack(logits_by_loop[1:], dim=0)
                pred_stack = torch.stack(preds[1:], dim=0)
                probs = F.softmax(logits_stack, dim=-1)
                top2 = probs.topk(2, dim=-1).values
                margin_stack = top2[:, :, 0] - top2[:, :, 1]
                max_prob_stack = top2[:, :, 0]
                for threshold, suffix in thresholds:
                    trusted = torch.zeros_like(target, dtype=torch.bool)
                    chosen_loops = torch.full_like(target, cfg.loop_steps)
                    pred_exit = full_pred
                    for i in range(cfg.loop_steps):
                        accept = (
                            (~trusted)
                            & (max_prob_stack[i] >= threshold)
                            & (margin_stack[i] >= 0.05)
                        )
                        loop_count = torch.full_like(chosen_loops, i + 1)
                        chosen_loops = torch.where(accept, loop_count, chosen_loops)
                        pred_exit = torch.where(accept, pred_stack[i], pred_exit)
                        trusted = trusted | accept
                    exit_metrics[f"early_{suffix}_acc"] += (pred_exit == target).float().mean().item()
                    exit_metrics[f"early_{suffix}_avg_loops"] += chosen_loops.float().mean().item()
                    exit_metrics[f"early_{suffix}_full_loop_rate"] += (~trusted).float().mean().item()
        model.train()
        final_acc = [x / cfg.eval_batches for x in final_acc]
        for k in list(exit_metrics):
            exit_metrics[k] /= cfg.eval_batches
        for _, suffix in thresholds:
            exit_metrics[f"early_{suffix}_avg_core_layers"] = (
                exit_metrics[f"early_{suffix}_avg_loops"] * float(cfg.core_layers)
            )
            exit_metrics[f"early_{suffix}_core_savings_vs_full_pct"] = (
                1.0 - exit_metrics[f"early_{suffix}_avg_loops"] / float(cfg.loop_steps)
            ) * 100.0
        step_acc_out = {
            str(i): (step_acc[i] / cfg.eval_batches if i > 0 else None)
            for i in range(cfg.loop_steps + 1)
        }
        out = {
            "audit_variant": audit_variant,
            "final_acc_by_loops": {str(i): final_acc[i] for i in range(cfg.loop_steps + 1)},
            "step_acc_by_loops": step_acc_out,
            "zero_loop_acc": final_acc[0],
            "one_loop_acc": final_acc[1],
            "full_loop_acc": final_acc[-1],
            "loop_gain_vs_zero": final_acc[-1] - final_acc[0],
            "loop_gain_vs_one": final_acc[-1] - final_acc[1],
            "full_by_hop": {k: (sum(v) / len(v) if v else None) for k, v in by_hop.items()},
            "full_by_task": {k: (sum(v) / len(v) if v else None) for k, v in by_task.items()},
            "reinject_gate": torch.sigmoid(model.reinject_logit).item(),
        }
        out.update(exit_metrics)
        return out

    eval_summary = eval_model()
    if best_teacher_summary is not None:
        eval_summary["teacher_val_best"] = best_teacher_summary
    print(f"[eval] {json.dumps(eval_summary, indent=2)}")

    prompt_audit_summary = None
    if audit_prompt_variants:
        py_random_state = random.getstate()
        try:
            prompt_audit_summary = {
                variant: eval_model(variant)
                for variant in audit_prompt_variants
            }
        finally:
            random.setstate(py_random_state)
        print(f"[prompt audit] {json.dumps(prompt_audit_summary, indent=2)}")

    def teacher_collect(input_ids, attention_mask, lengths):
        state0, layer_mask, position_ids, position_embeddings = model.encode(input_ids, attention_mask)
        hidden = state0
        states = []
        for i in range(cfg.loop_steps):
            hidden = model.loop_once(hidden, state0, layer_mask, position_ids, position_embeddings, i)
            states.append(hidden)
        logits = model.classify_state(hidden, lengths, layer_mask, position_ids, position_embeddings)
        return state0, layer_mask, position_ids, position_embeddings, states, logits

    jumprec = None
    jump_summary = None
    jumprec_prompt_audit_summary = None
    direct_summary = None
    has_loaded_jumprec = (
        cfg.load_jumprec_state
        and loaded_checkpoint is not None
        and loaded_checkpoint.get("jumprec_state") is not None
    )
    should_use_jumprec = cfg.jump_steps > 0 or (
        loaded_checkpoint is not None
        and not cfg.resume_teacher_training
        and has_loaded_jumprec
    )
    if should_use_jumprec:
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
                for p in self.jump_blocks.parameters():
                    p.requires_grad_(True)
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

            def forward_budget_encoded(
                self,
                state0,
                layer_mask,
                position_ids,
                position_embeddings,
                lengths,
                corrections: int,
            ):
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
                return final_logits, verifier_logit

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
        if has_loaded_jumprec:
            jumprec.load_state_dict(loaded_checkpoint["jumprec_state"])
            print("[checkpoint] loaded JumpRec", flush=True)
        else:
            opt_j = torch.optim.AdamW([p for p in jumprec.parameters() if p.requires_grad], lr=cfg.jump_lr)
            t1 = time.time()
            jumprec.train()
            for step in range(1, cfg.jump_steps + 1):
                input_ids, attention_mask, lengths, target, _, hops, _ = batch_encoded(cfg.batch_size)
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
                    ce_losses.append(weighted_ce(final_logits, target, hops))
                    distill_losses.append(weighted_kl(final_logits, soft_teacher, hops))
                    good = (final_logits.detach().argmax(dim=-1) == target).float()
                    verifier_losses.append(weighted_bce_with_logits(out["verify"][corrections], good, hops))
                loss = (
                    torch.stack(ce_losses).mean()
                    + cfg.distill_loss_weight * torch.stack(distill_losses).mean()
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
            save_checkpoint(model, jumprec)

        def eval_jumprec(audit_variant: str = "normal", include_heldout: bool = True):
            jumprec.eval()
            thresholds = [(0.80, "080"), (0.90, "090"), (0.95, "095")]
            router_policies = [("no_agree", False), ("agree", True)]
            full_core_layers = float(cfg.loop_steps * cfg.core_layers)
            metrics = {f"jump_c{c}_acc": 0.0 for c in range(cfg.max_correct + 1)}
            strict_080_by_hop = {str(i): [] for i in range(1, cfg.max_hops + 1)}
            strict_080_by_task = {name: [] for name in task_names}
            oracle = {
                "acc": 0.0,
                "avg_core_layers": 0.0,
                "full_loop_rate": 0.0,
                "any_jump_correct_rate": 0.0,
            }

            def threshold_key(threshold: float) -> str:
                return f"{threshold:.3f}".rstrip("0").rstrip(".")

            def calibration_bucket():
                return {
                    "count": 0,
                    "conf_sum": 0.0,
                    "correct_sum": 0.0,
                    "brier_sum": 0.0,
                    "bins": [
                        {"count": 0, "conf_sum": 0.0, "correct_sum": 0.0}
                        for _ in range(10)
                    ],
                }

            calibration = {
                "all": calibration_bucket(),
                "by_budget": {str(c): calibration_bucket() for c in range(cfg.max_correct + 1)},
            }

            def update_calibration(bucket, probs, correct):
                probs_f = probs.detach().float().flatten()
                correct_f = correct.detach().float().flatten()
                n = int(probs_f.numel())
                if n == 0:
                    return
                bucket["count"] += n
                bucket["conf_sum"] += probs_f.sum().item()
                bucket["correct_sum"] += correct_f.sum().item()
                bucket["brier_sum"] += ((probs_f - correct_f) ** 2).sum().item()
                bin_ids = torch.clamp((probs_f * 10).long(), max=9)
                for bin_id in range(10):
                    mask = bin_ids == bin_id
                    if mask.any():
                        bin_count = int(mask.sum().item())
                        bucket["bins"][bin_id]["count"] += bin_count
                        bucket["bins"][bin_id]["conf_sum"] += probs_f[mask].sum().item()
                        bucket["bins"][bin_id]["correct_sum"] += correct_f[mask].sum().item()

            def finish_calibration(bucket):
                n = bucket["count"]
                if n == 0:
                    return None
                ece = 0.0
                bins = []
                for bin_id, bin_bucket in enumerate(bucket["bins"]):
                    bin_count = bin_bucket["count"]
                    if bin_count:
                        mean_conf = bin_bucket["conf_sum"] / bin_count
                        empirical_acc = bin_bucket["correct_sum"] / bin_count
                        ece += (bin_count / n) * abs(mean_conf - empirical_acc)
                    else:
                        mean_conf = None
                        empirical_acc = None
                    bins.append(
                        {
                            "lo": bin_id / 10.0,
                            "hi": (bin_id + 1) / 10.0,
                            "count": bin_count,
                            "mean_conf": mean_conf,
                            "empirical_acc": empirical_acc,
                        }
                    )
                return {
                    "count": n,
                    "mean_conf": bucket["conf_sum"] / n,
                    "empirical_acc": bucket["correct_sum"] / n,
                    "brier": bucket["brier_sum"] / n,
                    "ece_10": ece,
                    "bins": bins,
                }

            def route_predictions(
                pred_stack,
                verify_stack,
                margin_stack,
                max_prob_stack,
                agree_stack,
                full_pred,
                target,
                threshold: float,
                need_agreement: bool,
            ):
                trusted = torch.zeros_like(target, dtype=torch.bool)
                chosen = torch.full_like(target, cfg.loop_steps)
                routed_pred = full_pred
                for c in range(cfg.max_correct + 1):
                    agree_ok = agree_stack[c] if need_agreement else torch.ones_like(trusted)
                    accept = (
                        (~trusted)
                        & (verify_stack[c] >= threshold)
                        & (margin_stack[c] >= 0.05)
                        & (max_prob_stack[c] >= 0.45)
                        & agree_ok
                    )
                    chosen = torch.where(accept, torch.full_like(chosen, c), chosen)
                    routed_pred = torch.where(accept, pred_stack[c], routed_pred)
                    trusted = trusted | accept
                core_layers = torch.where(
                    trusted,
                    torch.full_like(chosen.float(), float(cfg.jump_layers))
                    + chosen.float() * float(cfg.core_layers),
                    torch.full_like(chosen.float(), full_core_layers),
                )
                return routed_pred, trusted, chosen, core_layers

            def collect_router_grid(num_batches: int, threshold_values: List[float]):
                threshold_values = sorted(set(float(t) for t in threshold_values))
                out = {
                    "batches": num_batches,
                    "full_teacher_acc": 0.0,
                    "policies": {
                        policy_name: {
                            threshold_key(threshold): {
                                "threshold": threshold,
                                "acc": 0.0,
                                "full_loop_rate": 0.0,
                                "coverage": 0.0,
                                "avg_tail_loops": 0.0,
                                "avg_core_layers": 0.0,
                                "core_savings_vs_full_pct": 0.0,
                                "accepted_count": 0,
                                "accepted_precision": None,
                            }
                            for threshold in threshold_values
                        }
                        for policy_name, _ in router_policies
                    },
                }
                accepted_correct = {
                    policy_name: {threshold_key(threshold): 0.0 for threshold in threshold_values}
                    for policy_name, _ in router_policies
                }
                accepted_count = {
                    policy_name: {threshold_key(threshold): 0 for threshold in threshold_values}
                    for policy_name, _ in router_policies
                }
                with torch.no_grad():
                    for _ in range(num_batches):
                        input_ids, attention_mask, lengths, target, _, _, _ = batch_encoded(
                            cfg.batch_size,
                            hard_hop_sampling=False,
                            audit_variant=audit_variant,
                        )
                        state0, layer_mask, position_ids, position_embeddings, _, teacher_logits = teacher_collect(
                            input_ids,
                            attention_mask,
                            lengths,
                        )
                        full_pred = teacher_logits.argmax(dim=-1)
                        out["full_teacher_acc"] += (full_pred == target).float().mean().item()
                        jump_out = jumprec.forward_encoded(
                            state0,
                            layer_mask,
                            position_ids,
                            position_embeddings,
                            lengths,
                        )
                        pred_stack = torch.stack(
                            [logits_i.argmax(dim=-1) for logits_i in jump_out["logits"]],
                            dim=0,
                        )
                        verify_stack = torch.stack([torch.sigmoid(v) for v in jump_out["verify"]], dim=0)
                        logits_stack = torch.stack(jump_out["logits"], dim=0)
                        probs = F.softmax(logits_stack, dim=-1)
                        top2 = probs.topk(2, dim=-1).values
                        margin_stack = top2[:, :, 0] - top2[:, :, 1]
                        max_prob_stack = top2[:, :, 0]
                        agree_stack = torch.ones_like(verify_stack, dtype=torch.bool)
                        agree_stack[:-1] = pred_stack[:-1] == pred_stack[1:]
                        agree_stack[-1] = False
                        for threshold in threshold_values:
                            key = threshold_key(threshold)
                            for policy_name, need_agreement in router_policies:
                                routed_pred, trusted, chosen, core_layers = route_predictions(
                                    pred_stack,
                                    verify_stack,
                                    margin_stack,
                                    max_prob_stack,
                                    agree_stack,
                                    full_pred,
                                    target,
                                    threshold,
                                    need_agreement,
                                )
                                item = out["policies"][policy_name][key]
                                item["acc"] += (routed_pred == target).float().mean().item()
                                item["full_loop_rate"] += (~trusted).float().mean().item()
                                item["coverage"] += trusted.float().mean().item()
                                item["avg_tail_loops"] += torch.where(
                                    trusted,
                                    chosen.float(),
                                    torch.full_like(chosen.float(), float(cfg.loop_steps)),
                                ).mean().item()
                                item["avg_core_layers"] += core_layers.mean().item()
                                accepted_count[policy_name][key] += int(trusted.sum().item())
                                if trusted.any():
                                    accepted_correct[policy_name][key] += (
                                        routed_pred[trusted] == target[trusted]
                                    ).float().sum().item()
                out["full_teacher_acc"] /= num_batches
                for policy_name, _ in router_policies:
                    for key, item in out["policies"][policy_name].items():
                        item["acc"] /= num_batches
                        item["full_loop_rate"] /= num_batches
                        item["coverage"] /= num_batches
                        item["avg_tail_loops"] /= num_batches
                        item["avg_core_layers"] /= num_batches
                        item["core_savings_vs_full_pct"] = (
                            1.0 - item["avg_core_layers"] / full_core_layers
                        ) * 100.0
                        item["accepted_count"] = accepted_count[policy_name][key]
                        if accepted_count[policy_name][key] > 0:
                            item["accepted_precision"] = (
                                accepted_correct[policy_name][key] / accepted_count[policy_name][key]
                            )
                return out

            def choose_heldout_threshold(val_grid, policy_name: str):
                floor = val_grid["full_teacher_acc"] - cfg.router_val_max_drop
                candidates = list(val_grid["policies"][policy_name].items())
                acceptable = [(key, item) for key, item in candidates if item["acc"] >= floor]
                if acceptable:
                    key, item = min(
                        acceptable,
                        key=lambda pair: (pair[1]["avg_core_layers"], -pair[1]["acc"]),
                    )
                    reason = "within_val_drop"
                else:
                    key, item = max(
                        candidates,
                        key=lambda pair: (pair[1]["acc"], -pair[1]["avg_core_layers"]),
                    )
                    reason = "best_val_accuracy"
                return key, item, reason, floor

            for _, suffix in thresholds:
                metrics[f"strict_{suffix}_fallback_acc"] = 0.0
                metrics[f"strict_{suffix}_fallback_full_loop_rate"] = 0.0
                metrics[f"strict_{suffix}_fallback_avg_tail_loops"] = 0.0
                metrics[f"strict_{suffix}_fallback_avg_core_layers"] = 0.0
                for policy_name, _ in router_policies:
                    prefix = f"router_{suffix}_{policy_name}"
                    metrics[f"{prefix}_acc"] = 0.0
                    metrics[f"{prefix}_full_loop_rate"] = 0.0
                    metrics[f"{prefix}_avg_tail_loops"] = 0.0
                    metrics[f"{prefix}_avg_core_layers"] = 0.0
            with torch.no_grad():
                for _ in range(cfg.eval_batches):
                    input_ids, attention_mask, lengths, target, _, hops, task_ids = batch_encoded(
                        cfg.batch_size,
                        hard_hop_sampling=False,
                        audit_variant=audit_variant,
                    )
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
                        good = (pred == target).float()
                        update_calibration(calibration["by_budget"][str(c)], verify_stack[c], good)
                        update_calibration(calibration["all"], verify_stack[c], good)
                    correct_stack = pred_stack == target.unsqueeze(0)
                    any_jump_correct = correct_stack.any(dim=0)
                    first_correct = correct_stack.float().argmax(dim=0)
                    pred_oracle = full_pred
                    for c in range(cfg.max_correct + 1):
                        use_budget = any_jump_correct & (first_correct == c)
                        pred_oracle = torch.where(use_budget, pred_stack[c], pred_oracle)
                    oracle_core_layers = torch.where(
                        any_jump_correct,
                        torch.full_like(first_correct.float(), float(cfg.jump_layers))
                        + first_correct.float() * float(cfg.core_layers),
                        torch.full_like(first_correct.float(), full_core_layers),
                    )
                    oracle["acc"] += (pred_oracle == target).float().mean().item()
                    oracle["avg_core_layers"] += oracle_core_layers.mean().item()
                    oracle["full_loop_rate"] += (~any_jump_correct).float().mean().item()
                    oracle["any_jump_correct_rate"] += any_jump_correct.float().mean().item()
                    for threshold, suffix in thresholds:
                        pred_fb, trusted, chosen, core_layers = route_predictions(
                            pred_stack,
                            verify_stack,
                            margin_stack,
                            max_prob_stack,
                            agree_stack,
                            full_pred,
                            target,
                            threshold,
                            cfg.strict_need_agreement,
                        )
                        metrics[f"strict_{suffix}_fallback_acc"] += (pred_fb == target).float().mean().item()
                        metrics[f"strict_{suffix}_fallback_full_loop_rate"] += (~trusted).float().mean().item()
                        metrics[f"strict_{suffix}_fallback_avg_tail_loops"] += torch.where(
                            trusted,
                            chosen.float(),
                            torch.full_like(chosen.float(), float(cfg.loop_steps)),
                        ).mean().item()
                        metrics[f"strict_{suffix}_fallback_avg_core_layers"] += core_layers.mean().item()
                        for policy_name, need_agreement in router_policies:
                            pred_router, trusted_router, chosen_router, router_core_layers = route_predictions(
                                pred_stack,
                                verify_stack,
                                margin_stack,
                                max_prob_stack,
                                agree_stack,
                                full_pred,
                                target,
                                threshold,
                                need_agreement,
                            )
                            prefix = f"router_{suffix}_{policy_name}"
                            metrics[f"{prefix}_acc"] += (pred_router == target).float().mean().item()
                            metrics[f"{prefix}_full_loop_rate"] += (~trusted_router).float().mean().item()
                            metrics[f"{prefix}_avg_tail_loops"] += torch.where(
                                trusted_router,
                                chosen_router.float(),
                                torch.full_like(chosen_router.float(), float(cfg.loop_steps)),
                            ).mean().item()
                            metrics[f"{prefix}_avg_core_layers"] += router_core_layers.mean().item()
                        if suffix == "080":
                            for hop in range(1, cfg.max_hops + 1):
                                mask = hops == hop
                                if mask.any():
                                    strict_080_by_hop[str(hop)].append(
                                        (pred_fb[mask] == target[mask]).float().mean().item()
                                    )
                            for task_id, name in enumerate(task_names):
                                mask = task_ids == task_id
                                if mask.any():
                                    strict_080_by_task[name].append(
                                        (pred_fb[mask] == target[mask]).float().mean().item()
                                    )
            for k in list(metrics):
                metrics[k] /= cfg.eval_batches
            metrics["audit_variant"] = audit_variant
            metrics["full_core_layers"] = full_core_layers
            for _, suffix in thresholds:
                metrics[f"strict_{suffix}_fallback_core_savings_vs_full_pct"] = (
                    1.0 - metrics[f"strict_{suffix}_fallback_avg_core_layers"] / full_core_layers
                ) * 100.0
                for policy_name, _ in router_policies:
                    prefix = f"router_{suffix}_{policy_name}"
                    metrics[f"{prefix}_core_savings_vs_full_pct"] = (
                        1.0 - metrics[f"{prefix}_avg_core_layers"] / full_core_layers
                    ) * 100.0
            metrics["strict_080_by_hop"] = {
                k: (sum(v) / len(v) if v else None) for k, v in strict_080_by_hop.items()
            }
            metrics["strict_080_by_task"] = {
                k: (sum(v) / len(v) if v else None) for k, v in strict_080_by_task.items()
            }
            for key in list(oracle):
                oracle[key] /= cfg.eval_batches
            oracle["core_savings_vs_full_pct"] = (
                1.0 - oracle["avg_core_layers"] / full_core_layers
            ) * 100.0
            metrics["oracle_router"] = oracle
            metrics["verifier_calibration"] = {
                "all": finish_calibration(calibration["all"]),
                "by_budget": {
                    key: finish_calibration(bucket)
                    for key, bucket in calibration["by_budget"].items()
                },
            }
            if include_heldout and cfg.router_val_batches > 0:
                candidates = parse_float_list(cfg.router_threshold_candidates)
                if not candidates:
                    candidates = [threshold for threshold, _ in thresholds]
                val_grid = collect_router_grid(cfg.router_val_batches, candidates)
                selected = {}
                selected_thresholds = []
                for policy_name, _ in router_policies:
                    key, val_item, reason, floor = choose_heldout_threshold(val_grid, policy_name)
                    selected[policy_name] = {
                        "threshold": val_item["threshold"],
                        "threshold_key": key,
                        "selection_reason": reason,
                        "val_accuracy_floor": floor,
                        "val": val_item,
                    }
                    selected_thresholds.append(val_item["threshold"])
                final_grid = collect_router_grid(cfg.eval_batches, selected_thresholds)
                for policy_name, item in selected.items():
                    item["final"] = final_grid["policies"][policy_name][item["threshold_key"]]
                metrics["heldout_threshold_audit"] = {
                    "threshold_candidates": candidates,
                    "val_batches": cfg.router_val_batches,
                    "final_batches": cfg.eval_batches,
                    "max_val_drop_vs_teacher": cfg.router_val_max_drop,
                    "val_full_teacher_acc": val_grid["full_teacher_acc"],
                    "final_full_teacher_acc": final_grid["full_teacher_acc"],
                    "selected": selected,
                }
            jumprec.train()
            return metrics

        jump_summary = eval_jumprec()
        print(f"[jumprec eval] {json.dumps(jump_summary, indent=2)}")
        if audit_prompt_variants:
            py_random_state = random.getstate()
            try:
                jumprec_prompt_audit_summary = {
                    variant: eval_jumprec(variant, include_heldout=False)
                    for variant in audit_prompt_variants
                }
            finally:
                random.setstate(py_random_state)
            print(f"[jumprec prompt audit] {json.dumps(jumprec_prompt_audit_summary, indent=2)}")

    if cfg.direct_steps > 0:
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()

        class DirectControl(nn.Module):
            def __init__(self, teacher: RecurrentSmol):
                super().__init__()
                source_layers = [teacher.core[i % len(teacher.core)] for i in range(cfg.direct_layers)]
                self.blocks = nn.ModuleList([copy.deepcopy(layer) for layer in source_layers])
                for p in self.blocks.parameters():
                    p.requires_grad_(True)
                self.teacher = teacher

            def forward_encoded(self, state0, layer_mask, position_ids, position_embeddings, lengths):
                hidden = self.teacher.run_layers(self.blocks, state0, layer_mask, position_ids, position_embeddings)
                return self.teacher.classify_state(hidden, lengths, layer_mask, position_ids, position_embeddings)

        direct = DirectControl(model).to(device)
        print(f"[direct] trainable params={sum(p.numel() for p in direct.parameters() if p.requires_grad)/1e6:.3f}M")
        opt_d = torch.optim.AdamW([p for p in direct.parameters() if p.requires_grad], lr=cfg.direct_lr)
        t2 = time.time()
        direct.train()
        for step in range(1, cfg.direct_steps + 1):
            input_ids, attention_mask, lengths, target, _, hops, _ = batch_encoded(cfg.batch_size)
            with torch.no_grad():
                state0, layer_mask, position_ids, position_embeddings, _, _ = teacher_collect(
                    input_ids,
                    attention_mask,
                    lengths,
                )
            logits = direct.forward_encoded(state0.detach(), layer_mask, position_ids, position_embeddings, lengths)
            loss = weighted_ce(logits, target, hops)
            opt_d.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(direct.parameters(), 1.0)
            opt_d.step()
            if step % cfg.log_every == 0 or step == cfg.direct_steps:
                print(
                    f"  direct step {step:5d}/{cfg.direct_steps} "
                    f"loss {loss.item():.4f} acc {accuracy(logits, target)*100:.1f}% "
                    f"elapsed {time.time()-t2:.1f}s",
                    flush=True,
                )

        direct.eval()
        acc = 0.0
        direct_by_hop = {str(i): [] for i in range(1, cfg.max_hops + 1)}
        direct_by_task = {name: [] for name in task_names}
        with torch.no_grad():
            for _ in range(cfg.eval_batches):
                input_ids, attention_mask, lengths, target, _, hops, task_ids = batch_encoded(
                    cfg.batch_size,
                    hard_hop_sampling=False,
                )
                state0, layer_mask, position_ids, position_embeddings, _, _ = teacher_collect(
                    input_ids,
                    attention_mask,
                    lengths,
                )
                logits = direct.forward_encoded(state0, layer_mask, position_ids, position_embeddings, lengths)
                pred = logits.argmax(dim=-1)
                acc += (pred == target).float().mean().item()
                for hop in range(1, cfg.max_hops + 1):
                    mask = hops == hop
                    if mask.any():
                        direct_by_hop[str(hop)].append((pred[mask] == target[mask]).float().mean().item())
                for task_id, name in enumerate(task_names):
                    mask = task_ids == task_id
                    if mask.any():
                        direct_by_task[name].append((pred[mask] == target[mask]).float().mean().item())
        direct_summary = {
            "direct_acc": acc / cfg.eval_batches,
            "direct_layers": float(cfg.direct_layers),
            "direct_core_savings_vs_full_pct": (
                1.0 - float(cfg.direct_layers) / float(cfg.loop_steps * cfg.core_layers)
            ) * 100.0,
            "direct_by_hop": {k: (sum(v) / len(v) if v else None) for k, v in direct_by_hop.items()},
            "direct_by_task": {k: (sum(v) / len(v) if v else None) for k, v in direct_by_task.items()},
        }
        print(f"[direct eval] {json.dumps(direct_summary, indent=2)}")

    def benchmark():
        if cfg.timing_batches <= 0:
            return {}

        def sync():
            if device.type == "cuda":
                torch.cuda.synchronize()

        def time_fn(fn, timing_bsz: int | None = None):
            timing_bsz = timing_bsz or cfg.timing_batch_size or cfg.batch_size
            with torch.no_grad():
                for _ in range(2):
                    input_ids, attention_mask, lengths, _, _, _, _ = batch_encoded(
                        timing_bsz,
                        hard_hop_sampling=False,
                    )
                    fn(input_ids, attention_mask, lengths)
                sync()
                start = time.perf_counter()
                for _ in range(cfg.timing_batches):
                    input_ids, attention_mask, lengths, _, _, _, _ = batch_encoded(
                        timing_bsz,
                        hard_hop_sampling=False,
                    )
                    fn(input_ids, attention_mask, lengths)
                sync()
                return 1000.0 * (time.perf_counter() - start) / cfg.timing_batches

        model.eval()
        out = {
            "batch_size": float(cfg.batch_size),
            "timing_batch_size": float(cfg.timing_batch_size or cfg.batch_size),
            "one_loop_ms_per_batch": time_fn(lambda ids, mask, lens: model(ids, mask, lens, 1)),
            "full_loop_ms_per_batch": time_fn(lambda ids, mask, lens: model(ids, mask, lens, cfg.loop_steps)),
        }
        if jumprec is not None:
            out["jumprec_all_budgets_ms_per_batch"] = time_fn(
                lambda ids, mask, lens: jumprec.forward_encoded(*model.encode(ids, mask), lens)
            )

            def serial_jumprec(ids, mask, lens, threshold: float = 0.80):
                state0, layer_mask, position_ids, position_embeddings = model.encode(ids, mask)
                open_idx = torch.arange(ids.size(0), device=ids.device)
                last = None

                def take_batch(x, idx):
                    if x is None:
                        return None
                    if x.size(0) == ids.size(0):
                        return x.index_select(0, idx)
                    return x

                for corrections in range(cfg.max_correct + 1):
                    if open_idx.numel() == 0:
                        break
                    sub_state0 = state0.index_select(0, open_idx)
                    sub_mask = take_batch(layer_mask, open_idx)
                    sub_pos = take_batch(position_ids, open_idx)
                    sub_emb = tuple(take_batch(t, open_idx) for t in position_embeddings) if position_embeddings else None
                    sub_lens = lens.index_select(0, open_idx)
                    logits, verify = jumprec.forward_budget_encoded(
                        sub_state0,
                        sub_mask,
                        sub_pos,
                        sub_emb,
                        sub_lens,
                        corrections,
                    )
                    probs = F.softmax(logits, dim=-1)
                    top2 = probs.topk(2, dim=-1).values
                    accept = (
                        (torch.sigmoid(verify) >= threshold)
                        & ((top2[:, 0] - top2[:, 1]) >= 0.05)
                        & (top2[:, 0] >= 0.45)
                    )
                    last = logits
                    open_idx = open_idx[~accept]
                if open_idx.numel() > 0:
                    sub_state0 = state0.index_select(0, open_idx)
                    sub_mask = take_batch(layer_mask, open_idx)
                    sub_pos = take_batch(position_ids, open_idx)
                    sub_emb = tuple(take_batch(t, open_idx) for t in position_embeddings) if position_embeddings else None
                    sub_lens = lens.index_select(0, open_idx)
                    full = model.run_steps_from(sub_state0, sub_state0, sub_mask, sub_pos, sub_emb, 0, cfg.loop_steps)
                    last = model.classify_state(full, sub_lens, sub_mask, sub_pos, sub_emb)
                return last

            out["jumprec_serial_080_ms_per_batch"] = time_fn(serial_jumprec)
            out["jumprec_serial_090_ms_per_batch"] = time_fn(
                lambda ids, mask, lens: serial_jumprec(ids, mask, lens, 0.90)
            )
            out["jumprec_serial_095_ms_per_batch"] = time_fn(
                lambda ids, mask, lens: serial_jumprec(ids, mask, lens, 0.95)
            )

            def serial_jumprec_agree(ids, mask, lens, threshold: float = 0.80):
                state0, layer_mask, position_ids, position_embeddings = model.encode(ids, mask)
                open_idx = torch.arange(ids.size(0), device=ids.device)
                last = None

                def take_batch(x, idx):
                    if x is None:
                        return None
                    if x.size(0) == ids.size(0):
                        return x.index_select(0, idx)
                    return x

                for corrections in range(cfg.max_correct + 1):
                    if open_idx.numel() == 0:
                        break
                    sub_state0 = state0.index_select(0, open_idx)
                    sub_mask = take_batch(layer_mask, open_idx)
                    sub_pos = take_batch(position_ids, open_idx)
                    sub_emb = tuple(take_batch(t, open_idx) for t in position_embeddings) if position_embeddings else None
                    sub_lens = lens.index_select(0, open_idx)
                    logits, verify = jumprec.forward_budget_encoded(
                        sub_state0,
                        sub_mask,
                        sub_pos,
                        sub_emb,
                        sub_lens,
                        corrections,
                    )
                    probs = F.softmax(logits, dim=-1)
                    top2 = probs.topk(2, dim=-1).values
                    accept = (
                        (torch.sigmoid(verify) >= threshold)
                        & ((top2[:, 0] - top2[:, 1]) >= 0.05)
                        & (top2[:, 0] >= 0.45)
                    )
                    if corrections < cfg.max_correct:
                        next_logits, _ = jumprec.forward_budget_encoded(
                            sub_state0,
                            sub_mask,
                            sub_pos,
                            sub_emb,
                            sub_lens,
                            corrections + 1,
                        )
                        accept = accept & (logits.argmax(dim=-1) == next_logits.argmax(dim=-1))
                    else:
                        accept = torch.zeros_like(accept)
                    last = logits
                    open_idx = open_idx[~accept]
                if open_idx.numel() > 0:
                    sub_state0 = state0.index_select(0, open_idx)
                    sub_mask = take_batch(layer_mask, open_idx)
                    sub_pos = take_batch(position_ids, open_idx)
                    sub_emb = tuple(take_batch(t, open_idx) for t in position_embeddings) if position_embeddings else None
                    sub_lens = lens.index_select(0, open_idx)
                    full = model.run_steps_from(sub_state0, sub_state0, sub_mask, sub_pos, sub_emb, 0, cfg.loop_steps)
                    last = model.classify_state(full, sub_lens, sub_mask, sub_pos, sub_emb)
                return last

            out["jumprec_serial_agree_080_ms_per_batch"] = time_fn(serial_jumprec_agree)
        if cfg.timing_batch_sizes:
            sweep = {}
            for timing_bsz in [int(x.strip()) for x in cfg.timing_batch_sizes.split(",") if x.strip()]:
                item = {
                    "timing_batch_size": float(timing_bsz),
                    "one_loop_ms_per_batch": time_fn(
                        lambda ids, mask, lens: model(ids, mask, lens, 1),
                        timing_bsz,
                    ),
                    "full_loop_ms_per_batch": time_fn(
                        lambda ids, mask, lens: model(ids, mask, lens, cfg.loop_steps),
                        timing_bsz,
                    ),
                }
                if jumprec is not None:
                    item["jumprec_all_budgets_ms_per_batch"] = time_fn(
                        lambda ids, mask, lens: jumprec.forward_encoded(*model.encode(ids, mask), lens),
                        timing_bsz,
                    )
                    item["jumprec_serial_080_ms_per_batch"] = time_fn(serial_jumprec, timing_bsz)
                    item["jumprec_serial_090_ms_per_batch"] = time_fn(
                        lambda ids, mask, lens: serial_jumprec(ids, mask, lens, 0.90),
                        timing_bsz,
                    )
                    item["jumprec_serial_095_ms_per_batch"] = time_fn(
                        lambda ids, mask, lens: serial_jumprec(ids, mask, lens, 0.95),
                        timing_bsz,
                    )
                    item["jumprec_serial_agree_080_ms_per_batch"] = time_fn(
                        serial_jumprec_agree,
                        timing_bsz,
                    )
                sweep[str(timing_bsz)] = item
            out["batch_size_sweep"] = sweep
        model.train()
        return out

    timing_summary = benchmark()
    print(f"[timing] {json.dumps(timing_summary, indent=2)}")

    return {
        "config": asdict(cfg),
        "eval": eval_summary,
        "prompt_audit": prompt_audit_summary,
        "jumprec_eval": jump_summary,
        "jumprec_prompt_audit": jumprec_prompt_audit_summary,
        "direct_eval": direct_summary,
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
            "dry_hardhop",
            "dry_strathop",
            "dry_strathop_gate",
            "dry_strathop_polish",
            "dry_strathop_polish2",
            "dry_strathop_eval",
            "dry_strathop_polish2_eval",
            "dry_strathop_audit",
            "dry_strathop_polish2_audit",
            "dry_strathop_polish2_verifier_audit",
            "dry_sweep",
            "dry_sweep_reuse",
            "retrofit_probe",
            "jumprec_probe",
            "jumprec_no_adapter",
            "jumprec_no_agree",
            "direct_probe",
            "retrofit_no_reinject",
            "retrofit_core1",
            "retrofit_core3",
            "mixed_probe",
            "mixed_jumprec_direct",
            "mixed_core3_jumprec_direct",
            "mixed_core3_router_no_agree",
            "mixed_core3_router_no_agree_b1",
            "mixed_core3_router_bsize_sweep",
            "mixed_core3_router_bsize_sweep_reuse",
            "mixed_core3_router_verifier1",
            "retrofit_8n4h",
            "retrofit_8n4h_curriculum",
            "jumprec_8n4h_direct",
            "core3_8n4h_jumprec_direct",
            "core3_8n4h_hardhop_teacher",
            "core3_8n4h_hardhop_jumprec",
            "core3_8n4h_strathop_teacher",
            "core3_8n4h_strathop_gate_teacher",
            "core3_8n4h_strathop_polish_teacher",
            "core3_8n4h_strathop_polish2_teacher",
            "core3_8n4h_strathop_eval_teacher",
            "core3_8n4h_strathop_polish2_eval_teacher",
            "core3_8n4h_strathop_audit_teacher",
            "core3_8n4h_strathop_polish2_audit_teacher",
            "core3_8n4h_strathop_verifier_audit",
            "core3_8n4h_strathop_polish2_verifier_audit",
            "core3_8n4h_strathop_jumprec",
            "core3_8n4h_strathop_gate_jumprec",
            "core3_8n4h_strathop_polish_jumprec",
            "core3_8n4h_strathop_polish2_jumprec",
            "core3_8n4h_strathop_ablate_no_adapter",
            "core3_8n4h_strathop_ablate_no_distill",
            "core3_8n4h_strathop_ablate_no_verifier",
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
    if args.mode not in (
        "dry",
        "dry_hardhop",
        "dry_strathop",
        "dry_strathop_gate",
        "dry_strathop_polish",
        "dry_strathop_polish2",
        "dry_strathop_eval",
        "dry_strathop_polish2_eval",
        "dry_strathop_audit",
        "dry_strathop_polish2_audit",
        "dry_strathop_polish2_verifier_audit",
        "dry_sweep",
        "dry_sweep_reuse",
    ):
        raise SystemExit("Local CPU guard: only dry modes are allowed locally.")
    cfg = config_for_mode(args.mode)
    if args.seed is not None:
        cfg.seed = args.seed
    result = run_experiment(cfg, "cpu")
    print(json.dumps(result, indent=2))
