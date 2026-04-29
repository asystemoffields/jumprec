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
import itertools
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

NATGRAPH_NODE_NAMES = [
    "Aria",
    "Bryn",
    "Cato",
    "Dune",
    "Ezra",
    "Faye",
    "Galen",
    "Hale",
    "Iris",
    "Juno",
    "Kira",
    "Luca",
    "Mira",
    "Nico",
    "Orin",
    "Pax",
]


def route_label_names(prompt_style: str, n_nodes: int) -> List[str]:
    if prompt_style == "compact":
        return [chr(ord("A") + i) for i in range(n_nodes)]
    if prompt_style == "natural_graph":
        if n_nodes <= len(NATGRAPH_NODE_NAMES):
            return NATGRAPH_NODE_NAMES[:n_nodes]
        return NATGRAPH_NODE_NAMES + [f"Place{i}" for i in range(len(NATGRAPH_NODE_NAMES), n_nodes)]
    raise ValueError(f"unknown prompt_style: {prompt_style}")


def format_route_prompt(
    prompt_style: str,
    task: str,
    mapping_pairs: List[Tuple[int, int]],
    label_names: List[str],
    start_label: int,
    hops: int,
) -> str:
    mapping = " ".join(f"{label_names[src]}->{label_names[dst]}" for src, dst in mapping_pairs)
    if prompt_style == "compact":
        return (
            f"Task: {task}. Map: {mapping}. Start: {label_names[start_label]}. "
            f"Hops: {hops}. Answer:"
        )
    if prompt_style == "natural_graph":
        route_text = "; ".join(
            f"{label_names[src]} leads to {label_names[dst]}" for src, dst in mapping_pairs
        )
        instructions = {
            "forward": "follow each route arrow",
            "inverse": "move against each route arrow",
            "alternate": "first follow an arrow, then move against an arrow, alternating each move",
            "square": "treat one move as two route arrows in a row",
        }
        rule = instructions.get(task, f"use the {task} rule")
        return (
            "You are checking a route card. "
            f"Routes: {route_text}. "
            f"Start at {label_names[start_label]}. "
            f"Rule: {rule}. Moves: {hops}. "
            "Which place is reached? Answer:"
        )
    raise ValueError(f"unknown prompt_style: {prompt_style}")


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
    prompt_style: str = "compact"
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
    verifier_false_accept_weight: float = 1.0
    verifier_budget_savings_weight: float = 0.0
    verifier_ranking_loss_weight: float = 0.0
    verifier_ranking_margin: float = 0.25
    verifier_rank_later_weight: float = 1.0
    use_budget_controller: bool = False
    budget_controller_steps: int = 0
    budget_controller_lr: float = 4e-4
    budget_controller_target: str = "first_correct"
    budget_controller_train_threshold: float = 0.90
    budget_controller_under_weight: float = 0.0
    budget_controller_over_weight: float = 0.0
    use_stability_head: bool = False
    stability_steps: int = 0
    stability_lr: float = 4e-4
    stability_false_stable_weight: float = 2.0
    use_utility_router: bool = False
    utility_router_steps: int = 0
    utility_router_lr: float = 4e-4
    utility_false_accept_weight: float = 4.0
    utility_cost_weight: float = 0.25
    utility_correctness_bce_weight: float = 0.20
    utility_use_stability_feature: bool = False
    use_next_agreement_head: bool = False
    next_agreement_steps: int = 0
    next_agreement_lr: float = 4e-4
    use_consistency_head: bool = False
    consistency_steps: int = 0
    consistency_lr: float = 4e-4
    consistency_false_stable_weight: float = 4.0
    joint_halt_steps: int = 0
    joint_halt_lr: float = 1.5e-4
    joint_halt_candidate_ce_weight: float = 0.75
    joint_halt_candidate_distill_weight: float = 0.10
    joint_halt_verifier_bce_weight: float = 0.10
    joint_halt_stability_weight: float = 0.05
    joint_halt_agreement_bce_weight: float = 0.0
    joint_halt_agreement_distill_weight: float = 0.0
    joint_halt_agreement_route_weight: float = 0.0
    joint_halt_false_accept_weight_max: float = 0.0
    joint_halt_cost_weight_min: float = -1.0
    joint_halt_cost_weight_max: float = -1.0
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
    router_per_budget_audit: bool = False
    router_probe_audit: bool = False
    router_selective_agree_audit: bool = False
    router_probe_steps: int = 120
    router_probe_lr: float = 3e-3
    router_probe_hidden: int = 64

    @property
    def loop_steps(self) -> int:
        return self.max_hops + self.preserve_steps


def config_for_mode(mode: str) -> Config:
    cfg = Config(mode=mode)
    if mode in (
        "dry",
        "dry_sweep",
        "dry_sweep_reuse",
        "dry_natgraph_teacher",
        "dry_natgraph_joint_halt_quality_stability",
        "dry_natgraph_joint_halt_quality_stability_reuse",
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
        "dry_strathop_polish2_cost_controller",
        "dry_strathop_polish2_calib_controller",
        "dry_strathop_polish2_budget_controller",
        "dry_strathop_polish2_budget_controller_reuse",
        "dry_strathop_polish2_budget_verifytarget",
        "dry_strathop_polish2_budget_verifytarget_reuse",
        "dry_strathop_polish2_stability_router",
        "dry_strathop_polish2_stability_router_reuse",
        "dry_strathop_polish2_utility_router",
        "dry_strathop_polish2_utility_router_reuse",
        "dry_strathop_polish2_utility_stability_router",
        "dry_strathop_polish2_utility_stability_router_reuse",
        "dry_strathop_polish2_nextagree_router",
        "dry_strathop_polish2_nextagree_router_reuse",
        "dry_strathop_polish2_joint_halt",
        "dry_strathop_polish2_joint_halt_reuse",
        "dry_strathop_polish2_joint_halt_stability",
        "dry_strathop_polish2_joint_halt_stability_reuse",
        "dry_strathop_polish2_joint_halt_quality",
        "dry_strathop_polish2_joint_halt_quality_reuse",
        "dry_strathop_polish2_joint_halt_quality_agdistill",
        "dry_strathop_polish2_joint_halt_quality_agdistill_reuse",
        "dry_strathop_polish2_joint_halt_quality_cats",
        "dry_strathop_polish2_joint_halt_quality_cats_reuse",
        "dry_strathop_polish2_joint_halt_slo",
        "dry_strathop_polish2_joint_halt_slo_reuse",
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
        if mode == "dry_natgraph_teacher":
            cfg.n_nodes = 8
            cfg.max_hops = 4
            cfg.max_correct = 3
            cfg.max_length = 224
            cfg.prompt_style = "natural_graph"
            cfg.hop_sample_weights = "0.10,0.20,0.35,0.35"
            cfg.hop_loss_weights = "1.0,1.2,2.0,2.0"
            cfg.final_loop_loss_weight = 4.0
            cfg.jump_steps = 0
            cfg.direct_steps = 0
            cfg.save_checkpoints = True
            cfg.checkpoint_tag = "dry_natgraph_seed{seed}"
            cfg.timing_batch_sizes = "1,2"
        elif mode in (
            "dry_natgraph_joint_halt_quality_stability",
            "dry_natgraph_joint_halt_quality_stability_reuse",
        ):
            is_reuse = mode.endswith("_reuse")
            cfg.n_nodes = 8
            cfg.max_hops = 4
            cfg.max_correct = 3
            cfg.max_length = 224
            cfg.prompt_style = "natural_graph"
            cfg.hop_sample_weights = "0.10,0.20,0.35,0.35"
            cfg.hop_loss_weights = "1.0,1.2,2.0,2.0"
            cfg.final_loop_loss_weight = 4.0
            cfg.final_steps = 0
            cfg.recurrent_steps = 0
            cfg.jump_steps = 0 if is_reuse else 4
            cfg.joint_halt_steps = 0 if is_reuse else 4
            cfg.direct_steps = 0
            cfg.strict_need_agreement = False
            cfg.load_checkpoints = True
            cfg.load_checkpoint_tag = (
                "dry_natgraph_joint_halt_quality_stability_seed{seed}"
                if is_reuse
                else "dry_natgraph_seed{seed}"
            )
            cfg.save_checkpoints = not is_reuse
            cfg.checkpoint_tag = "dry_natgraph_joint_halt_quality_stability_seed{seed}"
            cfg.load_jumprec_state = is_reuse
            cfg.use_stability_head = True
            cfg.use_utility_router = True
            cfg.utility_use_stability_feature = True
            cfg.utility_false_accept_weight = 12.0
            cfg.utility_cost_weight = 0.08
            cfg.utility_correctness_bce_weight = 0.10
            cfg.joint_halt_candidate_ce_weight = 1.00
            cfg.joint_halt_candidate_distill_weight = 0.15
            cfg.joint_halt_verifier_bce_weight = 0.08
            cfg.joint_halt_stability_weight = 0.05
            cfg.joint_halt_agreement_bce_weight = 0.08
            cfg.router_val_batches = 1
            cfg.router_threshold_candidates = "0.10,0.30,0.50,0.70,0.90"
            cfg.router_probe_audit = is_reuse
            cfg.router_selective_agree_audit = is_reuse
            cfg.timing_batch_sizes = "1,2"
        elif mode == "dry_hardhop":
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
        elif mode == "dry_strathop_polish2_cost_controller":
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
            cfg.load_checkpoint_tag = "dry_strathop_polish2_seed{seed}"
            cfg.load_jumprec_state = False
            cfg.save_checkpoints = True
            cfg.checkpoint_tag = "dry_strathop_polish2_cost_controller_seed{seed}"
            cfg.verifier_false_accept_weight = 2.0
            cfg.verifier_budget_savings_weight = 1.0
            cfg.verifier_ranking_loss_weight = 0.05
            cfg.router_val_batches = 1
            cfg.timing_batch_sizes = "1,2"
        elif mode == "dry_strathop_polish2_calib_controller":
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
            cfg.load_checkpoint_tag = "dry_strathop_polish2_seed{seed}"
            cfg.load_jumprec_state = False
            cfg.save_checkpoints = True
            cfg.checkpoint_tag = "dry_strathop_polish2_calib_controller_seed{seed}"
            cfg.verifier_ranking_loss_weight = 0.02
            cfg.verifier_rank_later_weight = 0.0
            cfg.router_val_batches = 1
            cfg.timing_batch_sizes = "1,2"
        elif mode == "dry_strathop_polish2_budget_controller":
            cfg.n_nodes = 8
            cfg.max_hops = 4
            cfg.max_correct = 3
            cfg.final_steps = 0
            cfg.recurrent_steps = 0
            cfg.jump_steps = 4
            cfg.budget_controller_steps = 4
            cfg.direct_steps = 0
            cfg.hard_hop_fraction = 0.70
            cfg.hard_hop_loss_weight = 2.5
            cfg.final_loop_loss_weight = 8.0
            cfg.load_checkpoints = True
            cfg.load_checkpoint_tag = "dry_strathop_polish2_seed{seed}"
            cfg.load_jumprec_state = False
            cfg.save_checkpoints = True
            cfg.checkpoint_tag = "dry_strathop_polish2_budget_controller_seed{seed}"
            cfg.use_budget_controller = True
            cfg.router_val_batches = 1
            cfg.router_per_budget_audit = True
            cfg.timing_batch_sizes = "1,2"
        elif mode == "dry_strathop_polish2_budget_controller_reuse":
            cfg.n_nodes = 8
            cfg.max_hops = 4
            cfg.max_correct = 3
            cfg.final_steps = 0
            cfg.recurrent_steps = 0
            cfg.jump_steps = 0
            cfg.budget_controller_steps = 0
            cfg.direct_steps = 0
            cfg.hard_hop_fraction = 0.70
            cfg.hard_hop_loss_weight = 2.5
            cfg.final_loop_loss_weight = 8.0
            cfg.load_checkpoints = True
            cfg.checkpoint_tag = "dry_strathop_polish2_budget_controller_seed{seed}"
            cfg.load_jumprec_state = True
            cfg.use_budget_controller = True
            cfg.router_val_batches = 1
            cfg.router_per_budget_audit = True
            cfg.timing_batch_sizes = "1,2"
        elif mode == "dry_strathop_polish2_budget_verifytarget":
            cfg.n_nodes = 8
            cfg.max_hops = 4
            cfg.max_correct = 3
            cfg.final_steps = 0
            cfg.recurrent_steps = 0
            cfg.jump_steps = 4
            cfg.budget_controller_steps = 4
            cfg.direct_steps = 0
            cfg.hard_hop_fraction = 0.70
            cfg.hard_hop_loss_weight = 2.5
            cfg.final_loop_loss_weight = 8.0
            cfg.load_checkpoints = True
            cfg.load_checkpoint_tag = "dry_strathop_polish2_seed{seed}"
            cfg.load_jumprec_state = False
            cfg.save_checkpoints = True
            cfg.checkpoint_tag = "dry_strathop_polish2_budget_verifytarget_seed{seed}"
            cfg.use_budget_controller = True
            cfg.budget_controller_target = "first_acceptable"
            cfg.budget_controller_train_threshold = 0.90
            cfg.budget_controller_under_weight = 0.50
            cfg.budget_controller_over_weight = 0.05
            cfg.router_val_batches = 1
            cfg.router_per_budget_audit = True
            cfg.timing_batch_sizes = "1,2"
        elif mode == "dry_strathop_polish2_budget_verifytarget_reuse":
            cfg.n_nodes = 8
            cfg.max_hops = 4
            cfg.max_correct = 3
            cfg.final_steps = 0
            cfg.recurrent_steps = 0
            cfg.jump_steps = 0
            cfg.budget_controller_steps = 0
            cfg.direct_steps = 0
            cfg.hard_hop_fraction = 0.70
            cfg.hard_hop_loss_weight = 2.5
            cfg.final_loop_loss_weight = 8.0
            cfg.load_checkpoints = True
            cfg.checkpoint_tag = "dry_strathop_polish2_budget_verifytarget_seed{seed}"
            cfg.load_jumprec_state = True
            cfg.use_budget_controller = True
            cfg.budget_controller_target = "first_acceptable"
            cfg.budget_controller_train_threshold = 0.90
            cfg.budget_controller_under_weight = 0.50
            cfg.budget_controller_over_weight = 0.05
            cfg.router_val_batches = 1
            cfg.router_per_budget_audit = True
            cfg.timing_batch_sizes = "1,2"
        elif mode == "dry_strathop_polish2_stability_router":
            cfg.n_nodes = 8
            cfg.max_hops = 4
            cfg.max_correct = 3
            cfg.final_steps = 0
            cfg.recurrent_steps = 0
            cfg.jump_steps = 4
            cfg.stability_steps = 4
            cfg.direct_steps = 0
            cfg.hard_hop_fraction = 0.70
            cfg.hard_hop_loss_weight = 2.5
            cfg.final_loop_loss_weight = 8.0
            cfg.load_checkpoints = True
            cfg.load_checkpoint_tag = "dry_strathop_polish2_seed{seed}"
            cfg.load_jumprec_state = False
            cfg.save_checkpoints = True
            cfg.checkpoint_tag = "dry_strathop_polish2_stability_router_seed{seed}"
            cfg.use_stability_head = True
            cfg.router_val_batches = 1
            cfg.timing_batch_sizes = "1,2"
        elif mode == "dry_strathop_polish2_stability_router_reuse":
            cfg.n_nodes = 8
            cfg.max_hops = 4
            cfg.max_correct = 3
            cfg.final_steps = 0
            cfg.recurrent_steps = 0
            cfg.jump_steps = 0
            cfg.stability_steps = 0
            cfg.direct_steps = 0
            cfg.hard_hop_fraction = 0.70
            cfg.hard_hop_loss_weight = 2.5
            cfg.final_loop_loss_weight = 8.0
            cfg.load_checkpoints = True
            cfg.checkpoint_tag = "dry_strathop_polish2_stability_router_seed{seed}"
            cfg.load_jumprec_state = True
            cfg.use_stability_head = True
            cfg.router_val_batches = 1
            cfg.timing_batch_sizes = "1,2"
        elif mode == "dry_strathop_polish2_utility_router":
            cfg.n_nodes = 8
            cfg.max_hops = 4
            cfg.max_correct = 3
            cfg.final_steps = 0
            cfg.recurrent_steps = 0
            cfg.jump_steps = 4
            cfg.utility_router_steps = 4
            cfg.direct_steps = 0
            cfg.hard_hop_fraction = 0.70
            cfg.hard_hop_loss_weight = 2.5
            cfg.final_loop_loss_weight = 8.0
            cfg.load_checkpoints = True
            cfg.load_checkpoint_tag = "dry_strathop_polish2_seed{seed}"
            cfg.load_jumprec_state = False
            cfg.save_checkpoints = True
            cfg.checkpoint_tag = "dry_strathop_polish2_utility_router_seed{seed}"
            cfg.use_utility_router = True
            cfg.router_val_batches = 1
            cfg.router_threshold_candidates = "0.10,0.30,0.50,0.70,0.90"
            cfg.timing_batch_sizes = "1,2"
        elif mode == "dry_strathop_polish2_utility_router_reuse":
            cfg.n_nodes = 8
            cfg.max_hops = 4
            cfg.max_correct = 3
            cfg.final_steps = 0
            cfg.recurrent_steps = 0
            cfg.jump_steps = 0
            cfg.utility_router_steps = 0
            cfg.direct_steps = 0
            cfg.hard_hop_fraction = 0.70
            cfg.hard_hop_loss_weight = 2.5
            cfg.final_loop_loss_weight = 8.0
            cfg.load_checkpoints = True
            cfg.checkpoint_tag = "dry_strathop_polish2_utility_router_seed{seed}"
            cfg.load_jumprec_state = True
            cfg.use_utility_router = True
            cfg.router_val_batches = 1
            cfg.router_threshold_candidates = "0.10,0.30,0.50,0.70,0.90"
            cfg.timing_batch_sizes = "1,2"
        elif mode == "dry_strathop_polish2_utility_stability_router":
            cfg.n_nodes = 8
            cfg.max_hops = 4
            cfg.max_correct = 3
            cfg.final_steps = 0
            cfg.recurrent_steps = 0
            cfg.jump_steps = 0
            cfg.stability_steps = 0
            cfg.utility_router_steps = 4
            cfg.direct_steps = 0
            cfg.hard_hop_fraction = 0.70
            cfg.hard_hop_loss_weight = 2.5
            cfg.final_loop_loss_weight = 8.0
            cfg.load_checkpoints = True
            cfg.load_checkpoint_tag = "dry_strathop_polish2_stability_router_seed{seed}"
            cfg.load_jumprec_state = True
            cfg.save_checkpoints = True
            cfg.checkpoint_tag = "dry_strathop_polish2_utility_stability_router_seed{seed}"
            cfg.use_stability_head = True
            cfg.use_utility_router = True
            cfg.utility_use_stability_feature = True
            cfg.router_val_batches = 1
            cfg.router_threshold_candidates = "0.10,0.30,0.50,0.70,0.90"
            cfg.timing_batch_sizes = "1,2"
        elif mode == "dry_strathop_polish2_utility_stability_router_reuse":
            cfg.n_nodes = 8
            cfg.max_hops = 4
            cfg.max_correct = 3
            cfg.final_steps = 0
            cfg.recurrent_steps = 0
            cfg.jump_steps = 0
            cfg.stability_steps = 0
            cfg.utility_router_steps = 0
            cfg.direct_steps = 0
            cfg.hard_hop_fraction = 0.70
            cfg.hard_hop_loss_weight = 2.5
            cfg.final_loop_loss_weight = 8.0
            cfg.load_checkpoints = True
            cfg.checkpoint_tag = "dry_strathop_polish2_utility_stability_router_seed{seed}"
            cfg.load_jumprec_state = True
            cfg.use_stability_head = True
            cfg.use_utility_router = True
            cfg.utility_use_stability_feature = True
            cfg.router_val_batches = 1
            cfg.router_threshold_candidates = "0.10,0.30,0.50,0.70,0.90"
            cfg.timing_batch_sizes = "1,2"
        elif mode == "dry_strathop_polish2_nextagree_router":
            cfg.n_nodes = 8
            cfg.max_hops = 4
            cfg.max_correct = 3
            cfg.final_steps = 0
            cfg.recurrent_steps = 0
            cfg.jump_steps = 4
            cfg.next_agreement_steps = 4
            cfg.direct_steps = 0
            cfg.hard_hop_fraction = 0.70
            cfg.hard_hop_loss_weight = 2.5
            cfg.final_loop_loss_weight = 8.0
            cfg.load_checkpoints = True
            cfg.load_checkpoint_tag = "dry_strathop_polish2_seed{seed}"
            cfg.load_jumprec_state = False
            cfg.save_checkpoints = True
            cfg.checkpoint_tag = "dry_strathop_polish2_nextagree_router_seed{seed}"
            cfg.use_next_agreement_head = True
            cfg.router_val_batches = 1
            cfg.router_threshold_candidates = "0.10,0.30,0.50,0.70,0.90"
            cfg.timing_batch_sizes = "1,2"
        elif mode == "dry_strathop_polish2_nextagree_router_reuse":
            cfg.n_nodes = 8
            cfg.max_hops = 4
            cfg.max_correct = 3
            cfg.final_steps = 0
            cfg.recurrent_steps = 0
            cfg.jump_steps = 0
            cfg.next_agreement_steps = 0
            cfg.direct_steps = 0
            cfg.hard_hop_fraction = 0.70
            cfg.hard_hop_loss_weight = 2.5
            cfg.final_loop_loss_weight = 8.0
            cfg.load_checkpoints = True
            cfg.checkpoint_tag = "dry_strathop_polish2_nextagree_router_seed{seed}"
            cfg.load_jumprec_state = True
            cfg.use_next_agreement_head = True
            cfg.router_val_batches = 1
            cfg.router_threshold_candidates = "0.10,0.30,0.50,0.70,0.90"
            cfg.timing_batch_sizes = "1,2"
        elif mode in (
            "dry_strathop_polish2_joint_halt",
            "dry_strathop_polish2_joint_halt_stability",
            "dry_strathop_polish2_joint_halt_quality",
            "dry_strathop_polish2_joint_halt_quality_agdistill",
            "dry_strathop_polish2_joint_halt_slo",
        ):
            use_joint_stability = mode.endswith("_stability")
            use_quality_objective = "_quality" in mode
            use_agreement_distill = "_agdistill" in mode
            use_slo_objective = "_slo" in mode
            cfg.n_nodes = 8
            cfg.max_hops = 4
            cfg.max_correct = 3
            cfg.final_steps = 0
            cfg.recurrent_steps = 0
            cfg.jump_steps = 4
            cfg.joint_halt_steps = 4
            cfg.direct_steps = 0
            cfg.hard_hop_fraction = 0.70
            cfg.hard_hop_loss_weight = 2.5
            cfg.final_loop_loss_weight = 8.0
            cfg.load_checkpoints = True
            cfg.load_checkpoint_tag = "dry_strathop_polish2_seed{seed}"
            cfg.save_checkpoints = True
            cfg.checkpoint_tag = (
                "dry_strathop_polish2_joint_halt_stability_seed{seed}"
                if use_joint_stability
                else "dry_strathop_polish2_joint_halt_quality_agdistill_seed{seed}"
                if use_agreement_distill
                else "dry_strathop_polish2_joint_halt_quality_seed{seed}"
                if use_quality_objective
                else "dry_strathop_polish2_joint_halt_slo_seed{seed}"
                if use_slo_objective
                else "dry_strathop_polish2_joint_halt_seed{seed}"
            )
            cfg.load_jumprec_state = False
            cfg.use_stability_head = use_joint_stability
            cfg.use_utility_router = True
            cfg.utility_use_stability_feature = use_joint_stability
            cfg.utility_false_accept_weight = 6.0
            cfg.utility_cost_weight = 0.20
            cfg.utility_correctness_bce_weight = 0.10
            if use_quality_objective:
                cfg.utility_false_accept_weight = 12.0
                cfg.utility_cost_weight = 0.08
                cfg.joint_halt_candidate_ce_weight = 1.00
                cfg.joint_halt_candidate_distill_weight = 0.15
                cfg.joint_halt_verifier_bce_weight = 0.08
                cfg.joint_halt_agreement_bce_weight = 0.08
            if use_agreement_distill:
                cfg.joint_halt_agreement_distill_weight = 0.10
                cfg.joint_halt_agreement_route_weight = 0.75
            if use_slo_objective:
                cfg.utility_false_accept_weight = 8.0
                cfg.utility_cost_weight = 0.12
                cfg.joint_halt_false_accept_weight_max = 18.0
                cfg.joint_halt_cost_weight_min = 0.04
                cfg.joint_halt_cost_weight_max = 0.18
                cfg.joint_halt_candidate_ce_weight = 1.00
                cfg.joint_halt_candidate_distill_weight = 0.15
                cfg.joint_halt_verifier_bce_weight = 0.08
                cfg.joint_halt_agreement_bce_weight = 0.10
            cfg.router_val_batches = 1
            cfg.router_threshold_candidates = "0.10,0.30,0.50,0.70,0.90"
            cfg.timing_batch_sizes = "1,2"
            cfg.router_probe_audit = use_joint_stability
            cfg.router_selective_agree_audit = use_joint_stability
            cfg.router_probe_steps = 12
        elif mode in (
            "dry_strathop_polish2_joint_halt_reuse",
            "dry_strathop_polish2_joint_halt_stability_reuse",
            "dry_strathop_polish2_joint_halt_quality_reuse",
            "dry_strathop_polish2_joint_halt_quality_agdistill_reuse",
            "dry_strathop_polish2_joint_halt_slo_reuse",
        ):
            use_joint_stability = mode.endswith("_stability_reuse")
            use_quality_objective = "_quality" in mode
            use_agreement_distill = "_agdistill" in mode
            use_slo_objective = "_slo" in mode
            cfg.n_nodes = 8
            cfg.max_hops = 4
            cfg.max_correct = 3
            cfg.final_steps = 0
            cfg.recurrent_steps = 0
            cfg.jump_steps = 0
            cfg.joint_halt_steps = 0
            cfg.direct_steps = 0
            cfg.hard_hop_fraction = 0.70
            cfg.hard_hop_loss_weight = 2.5
            cfg.final_loop_loss_weight = 8.0
            cfg.load_checkpoints = True
            cfg.checkpoint_tag = (
                "dry_strathop_polish2_joint_halt_stability_seed{seed}"
                if use_joint_stability
                else "dry_strathop_polish2_joint_halt_quality_agdistill_seed{seed}"
                if use_agreement_distill
                else "dry_strathop_polish2_joint_halt_quality_seed{seed}"
                if use_quality_objective
                else "dry_strathop_polish2_joint_halt_slo_seed{seed}"
                if use_slo_objective
                else "dry_strathop_polish2_joint_halt_seed{seed}"
            )
            cfg.load_jumprec_state = True
            cfg.use_stability_head = use_joint_stability
            cfg.use_utility_router = True
            cfg.utility_use_stability_feature = use_joint_stability
            cfg.utility_false_accept_weight = 6.0
            cfg.utility_cost_weight = 0.20
            cfg.utility_correctness_bce_weight = 0.10
            if use_quality_objective:
                cfg.utility_false_accept_weight = 12.0
                cfg.utility_cost_weight = 0.08
                cfg.joint_halt_candidate_ce_weight = 1.00
                cfg.joint_halt_candidate_distill_weight = 0.15
                cfg.joint_halt_verifier_bce_weight = 0.08
                cfg.joint_halt_agreement_bce_weight = 0.08
            if use_agreement_distill:
                cfg.joint_halt_agreement_distill_weight = 0.10
                cfg.joint_halt_agreement_route_weight = 0.75
            if use_slo_objective:
                cfg.utility_false_accept_weight = 8.0
                cfg.utility_cost_weight = 0.12
                cfg.joint_halt_false_accept_weight_max = 18.0
                cfg.joint_halt_cost_weight_min = 0.04
                cfg.joint_halt_cost_weight_max = 0.18
                cfg.joint_halt_candidate_ce_weight = 1.00
                cfg.joint_halt_candidate_distill_weight = 0.15
                cfg.joint_halt_verifier_bce_weight = 0.08
                cfg.joint_halt_agreement_bce_weight = 0.10
            cfg.router_val_batches = 1
            cfg.router_threshold_candidates = "0.10,0.30,0.50,0.70,0.90"
            cfg.timing_batch_sizes = "1,2"
            cfg.router_probe_audit = use_joint_stability
            cfg.router_selective_agree_audit = use_joint_stability
            cfg.router_probe_steps = 12
        elif mode in (
            "dry_strathop_polish2_joint_halt_quality_cats",
            "dry_strathop_polish2_joint_halt_quality_cats_reuse",
        ):
            is_cats_reuse = mode.endswith("_reuse")
            cfg.n_nodes = 8
            cfg.max_hops = 4
            cfg.max_correct = 3
            cfg.final_steps = 0
            cfg.recurrent_steps = 0
            cfg.jump_steps = 0
            cfg.joint_halt_steps = 0
            cfg.consistency_steps = 0 if is_cats_reuse else 4
            cfg.direct_steps = 0
            cfg.hard_hop_fraction = 0.70
            cfg.hard_hop_loss_weight = 2.5
            cfg.final_loop_loss_weight = 8.0
            cfg.load_checkpoints = True
            cfg.load_checkpoint_tag = (
                "dry_strathop_polish2_joint_halt_quality_cats_seed{seed}"
                if is_cats_reuse
                else "dry_strathop_polish2_joint_halt_quality_seed{seed}"
            )
            cfg.checkpoint_tag = "dry_strathop_polish2_joint_halt_quality_cats_seed{seed}"
            cfg.save_checkpoints = not is_cats_reuse
            cfg.load_jumprec_state = True
            cfg.use_utility_router = True
            cfg.use_consistency_head = True
            cfg.utility_false_accept_weight = 12.0
            cfg.utility_cost_weight = 0.08
            cfg.utility_correctness_bce_weight = 0.10
            cfg.joint_halt_candidate_ce_weight = 1.00
            cfg.joint_halt_candidate_distill_weight = 0.15
            cfg.joint_halt_verifier_bce_weight = 0.08
            cfg.joint_halt_agreement_bce_weight = 0.08
            cfg.router_val_batches = 1
            cfg.router_threshold_candidates = "0.10,0.30,0.50,0.70,0.90"
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
        cfg.save_checkpoints = not is_budget_reuse
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
    elif mode == "core3_8n4h_natgraph_teacher":
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.max_length = 224
        cfg.prompt_style = "natural_graph"
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 6500
        cfg.recurrent_steps = 10000
        cfg.hop_sample_weights = "0.10,0.20,0.35,0.35"
        cfg.hop_loss_weights = "1.0,1.2,2.0,2.0"
        cfg.final_loop_loss_weight = 4.0
        cfg.jump_steps = 0
        cfg.direct_steps = 0
        cfg.save_checkpoints = True
        cfg.checkpoint_tag = "core3_8n4h_natgraph_seed{seed}"
        cfg.teacher_val_every = 500
        cfg.teacher_val_batches = 16
        cfg.teacher_gate_min_full = 0.995
        cfg.teacher_gate_min_worst_hop = 0.98
        cfg.eval_batches = 96
        cfg.log_every = 500
    elif mode == "core3_8n4h_natgraph_teacher_resume":
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.max_length = 224
        cfg.prompt_style = "natural_graph"
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 0
        cfg.recurrent_steps = 9500
        cfg.hop_sample_weights = "0.10,0.20,0.35,0.35"
        cfg.hop_loss_weights = "1.0,1.2,2.0,2.0"
        cfg.final_loop_loss_weight = 4.0
        cfg.jump_steps = 0
        cfg.direct_steps = 0
        cfg.load_checkpoints = True
        cfg.resume_teacher_training = True
        cfg.load_checkpoint_tag = "core3_8n4h_natgraph_seed{seed}"
        cfg.save_checkpoints = True
        cfg.checkpoint_tag = "core3_8n4h_natgraph_seed{seed}"
        cfg.teacher_val_every = 500
        cfg.teacher_val_batches = 16
        cfg.teacher_gate_min_full = 0.995
        cfg.teacher_gate_min_worst_hop = 0.98
        cfg.eval_batches = 96
        cfg.log_every = 500
    elif mode == "core3_8n4h_natgraph_polish_teacher":
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.max_length = 224
        cfg.prompt_style = "natural_graph"
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 0
        cfg.recurrent_steps = 6000
        cfg.lr_blocks = 2e-5
        cfg.lr_head = 1.5e-4
        cfg.hard_hop_fraction = 0.70
        cfg.hard_hop_loss_weight = 2.5
        cfg.final_loop_loss_weight = 8.0
        cfg.jump_steps = 0
        cfg.direct_steps = 0
        cfg.load_checkpoints = True
        cfg.resume_teacher_training = True
        cfg.load_checkpoint_tag = "core3_8n4h_natgraph_seed{seed}"
        cfg.save_checkpoints = True
        cfg.checkpoint_tag = "core3_8n4h_natgraph_polish_seed{seed}"
        cfg.teacher_val_every = 250
        cfg.teacher_val_batches = 16
        cfg.teacher_gate_min_full = 0.995
        cfg.teacher_gate_min_worst_hop = 0.98
        cfg.eval_batches = 96
        cfg.log_every = 500
    elif mode == "core3_8n4h_natgraph_polish_eval_teacher":
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.max_length = 224
        cfg.prompt_style = "natural_graph"
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 0
        cfg.recurrent_steps = 0
        cfg.lr_blocks = 2e-5
        cfg.lr_head = 1.5e-4
        cfg.hard_hop_fraction = 0.70
        cfg.hard_hop_loss_weight = 2.5
        cfg.final_loop_loss_weight = 8.0
        cfg.jump_steps = 0
        cfg.direct_steps = 0
        cfg.load_checkpoints = True
        cfg.checkpoint_tag = "core3_8n4h_natgraph_polish_seed{seed}"
        cfg.eval_batches = 256
        cfg.timing_batches = 16
        cfg.log_every = 500
    elif mode == "core3_8n4h_natgraph_polish2_teacher":
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.max_length = 224
        cfg.prompt_style = "natural_graph"
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 0
        cfg.recurrent_steps = 4000
        cfg.lr_blocks = 1e-5
        cfg.lr_head = 7.5e-5
        cfg.hard_hop_fraction = 0.80
        cfg.hard_hop_loss_weight = 3.0
        cfg.final_loop_loss_weight = 10.0
        cfg.jump_steps = 0
        cfg.direct_steps = 0
        cfg.load_checkpoints = True
        cfg.resume_teacher_training = True
        cfg.load_checkpoint_tag = "core3_8n4h_natgraph_polish_seed{seed}"
        cfg.save_checkpoints = True
        cfg.checkpoint_tag = "core3_8n4h_natgraph_polish2_seed{seed}"
        cfg.teacher_val_every = 250
        cfg.teacher_val_batches = 16
        cfg.teacher_gate_min_full = 0.995
        cfg.teacher_gate_min_worst_hop = 0.98
        cfg.eval_batches = 96
        cfg.log_every = 500
    elif mode == "core3_8n4h_natgraph_polish2_eval_teacher":
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.max_length = 224
        cfg.prompt_style = "natural_graph"
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 0
        cfg.recurrent_steps = 0
        cfg.lr_blocks = 1e-5
        cfg.lr_head = 7.5e-5
        cfg.hard_hop_fraction = 0.80
        cfg.hard_hop_loss_weight = 3.0
        cfg.final_loop_loss_weight = 10.0
        cfg.jump_steps = 0
        cfg.direct_steps = 0
        cfg.load_checkpoints = True
        cfg.checkpoint_tag = "core3_8n4h_natgraph_polish2_seed{seed}"
        cfg.eval_batches = 256
        cfg.timing_batches = 16
        cfg.log_every = 500
    elif mode == "core3_8n4h_natgraph_polish2_audit_teacher":
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.max_length = 224
        cfg.prompt_style = "natural_graph"
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 0
        cfg.recurrent_steps = 0
        cfg.lr_blocks = 1e-5
        cfg.lr_head = 7.5e-5
        cfg.hard_hop_fraction = 0.80
        cfg.hard_hop_loss_weight = 3.0
        cfg.final_loop_loss_weight = 10.0
        cfg.jump_steps = 0
        cfg.direct_steps = 0
        cfg.load_checkpoints = True
        cfg.checkpoint_tag = "core3_8n4h_natgraph_polish2_seed{seed}"
        cfg.audit_prompt_variants = "normal,relabel,map_scramble,hop_random"
        cfg.eval_batches = 256
        cfg.timing_batches = 8
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
    elif mode == "core3_8n4h_natgraph_eval_teacher":
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.max_length = 224
        cfg.prompt_style = "natural_graph"
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
        cfg.checkpoint_tag = "core3_8n4h_natgraph_seed{seed}"
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
        cfg.timing_batch_sizes = "1,2,4,8,16,32,64"
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
        cfg.timing_batch_sizes = "1,2,4,8,16,32,64"
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
        "core3_8n4h_strathop_cost_controller",
        "core3_8n4h_strathop_polish2_cost_controller",
        "core3_8n4h_strathop_calib_controller",
        "core3_8n4h_strathop_polish2_calib_controller",
        "core3_8n4h_strathop_budget_controller",
        "core3_8n4h_strathop_polish2_budget_controller",
        "core3_8n4h_strathop_budget_controller_reuse",
        "core3_8n4h_strathop_polish2_budget_controller_reuse",
        "core3_8n4h_strathop_budget_verifytarget",
        "core3_8n4h_strathop_polish2_budget_verifytarget",
        "core3_8n4h_strathop_budget_verifytarget_reuse",
        "core3_8n4h_strathop_polish2_budget_verifytarget_reuse",
    ):
        is_polish2_controller = mode.startswith("core3_8n4h_strathop_polish2")
        is_budget_controller = "_budget_controller" in mode or "_budget_verifytarget" in mode
        is_budget_reuse = mode.endswith("_budget_controller_reuse") or mode.endswith("_budget_verifytarget_reuse")
        is_budget_verifytarget = "_budget_verifytarget" in mode
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 0
        cfg.recurrent_steps = 0
        if is_polish2_controller:
            cfg.hard_hop_fraction = 0.70
            cfg.hard_hop_loss_weight = 2.5
            cfg.final_loop_loss_weight = 8.0
            cfg.load_checkpoint_tag = (
                (
                    "core3_8n4h_strathop_polish2_budget_verifytarget_seed{seed}"
                    if is_budget_verifytarget
                    else "core3_8n4h_strathop_polish2_budget_controller_seed{seed}"
                )
                if is_budget_reuse
                else "core3_8n4h_strathop_polish2_seed{seed}"
            )
        else:
            cfg.hop_sample_weights = "0.10,0.20,0.35,0.35"
            cfg.hop_loss_weights = "1.0,1.2,2.0,2.0"
            cfg.final_loop_loss_weight = 4.0
            cfg.load_checkpoint_tag = (
                (
                    "core3_8n4h_strathop_budget_verifytarget_seed{seed}"
                    if is_budget_verifytarget
                    else "core3_8n4h_strathop_budget_controller_seed{seed}"
                )
                if is_budget_reuse
                else "core3_8n4h_strathop_seed{seed}"
            )
        cfg.jump_steps = 0 if is_budget_controller else 4500
        cfg.budget_controller_steps = 0 if is_budget_reuse else (2000 if is_budget_controller else 0)
        cfg.direct_steps = 0
        cfg.direct_layers = 3
        cfg.strict_need_agreement = False
        cfg.load_checkpoints = True
        cfg.save_checkpoints = True
        cfg.checkpoint_tag = f"{mode}_seed{{seed}}"
        if is_budget_controller:
            cfg.use_budget_controller = True
            cfg.load_jumprec_state = True
            if is_budget_verifytarget:
                cfg.budget_controller_target = "first_acceptable"
                cfg.budget_controller_train_threshold = 0.90
                cfg.budget_controller_under_weight = 0.50
                cfg.budget_controller_over_weight = 0.05
        else:
            cfg.load_jumprec_state = False
        if "_calib_controller" in mode:
            cfg.verifier_ranking_loss_weight = 0.02
            cfg.verifier_rank_later_weight = 0.0
        elif "_cost_controller" in mode:
            cfg.verifier_false_accept_weight = 2.0
            cfg.verifier_budget_savings_weight = 1.0
            cfg.verifier_ranking_loss_weight = 0.05
        cfg.router_val_batches = 64
        cfg.router_per_budget_audit = True
        cfg.timing_batches = 16
        cfg.timing_batch_sizes = "1,2,4,8,16,32,64"
        cfg.eval_batches = 128
        cfg.log_every = 500
    elif mode in (
        "core3_8n4h_strathop_stability_router",
        "core3_8n4h_strathop_polish2_stability_router",
        "core3_8n4h_strathop_stability_router_reuse",
        "core3_8n4h_strathop_polish2_stability_router_reuse",
    ):
        is_polish2_stability = mode.startswith("core3_8n4h_strathop_polish2")
        is_stability_reuse = mode.endswith("_stability_router_reuse")
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 0
        cfg.recurrent_steps = 0
        if is_polish2_stability:
            cfg.hard_hop_fraction = 0.70
            cfg.hard_hop_loss_weight = 2.5
            cfg.final_loop_loss_weight = 8.0
            cfg.load_checkpoint_tag = (
                "core3_8n4h_strathop_polish2_stability_router_seed{seed}"
                if is_stability_reuse
                else "core3_8n4h_strathop_polish2_seed{seed}"
            )
        else:
            cfg.hop_sample_weights = "0.10,0.20,0.35,0.35"
            cfg.hop_loss_weights = "1.0,1.2,2.0,2.0"
            cfg.final_loop_loss_weight = 4.0
            cfg.load_checkpoint_tag = (
                "core3_8n4h_strathop_stability_router_seed{seed}"
                if is_stability_reuse
                else "core3_8n4h_strathop_seed{seed}"
            )
        cfg.jump_steps = 0
        cfg.stability_steps = 0 if is_stability_reuse else 2000
        cfg.direct_steps = 0
        cfg.direct_layers = 3
        cfg.strict_need_agreement = False
        cfg.load_checkpoints = True
        cfg.save_checkpoints = not is_stability_reuse
        cfg.checkpoint_tag = f"{mode}_seed{{seed}}"
        cfg.load_jumprec_state = True
        cfg.use_stability_head = True
        cfg.router_val_batches = 64
        cfg.timing_batches = 16
        cfg.timing_batch_sizes = "1,2,4,8,16,32,64"
        cfg.eval_batches = 128
        cfg.log_every = 500
    elif mode in (
        "core3_8n4h_strathop_utility_router",
        "core3_8n4h_strathop_polish2_utility_router",
        "core3_8n4h_strathop_utility_router_reuse",
        "core3_8n4h_strathop_polish2_utility_router_reuse",
    ):
        is_polish2_utility = mode.startswith("core3_8n4h_strathop_polish2")
        is_utility_reuse = mode.endswith("_utility_router_reuse")
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 0
        cfg.recurrent_steps = 0
        if is_polish2_utility:
            cfg.hard_hop_fraction = 0.70
            cfg.hard_hop_loss_weight = 2.5
            cfg.final_loop_loss_weight = 8.0
            cfg.load_checkpoint_tag = (
                "core3_8n4h_strathop_polish2_utility_router_seed{seed}"
                if is_utility_reuse
                else "core3_8n4h_strathop_polish2_seed{seed}"
            )
        else:
            cfg.hop_sample_weights = "0.10,0.20,0.35,0.35"
            cfg.hop_loss_weights = "1.0,1.2,2.0,2.0"
            cfg.final_loop_loss_weight = 4.0
            cfg.load_checkpoint_tag = (
                "core3_8n4h_strathop_utility_router_seed{seed}"
                if is_utility_reuse
                else "core3_8n4h_strathop_seed{seed}"
            )
        cfg.jump_steps = 0
        cfg.utility_router_steps = 0 if is_utility_reuse else 2000
        cfg.direct_steps = 0
        cfg.direct_layers = 3
        cfg.strict_need_agreement = False
        cfg.load_checkpoints = True
        cfg.save_checkpoints = not is_utility_reuse
        cfg.checkpoint_tag = f"{mode}_seed{{seed}}"
        cfg.load_jumprec_state = True
        cfg.use_utility_router = True
        cfg.router_val_batches = 64
        cfg.router_threshold_candidates = "0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.85,0.90,0.95,0.97,0.99"
        cfg.timing_batches = 16
        cfg.timing_batch_sizes = "1,2,4,8,16,32,64"
        cfg.eval_batches = 128
        cfg.log_every = 500
    elif mode in (
        "core3_8n4h_strathop_utility_stability_router",
        "core3_8n4h_strathop_polish2_utility_stability_router",
        "core3_8n4h_strathop_utility_stability_router_reuse",
        "core3_8n4h_strathop_polish2_utility_stability_router_reuse",
    ):
        is_polish2_utility_stability = mode.startswith("core3_8n4h_strathop_polish2")
        is_utility_stability_reuse = mode.endswith("_utility_stability_router_reuse")
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 0
        cfg.recurrent_steps = 0
        if is_polish2_utility_stability:
            cfg.hard_hop_fraction = 0.70
            cfg.hard_hop_loss_weight = 2.5
            cfg.final_loop_loss_weight = 8.0
            cfg.load_checkpoint_tag = (
                "core3_8n4h_strathop_polish2_utility_stability_router_seed{seed}"
                if is_utility_stability_reuse
                else "core3_8n4h_strathop_polish2_stability_router_seed{seed}"
            )
        else:
            cfg.hop_sample_weights = "0.10,0.20,0.35,0.35"
            cfg.hop_loss_weights = "1.0,1.2,2.0,2.0"
            cfg.final_loop_loss_weight = 4.0
            cfg.load_checkpoint_tag = (
                "core3_8n4h_strathop_utility_stability_router_seed{seed}"
                if is_utility_stability_reuse
                else "core3_8n4h_strathop_stability_router_seed{seed}"
            )
        cfg.jump_steps = 0
        cfg.stability_steps = 0
        cfg.utility_router_steps = 0 if is_utility_stability_reuse else 2000
        cfg.direct_steps = 0
        cfg.direct_layers = 3
        cfg.strict_need_agreement = False
        cfg.load_checkpoints = True
        cfg.save_checkpoints = not is_utility_stability_reuse
        cfg.checkpoint_tag = f"{mode}_seed{{seed}}"
        cfg.load_jumprec_state = True
        cfg.use_stability_head = True
        cfg.use_utility_router = True
        cfg.utility_use_stability_feature = True
        cfg.router_val_batches = 64
        cfg.router_threshold_candidates = "0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.85,0.90,0.95,0.97,0.99"
        cfg.timing_batches = 16
        cfg.timing_batch_sizes = "1,2,4,8,16,32,64"
        cfg.eval_batches = 128
        cfg.log_every = 500
    elif mode in (
        "core3_8n4h_strathop_nextagree_router",
        "core3_8n4h_strathop_polish2_nextagree_router",
        "core3_8n4h_strathop_nextagree_router_reuse",
        "core3_8n4h_strathop_polish2_nextagree_router_reuse",
    ):
        is_polish2_nextagree = mode.startswith("core3_8n4h_strathop_polish2")
        is_nextagree_reuse = mode.endswith("_nextagree_router_reuse")
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 0
        cfg.recurrent_steps = 0
        if is_polish2_nextagree:
            cfg.hard_hop_fraction = 0.70
            cfg.hard_hop_loss_weight = 2.5
            cfg.final_loop_loss_weight = 8.0
            cfg.load_checkpoint_tag = (
                "core3_8n4h_strathop_polish2_nextagree_router_seed{seed}"
                if is_nextagree_reuse
                else "core3_8n4h_strathop_polish2_seed{seed}"
            )
        else:
            cfg.hop_sample_weights = "0.10,0.20,0.35,0.35"
            cfg.hop_loss_weights = "1.0,1.2,2.0,2.0"
            cfg.final_loop_loss_weight = 4.0
            cfg.load_checkpoint_tag = (
                "core3_8n4h_strathop_nextagree_router_seed{seed}"
                if is_nextagree_reuse
                else "core3_8n4h_strathop_seed{seed}"
            )
        cfg.jump_steps = 0
        cfg.next_agreement_steps = 0 if is_nextagree_reuse else 2000
        cfg.direct_steps = 0
        cfg.direct_layers = 3
        cfg.strict_need_agreement = False
        cfg.load_checkpoints = True
        cfg.save_checkpoints = not is_nextagree_reuse
        cfg.checkpoint_tag = f"{mode}_seed{{seed}}"
        cfg.load_jumprec_state = True
        cfg.use_next_agreement_head = True
        cfg.router_val_batches = 64
        cfg.router_threshold_candidates = "0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.85,0.90,0.95,0.97,0.99"
        cfg.timing_batches = 16
        cfg.timing_batch_sizes = "1,2,4,8,16,32,64"
        cfg.eval_batches = 128
        cfg.log_every = 500
    elif mode in (
        "core3_8n4h_strathop_joint_halt_quality_cats",
        "core3_8n4h_strathop_polish2_joint_halt_quality_cats",
        "core3_8n4h_strathop_joint_halt_quality_cats_reuse",
        "core3_8n4h_strathop_polish2_joint_halt_quality_cats_reuse",
        "core3_8n4h_strathop_joint_halt_quality_cats_reuse_highval",
        "core3_8n4h_strathop_polish2_joint_halt_quality_cats_reuse_highval",
    ):
        is_polish2_cats = mode.startswith("core3_8n4h_strathop_polish2")
        is_cats_reuse = "_reuse" in mode
        is_highval_reuse = mode.endswith("_reuse_highval")
        family_prefix = "core3_8n4h_strathop_polish2" if is_polish2_cats else "core3_8n4h_strathop"
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 0
        cfg.recurrent_steps = 0
        if is_polish2_cats:
            cfg.hard_hop_fraction = 0.70
            cfg.hard_hop_loss_weight = 2.5
            cfg.final_loop_loss_weight = 8.0
        else:
            cfg.hop_sample_weights = "0.10,0.20,0.35,0.35"
            cfg.hop_loss_weights = "1.0,1.2,2.0,2.0"
            cfg.final_loop_loss_weight = 4.0
        cfg.jump_steps = 0
        cfg.joint_halt_steps = 0
        cfg.consistency_steps = 0 if is_cats_reuse else 2000
        cfg.direct_steps = 0
        cfg.direct_layers = 3
        cfg.strict_need_agreement = False
        cfg.load_checkpoints = True
        cfg.load_checkpoint_tag = (
            f"{family_prefix}_joint_halt_quality_cats_seed{{seed}}"
            if is_cats_reuse
            else f"{family_prefix}_joint_halt_quality_seed{{seed}}"
        )
        cfg.checkpoint_tag = f"{family_prefix}_joint_halt_quality_cats_seed{{seed}}"
        cfg.save_checkpoints = not is_cats_reuse
        cfg.load_jumprec_state = True
        cfg.use_utility_router = True
        cfg.use_consistency_head = True
        cfg.utility_false_accept_weight = 12.0
        cfg.utility_cost_weight = 0.08
        cfg.utility_correctness_bce_weight = 0.10
        cfg.joint_halt_candidate_ce_weight = 1.00
        cfg.joint_halt_candidate_distill_weight = 0.15
        cfg.joint_halt_verifier_bce_weight = 0.08
        cfg.joint_halt_agreement_bce_weight = 0.08
        cfg.router_val_batches = 64
        cfg.router_threshold_candidates = "0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.85,0.90,0.95,0.97,0.99"
        cfg.timing_batches = 16
        cfg.timing_batch_sizes = "1,2,4,8,16,32,64"
        cfg.eval_batches = 128
        if is_highval_reuse:
            cfg.router_val_batches = 256
            cfg.eval_batches = 256
            cfg.router_threshold_candidates = (
                "0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,"
                "0.50,0.60,0.70,0.80,0.85,0.90,0.95,0.97,0.99"
            )
            cfg.timing_batches = 8
            cfg.timing_batch_sizes = "1,64"
        cfg.log_every = 500
    elif mode in (
        "core3_8n4h_natgraph_joint_halt_quality_stability",
        "core3_8n4h_natgraph_joint_halt_quality_stability_reuse",
        "core3_8n4h_natgraph_joint_halt_quality_stability_reuse_audit",
        "core3_8n4h_natgraph_joint_halt_quality_stability_reuse_highval",
        "core3_8n4h_natgraph_polish2_joint_halt_quality_stability",
        "core3_8n4h_natgraph_polish2_joint_halt_quality_stability_reuse",
        "core3_8n4h_natgraph_polish2_joint_halt_quality_stability_reuse_audit",
        "core3_8n4h_natgraph_polish2_joint_halt_quality_stability_reuse_highval",
    ):
        is_natgraph_reuse = "_reuse" in mode
        is_audit_reuse = mode.endswith("_reuse_audit")
        is_highval_reuse = mode.endswith("_reuse_highval")
        is_polish2_natgraph = mode.startswith("core3_8n4h_natgraph_polish2_")
        natgraph_prefix = "core3_8n4h_natgraph_polish2" if is_polish2_natgraph else "core3_8n4h_natgraph"
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.max_length = 224
        cfg.prompt_style = "natural_graph"
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 0
        cfg.recurrent_steps = 0
        if is_polish2_natgraph:
            cfg.hard_hop_fraction = 0.80
            cfg.hard_hop_loss_weight = 3.0
            cfg.final_loop_loss_weight = 10.0
        else:
            cfg.hop_sample_weights = "0.10,0.20,0.35,0.35"
            cfg.hop_loss_weights = "1.0,1.2,2.0,2.0"
            cfg.final_loop_loss_weight = 4.0
        cfg.load_checkpoint_tag = (
            f"{natgraph_prefix}_joint_halt_quality_stability_seed{{seed}}"
            if is_natgraph_reuse
            else f"{natgraph_prefix}_seed{{seed}}"
        )
        cfg.jump_steps = 0 if is_natgraph_reuse else 4500
        cfg.joint_halt_steps = 0 if is_natgraph_reuse else 2000
        cfg.direct_steps = 0
        cfg.direct_layers = 3
        cfg.strict_need_agreement = False
        cfg.load_checkpoints = True
        cfg.save_checkpoints = not is_natgraph_reuse
        cfg.checkpoint_tag = f"{natgraph_prefix}_joint_halt_quality_stability_seed{{seed}}"
        cfg.load_jumprec_state = is_natgraph_reuse
        cfg.use_stability_head = True
        cfg.use_utility_router = True
        cfg.utility_use_stability_feature = True
        cfg.utility_false_accept_weight = 12.0
        cfg.utility_cost_weight = 0.08
        cfg.utility_correctness_bce_weight = 0.10
        cfg.joint_halt_candidate_ce_weight = 1.00
        cfg.joint_halt_candidate_distill_weight = 0.15
        cfg.joint_halt_verifier_bce_weight = 0.08
        cfg.joint_halt_stability_weight = 0.05
        cfg.joint_halt_agreement_bce_weight = 0.08
        cfg.router_val_batches = 64
        cfg.router_threshold_candidates = "0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.85,0.90,0.95,0.97,0.99"
        cfg.timing_batches = 16
        cfg.timing_batch_sizes = "1,2,4,8,16,32,64"
        cfg.eval_batches = 128
        if is_highval_reuse:
            cfg.router_val_batches = 256
            cfg.eval_batches = 256
            cfg.router_threshold_candidates = (
                "0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,"
                "0.50,0.60,0.70,0.80,0.85,0.90,0.95,0.97,0.99"
            )
            cfg.timing_batches = 8
            cfg.timing_batch_sizes = "1,64"
            cfg.router_probe_audit = True
            cfg.router_selective_agree_audit = True
        if is_audit_reuse:
            cfg.router_val_batches = 0
            cfg.eval_batches = 256
            cfg.audit_prompt_variants = "normal,relabel,map_scramble,hop_random"
            cfg.timing_batches = 4
            cfg.timing_batch_sizes = "1,64"
        cfg.log_every = 500
    elif mode in (
        "core3_8n4h_strathop_joint_halt",
        "core3_8n4h_strathop_polish2_joint_halt",
        "core3_8n4h_strathop_joint_halt_reuse",
        "core3_8n4h_strathop_polish2_joint_halt_reuse",
        "core3_8n4h_strathop_joint_halt_stability",
        "core3_8n4h_strathop_polish2_joint_halt_stability",
        "core3_8n4h_strathop_joint_halt_stability_reuse",
        "core3_8n4h_strathop_polish2_joint_halt_stability_reuse",
        "core3_8n4h_strathop_joint_halt_reuse_highval",
        "core3_8n4h_strathop_polish2_joint_halt_reuse_highval",
        "core3_8n4h_strathop_joint_halt_stability_reuse_highval",
        "core3_8n4h_strathop_polish2_joint_halt_stability_reuse_highval",
        "core3_8n4h_strathop_joint_halt_quality",
        "core3_8n4h_strathop_polish2_joint_halt_quality",
        "core3_8n4h_strathop_joint_halt_quality_reuse",
        "core3_8n4h_strathop_polish2_joint_halt_quality_reuse",
        "core3_8n4h_strathop_joint_halt_quality_reuse_highval",
        "core3_8n4h_strathop_polish2_joint_halt_quality_reuse_highval",
        "core3_8n4h_strathop_joint_halt_quality_agdistill",
        "core3_8n4h_strathop_polish2_joint_halt_quality_agdistill",
        "core3_8n4h_strathop_joint_halt_quality_agdistill_reuse",
        "core3_8n4h_strathop_polish2_joint_halt_quality_agdistill_reuse",
        "core3_8n4h_strathop_joint_halt_quality_agdistill_reuse_highval",
        "core3_8n4h_strathop_polish2_joint_halt_quality_agdistill_reuse_highval",
        "core3_8n4h_strathop_joint_halt_quality_stability",
        "core3_8n4h_strathop_polish2_joint_halt_quality_stability",
        "core3_8n4h_strathop_joint_halt_quality_stability_reuse",
        "core3_8n4h_strathop_polish2_joint_halt_quality_stability_reuse",
        "core3_8n4h_strathop_joint_halt_quality_stability_reuse_highval",
        "core3_8n4h_strathop_polish2_joint_halt_quality_stability_reuse_highval",
        "core3_8n4h_strathop_joint_halt_slo",
        "core3_8n4h_strathop_polish2_joint_halt_slo",
        "core3_8n4h_strathop_joint_halt_slo_reuse",
        "core3_8n4h_strathop_polish2_joint_halt_slo_reuse",
        "core3_8n4h_strathop_joint_halt_slo_reuse_highval",
        "core3_8n4h_strathop_polish2_joint_halt_slo_reuse_highval",
        "core3_8n4h_strathop_joint_halt_slo_stability",
        "core3_8n4h_strathop_polish2_joint_halt_slo_stability",
        "core3_8n4h_strathop_joint_halt_slo_stability_reuse",
        "core3_8n4h_strathop_polish2_joint_halt_slo_stability_reuse",
        "core3_8n4h_strathop_joint_halt_slo_stability_reuse_highval",
        "core3_8n4h_strathop_polish2_joint_halt_slo_stability_reuse_highval",
    ):
        is_polish2_joint = mode.startswith("core3_8n4h_strathop_polish2")
        is_joint_reuse = "_reuse" in mode
        is_highval_reuse = mode.endswith("_reuse_highval")
        use_joint_stability = "_stability" in mode
        use_quality_objective = "_quality" in mode
        use_agreement_distill = "_agdistill" in mode
        use_slo_objective = "_slo" in mode
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 0
        cfg.recurrent_steps = 0
        family_prefix = "core3_8n4h_strathop_polish2" if is_polish2_joint else "core3_8n4h_strathop"
        base_teacher_tag = (
            "core3_8n4h_strathop_polish2_seed{seed}"
            if is_polish2_joint
            else "core3_8n4h_strathop_seed{seed}"
        )
        if use_quality_objective:
            reuse_tag = (
                f"{family_prefix}_joint_halt_quality_stability_seed{{seed}}"
                if use_joint_stability
                else f"{family_prefix}_joint_halt_quality_agdistill_seed{{seed}}"
                if use_agreement_distill
                else f"{family_prefix}_joint_halt_quality_seed{{seed}}"
            )
        elif use_slo_objective:
            reuse_tag = (
                f"{family_prefix}_joint_halt_slo_stability_seed{{seed}}"
                if use_joint_stability
                else f"{family_prefix}_joint_halt_slo_seed{{seed}}"
            )
        else:
            reuse_tag = (
                f"{family_prefix}_joint_halt_stability_seed{{seed}}"
                if use_joint_stability
                else f"{family_prefix}_joint_halt_seed{{seed}}"
            )
        if is_polish2_joint:
            cfg.hard_hop_fraction = 0.70
            cfg.hard_hop_loss_weight = 2.5
            cfg.final_loop_loss_weight = 8.0
        else:
            cfg.hop_sample_weights = "0.10,0.20,0.35,0.35"
            cfg.hop_loss_weights = "1.0,1.2,2.0,2.0"
            cfg.final_loop_loss_weight = 4.0
        cfg.load_checkpoint_tag = reuse_tag if is_joint_reuse else base_teacher_tag
        cfg.jump_steps = 0
        cfg.joint_halt_steps = 0 if is_joint_reuse else 2000
        cfg.direct_steps = 0
        cfg.direct_layers = 3
        cfg.strict_need_agreement = False
        cfg.load_checkpoints = True
        cfg.save_checkpoints = not is_joint_reuse
        cfg.checkpoint_tag = f"{mode}_seed{{seed}}"
        cfg.load_jumprec_state = True
        cfg.use_stability_head = use_joint_stability
        cfg.use_utility_router = True
        cfg.utility_use_stability_feature = use_joint_stability
        cfg.utility_false_accept_weight = 6.0
        cfg.utility_cost_weight = 0.20
        cfg.utility_correctness_bce_weight = 0.10
        if use_quality_objective:
            cfg.utility_false_accept_weight = 12.0
            cfg.utility_cost_weight = 0.08
            cfg.joint_halt_candidate_ce_weight = 1.00
            cfg.joint_halt_candidate_distill_weight = 0.15
            cfg.joint_halt_verifier_bce_weight = 0.08
            cfg.joint_halt_agreement_bce_weight = 0.08
        if use_agreement_distill:
            cfg.joint_halt_agreement_distill_weight = 0.10
            cfg.joint_halt_agreement_route_weight = 0.75
        if use_slo_objective:
            cfg.utility_false_accept_weight = 8.0
            cfg.utility_cost_weight = 0.12
            cfg.joint_halt_false_accept_weight_max = 18.0
            cfg.joint_halt_cost_weight_min = 0.04
            cfg.joint_halt_cost_weight_max = 0.18
            cfg.joint_halt_candidate_ce_weight = 1.00
            cfg.joint_halt_candidate_distill_weight = 0.15
            cfg.joint_halt_verifier_bce_weight = 0.08
            cfg.joint_halt_agreement_bce_weight = 0.10
        cfg.router_val_batches = 64
        cfg.router_threshold_candidates = "0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.85,0.90,0.95,0.97,0.99"
        cfg.timing_batches = 16
        cfg.timing_batch_sizes = "1,2,4,8,16,32,64"
        cfg.eval_batches = 128
        if is_highval_reuse:
            cfg.router_val_batches = 256
            cfg.eval_batches = 256
            cfg.router_threshold_candidates = (
                "0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,"
                "0.50,0.60,0.70,0.80,0.85,0.90,0.95,0.97,0.99"
            )
            cfg.timing_batches = 8
            cfg.timing_batch_sizes = "1,64"
            cfg.router_probe_audit = use_quality_objective and use_joint_stability
            cfg.router_selective_agree_audit = use_quality_objective and use_joint_stability
        cfg.log_every = 500
    elif mode in (
        "core3_8n4h_strathop_ablate_no_adapter",
        "core3_8n4h_strathop_ablate_no_distill",
        "core3_8n4h_strathop_ablate_no_verifier",
        "core3_8n4h_strathop_polish2_ablate_no_adapter",
        "core3_8n4h_strathop_polish2_ablate_no_distill",
        "core3_8n4h_strathop_polish2_ablate_no_verifier",
    ):
        is_polish2_ablation = mode.startswith("core3_8n4h_strathop_polish2_ablate")
        cfg.n_nodes = 8
        cfg.max_hops = 4
        cfg.preserve_steps = 2
        cfg.core_layers = 3
        cfg.coda_start = 27
        cfg.coda_layers = 3
        cfg.final_steps = 0
        cfg.recurrent_steps = 0
        if is_polish2_ablation:
            cfg.hard_hop_fraction = 0.70
            cfg.hard_hop_loss_weight = 2.5
            cfg.final_loop_loss_weight = 8.0
            cfg.load_checkpoint_tag = "core3_8n4h_strathop_polish2_seed{seed}"
        else:
            cfg.hop_sample_weights = "0.10,0.20,0.35,0.35"
            cfg.hop_loss_weights = "1.0,1.2,2.0,2.0"
            cfg.final_loop_loss_weight = 4.0
            cfg.load_checkpoint_tag = "core3_8n4h_strathop_seed{seed}"
        cfg.jump_steps = 4500
        cfg.direct_steps = 0
        cfg.direct_layers = 3
        cfg.strict_need_agreement = False
        cfg.load_checkpoints = True
        cfg.load_jumprec_state = False
        cfg.save_checkpoints = True
        cfg.checkpoint_tag = f"{mode}_seed{{seed}}"
        cfg.timing_batches = 16
        cfg.eval_batches = 96
        cfg.log_every = 500
        if mode.endswith("_ablate_no_adapter"):
            cfg.use_temp_adapter = False
        elif mode.endswith("_ablate_no_distill"):
            cfg.distill_loss_weight = 0.0
        elif mode.endswith("_ablate_no_verifier"):
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

    if cfg.prompt_style not in {"compact", "natural_graph"}:
        raise ValueError(f"unknown prompt_style: {cfg.prompt_style}")
    label_names = route_label_names(cfg.prompt_style, cfg.n_nodes)
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
            task = task_names[display_task_id]
            text = format_route_prompt(
                cfg.prompt_style,
                task,
                mapping_pairs,
                label_names,
                label_map[start],
                display_hops,
            )
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

    def stability_bce_with_logits(logits, target, hops):
        weights = example_weights(hops)
        if cfg.stability_false_stable_weight != 1.0:
            false_stable_weight = torch.full_like(target, float(cfg.stability_false_stable_weight))
            weights = weights * torch.where(target > 0.5, torch.ones_like(target), false_stable_weight)
        losses = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return (losses * weights).sum() / weights.sum().clamp_min(1e-6)

    def cost_aware_verifier_bce(logits, target, hops, corrections: int):
        weights = example_weights(hops)
        if cfg.verifier_false_accept_weight != 1.0:
            false_accept_weight = torch.full_like(target, float(cfg.verifier_false_accept_weight))
            weights = weights * torch.where(target > 0.5, torch.ones_like(target), false_accept_weight)
        if cfg.verifier_budget_savings_weight != 0.0:
            full_core_layers = float(cfg.loop_steps * cfg.core_layers)
            budget_core_layers = float(cfg.jump_layers + corrections * cfg.core_layers)
            savings = max(0.0, 1.0 - budget_core_layers / full_core_layers)
            positive_weight = torch.full_like(target, 1.0 + cfg.verifier_budget_savings_weight * savings)
            weights = weights * torch.where(target > 0.5, positive_weight, torch.ones_like(target))
        losses = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return (losses * weights).sum() / weights.sum().clamp_min(1e-6)

    def verifier_budget_ranking_loss(verify_logits, good_stack, hops):
        if cfg.verifier_ranking_loss_weight <= 0.0:
            return verify_logits[0].new_tensor(0.0)
        good_bool = good_stack.bool()
        any_good = good_bool.any(dim=0)
        if not any_good.any():
            return verify_logits[0].new_tensor(0.0)
        num_budgets, batch_size = good_stack.shape
        budget_ids = torch.arange(num_budgets, device=verify_logits.device).view(num_budgets, 1)
        fallback_idx = torch.full(
            (num_budgets, batch_size),
            num_budgets,
            dtype=torch.long,
            device=verify_logits.device,
        )
        first_good = torch.where(good_bool, budget_ids.expand(num_budgets, batch_size), fallback_idx).min(dim=0).values
        sample_ids = torch.arange(batch_size, device=verify_logits.device)[any_good]
        first_good = first_good[any_good]
        selected_logits = verify_logits[first_good, sample_ids].view(1, -1)
        compare_logits = verify_logits[:, sample_ids]
        budget_grid = budget_ids.expand_as(compare_logits)
        earlier = budget_grid < first_good.view(1, -1)
        later = budget_grid > first_good.view(1, -1)
        active = earlier | (later & (cfg.verifier_rank_later_weight > 0.0))
        losses = F.softplus(compare_logits - selected_logits + cfg.verifier_ranking_margin)
        rank_weights = torch.where(
            earlier,
            torch.ones_like(losses),
            torch.full_like(losses, float(cfg.verifier_rank_later_weight)),
        )
        pair_weights = example_weights(hops)[sample_ids].view(1, -1).expand_as(losses)
        weights = pair_weights * rank_weights
        return (losses[active] * weights[active]).sum() / weights[active].sum().clamp_min(1e-6)

    def first_sufficient_budget(correct_stack):
        num_budgets, batch_size = correct_stack.shape
        fallback = cfg.max_correct + 1
        budget_ids = torch.arange(num_budgets, dtype=torch.long, device=correct_stack.device).view(num_budgets, 1)
        fallback_idx = torch.full(
            (num_budgets, batch_size),
            fallback,
            dtype=torch.long,
            device=correct_stack.device,
        )
        return torch.where(correct_stack.bool(), budget_ids.expand(num_budgets, batch_size), fallback_idx).min(dim=0).values

    def first_acceptable_budget(correct_stack, verify_stack, margin_stack, max_prob_stack, threshold: float):
        acceptable_stack = (
            correct_stack.bool()
            & (verify_stack >= threshold)
            & (margin_stack >= 0.05)
            & (max_prob_stack >= 0.45)
        )
        return first_sufficient_budget(acceptable_stack)

    def budget_tail_loops(budget_index):
        fallback = cfg.max_correct + 1
        return torch.where(
            budget_index == fallback,
            torch.full_like(budget_index, cfg.loop_steps),
            budget_index,
        ).float()

    def budget_controller_loss(logits, target, hops):
        loss = weighted_ce(logits, target, hops)
        if cfg.budget_controller_under_weight <= 0.0 and cfg.budget_controller_over_weight <= 0.0:
            return loss
        weights = example_weights(hops)
        probs = F.softmax(logits, dim=-1)
        class_ids = torch.arange(cfg.max_correct + 2, dtype=torch.long, device=logits.device)
        class_tail = budget_tail_loops(class_ids).view(1, -1)
        target_tail = budget_tail_loops(target).view(-1, 1)
        expected_under = (probs * (target_tail - class_tail).clamp_min(0.0)).sum(dim=-1)
        expected_over = (probs * (class_tail - target_tail).clamp_min(0.0)).sum(dim=-1)
        asymmetric = (
            float(cfg.budget_controller_under_weight) * expected_under
            + float(cfg.budget_controller_over_weight) * expected_over
        )
        return loss + (asymmetric * weights).sum() / weights.sum().clamp_min(1e-6)

    def utility_router_expected_loss(accept_logits, correct_stack, full_correct, hops):
        weights = example_weights(hops)
        accept_prob = torch.sigmoid(accept_logits)
        survival = torch.ones_like(accept_prob[0])
        expected_wrong = torch.zeros_like(survival)
        expected_core = torch.zeros_like(survival)
        full_core_layers = float(cfg.loop_steps * cfg.core_layers)
        for corrections in range(cfg.max_correct + 1):
            route_prob = survival * accept_prob[corrections]
            wrong = (~correct_stack[corrections].bool()).float()
            core_frac = (float(cfg.jump_layers) + float(corrections * cfg.core_layers)) / full_core_layers
            expected_wrong = expected_wrong + route_prob * wrong * float(cfg.utility_false_accept_weight)
            expected_core = expected_core + route_prob * core_frac
            survival = survival * (1.0 - accept_prob[corrections])
        fallback_wrong = (~full_correct.bool()).float()
        expected_wrong = expected_wrong + survival * fallback_wrong
        expected_core = expected_core + survival
        utility_loss = expected_wrong + float(cfg.utility_cost_weight) * expected_core
        return (utility_loss * weights).sum() / weights.sum().clamp_min(1e-6), expected_wrong, expected_core

    def joint_halt_expected_loss(
        accept_logits,
        logits_stack,
        full_correct,
        target,
        hops,
        false_accept_weight: float | None = None,
        cost_weight: float | None = None,
        agreement_prob_stack=None,
        agreement_route_weight: float = 0.0,
    ):
        if false_accept_weight is None:
            false_accept_weight = float(cfg.utility_false_accept_weight)
        if cost_weight is None:
            cost_weight = float(cfg.utility_cost_weight)
        weights = example_weights(hops)
        accept_prob = torch.sigmoid(accept_logits)
        target_idx = target.view(1, -1, 1).expand(logits_stack.size(0), -1, 1)
        target_prob = F.softmax(logits_stack, dim=-1).gather(-1, target_idx).squeeze(-1)
        survival = torch.ones_like(accept_prob[0])
        expected_wrong = torch.zeros_like(survival)
        expected_core = torch.zeros_like(survival)
        full_core_layers = float(cfg.loop_steps * cfg.core_layers)
        for corrections in range(cfg.max_correct + 1):
            route_prob = survival * accept_prob[corrections]
            wrong_prob = (1.0 - target_prob[corrections]).clamp(0.0, 1.0)
            if agreement_prob_stack is not None and agreement_route_weight > 0.0:
                stability_penalty = target_prob[corrections] * (
                    1.0 - agreement_prob_stack[corrections].clamp(0.0, 1.0)
                )
                wrong_prob = (
                    wrong_prob + float(agreement_route_weight) * stability_penalty
                ).clamp(0.0, 1.0)
            core_frac = (float(cfg.jump_layers) + float(corrections * cfg.core_layers)) / full_core_layers
            expected_wrong = expected_wrong + route_prob * wrong_prob * float(false_accept_weight)
            expected_core = expected_core + route_prob * core_frac
            survival = survival * (1.0 - accept_prob[corrections])
        fallback_wrong = (~full_correct.bool()).float()
        expected_wrong = expected_wrong + survival * fallback_wrong
        expected_core = expected_core + survival
        route_loss = expected_wrong + float(cost_weight) * expected_core
        return (route_loss * weights).sum() / weights.sum().clamp_min(1e-6), expected_wrong, expected_core

    def candidate_agreement_prob_stack(logits_stack, soft_teacher):
        probs = F.softmax(logits_stack, dim=-1)
        agreement_prob = torch.zeros_like(probs[:, :, 0])
        if cfg.max_correct > 0:
            next_probs = probs[1:].detach()
            agreement_prob[:-1] = (probs[:-1] * next_probs).sum(dim=-1)
        agreement_prob[-1] = (probs[-1] * soft_teacher.detach()).sum(dim=-1)
        return agreement_prob

    def utility_router_aux_bce(accept_logits, correct_stack, hops):
        if cfg.utility_correctness_bce_weight <= 0.0:
            return accept_logits.new_tensor(0.0)
        weights = example_weights(hops).view(1, -1).expand_as(accept_logits)
        target = correct_stack.float()
        if cfg.utility_false_accept_weight != 1.0:
            false_accept_weight = torch.full_like(target, float(cfg.utility_false_accept_weight))
            weights = weights * torch.where(target > 0.5, torch.ones_like(target), false_accept_weight)
        losses = F.binary_cross_entropy_with_logits(accept_logits, target, reduction="none")
        return float(cfg.utility_correctness_bce_weight) * (losses * weights).sum() / weights.sum().clamp_min(1e-6)

    def utility_router_agreement_bce(accept_logits, correct_stack, pred_stack, hops):
        if cfg.joint_halt_agreement_bce_weight <= 0.0:
            return accept_logits.new_tensor(0.0)
        agree_stack = torch.zeros_like(correct_stack, dtype=torch.bool)
        if cfg.max_correct > 0:
            agree_stack[:-1] = pred_stack[:-1] == pred_stack[1:]
        safe_target = (correct_stack.bool() & agree_stack).float()
        weights = example_weights(hops).view(1, -1).expand_as(accept_logits)
        weights = weights.clone()
        weights[-1] = 0.0
        false_accept_weight = torch.full_like(safe_target, float(cfg.utility_false_accept_weight))
        weights = weights * torch.where(safe_target > 0.5, torch.ones_like(safe_target), false_accept_weight)
        losses = F.binary_cross_entropy_with_logits(accept_logits, safe_target, reduction="none")
        return float(cfg.joint_halt_agreement_bce_weight) * (losses * weights).sum() / weights.sum().clamp_min(1e-6)

    def joint_halt_agreement_distill_loss(logits_stack, soft_teacher, target, hops):
        if cfg.joint_halt_agreement_distill_weight <= 0.0:
            return logits_stack.new_tensor(0.0)
        weights = example_weights(hops)
        losses = []
        for corrections in range(cfg.max_correct):
            with torch.no_grad():
                next_logits = logits_stack[corrections + 1].detach()
                next_target = F.softmax(next_logits, dim=-1)
                next_correct = (next_logits.argmax(dim=-1) == target).float()
            pair_loss = F.kl_div(
                F.log_softmax(logits_stack[corrections], dim=-1),
                next_target,
                reduction="none",
            ).sum(dim=-1)
            pair_weights = weights * next_correct
            losses.append((pair_loss * pair_weights).sum() / pair_weights.sum().clamp_min(1e-6))
        with torch.no_grad():
            teacher_correct = (soft_teacher.argmax(dim=-1) == target).float()
        final_loss = F.kl_div(
            F.log_softmax(logits_stack[-1], dim=-1),
            soft_teacher.detach(),
            reduction="none",
        ).sum(dim=-1)
        final_weights = weights * teacher_correct
        losses.append((final_loss * final_weights).sum() / final_weights.sum().clamp_min(1e-6))
        return float(cfg.joint_halt_agreement_distill_weight) * torch.stack(losses).mean()

    def consistency_bce_with_logits(consistency_logits, stable_target, hops):
        weights = example_weights(hops).view(1, -1).expand_as(consistency_logits)
        false_stable_weight = torch.full_like(stable_target, float(cfg.consistency_false_stable_weight))
        weights = weights * torch.where(stable_target > 0.5, torch.ones_like(stable_target), false_stable_weight)
        losses = F.binary_cross_entropy_with_logits(consistency_logits, stable_target, reduction="none")
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
    should_use_jumprec = (
        cfg.jump_steps > 0
        or cfg.budget_controller_steps > 0
        or cfg.stability_steps > 0
        or cfg.utility_router_steps > 0
        or cfg.next_agreement_steps > 0
        or cfg.consistency_steps > 0
        or cfg.joint_halt_steps > 0
        or (
            loaded_checkpoint is not None
            and not cfg.resume_teacher_training
            and has_loaded_jumprec
        )
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
                self.budget_controller = None
                if cfg.use_budget_controller:
                    self.budget_controller = nn.Sequential(
                        nn.LayerNorm(cfg.d_model),
                        nn.Linear(cfg.d_model, cfg.d_model),
                        nn.GELU(),
                        nn.Linear(cfg.d_model, cfg.max_correct + 2),
                    )
                self.stability_heads = None
                if cfg.use_stability_head:
                    self.stability_heads = nn.ModuleList(
                        [
                            nn.Sequential(
                                nn.Linear(verifier_in, cfg.d_model),
                                nn.GELU(),
                                nn.Linear(cfg.d_model, 1),
                            )
                            for _ in range(cfg.max_correct + 1)
                        ]
                    )
                self.consistency_heads = None
                if cfg.use_consistency_head:
                    consistency_in = verifier_in + 3
                    self.consistency_heads = nn.ModuleList(
                        [
                            nn.Sequential(
                                nn.Linear(consistency_in, cfg.d_model),
                                nn.GELU(),
                                nn.Linear(cfg.d_model, 1),
                            )
                            for _ in range(cfg.max_correct + 1)
                        ]
                    )
                self.utility_router = None
                if cfg.use_utility_router:
                    utility_in = verifier_in + 3
                    if cfg.utility_use_stability_feature:
                        utility_in += 1
                    self.utility_router = nn.Sequential(
                        nn.Linear(utility_in, cfg.d_model),
                        nn.GELU(),
                        nn.Linear(cfg.d_model, 1),
                    )
                self.next_pred_heads = None
                if cfg.use_next_agreement_head:
                    self.next_pred_heads = nn.ModuleList(
                        [
                            nn.Sequential(
                                nn.Linear(verifier_in, cfg.d_model),
                                nn.GELU(),
                                nn.Linear(cfg.d_model, cfg.n_nodes),
                            )
                            for _ in range(cfg.max_correct)
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

            def budget_logits(self, state0, lengths):
                if self.budget_controller is None:
                    raise RuntimeError("budget controller is disabled")
                return self.budget_controller(self.teacher.read_last(state0, lengths))

            def stability_logit(self, corrections: int, features):
                if self.stability_heads is None:
                    raise RuntimeError("stability head is disabled")
                return self.stability_heads[corrections](features).squeeze(-1)

            def consistency_logit(self, corrections: int, features, verifier_logit):
                if self.consistency_heads is None:
                    raise RuntimeError("consistency head is disabled")
                batch_size = features.size(0)
                budget_frac = corrections / max(1, cfg.max_correct)
                full_core_layers = float(cfg.loop_steps * cfg.core_layers)
                core_frac = (float(cfg.jump_layers) + float(corrections * cfg.core_layers)) / full_core_layers
                extras = [
                    verifier_logit.unsqueeze(-1),
                    torch.full((batch_size, 1), budget_frac, dtype=features.dtype, device=features.device),
                    torch.full((batch_size, 1), core_frac, dtype=features.dtype, device=features.device),
                ]
                return self.consistency_heads[corrections](torch.cat([features] + extras, dim=-1)).squeeze(-1)

            def utility_logit(self, corrections: int, features, verifier_logit, stability_logit=None):
                if self.utility_router is None:
                    raise RuntimeError("utility router is disabled")
                batch_size = features.size(0)
                budget_frac = corrections / max(1, cfg.max_correct)
                full_core_layers = float(cfg.loop_steps * cfg.core_layers)
                core_frac = (float(cfg.jump_layers) + float(corrections * cfg.core_layers)) / full_core_layers
                extras = [
                    verifier_logit.unsqueeze(-1),
                    torch.full((batch_size, 1), budget_frac, dtype=features.dtype, device=features.device),
                    torch.full((batch_size, 1), core_frac, dtype=features.dtype, device=features.device),
                ]
                if cfg.utility_use_stability_feature:
                    if stability_logit is None:
                        stability_logit = torch.zeros(batch_size, dtype=features.dtype, device=features.device)
                    extras.append(stability_logit.unsqueeze(-1))
                return self.utility_router(torch.cat([features] + extras, dim=-1)).squeeze(-1)

            def next_pred_logits(self, corrections: int, features):
                if self.next_pred_heads is None:
                    raise RuntimeError("next-agreement head is disabled")
                if corrections >= cfg.max_correct:
                    raise RuntimeError("next-agreement head has no final-budget prediction")
                return self.next_pred_heads[corrections](features)

            def forward_encoded(self, state0, layer_mask, position_ids, position_embeddings, lengths):
                landing_states, final_states, logits, verify, stability, consistency, utility, next_pred = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
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
                    features = self.verifier_features(final, final_logits, lengths)
                    verifier_logit = self.verifiers[corrections](features).squeeze(-1)
                    landing_states.append(landing)
                    final_states.append(final)
                    logits.append(final_logits)
                    verify.append(verifier_logit)
                    stability_logit = None
                    if self.stability_heads is not None:
                        stability_logit = self.stability_logit(corrections, features)
                        stability.append(stability_logit)
                    if self.consistency_heads is not None:
                        consistency.append(self.consistency_logit(corrections, features, verifier_logit))
                    if self.utility_router is not None:
                        utility.append(
                            self.utility_logit(corrections, features, verifier_logit, stability_logit)
                        )
                    if self.next_pred_heads is not None and corrections < cfg.max_correct:
                        next_pred.append(self.next_pred_logits(corrections, features))
                return {
                    "landing_states": landing_states,
                    "final_states": final_states,
                    "logits": logits,
                    "verify": verify,
                    "stability": stability if self.stability_heads is not None else None,
                    "consistency": consistency if self.consistency_heads is not None else None,
                    "utility": utility if self.utility_router is not None else None,
                    "next_pred": next_pred if self.next_pred_heads is not None else None,
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
                features = self.verifier_features(final, final_logits, lengths)
                verifier_logit = self.verifiers[corrections](features).squeeze(-1)
                result = [final_logits, verifier_logit]
                if self.stability_heads is not None:
                    result.append(self.stability_logit(corrections, features))
                if self.consistency_heads is not None:
                    result.append(self.consistency_logit(corrections, features, verifier_logit))
                if self.utility_router is not None:
                    stability_logit = result[2] if self.stability_heads is not None else None
                    result.append(self.utility_logit(corrections, features, verifier_logit, stability_logit))
                if self.next_pred_heads is not None and corrections < cfg.max_correct:
                    result.append(self.next_pred_logits(corrections, features))
                return tuple(result)

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
            load_result = jumprec.load_state_dict(
                loaded_checkpoint["jumprec_state"],
                strict=not (
                    cfg.use_budget_controller
                    or cfg.use_stability_head
                    or cfg.use_utility_router
                    or cfg.use_next_agreement_head
                    or cfg.use_consistency_head
                ),
            )
            if (
                cfg.use_budget_controller
                or cfg.use_stability_head
                or cfg.use_utility_router
                or cfg.use_next_agreement_head
                or cfg.use_consistency_head
            ) and (load_result.missing_keys or load_result.unexpected_keys):
                print(
                    "[checkpoint] partial JumpRec load "
                    f"missing={load_result.missing_keys} unexpected={load_result.unexpected_keys}",
                    flush=True,
                )
            print("[checkpoint] loaded JumpRec", flush=True)
        else:
            if cfg.jump_steps <= 0:
                raise RuntimeError("JumpRec checkpoint is required when jump_steps is 0")
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
                ce_losses, distill_losses, verifier_losses, good_labels = [], [], [], []
                for corrections in range(cfg.max_correct + 1):
                    final_logits = out["logits"][corrections]
                    ce_losses.append(weighted_ce(final_logits, target, hops))
                    distill_losses.append(weighted_kl(final_logits, soft_teacher, hops))
                    good = (final_logits.detach().argmax(dim=-1) == target).float()
                    good_labels.append(good)
                    verifier_losses.append(
                        cost_aware_verifier_bce(out["verify"][corrections], good, hops, corrections)
                    )
                verify_logits = torch.stack(out["verify"], dim=0)
                good_stack = torch.stack(good_labels, dim=0)
                ranking_loss = verifier_budget_ranking_loss(verify_logits, good_stack, hops)
                loss = (
                    torch.stack(ce_losses).mean()
                    + cfg.distill_loss_weight * torch.stack(distill_losses).mean()
                    + cfg.verifier_loss_weight * torch.stack(verifier_losses).mean()
                    + cfg.verifier_ranking_loss_weight * ranking_loss
                )
                opt_j.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(jumprec.parameters(), 1.0)
                opt_j.step()
                if step % cfg.log_every == 0 or step == cfg.jump_steps:
                    print(
                        f"  jump step {step:5d}/{cfg.jump_steps} loss {loss.item():.4f} "
                        f"ce0 {ce_losses[0].item():.3f} ce{cfg.max_correct} {ce_losses[-1].item():.3f} "
                        f"vf {torch.stack(verifier_losses).mean().item():.3f} rank {ranking_loss.item():.3f} "
                        f"elapsed {time.time()-t1:.1f}s",
                        flush=True,
                    )
            save_checkpoint(model, jumprec)

        if cfg.use_utility_router and cfg.joint_halt_steps > 0:
            if jumprec.utility_router is None:
                raise RuntimeError("joint_halt_steps requires cfg.use_utility_router")
            opt_joint = torch.optim.AdamW(
                [p for p in jumprec.parameters() if p.requires_grad],
                lr=cfg.joint_halt_lr,
            )
            t_joint = time.time()
            jumprec.train()
            for step in range(1, cfg.joint_halt_steps + 1):
                input_ids, attention_mask, lengths, target, _, hops, _ = batch_encoded(cfg.batch_size)
                with torch.no_grad():
                    state0, layer_mask, position_ids, position_embeddings, _, teacher_logits = teacher_collect(
                        input_ids,
                        attention_mask,
                        lengths,
                    )
                    soft_teacher = F.softmax(teacher_logits, dim=-1)
                    full_pred = teacher_logits.argmax(dim=-1)
                out = jumprec.forward_encoded(
                    state0.detach(),
                    layer_mask,
                    position_ids,
                    position_embeddings,
                    lengths,
                )
                logits_stack = torch.stack(out["logits"], dim=0)
                utility_logits = torch.stack(out["utility"], dim=0)
                pred_stack = logits_stack.detach().argmax(dim=-1)
                correct_stack = pred_stack == target.unsqueeze(0)
                route_false_accept_weight = float(cfg.utility_false_accept_weight)
                if cfg.joint_halt_false_accept_weight_max > route_false_accept_weight:
                    route_false_accept_weight = random.uniform(
                        route_false_accept_weight,
                        float(cfg.joint_halt_false_accept_weight_max),
                    )
                route_cost_weight = float(cfg.utility_cost_weight)
                if cfg.joint_halt_cost_weight_min >= 0.0 and cfg.joint_halt_cost_weight_max >= cfg.joint_halt_cost_weight_min:
                    route_cost_weight = random.uniform(
                        float(cfg.joint_halt_cost_weight_min),
                        float(cfg.joint_halt_cost_weight_max),
                    )
                agreement_prob_stack = (
                    candidate_agreement_prob_stack(logits_stack, soft_teacher)
                    if cfg.joint_halt_agreement_route_weight > 0.0
                    else None
                )
                route_loss, expected_wrong, expected_core = joint_halt_expected_loss(
                    utility_logits,
                    logits_stack,
                    full_pred == target,
                    target,
                    hops,
                    false_accept_weight=route_false_accept_weight,
                    cost_weight=route_cost_weight,
                    agreement_prob_stack=agreement_prob_stack,
                    agreement_route_weight=float(cfg.joint_halt_agreement_route_weight),
                )
                ce_loss = torch.stack(
                    [weighted_ce(logits_stack[c], target, hops) for c in range(cfg.max_correct + 1)]
                ).mean()
                distill_loss = torch.stack(
                    [weighted_kl(logits_stack[c], soft_teacher, hops) for c in range(cfg.max_correct + 1)]
                ).mean()
                verifier_loss = torch.stack(
                    [
                        cost_aware_verifier_bce(out["verify"][c], correct_stack[c].float(), hops, c)
                        for c in range(cfg.max_correct + 1)
                    ]
                ).mean()
                aux_bce = utility_router_aux_bce(utility_logits, correct_stack, hops)
                agreement_bce = utility_router_agreement_bce(
                    utility_logits,
                    correct_stack,
                    pred_stack,
                    hops,
                )
                agreement_distill = joint_halt_agreement_distill_loss(
                    logits_stack,
                    soft_teacher,
                    target,
                    hops,
                )
                joint_loss = (
                    route_loss
                    + float(cfg.joint_halt_candidate_ce_weight) * ce_loss
                    + float(cfg.joint_halt_candidate_distill_weight) * distill_loss
                    + float(cfg.joint_halt_verifier_bce_weight) * verifier_loss
                    + aux_bce
                    + agreement_bce
                    + agreement_distill
                )
                stability_loss = utility_logits.new_tensor(0.0)
                if cfg.use_stability_head and out["stability"] is not None and cfg.joint_halt_stability_weight > 0.0:
                    stability_targets = []
                    for corrections in range(cfg.max_correct + 1):
                        if corrections < cfg.max_correct:
                            stability_targets.append(
                                (pred_stack[corrections] == pred_stack[corrections + 1]).float()
                            )
                        else:
                            stability_targets.append(torch.zeros_like(target, dtype=torch.float32))
                    stability_loss = torch.stack(
                        [
                            stability_bce_with_logits(out["stability"][c], stability_targets[c], hops)
                            for c in range(cfg.max_correct + 1)
                        ]
                    ).mean()
                    joint_loss = joint_loss + float(cfg.joint_halt_stability_weight) * stability_loss
                opt_joint.zero_grad()
                joint_loss.backward()
                torch.nn.utils.clip_grad_norm_(jumprec.parameters(), 1.0)
                opt_joint.step()
                if step % cfg.log_every == 0 or step == cfg.joint_halt_steps:
                    utility_prob = torch.sigmoid(utility_logits.detach())
                    trusted = torch.zeros_like(target, dtype=torch.bool)
                    chosen = torch.full_like(target, cfg.loop_steps)
                    routed_pred = full_pred
                    for corrections in range(cfg.max_correct + 1):
                        accept = (~trusted) & (utility_prob[corrections] >= 0.5)
                        chosen = torch.where(accept, torch.full_like(chosen, corrections), chosen)
                        routed_pred = torch.where(accept, pred_stack[corrections], routed_pred)
                        trusted = trusted | accept
                    full_core_layers = float(cfg.loop_steps * cfg.core_layers)
                    core_layers = torch.where(
                        trusted,
                        torch.full_like(chosen.float(), float(cfg.jump_layers))
                        + chosen.float() * float(cfg.core_layers),
                        torch.full_like(chosen.float(), full_core_layers),
                    )
                    precision_denom = trusted.float().sum().clamp_min(1.0)
                    precision = (routed_pred[trusted] == target[trusted]).float().sum().item() / precision_denom.item()
                    print(
                        f"  joint halt step {step:5d}/{cfg.joint_halt_steps} "
                        f"loss {joint_loss.item():.4f} route {route_loss.item():.4f} "
                        f"ce {ce_loss.item():.4f} distill {distill_loss.item():.4f} "
                        f"vf {verifier_loss.item():.4f} aux {aux_bce.item():.4f} "
                        f"agree_aux {agreement_bce.item():.4f} "
                        f"agree_distill {agreement_distill.item():.4f} "
                        f"stab {stability_loss.item():.4f} "
                        f"wrong {expected_wrong.mean().item():.4f} core {expected_core.mean().item():.4f} "
                        f"agree_prob {(agreement_prob_stack.mean().item() if agreement_prob_stack is not None else 0.0):.4f} "
                        f"fa_w {route_false_accept_weight:.2f} cost_w {route_cost_weight:.3f} "
                        f"acc {(routed_pred == target).float().mean().item()*100:.1f}% "
                        f"coverage {trusted.float().mean().item()*100:.1f}% "
                        f"precision {precision*100:.1f}% "
                        f"avg_core {core_layers.float().mean().item():.2f} "
                        f"elapsed {time.time()-t_joint:.1f}s",
                        flush=True,
                    )
            save_checkpoint(model, jumprec)

        if cfg.use_budget_controller and cfg.budget_controller_steps > 0:
            if jumprec.budget_controller is None:
                raise RuntimeError("budget_controller_steps requires cfg.use_budget_controller")
            budget_restore_params = list(jumprec.parameters())
            budget_restore_requires_grad = [p.requires_grad for p in budget_restore_params]
            for p in budget_restore_params:
                p.requires_grad_(False)
            for p in jumprec.budget_controller.parameters():
                p.requires_grad_(True)
            opt_budget = torch.optim.AdamW(jumprec.budget_controller.parameters(), lr=cfg.budget_controller_lr)
            t_budget = time.time()
            jumprec.eval()
            jumprec.budget_controller.train()
            for step in range(1, cfg.budget_controller_steps + 1):
                input_ids, attention_mask, lengths, target, _, hops, _ = batch_encoded(cfg.batch_size)
                with torch.no_grad():
                    state0, layer_mask, position_ids, position_embeddings, _, _ = teacher_collect(
                        input_ids,
                        attention_mask,
                        lengths,
                    )
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
                    correct_stack = pred_stack == target.unsqueeze(0)
                    if cfg.budget_controller_target == "first_correct":
                        budget_target = first_sufficient_budget(correct_stack)
                    elif cfg.budget_controller_target == "first_acceptable":
                        logits_stack = torch.stack(jump_out["logits"], dim=0)
                        probs = F.softmax(logits_stack, dim=-1)
                        top2 = probs.topk(2, dim=-1).values
                        verify_stack = torch.stack([torch.sigmoid(v) for v in jump_out["verify"]], dim=0)
                        budget_target = first_acceptable_budget(
                            correct_stack,
                            verify_stack,
                            top2[:, :, 0] - top2[:, :, 1],
                            top2[:, :, 0],
                            cfg.budget_controller_train_threshold,
                        )
                    else:
                        raise ValueError(f"unknown budget_controller_target={cfg.budget_controller_target!r}")
                budget_logits = jumprec.budget_logits(state0.detach(), lengths)
                budget_loss = budget_controller_loss(budget_logits, budget_target, hops)
                opt_budget.zero_grad()
                budget_loss.backward()
                torch.nn.utils.clip_grad_norm_(jumprec.budget_controller.parameters(), 1.0)
                opt_budget.step()
                if step % cfg.log_every == 0 or step == cfg.budget_controller_steps:
                    budget_pred = budget_logits.detach().argmax(dim=-1)
                    pred_tail = budget_tail_loops(budget_pred)
                    target_tail = budget_tail_loops(budget_target)
                    print(
                        f"  budget step {step:5d}/{cfg.budget_controller_steps} "
                        f"loss {budget_loss.item():.4f} "
                        f"acc {(budget_pred == budget_target).float().mean().item()*100:.1f}% "
                        f"under {(pred_tail < target_tail).float().mean().item()*100:.1f}% "
                        f"over {(pred_tail > target_tail).float().mean().item()*100:.1f}% "
                        f"pred_tail {pred_tail.float().mean().item():.2f} "
                        f"target_tail {target_tail.float().mean().item():.2f} "
                        f"elapsed {time.time()-t_budget:.1f}s",
                        flush=True,
                    )
            for p, requires_grad in zip(budget_restore_params, budget_restore_requires_grad):
                p.requires_grad_(requires_grad)
            save_checkpoint(model, jumprec)

        if cfg.use_stability_head and cfg.stability_steps > 0:
            if jumprec.stability_heads is None:
                raise RuntimeError("stability_steps requires cfg.use_stability_head")
            stability_restore_params = list(jumprec.parameters())
            stability_restore_requires_grad = [p.requires_grad for p in stability_restore_params]
            for p in stability_restore_params:
                p.requires_grad_(False)
            for p in jumprec.stability_heads.parameters():
                p.requires_grad_(True)
            opt_stability = torch.optim.AdamW(jumprec.stability_heads.parameters(), lr=cfg.stability_lr)
            t_stability = time.time()
            jumprec.eval()
            jumprec.stability_heads.train()
            for step in range(1, cfg.stability_steps + 1):
                input_ids, attention_mask, lengths, target, _, hops, _ = batch_encoded(cfg.batch_size)
                with torch.no_grad():
                    state0, layer_mask, position_ids, position_embeddings, _, _ = teacher_collect(
                        input_ids,
                        attention_mask,
                        lengths,
                    )
                out = jumprec.forward_encoded(
                    state0.detach(),
                    layer_mask,
                    position_ids,
                    position_embeddings,
                    lengths,
                )
                pred_stack = torch.stack(
                    [logits_i.detach().argmax(dim=-1) for logits_i in out["logits"]],
                    dim=0,
                )
                stability_targets = []
                for corrections in range(cfg.max_correct + 1):
                    if corrections < cfg.max_correct:
                        stability_targets.append(
                            (pred_stack[corrections] == pred_stack[corrections + 1]).float()
                        )
                    else:
                        stability_targets.append(torch.zeros_like(target, dtype=torch.float32))
                stability_losses = [
                    stability_bce_with_logits(out["stability"][c], stability_targets[c], hops)
                    for c in range(cfg.max_correct + 1)
                ]
                stability_loss = torch.stack(stability_losses).mean()
                opt_stability.zero_grad()
                stability_loss.backward()
                torch.nn.utils.clip_grad_norm_(jumprec.stability_heads.parameters(), 1.0)
                opt_stability.step()
                if step % cfg.log_every == 0 or step == cfg.stability_steps:
                    stability_logits = torch.stack([x.detach() for x in out["stability"]], dim=0)
                    stability_pred = torch.sigmoid(stability_logits) >= 0.5
                    stability_target = torch.stack(stability_targets, dim=0).bool()
                    pred_pos = stability_pred.float().mean().item()
                    target_pos = stability_target.float().mean().item()
                    acc = (stability_pred == stability_target).float().mean().item()
                    precision_denom = stability_pred.float().sum().clamp_min(1.0)
                    precision = (stability_pred & stability_target).float().sum().item() / precision_denom.item()
                    print(
                        f"  stability step {step:5d}/{cfg.stability_steps} "
                        f"loss {stability_loss.item():.4f} "
                        f"acc {acc*100:.1f}% pred_pos {pred_pos*100:.1f}% "
                        f"target_pos {target_pos*100:.1f}% precision {precision*100:.1f}% "
                        f"elapsed {time.time()-t_stability:.1f}s",
                        flush=True,
                    )
            for p, requires_grad in zip(stability_restore_params, stability_restore_requires_grad):
                p.requires_grad_(requires_grad)
            save_checkpoint(model, jumprec)

        if cfg.use_utility_router and cfg.utility_router_steps > 0:
            if jumprec.utility_router is None:
                raise RuntimeError("utility_router_steps requires cfg.use_utility_router")
            utility_restore_params = list(jumprec.parameters())
            utility_restore_requires_grad = [p.requires_grad for p in utility_restore_params]
            for p in utility_restore_params:
                p.requires_grad_(False)
            for p in jumprec.utility_router.parameters():
                p.requires_grad_(True)
            opt_utility = torch.optim.AdamW(jumprec.utility_router.parameters(), lr=cfg.utility_router_lr)
            t_utility = time.time()
            jumprec.eval()
            jumprec.utility_router.train()
            for step in range(1, cfg.utility_router_steps + 1):
                input_ids, attention_mask, lengths, target, _, hops, _ = batch_encoded(cfg.batch_size)
                with torch.no_grad():
                    state0, layer_mask, position_ids, position_embeddings, _, teacher_logits = teacher_collect(
                        input_ids,
                        attention_mask,
                        lengths,
                    )
                    full_pred = teacher_logits.argmax(dim=-1)
                out = jumprec.forward_encoded(
                    state0.detach(),
                    layer_mask,
                    position_ids,
                    position_embeddings,
                    lengths,
                )
                pred_stack = torch.stack(
                    [logits_i.detach().argmax(dim=-1) for logits_i in out["logits"]],
                    dim=0,
                )
                correct_stack = pred_stack == target.unsqueeze(0)
                utility_logits = torch.stack(out["utility"], dim=0)
                expected_loss, expected_wrong, expected_core = utility_router_expected_loss(
                    utility_logits,
                    correct_stack,
                    full_pred == target,
                    hops,
                )
                aux_bce = utility_router_aux_bce(utility_logits, correct_stack, hops)
                utility_loss = expected_loss + aux_bce
                opt_utility.zero_grad()
                utility_loss.backward()
                torch.nn.utils.clip_grad_norm_(jumprec.utility_router.parameters(), 1.0)
                opt_utility.step()
                if step % cfg.log_every == 0 or step == cfg.utility_router_steps:
                    utility_prob = torch.sigmoid(utility_logits.detach())
                    trusted = torch.zeros_like(target, dtype=torch.bool)
                    chosen = torch.full_like(target, cfg.loop_steps)
                    routed_pred = full_pred
                    for corrections in range(cfg.max_correct + 1):
                        accept = (~trusted) & (utility_prob[corrections] >= 0.5)
                        chosen = torch.where(accept, torch.full_like(chosen, corrections), chosen)
                        routed_pred = torch.where(accept, pred_stack[corrections], routed_pred)
                        trusted = trusted | accept
                    full_core_layers = float(cfg.loop_steps * cfg.core_layers)
                    core_layers = torch.where(
                        trusted,
                        torch.full_like(chosen.float(), float(cfg.jump_layers))
                        + chosen.float() * float(cfg.core_layers),
                        torch.full_like(chosen.float(), full_core_layers),
                    )
                    precision_denom = trusted.float().sum().clamp_min(1.0)
                    precision = (routed_pred[trusted] == target[trusted]).float().sum().item() / precision_denom.item()
                    print(
                        f"  utility step {step:5d}/{cfg.utility_router_steps} "
                        f"loss {utility_loss.item():.4f} expected {expected_loss.item():.4f} "
                        f"bce {aux_bce.item():.4f} wrong {expected_wrong.mean().item():.4f} "
                        f"core {expected_core.mean().item():.4f} "
                        f"acc {(routed_pred == target).float().mean().item()*100:.1f}% "
                        f"coverage {trusted.float().mean().item()*100:.1f}% "
                        f"precision {precision*100:.1f}% "
                        f"avg_core {core_layers.float().mean().item():.2f} "
                        f"elapsed {time.time()-t_utility:.1f}s",
                        flush=True,
                    )
            for p, requires_grad in zip(utility_restore_params, utility_restore_requires_grad):
                p.requires_grad_(requires_grad)
            save_checkpoint(model, jumprec)

        if cfg.use_next_agreement_head and cfg.next_agreement_steps > 0:
            if jumprec.next_pred_heads is None:
                raise RuntimeError("next_agreement_steps requires cfg.use_next_agreement_head")
            next_restore_params = list(jumprec.parameters())
            next_restore_requires_grad = [p.requires_grad for p in next_restore_params]
            for p in next_restore_params:
                p.requires_grad_(False)
            for p in jumprec.next_pred_heads.parameters():
                p.requires_grad_(True)
            opt_next = torch.optim.AdamW(jumprec.next_pred_heads.parameters(), lr=cfg.next_agreement_lr)
            t_next = time.time()
            jumprec.eval()
            jumprec.next_pred_heads.train()
            for step in range(1, cfg.next_agreement_steps + 1):
                input_ids, attention_mask, lengths, target, _, hops, _ = batch_encoded(cfg.batch_size)
                with torch.no_grad():
                    state0, layer_mask, position_ids, position_embeddings, _, _ = teacher_collect(
                        input_ids,
                        attention_mask,
                        lengths,
                    )
                out = jumprec.forward_encoded(
                    state0.detach(),
                    layer_mask,
                    position_ids,
                    position_embeddings,
                    lengths,
                )
                pred_stack = torch.stack(
                    [logits_i.detach().argmax(dim=-1) for logits_i in out["logits"]],
                    dim=0,
                )
                next_losses = [
                    weighted_ce(out["next_pred"][c], pred_stack[c + 1], hops)
                    for c in range(cfg.max_correct)
                ]
                next_loss = torch.stack(next_losses).mean()
                opt_next.zero_grad()
                next_loss.backward()
                torch.nn.utils.clip_grad_norm_(jumprec.next_pred_heads.parameters(), 1.0)
                opt_next.step()
                if step % cfg.log_every == 0 or step == cfg.next_agreement_steps:
                    next_logits = torch.stack([x.detach() for x in out["next_pred"]], dim=0)
                    next_probs = F.softmax(next_logits, dim=-1)
                    next_conf, next_pred = next_probs.max(dim=-1)
                    target_next = pred_stack[1:]
                    next_acc = (next_pred == target_next).float().mean().item()
                    proxy_agree = next_pred == pred_stack[:-1]
                    true_agree = pred_stack[:-1] == pred_stack[1:]
                    precision_denom = proxy_agree.float().sum().clamp_min(1.0)
                    proxy_precision = (proxy_agree & true_agree).float().sum().item() / precision_denom.item()
                    print(
                        f"  nextagree step {step:5d}/{cfg.next_agreement_steps} "
                        f"loss {next_loss.item():.4f} "
                        f"next_acc {next_acc*100:.1f}% "
                        f"proxy_agree {proxy_agree.float().mean().item()*100:.1f}% "
                        f"proxy_precision {proxy_precision*100:.1f}% "
                        f"next_conf {next_conf.float().mean().item():.3f} "
                        f"elapsed {time.time()-t_next:.1f}s",
                        flush=True,
                    )
            for p, requires_grad in zip(next_restore_params, next_restore_requires_grad):
                p.requires_grad_(requires_grad)
            save_checkpoint(model, jumprec)

        if cfg.use_consistency_head and cfg.consistency_steps > 0:
            if jumprec.consistency_heads is None:
                raise RuntimeError("consistency_steps requires cfg.use_consistency_head")
            consistency_restore_params = list(jumprec.parameters())
            consistency_restore_requires_grad = [p.requires_grad for p in consistency_restore_params]
            for p in consistency_restore_params:
                p.requires_grad_(False)
            for p in jumprec.consistency_heads.parameters():
                p.requires_grad_(True)
            opt_consistency = torch.optim.AdamW(jumprec.consistency_heads.parameters(), lr=cfg.consistency_lr)
            t_consistency = time.time()
            jumprec.eval()
            jumprec.consistency_heads.train()
            for step in range(1, cfg.consistency_steps + 1):
                input_ids, attention_mask, lengths, target, _, hops, _ = batch_encoded(cfg.batch_size)
                with torch.no_grad():
                    state0, layer_mask, position_ids, position_embeddings, _, teacher_logits = teacher_collect(
                        input_ids,
                        attention_mask,
                        lengths,
                    )
                    full_pred = teacher_logits.argmax(dim=-1)
                out = jumprec.forward_encoded(
                    state0.detach(),
                    layer_mask,
                    position_ids,
                    position_embeddings,
                    lengths,
                )
                pred_stack = torch.stack(
                    [logits_i.detach().argmax(dim=-1) for logits_i in out["logits"]],
                    dim=0,
                )
                consistency_logits = torch.stack(out["consistency"], dim=0)
                stable_target = torch.zeros_like(consistency_logits)
                if cfg.max_correct > 0:
                    stable_target[:-1] = (pred_stack[:-1] == pred_stack[1:]).float()
                stable_target[-1] = (pred_stack[-1] == full_pred).float()
                consistency_loss = consistency_bce_with_logits(consistency_logits, stable_target, hops)
                opt_consistency.zero_grad()
                consistency_loss.backward()
                torch.nn.utils.clip_grad_norm_(jumprec.consistency_heads.parameters(), 1.0)
                opt_consistency.step()
                if step % cfg.log_every == 0 or step == cfg.consistency_steps:
                    consistency_prob = torch.sigmoid(consistency_logits.detach())
                    consistency_pred = consistency_prob >= 0.5
                    stable_bool = stable_target.bool()
                    pred_pos = consistency_pred.float().mean().item()
                    target_pos = stable_target.float().mean().item()
                    acc = (consistency_pred == stable_bool).float().mean().item()
                    precision_denom = consistency_pred.float().sum().clamp_min(1.0)
                    precision = (consistency_pred & stable_bool).float().sum().item() / precision_denom.item()
                    final_pred_pos = consistency_pred[-1].float().mean().item()
                    final_target_pos = stable_target[-1].float().mean().item()
                    print(
                        f"  cats step {step:5d}/{cfg.consistency_steps} "
                        f"loss {consistency_loss.item():.4f} "
                        f"acc {acc*100:.1f}% pred_stable {pred_pos*100:.1f}% "
                        f"target_stable {target_pos*100:.1f}% precision {precision*100:.1f}% "
                        f"final_pred {final_pred_pos*100:.1f}% final_target {final_target_pos*100:.1f}% "
                        f"elapsed {time.time()-t_consistency:.1f}s",
                        flush=True,
                    )
            for p, requires_grad in zip(consistency_restore_params, consistency_restore_requires_grad):
                p.requires_grad_(requires_grad)
            save_checkpoint(model, jumprec)

        def eval_jumprec(audit_variant: str = "normal", include_heldout: bool = True):
            jumprec.eval()
            thresholds = [(0.80, "080"), (0.90, "090"), (0.95, "095")]
            router_policies = [("no_agree", False), ("agree", True)]
            stability_policy_thresholds = [0.50, 0.70, 0.90]
            stability_policy_names = (
                [f"stable_{int(threshold * 100):03d}" for threshold in stability_policy_thresholds]
                if cfg.use_stability_head
                else []
            )
            budget_policy_names = (
                ["budget", "budget_escalate", "budget_open", "budget_scan_up"]
                if cfg.use_budget_controller
                else []
            )
            utility_policy_defs = [("utility", False), ("utility_guarded", True)] if cfg.use_utility_router else []
            utility_policy_names = [name for name, _ in utility_policy_defs]
            utility_then_agree_floors = [0.00, 0.30, 0.50]
            utility_then_agree_policy_names = (
                [
                    f"utility_then_agree_{int(confirm_floor * 100):03d}"
                    for confirm_floor in utility_then_agree_floors
                ]
                if cfg.use_utility_router
                else []
            )
            agree_then_utility_floors = [0.90, 0.95, 0.99]
            agree_then_utility_policy_names = (
                [
                    f"agree_then_utility_{int(utility_floor * 100):03d}"
                    for utility_floor in agree_then_utility_floors
                ]
                if cfg.use_utility_router
                else []
            )
            consistency_policy_thresholds = [0.50, 0.70, 0.90]
            consistency_policy_names = (
                [f"utility_cats_{int(threshold * 100):03d}" for threshold in consistency_policy_thresholds]
                if cfg.use_consistency_head and cfg.use_utility_router
                else []
            )
            next_agreement_thresholds = [0.00, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.90]
            next_agreement_policy_names = (
                [f"nextagree_{int(threshold * 100):03d}" for threshold in next_agreement_thresholds]
                if cfg.use_next_agreement_head
                else []
            )
            all_policy_names = [
                policy_name for policy_name, _ in router_policies
            ]
            all_policy_names += stability_policy_names
            all_policy_names += budget_policy_names
            all_policy_names += utility_policy_names
            all_policy_names += utility_then_agree_policy_names
            all_policy_names += agree_then_utility_policy_names
            all_policy_names += consistency_policy_names
            all_policy_names += next_agreement_policy_names
            full_core_layers = float(cfg.loop_steps * cfg.core_layers)
            metrics = {f"jump_c{c}_acc": 0.0 for c in range(cfg.max_correct + 1)}
            if cfg.use_budget_controller:
                metrics.update(
                    {
                        "budget_controller_target_acc": 0.0,
                        "budget_controller_under_rate": 0.0,
                        "budget_controller_over_rate": 0.0,
                        "budget_controller_within_one_rate": 0.0,
                        "budget_controller_avg_abs_tail_error": 0.0,
                        "budget_controller_pred_fallback_rate": 0.0,
                        "budget_controller_target_fallback_rate": 0.0,
                        "budget_controller_avg_pred_tail_loops": 0.0,
                        "budget_controller_avg_target_tail_loops": 0.0,
                    }
                )
                budget_confusion = torch.zeros(
                    cfg.max_correct + 2,
                    cfg.max_correct + 2,
                    dtype=torch.long,
                )
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

            def route_stability_predictions(
                pred_stack,
                verify_stack,
                margin_stack,
                max_prob_stack,
                stability_stack,
                full_pred,
                target,
                threshold: float,
                stability_threshold: float,
            ):
                trusted = torch.zeros_like(target, dtype=torch.bool)
                chosen = torch.full_like(target, cfg.loop_steps)
                routed_pred = full_pred
                for c in range(cfg.max_correct + 1):
                    accept = (
                        (~trusted)
                        & (c < cfg.max_correct)
                        & (verify_stack[c] >= threshold)
                        & (margin_stack[c] >= 0.05)
                        & (max_prob_stack[c] >= 0.45)
                        & (stability_stack[c] >= stability_threshold)
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

            def route_utility_predictions(
                pred_stack,
                utility_stack,
                margin_stack,
                max_prob_stack,
                full_pred,
                target,
                threshold: float,
                guarded: bool,
            ):
                trusted = torch.zeros_like(target, dtype=torch.bool)
                chosen = torch.full_like(target, cfg.loop_steps)
                routed_pred = full_pred
                for c in range(cfg.max_correct + 1):
                    accept = (~trusted) & (utility_stack[c] >= threshold)
                    if guarded:
                        accept = accept & (margin_stack[c] >= 0.05) & (max_prob_stack[c] >= 0.45)
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

            def route_utility_then_agree_predictions(
                pred_stack,
                verify_stack,
                utility_stack,
                margin_stack,
                max_prob_stack,
                agree_stack,
                full_pred,
                target,
                threshold: float,
                confirm_floor: float,
            ):
                trusted = torch.zeros_like(target, dtype=torch.bool)
                chosen = torch.full_like(target, cfg.loop_steps)
                routed_pred = full_pred
                for c in range(cfg.max_correct + 1):
                    direct = utility_stack[c] >= threshold
                    confirmed = (
                        (utility_stack[c] >= confirm_floor)
                        & (verify_stack[c] >= threshold)
                        & (margin_stack[c] >= 0.05)
                        & (max_prob_stack[c] >= 0.45)
                        & agree_stack[c]
                    )
                    accept = (~trusted) & (direct | confirmed)
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

            def route_agree_then_utility_predictions(
                pred_stack,
                verify_stack,
                utility_stack,
                margin_stack,
                max_prob_stack,
                agree_stack,
                full_pred,
                target,
                threshold: float,
                utility_floor: float,
            ):
                trusted = torch.zeros_like(target, dtype=torch.bool)
                chosen = torch.full_like(target, cfg.loop_steps)
                routed_pred = full_pred
                for c in range(cfg.max_correct + 1):
                    agreed = (
                        (verify_stack[c] >= threshold)
                        & (margin_stack[c] >= 0.05)
                        & (max_prob_stack[c] >= 0.45)
                        & agree_stack[c]
                    )
                    direct = utility_stack[c] >= utility_floor
                    accept = (~trusted) & (agreed | direct)
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

            def route_utility_consistency_predictions(
                pred_stack,
                utility_stack,
                consistency_stack,
                full_pred,
                target,
                utility_threshold: float,
                consistency_threshold: float,
            ):
                trusted = torch.zeros_like(target, dtype=torch.bool)
                chosen = torch.full_like(target, cfg.loop_steps)
                routed_pred = full_pred
                for c in range(cfg.max_correct + 1):
                    accept = (
                        (~trusted)
                        & (utility_stack[c] >= utility_threshold)
                        & (consistency_stack[c] >= consistency_threshold)
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

            def route_nextagreement_predictions(
                pred_stack,
                verify_stack,
                margin_stack,
                max_prob_stack,
                next_pred_class,
                next_pred_conf,
                full_pred,
                target,
                threshold: float,
                next_threshold: float,
            ):
                trusted = torch.zeros_like(target, dtype=torch.bool)
                chosen = torch.full_like(target, cfg.loop_steps)
                routed_pred = full_pred
                for c in range(cfg.max_correct):
                    accept = (
                        (~trusted)
                        & (verify_stack[c] >= threshold)
                        & (margin_stack[c] >= 0.05)
                        & (max_prob_stack[c] >= 0.45)
                        & (next_pred_conf[c] >= next_threshold)
                        & (next_pred_class[c] == pred_stack[c])
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

            def route_budget_predictions(
                pred_stack,
                verify_stack,
                margin_stack,
                max_prob_stack,
                full_pred,
                threshold: float,
                budget_choice,
            ):
                fallback = cfg.max_correct + 1
                budget_open = budget_choice < fallback
                clamped_choice = budget_choice.clamp(0, cfg.max_correct)
                gather_idx = clamped_choice.view(1, -1)
                selected_pred = pred_stack.gather(0, gather_idx).squeeze(0)
                selected_verify = verify_stack.gather(0, gather_idx).squeeze(0)
                selected_margin = margin_stack.gather(0, gather_idx).squeeze(0)
                selected_max_prob = max_prob_stack.gather(0, gather_idx).squeeze(0)
                trusted = (
                    budget_open
                    & (selected_verify >= threshold)
                    & (selected_margin >= 0.05)
                    & (selected_max_prob >= 0.45)
                )
                chosen = torch.where(trusted, clamped_choice, torch.full_like(clamped_choice, cfg.loop_steps))
                routed_pred = torch.where(trusted, selected_pred, full_pred)
                core_layers = torch.where(
                    trusted,
                    torch.full_like(chosen.float(), float(cfg.jump_layers))
                    + chosen.float() * float(cfg.core_layers),
                    torch.full_like(chosen.float(), full_core_layers),
                )
                return routed_pred, trusted, chosen, core_layers

            def route_budget_escalate_predictions(
                pred_stack,
                verify_stack,
                margin_stack,
                max_prob_stack,
                full_pred,
                threshold: float,
                budget_choice,
            ):
                fallback = cfg.max_correct + 1
                budget_open = budget_choice < fallback
                first_choice = budget_choice.clamp(0, cfg.max_correct)
                first_idx = first_choice.view(1, -1)
                first_pred = pred_stack.gather(0, first_idx).squeeze(0)
                first_verify = verify_stack.gather(0, first_idx).squeeze(0)
                first_margin = margin_stack.gather(0, first_idx).squeeze(0)
                first_max_prob = max_prob_stack.gather(0, first_idx).squeeze(0)
                first_trusted = (
                    budget_open
                    & (first_verify >= threshold)
                    & (first_margin >= 0.05)
                    & (first_max_prob >= 0.45)
                )

                second_choice = (first_choice + 1).clamp(max=cfg.max_correct)
                second_idx = second_choice.view(1, -1)
                second_pred = pred_stack.gather(0, second_idx).squeeze(0)
                second_verify = verify_stack.gather(0, second_idx).squeeze(0)
                second_margin = margin_stack.gather(0, second_idx).squeeze(0)
                second_max_prob = max_prob_stack.gather(0, second_idx).squeeze(0)
                can_escalate = (~first_trusted) & (budget_choice < cfg.max_correct)
                second_trusted = (
                    can_escalate
                    & (second_verify >= threshold)
                    & (second_margin >= 0.05)
                    & (second_max_prob >= 0.45)
                )

                trusted = first_trusted | second_trusted
                chosen = torch.where(
                    first_trusted,
                    first_choice,
                    torch.where(second_trusted, second_choice, torch.full_like(first_choice, cfg.loop_steps)),
                )
                routed_pred = torch.where(
                    first_trusted,
                    first_pred,
                    torch.where(second_trusted, second_pred, full_pred),
                )
                core_layers = torch.where(
                    trusted,
                    torch.full_like(chosen.float(), float(cfg.jump_layers))
                    + chosen.float() * float(cfg.core_layers),
                    torch.full_like(chosen.float(), full_core_layers),
                )
                return routed_pred, trusted, chosen, core_layers

            def route_budget_open_predictions(
                pred_stack,
                full_pred,
                budget_choice,
            ):
                fallback = cfg.max_correct + 1
                budget_open = budget_choice < fallback
                clamped_choice = budget_choice.clamp(0, cfg.max_correct)
                selected_pred = pred_stack.gather(0, clamped_choice.view(1, -1)).squeeze(0)
                chosen = torch.where(
                    budget_open,
                    clamped_choice,
                    torch.full_like(clamped_choice, cfg.loop_steps),
                )
                routed_pred = torch.where(budget_open, selected_pred, full_pred)
                core_layers = torch.where(
                    budget_open,
                    torch.full_like(chosen.float(), float(cfg.jump_layers))
                    + chosen.float() * float(cfg.core_layers),
                    torch.full_like(chosen.float(), full_core_layers),
                )
                return routed_pred, budget_open, chosen, core_layers

            def route_budget_scan_up_predictions(
                pred_stack,
                verify_stack,
                margin_stack,
                max_prob_stack,
                full_pred,
                threshold: float,
                budget_choice,
            ):
                fallback = cfg.max_correct + 1
                budget_open = budget_choice < fallback
                trusted = torch.zeros_like(budget_choice, dtype=torch.bool)
                chosen = torch.full_like(budget_choice, cfg.loop_steps)
                routed_pred = full_pred
                for c in range(cfg.max_correct + 1):
                    eligible = budget_open & (~trusted) & (budget_choice <= c)
                    accept = (
                        eligible
                        & (verify_stack[c] >= threshold)
                        & (margin_stack[c] >= 0.05)
                        & (max_prob_stack[c] >= 0.45)
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
                                "false_accept_rate": None,
                                "fallback_count": 0,
                                "accepted_by_budget": [0 for _ in range(cfg.max_correct + 1)],
                                "accepted_correct_by_budget": [0.0 for _ in range(cfg.max_correct + 1)],
                                "accepted_precision_by_budget": [None for _ in range(cfg.max_correct + 1)],
                                "accepted_share_by_budget": [0.0 for _ in range(cfg.max_correct + 1)],
                            }
                            for threshold in threshold_values
                        }
                        for policy_name in all_policy_names
                    },
                }
                accepted_correct = {
                    policy_name: {threshold_key(threshold): 0.0 for threshold in threshold_values}
                    for policy_name in all_policy_names
                }
                accepted_count = {
                    policy_name: {threshold_key(threshold): 0 for threshold in threshold_values}
                    for policy_name in all_policy_names
                }
                fallback_count = {
                    policy_name: {threshold_key(threshold): 0 for threshold in threshold_values}
                    for policy_name in all_policy_names
                }
                accepted_by_budget = {
                    policy_name: {
                        threshold_key(threshold): [0 for _ in range(cfg.max_correct + 1)]
                        for threshold in threshold_values
                    }
                    for policy_name in all_policy_names
                }
                accepted_correct_by_budget = {
                    policy_name: {
                        threshold_key(threshold): [0.0 for _ in range(cfg.max_correct + 1)]
                        for threshold in threshold_values
                    }
                    for policy_name in all_policy_names
                }

                def record_accepts(policy_name, key, trusted, chosen, routed_pred, target):
                    accepted_count[policy_name][key] += int(trusted.sum().item())
                    fallback_count[policy_name][key] += int((~trusted).sum().item())
                    if not trusted.any():
                        return
                    accepted_is_correct = (routed_pred[trusted] == target[trusted]).float()
                    accepted_correct[policy_name][key] += accepted_is_correct.sum().item()
                    accepted_budget = chosen[trusted].clamp(0, cfg.max_correct).long()
                    budget_counts = torch.bincount(
                        accepted_budget.detach().cpu(),
                        minlength=cfg.max_correct + 1,
                    )
                    budget_correct = torch.bincount(
                        accepted_budget.detach().cpu(),
                        weights=accepted_is_correct.detach().cpu(),
                        minlength=cfg.max_correct + 1,
                    )
                    for budget in range(cfg.max_correct + 1):
                        accepted_by_budget[policy_name][key][budget] += int(budget_counts[budget].item())
                        accepted_correct_by_budget[policy_name][key][budget] += float(
                            budget_correct[budget].item()
                        )

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
                        stability_stack = None
                        if cfg.use_stability_head:
                            stability_stack = torch.stack(
                                [torch.sigmoid(v) for v in jump_out["stability"]],
                                dim=0,
                            )
                        utility_stack = None
                        if cfg.use_utility_router:
                            utility_stack = torch.stack(
                                [torch.sigmoid(v) for v in jump_out["utility"]],
                                dim=0,
                            )
                        consistency_stack = None
                        if cfg.use_consistency_head:
                            consistency_stack = torch.stack(
                                [torch.sigmoid(v) for v in jump_out["consistency"]],
                                dim=0,
                            )
                        next_pred_class = None
                        next_pred_conf = None
                        if cfg.use_next_agreement_head:
                            next_logits_stack = torch.stack(jump_out["next_pred"], dim=0)
                            next_probs = F.softmax(next_logits_stack, dim=-1)
                            next_pred_conf, next_pred_class = next_probs.max(dim=-1)
                        budget_choice = None
                        if cfg.use_budget_controller:
                            budget_choice = jumprec.budget_logits(state0, lengths).argmax(dim=-1)
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
                                record_accepts(policy_name, key, trusted, chosen, routed_pred, target)
                            if cfg.use_stability_head:
                                for policy_name, stability_threshold in zip(
                                    stability_policy_names,
                                    stability_policy_thresholds,
                                ):
                                    routed_pred, trusted, chosen, core_layers = route_stability_predictions(
                                        pred_stack,
                                        verify_stack,
                                        margin_stack,
                                        max_prob_stack,
                                        stability_stack,
                                        full_pred,
                                        target,
                                        threshold,
                                        stability_threshold,
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
                                    record_accepts(policy_name, key, trusted, chosen, routed_pred, target)
                            if cfg.use_utility_router:
                                for policy_name, guarded in utility_policy_defs:
                                    routed_pred, trusted, chosen, core_layers = route_utility_predictions(
                                        pred_stack,
                                        utility_stack,
                                        margin_stack,
                                        max_prob_stack,
                                        full_pred,
                                        target,
                                        threshold,
                                        guarded,
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
                                    record_accepts(policy_name, key, trusted, chosen, routed_pred, target)
                                for policy_name, confirm_floor in zip(
                                    utility_then_agree_policy_names,
                                    utility_then_agree_floors,
                                ):
                                    routed_pred, trusted, chosen, core_layers = (
                                        route_utility_then_agree_predictions(
                                            pred_stack,
                                            verify_stack,
                                            utility_stack,
                                            margin_stack,
                                            max_prob_stack,
                                            agree_stack,
                                            full_pred,
                                            target,
                                            threshold,
                                            confirm_floor,
                                        )
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
                                    record_accepts(policy_name, key, trusted, chosen, routed_pred, target)
                                for policy_name, utility_floor in zip(
                                    agree_then_utility_policy_names,
                                    agree_then_utility_floors,
                                ):
                                    routed_pred, trusted, chosen, core_layers = (
                                        route_agree_then_utility_predictions(
                                            pred_stack,
                                            verify_stack,
                                            utility_stack,
                                            margin_stack,
                                            max_prob_stack,
                                            agree_stack,
                                            full_pred,
                                            target,
                                            threshold,
                                            utility_floor,
                                        )
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
                                    record_accepts(policy_name, key, trusted, chosen, routed_pred, target)
                            if cfg.use_consistency_head and cfg.use_utility_router:
                                for policy_name, consistency_threshold in zip(
                                    consistency_policy_names,
                                    consistency_policy_thresholds,
                                ):
                                    routed_pred, trusted, chosen, core_layers = (
                                        route_utility_consistency_predictions(
                                            pred_stack,
                                            utility_stack,
                                            consistency_stack,
                                            full_pred,
                                            target,
                                            threshold,
                                            consistency_threshold,
                                        )
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
                                    record_accepts(policy_name, key, trusted, chosen, routed_pred, target)
                            if cfg.use_next_agreement_head:
                                for policy_name, next_threshold in zip(
                                    next_agreement_policy_names,
                                    next_agreement_thresholds,
                                ):
                                    routed_pred, trusted, chosen, core_layers = route_nextagreement_predictions(
                                        pred_stack,
                                        verify_stack,
                                        margin_stack,
                                        max_prob_stack,
                                        next_pred_class,
                                        next_pred_conf,
                                        full_pred,
                                        target,
                                        threshold,
                                        next_threshold,
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
                                    record_accepts(policy_name, key, trusted, chosen, routed_pred, target)
                            if cfg.use_budget_controller:
                                routed_pred, trusted, chosen, core_layers = route_budget_predictions(
                                    pred_stack,
                                    verify_stack,
                                    margin_stack,
                                    max_prob_stack,
                                    full_pred,
                                    threshold,
                                    budget_choice,
                                )
                                item = out["policies"]["budget"][key]
                                item["acc"] += (routed_pred == target).float().mean().item()
                                item["full_loop_rate"] += (~trusted).float().mean().item()
                                item["coverage"] += trusted.float().mean().item()
                                item["avg_tail_loops"] += torch.where(
                                    trusted,
                                    chosen.float(),
                                    torch.full_like(chosen.float(), float(cfg.loop_steps)),
                                ).mean().item()
                                item["avg_core_layers"] += core_layers.mean().item()
                                record_accepts("budget", key, trusted, chosen, routed_pred, target)
                                routed_pred, trusted, chosen, core_layers = route_budget_escalate_predictions(
                                    pred_stack,
                                    verify_stack,
                                    margin_stack,
                                    max_prob_stack,
                                    full_pred,
                                    threshold,
                                    budget_choice,
                                )
                                item = out["policies"]["budget_escalate"][key]
                                item["acc"] += (routed_pred == target).float().mean().item()
                                item["full_loop_rate"] += (~trusted).float().mean().item()
                                item["coverage"] += trusted.float().mean().item()
                                item["avg_tail_loops"] += torch.where(
                                    trusted,
                                    chosen.float(),
                                    torch.full_like(chosen.float(), float(cfg.loop_steps)),
                                ).mean().item()
                                item["avg_core_layers"] += core_layers.mean().item()
                                record_accepts("budget_escalate", key, trusted, chosen, routed_pred, target)
                                routed_pred, trusted, chosen, core_layers = route_budget_open_predictions(
                                    pred_stack,
                                    full_pred,
                                    budget_choice,
                                )
                                item = out["policies"]["budget_open"][key]
                                item["acc"] += (routed_pred == target).float().mean().item()
                                item["full_loop_rate"] += (~trusted).float().mean().item()
                                item["coverage"] += trusted.float().mean().item()
                                item["avg_tail_loops"] += torch.where(
                                    trusted,
                                    chosen.float(),
                                    torch.full_like(chosen.float(), float(cfg.loop_steps)),
                                ).mean().item()
                                item["avg_core_layers"] += core_layers.mean().item()
                                record_accepts("budget_open", key, trusted, chosen, routed_pred, target)
                                routed_pred, trusted, chosen, core_layers = route_budget_scan_up_predictions(
                                    pred_stack,
                                    verify_stack,
                                    margin_stack,
                                    max_prob_stack,
                                    full_pred,
                                    threshold,
                                    budget_choice,
                                )
                                item = out["policies"]["budget_scan_up"][key]
                                item["acc"] += (routed_pred == target).float().mean().item()
                                item["full_loop_rate"] += (~trusted).float().mean().item()
                                item["coverage"] += trusted.float().mean().item()
                                item["avg_tail_loops"] += torch.where(
                                    trusted,
                                    chosen.float(),
                                    torch.full_like(chosen.float(), float(cfg.loop_steps)),
                                ).mean().item()
                                item["avg_core_layers"] += core_layers.mean().item()
                                record_accepts("budget_scan_up", key, trusted, chosen, routed_pred, target)
                out["full_teacher_acc"] /= num_batches
                for policy_name in all_policy_names:
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
                        item["fallback_count"] = fallback_count[policy_name][key]
                        item["accepted_by_budget"] = accepted_by_budget[policy_name][key]
                        item["accepted_correct_by_budget"] = accepted_correct_by_budget[policy_name][key]
                        if accepted_count[policy_name][key] > 0:
                            item["accepted_precision"] = (
                                accepted_correct[policy_name][key] / accepted_count[policy_name][key]
                            )
                            item["false_accept_rate"] = 1.0 - item["accepted_precision"]
                            item["accepted_share_by_budget"] = [
                                count / accepted_count[policy_name][key]
                                for count in accepted_by_budget[policy_name][key]
                            ]
                        item["accepted_precision_by_budget"] = [
                            (
                                accepted_correct_by_budget[policy_name][key][budget]
                                / accepted_by_budget[policy_name][key][budget]
                            )
                            if accepted_by_budget[policy_name][key][budget] > 0
                            else None
                            for budget in range(cfg.max_correct + 1)
                        ]
                return out

            def collect_router_records(num_batches: int):
                records = {
                    "full_correct": [],
                    "pred_correct": [],
                    "verify": [],
                    "margin": [],
                    "max_prob": [],
                }
                if cfg.use_utility_router:
                    records["utility"] = []
                needs_agreement_records = cfg.router_probe_audit or cfg.router_selective_agree_audit
                if needs_agreement_records:
                    records["agree"] = []
                if cfg.router_probe_audit:
                    records["entropy"] = []
                    records["probs"] = []
                    records["features"] = []
                    if cfg.use_stability_head:
                        records["stability"] = []
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
                        logits_stack = torch.stack(jump_out["logits"], dim=0)
                        probs = F.softmax(logits_stack, dim=-1)
                        top2 = probs.topk(2, dim=-1).values
                        records["full_correct"].append((full_pred == target).detach().cpu())
                        records["pred_correct"].append((pred_stack == target.unsqueeze(0)).detach().cpu())
                        records["verify"].append(
                            torch.stack([torch.sigmoid(v) for v in jump_out["verify"]], dim=0).detach().cpu()
                        )
                        if cfg.use_utility_router:
                            records["utility"].append(
                                torch.stack(
                                    [torch.sigmoid(v) for v in jump_out["utility"]],
                                    dim=0,
                                )
                                .detach()
                                .cpu()
                            )
                        records["margin"].append((top2[:, :, 0] - top2[:, :, 1]).detach().cpu())
                        records["max_prob"].append(top2[:, :, 0].detach().cpu())
                        if needs_agreement_records:
                            agree_stack = torch.zeros_like(pred_stack, dtype=torch.bool)
                            if cfg.max_correct > 0:
                                agree_stack[:-1] = pred_stack[:-1] == pred_stack[1:]
                            records["agree"].append(agree_stack.detach().cpu())
                        if cfg.router_probe_audit:
                            log_probs = F.log_softmax(logits_stack, dim=-1)
                            entropy = -(probs * log_probs).sum(dim=-1) / math.log(cfg.n_nodes)
                            feature_stack = torch.stack(
                                [
                                    jumprec.verifier_features(final_state, logits_i, lengths)
                                    for final_state, logits_i in zip(
                                        jump_out["final_states"],
                                        jump_out["logits"],
                                    )
                                ],
                                dim=0,
                            )
                            records["entropy"].append(entropy.detach().cpu())
                            records["probs"].append(probs.detach().cpu())
                            records["features"].append(feature_stack.detach().cpu())
                            if cfg.use_stability_head:
                                records["stability"].append(
                                    torch.stack(
                                        [torch.sigmoid(v) for v in jump_out["stability"]],
                                        dim=0,
                                    )
                                    .detach()
                                    .cpu()
                                )
                out_records = {
                    "full_correct": torch.cat(records["full_correct"], dim=0),
                    "pred_correct": torch.cat(records["pred_correct"], dim=1),
                    "verify": torch.cat(records["verify"], dim=1),
                    "margin": torch.cat(records["margin"], dim=1),
                    "max_prob": torch.cat(records["max_prob"], dim=1),
                }
                if cfg.use_utility_router:
                    out_records["utility"] = torch.cat(records["utility"], dim=1)
                if needs_agreement_records:
                    out_records["agree"] = torch.cat(records["agree"], dim=1)
                if cfg.router_probe_audit:
                    out_records["entropy"] = torch.cat(records["entropy"], dim=1)
                    out_records["probs"] = torch.cat(records["probs"], dim=1)
                    out_records["features"] = torch.cat(records["features"], dim=1)
                    if cfg.use_stability_head:
                        out_records["stability"] = torch.cat(records["stability"], dim=1)
                return out_records

            def per_budget_route_item(thresholds_by_budget, trusted, chosen, routed_correct):
                n = routed_correct.numel()
                core_layers = torch.where(
                    trusted,
                    torch.full((n,), float(cfg.jump_layers)) + chosen.float() * float(cfg.core_layers),
                    torch.full((n,), full_core_layers),
                )
                accepted_count_i = int(trusted.sum().item())
                accepted_correct_i = (
                    float(routed_correct[trusted].float().sum().item()) if accepted_count_i > 0 else 0.0
                )
                if accepted_count_i > 0:
                    accepted_budget = chosen[trusted].clamp(0, cfg.max_correct).long()
                    accepted_by_budget = torch.bincount(
                        accepted_budget,
                        minlength=cfg.max_correct + 1,
                    ).tolist()
                    accepted_correct_by_budget = torch.bincount(
                        accepted_budget,
                        weights=routed_correct[trusted].float(),
                        minlength=cfg.max_correct + 1,
                    ).tolist()
                else:
                    accepted_by_budget = [0 for _ in range(cfg.max_correct + 1)]
                    accepted_correct_by_budget = [0.0 for _ in range(cfg.max_correct + 1)]
                item = {
                    "thresholds": [float(t) for t in thresholds_by_budget],
                    "threshold_key": ",".join(threshold_key(t) for t in thresholds_by_budget),
                    "acc": routed_correct.float().mean().item(),
                    "full_loop_rate": (~trusted).float().mean().item(),
                    "coverage": trusted.float().mean().item(),
                    "avg_tail_loops": torch.where(
                        trusted,
                        chosen.float(),
                        torch.full_like(chosen.float(), float(cfg.loop_steps)),
                    ).mean().item(),
                    "avg_core_layers": core_layers.mean().item(),
                    "core_savings_vs_full_pct": (
                        1.0 - core_layers.mean().item() / full_core_layers
                    )
                    * 100.0,
                    "accepted_count": accepted_count_i,
                    "accepted_precision": (
                        accepted_correct_i / accepted_count_i if accepted_count_i > 0 else None
                    ),
                    "false_accept_rate": (
                        1.0 - accepted_correct_i / accepted_count_i if accepted_count_i > 0 else None
                    ),
                    "accepted_by_budget": accepted_by_budget,
                    "accepted_correct_by_budget": accepted_correct_by_budget,
                    "accepted_precision_by_budget": [
                        (
                            accepted_correct_by_budget[budget] / accepted_by_budget[budget]
                            if accepted_by_budget[budget] > 0
                            else None
                        )
                        for budget in range(cfg.max_correct + 1)
                    ],
                    "accepted_share_by_budget": [
                        (
                            accepted_by_budget[budget] / accepted_count_i
                            if accepted_count_i > 0
                            else 0.0
                        )
                        for budget in range(cfg.max_correct + 1)
                    ],
                }
                return item

            def evaluate_per_budget_thresholds(records, thresholds_by_budget):
                full_correct = records["full_correct"].bool()
                pred_correct = records["pred_correct"].bool()
                verify = records["verify"]
                margin = records["margin"]
                max_prob = records["max_prob"]
                n = full_correct.numel()
                trusted = torch.zeros(n, dtype=torch.bool)
                chosen = torch.full((n,), cfg.loop_steps, dtype=torch.long)
                routed_correct = full_correct.clone()
                for c, threshold in enumerate(thresholds_by_budget):
                    accept = (
                        (~trusted)
                        & (verify[c] >= float(threshold))
                        & (margin[c] >= 0.05)
                        & (max_prob[c] >= 0.45)
                    )
                    chosen[accept] = c
                    routed_correct[accept] = pred_correct[c, accept]
                    trusted |= accept
                return per_budget_route_item(thresholds_by_budget, trusted, chosen, routed_correct)

            def evaluate_per_budget_utility_thresholds(records, thresholds_by_budget, guarded: bool = False):
                full_correct = records["full_correct"].bool()
                pred_correct = records["pred_correct"].bool()
                utility = records["utility"]
                margin = records["margin"]
                max_prob = records["max_prob"]
                n = full_correct.numel()
                trusted = torch.zeros(n, dtype=torch.bool)
                chosen = torch.full((n,), cfg.loop_steps, dtype=torch.long)
                routed_correct = full_correct.clone()
                for c, threshold in enumerate(thresholds_by_budget):
                    accept = (~trusted) & (utility[c] >= float(threshold))
                    if guarded:
                        accept = accept & (margin[c] >= 0.05) & (max_prob[c] >= 0.45)
                    chosen[accept] = c
                    routed_correct[accept] = pred_correct[c, accept]
                    trusted |= accept
                return per_budget_route_item(thresholds_by_budget, trusted, chosen, routed_correct)

            def selective_agree_item(
                verify_threshold,
                direct_floor,
                agree_floor,
                trusted,
                chosen,
                routed_correct,
                agreement_checks,
                agreement_check_core,
            ):
                item = per_budget_route_item(
                    [float(verify_threshold) for _ in range(cfg.max_correct + 1)],
                    trusted,
                    chosen,
                    routed_correct,
                )
                n = max(1, routed_correct.numel())
                item["verify_threshold"] = float(verify_threshold)
                item["direct_floor"] = float(direct_floor)
                item["agree_floor"] = float(agree_floor)
                item["agreement_check_count"] = int(agreement_checks.sum().item())
                item["agreement_check_rate"] = float(agreement_checks.float().mean().item())
                item["agreement_check_core_layers"] = float(agreement_check_core.float().sum().item() / n)
                item["effective_core_layers_with_checks"] = (
                    item["avg_core_layers"] + item["agreement_check_core_layers"]
                )
                item["effective_core_savings_vs_full_pct"] = (
                    1.0 - item["effective_core_layers_with_checks"] / full_core_layers
                ) * 100.0
                return item

            def evaluate_selective_agree(
                records,
                verify_threshold: float,
                direct_floor: float,
                agree_floor: float,
            ):
                full_correct = records["full_correct"].bool()
                pred_correct = records["pred_correct"].bool()
                utility = records["utility"]
                verify = records["verify"]
                margin = records["margin"]
                max_prob = records["max_prob"]
                agree = records["agree"].bool()
                n = full_correct.numel()
                trusted = torch.zeros(n, dtype=torch.bool)
                chosen = torch.full((n,), cfg.loop_steps, dtype=torch.long)
                routed_correct = full_correct.clone()
                agreement_checks = torch.zeros(n, dtype=torch.long)
                agreement_check_core = torch.zeros(n, dtype=torch.float32)
                for c in range(cfg.max_correct + 1):
                    direct = (~trusted) & (utility[c] >= float(direct_floor))
                    chosen[direct] = c
                    routed_correct[direct] = pred_correct[c, direct]
                    trusted |= direct

                    # The final budget has no adjacent-budget partner, so it can
                    # only be accepted directly or fall through to the teacher.
                    if c >= cfg.max_correct:
                        continue
                    check = (
                        (~trusted)
                        & (utility[c] >= float(agree_floor))
                        & (verify[c] >= float(verify_threshold))
                        & (margin[c] >= 0.05)
                        & (max_prob[c] >= 0.45)
                    )
                    agreement_checks += check.long()
                    agreement_check_core += check.float() * (
                        float(cfg.jump_layers) + float((c + 1) * cfg.core_layers)
                    )
                    accept = check & agree[c]
                    chosen[accept] = c
                    routed_correct[accept] = pred_correct[c, accept]
                    trusted |= accept
                return selective_agree_item(
                    verify_threshold,
                    direct_floor,
                    agree_floor,
                    trusted,
                    chosen,
                    routed_correct,
                    agreement_checks,
                    agreement_check_core,
                )

            def choose_per_budget_thresholds(records, threshold_values, monotone: bool = False):
                full_teacher_acc = records["full_correct"].float().mean().item()
                floor = full_teacher_acc - cfg.router_val_max_drop
                best_acceptable = None
                best_overall = None
                for thresholds_by_budget in itertools.product(threshold_values, repeat=cfg.max_correct + 1):
                    if monotone and any(
                        thresholds_by_budget[i] < thresholds_by_budget[i + 1]
                        for i in range(len(thresholds_by_budget) - 1)
                    ):
                        continue
                    item = evaluate_per_budget_thresholds(records, thresholds_by_budget)
                    acceptable_key = (item["avg_core_layers"], -item["acc"])
                    overall_key = (item["acc"], -item["avg_core_layers"])
                    if item["acc"] >= floor and (
                        best_acceptable is None or acceptable_key < best_acceptable[0]
                    ):
                        best_acceptable = (acceptable_key, thresholds_by_budget, item)
                    if best_overall is None or overall_key > best_overall[0]:
                        best_overall = (overall_key, thresholds_by_budget, item)
                if best_acceptable is not None:
                    _, thresholds_by_budget, item = best_acceptable
                    return thresholds_by_budget, item, "within_val_drop", floor, full_teacher_acc
                _, thresholds_by_budget, item = best_overall
                return thresholds_by_budget, item, "best_val_accuracy", floor, full_teacher_acc

            def choose_per_budget_utility_thresholds(
                records,
                threshold_values,
                monotone: bool = False,
                guarded: bool = False,
                max_exhaustive_combinations: int = 20000,
                max_drop: float | None = None,
            ):
                threshold_values = sorted(set(float(t) for t in threshold_values))
                full_teacher_acc = records["full_correct"].float().mean().item()
                floor = full_teacher_acc - (cfg.router_val_max_drop if max_drop is None else max_drop)
                best_acceptable = None
                best_overall = None
                num_budgets = cfg.max_correct + 1

                def monotone_ok(thresholds_by_budget):
                    return not monotone or not any(
                        thresholds_by_budget[i] < thresholds_by_budget[i + 1]
                        for i in range(len(thresholds_by_budget) - 1)
                    )

                def consider(thresholds_by_budget):
                    nonlocal best_acceptable, best_overall
                    thresholds_by_budget = tuple(float(t) for t in thresholds_by_budget)
                    if not monotone_ok(thresholds_by_budget):
                        return
                    item = evaluate_per_budget_utility_thresholds(
                        records,
                        thresholds_by_budget,
                        guarded=guarded,
                    )
                    acceptable_key = (item["avg_core_layers"], -item["acc"])
                    overall_key = (item["acc"], -item["avg_core_layers"])
                    if item["acc"] >= floor and (
                        best_acceptable is None or acceptable_key < best_acceptable[0]
                    ):
                        best_acceptable = (acceptable_key, thresholds_by_budget, item)
                    if best_overall is None or overall_key > best_overall[0]:
                        best_overall = (overall_key, thresholds_by_budget, item)

                def current_best():
                    if best_acceptable is not None:
                        return best_acceptable[1], best_acceptable[2], "within_val_drop"
                    return best_overall[1], best_overall[2], "best_val_accuracy"

                combo_count = len(threshold_values) ** num_budgets
                search_method = "exhaustive"
                if combo_count <= max_exhaustive_combinations:
                    for thresholds_by_budget in itertools.product(threshold_values, repeat=num_budgets):
                        consider(thresholds_by_budget)
                else:
                    max_coarse_values = 9
                    stride = max(1, math.ceil(len(threshold_values) / max_coarse_values))
                    coarse_values = sorted(set(threshold_values[::stride] + [threshold_values[-1]]))
                    while len(coarse_values) ** num_budgets > max_exhaustive_combinations and len(coarse_values) > 2:
                        stride += 1
                        coarse_values = sorted(set(threshold_values[::stride] + [threshold_values[-1]]))
                    for thresholds_by_budget in itertools.product(coarse_values, repeat=num_budgets):
                        consider(thresholds_by_budget)
                    start_thresholds, _, _ = current_best()
                    current = list(start_thresholds)
                    for _ in range(4):
                        improved = False
                        for budget in range(num_budgets):
                            local_best = None
                            for threshold in threshold_values:
                                trial = list(current)
                                trial[budget] = threshold
                                trial = tuple(trial)
                                if not monotone_ok(trial):
                                    continue
                                item = evaluate_per_budget_utility_thresholds(
                                    records,
                                    trial,
                                    guarded=guarded,
                                )
                                if item["acc"] >= floor:
                                    key = (0, item["avg_core_layers"], -item["acc"])
                                else:
                                    key = (1, -item["acc"], item["avg_core_layers"])
                                if local_best is None or key < local_best[0]:
                                    local_best = (key, trial)
                            if local_best is not None and tuple(current) != local_best[1]:
                                current = list(local_best[1])
                                improved = True
                                consider(current)
                        if not improved:
                            break
                    search_method = f"coarse_coordinate_{len(coarse_values)}"
                thresholds_by_budget, item, reason = current_best()
                item["threshold_search"] = search_method
                return thresholds_by_budget, item, reason, floor, full_teacher_acc

            def choose_selective_agree_thresholds(
                records,
                verify_threshold_values,
                max_drop: float | None = None,
                check_penalty: float = 0.0,
            ):
                full_teacher_acc = records["full_correct"].float().mean().item()
                floor = full_teacher_acc - (cfg.router_val_max_drop if max_drop is None else max_drop)
                utility_values = [0.0, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.97, 0.99]
                direct_values = [0.90, 0.95, 0.97, 0.99, 1.01]
                best_acceptable = None
                best_overall = None
                for verify_threshold in verify_threshold_values:
                    for direct_floor in direct_values:
                        for agree_floor in utility_values:
                            if direct_floor <= agree_floor:
                                continue
                            item = evaluate_selective_agree(
                                records,
                                verify_threshold,
                                direct_floor,
                                agree_floor,
                            )
                            cost = item["avg_core_layers"] + float(check_penalty) * item["agreement_check_rate"]
                            effective_cost = item["effective_core_layers_with_checks"]
                            acceptable_key = (
                                cost,
                                item["agreement_check_rate"],
                                effective_cost,
                                -item["acc"],
                            )
                            overall_key = (
                                item["acc"],
                                -cost,
                                -item["agreement_check_rate"],
                            )
                            if item["acc"] >= floor and (
                                best_acceptable is None or acceptable_key < best_acceptable[0]
                            ):
                                best_acceptable = (
                                    acceptable_key,
                                    verify_threshold,
                                    direct_floor,
                                    agree_floor,
                                    item,
                                )
                            if best_overall is None or overall_key > best_overall[0]:
                                best_overall = (
                                    overall_key,
                                    verify_threshold,
                                    direct_floor,
                                    agree_floor,
                                    item,
                                )
                if best_acceptable is not None:
                    _, verify_threshold, direct_floor, agree_floor, item = best_acceptable
                    return verify_threshold, direct_floor, agree_floor, item, "within_val_drop", floor, full_teacher_acc
                _, verify_threshold, direct_floor, agree_floor, item = best_overall
                return verify_threshold, direct_floor, agree_floor, item, "best_val_accuracy", floor, full_teacher_acc

            def slice_router_records(records, start: int, end: int):
                n = int(records["full_correct"].numel())
                sliced = {}
                for key, value in records.items():
                    if not torch.is_tensor(value):
                        sliced[key] = value
                    elif value.dim() == 1 and value.size(0) == n:
                        sliced[key] = value[start:end].clone()
                    elif (
                        value.dim() >= 2
                        and value.size(0) == cfg.max_correct + 1
                        and value.size(1) == n
                    ):
                        sliced[key] = value[:, start:end].clone()
                    else:
                        sliced[key] = value.clone()
                return sliced

            def with_probe_scores(records, score_stack):
                scored = dict(records)
                scored["utility"] = score_stack.detach().cpu()
                return scored

            def build_probe_features(records, feature_set: str):
                verify = records["verify"].float()
                num_budgets, num_examples = verify.shape
                budget_idx = torch.arange(num_budgets, dtype=verify.dtype).view(num_budgets, 1, 1)
                budget_frac = budget_idx.expand(num_budgets, num_examples, 1) / max(1, num_budgets - 1)
                core_frac_values = (
                    float(cfg.jump_layers)
                    + torch.arange(num_budgets, dtype=verify.dtype) * float(cfg.core_layers)
                ) / full_core_layers
                core_frac = core_frac_values.view(num_budgets, 1, 1).expand(num_budgets, num_examples, 1)
                scalar_parts = [
                    verify.unsqueeze(-1),
                    records["margin"].float().unsqueeze(-1),
                    records["max_prob"].float().unsqueeze(-1),
                    records["entropy"].float().unsqueeze(-1),
                    budget_frac,
                    core_frac,
                ]
                if "utility" in records:
                    scalar_parts.append(records["utility"].float().unsqueeze(-1))
                if "stability" in records:
                    scalar_parts.append(records["stability"].float().unsqueeze(-1))
                if feature_set == "scalar":
                    return torch.cat(scalar_parts, dim=-1)
                if feature_set == "rich":
                    budget_onehot = F.one_hot(torch.arange(num_budgets), num_classes=num_budgets).float()
                    budget_onehot = budget_onehot.view(num_budgets, 1, num_budgets).expand(
                        num_budgets,
                        num_examples,
                        num_budgets,
                    )
                    return torch.cat(
                        [
                            records["features"].float(),
                            records["probs"].float(),
                            torch.cat(scalar_parts, dim=-1),
                            budget_onehot,
                        ],
                        dim=-1,
                    )
                raise ValueError(f"unknown probe feature set: {feature_set}")

            def probe_target_stack(records, target_name: str):
                if target_name == "agreement":
                    return records["agree"].float()
                if target_name == "correct":
                    return records["pred_correct"].float()
                raise ValueError(f"unknown probe target: {target_name}")

            def binary_rank_metrics(scores, labels):
                scores = scores.detach().flatten().float().cpu()
                labels = labels.detach().flatten().bool().cpu()
                total = int(labels.numel())
                pos = int(labels.sum().item())
                neg = total - pos
                out = {
                    "examples": total,
                    "positive_rate": pos / total if total else None,
                    "auc": None,
                    "ap": None,
                }
                if pos == 0 or neg == 0:
                    return out
                order_desc = torch.argsort(scores, descending=True)
                sorted_labels = labels[order_desc].float()
                rank = torch.arange(1, total + 1, dtype=torch.float32)
                precision_at_rank = torch.cumsum(sorted_labels, dim=0) / rank
                out["ap"] = float((precision_at_rank * sorted_labels).sum().item() / pos)
                order_asc = torch.argsort(scores)
                asc_labels = labels[order_asc]
                asc_ranks = torch.arange(1, total + 1, dtype=torch.float32)
                rank_sum = asc_ranks[asc_labels].sum().item()
                auc = (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)
                out["auc"] = float(auc)
                return out

            def fit_probe_scores(train_records, tune_records, final_records, target_name: str, feature_set: str, model_kind: str):
                train_features = build_probe_features(train_records, feature_set)
                tune_features = build_probe_features(tune_records, feature_set)
                final_features = build_probe_features(final_records, feature_set)
                num_budgets, _, feature_dim = train_features.shape
                x_train = train_features.reshape(-1, feature_dim).to(device)
                y_train = probe_target_stack(train_records, target_name).reshape(-1).to(device)
                mean = x_train.mean(dim=0, keepdim=True)
                std = x_train.std(dim=0, keepdim=True).clamp_min(1e-4)
                x_train = (x_train - mean) / std
                pos = float(y_train.sum().item())
                total = float(y_train.numel())
                neg = total - pos

                def score_features(feature_stack, model_obj=None, constant_prob=None):
                    x = feature_stack.reshape(-1, feature_dim).to(device)
                    x = (x - mean) / std
                    if constant_prob is not None:
                        scores = torch.full((x.size(0),), float(constant_prob), dtype=torch.float32, device=device)
                    else:
                        with torch.no_grad():
                            scores = torch.sigmoid(model_obj(x).squeeze(-1))
                    return scores.view(num_budgets, -1).detach().cpu()

                if pos <= 0.0 or neg <= 0.0:
                    constant = pos / max(1.0, total)
                    return {
                        "tune_scores": score_features(tune_features, constant_prob=constant),
                        "final_scores": score_features(final_features, constant_prob=constant),
                        "feature_dim": feature_dim,
                        "loss": None,
                        "constant": constant,
                    }

                if model_kind == "logistic":
                    model_obj = nn.Linear(feature_dim, 1).to(device)
                elif model_kind == "mlp":
                    model_obj = nn.Sequential(
                        nn.Linear(feature_dim, cfg.router_probe_hidden),
                        nn.GELU(),
                        nn.Linear(cfg.router_probe_hidden, 1),
                    ).to(device)
                else:
                    raise ValueError(f"unknown probe model: {model_kind}")
                pos_weight = torch.tensor([neg / max(1.0, pos)], dtype=torch.float32, device=device)
                opt_probe = torch.optim.AdamW(model_obj.parameters(), lr=cfg.router_probe_lr, weight_decay=1e-4)
                loss_value = 0.0
                for _ in range(cfg.router_probe_steps):
                    opt_probe.zero_grad(set_to_none=True)
                    logits = model_obj(x_train).squeeze(-1)
                    loss = F.binary_cross_entropy_with_logits(logits, y_train, pos_weight=pos_weight)
                    loss.backward()
                    opt_probe.step()
                    loss_value = float(loss.item())
                return {
                    "tune_scores": score_features(tune_features, model_obj=model_obj),
                    "final_scores": score_features(final_features, model_obj=model_obj),
                    "feature_dim": feature_dim,
                    "loss": loss_value,
                    "constant": None,
                }

            def choose_probe_global_threshold(records, score_stack, threshold_values, max_drop: float, guarded: bool = False):
                scored_records = with_probe_scores(records, score_stack)
                full_teacher_acc = records["full_correct"].float().mean().item()
                floor = full_teacher_acc - max_drop
                best_acceptable = None
                best_overall = None
                for threshold in threshold_values:
                    thresholds_by_budget = [float(threshold) for _ in range(cfg.max_correct + 1)]
                    item = evaluate_per_budget_utility_thresholds(
                        scored_records,
                        thresholds_by_budget,
                        guarded=guarded,
                    )
                    item["threshold"] = float(threshold)
                    acceptable_key = (item["avg_core_layers"], -item["acc"])
                    overall_key = (item["acc"], -item["avg_core_layers"])
                    if item["acc"] >= floor and (
                        best_acceptable is None or acceptable_key < best_acceptable[0]
                    ):
                        best_acceptable = (acceptable_key, threshold, item)
                    if best_overall is None or overall_key > best_overall[0]:
                        best_overall = (overall_key, threshold, item)
                if best_acceptable is not None:
                    _, threshold, item = best_acceptable
                    return threshold, item, "within_val_drop", floor, full_teacher_acc
                _, threshold, item = best_overall
                return threshold, item, "best_val_accuracy", floor, full_teacher_acc

            def run_router_probe_audit(val_records, final_records, threshold_values, selection_drops):
                num_examples = int(val_records["full_correct"].numel())
                split = max(1, num_examples // 2)
                split = min(split, num_examples - 1)
                train_records = slice_router_records(val_records, 0, split)
                tune_records = slice_router_records(val_records, split, num_examples)
                probe_summary = {
                    "train_examples": int(train_records["full_correct"].numel()),
                    "tune_examples": int(tune_records["full_correct"].numel()),
                    "final_examples": int(final_records["full_correct"].numel()),
                    "probe_steps": cfg.router_probe_steps,
                    "probes": {},
                }
                probe_specs = [
                    ("agreement", "scalar", "logistic"),
                    ("agreement", "scalar", "mlp"),
                    ("agreement", "rich", "logistic"),
                    ("agreement", "rich", "mlp"),
                    ("correct", "scalar", "logistic"),
                    ("correct", "scalar", "mlp"),
                    ("correct", "rich", "logistic"),
                    ("correct", "rich", "mlp"),
                ]
                for target_name, feature_set, model_kind in probe_specs:
                    audit_name = f"{target_name}_{feature_set}_{model_kind}"
                    fit = fit_probe_scores(
                        train_records,
                        tune_records,
                        final_records,
                        target_name,
                        feature_set,
                        model_kind,
                    )
                    scored_tune = with_probe_scores(tune_records, fit["tune_scores"])
                    scored_final = with_probe_scores(final_records, fit["final_scores"])
                    target_final = probe_target_stack(final_records, target_name)
                    item = {
                        "target": target_name,
                        "feature_set": feature_set,
                        "model": model_kind,
                        "feature_dim": fit["feature_dim"],
                        "train_loss": fit["loss"],
                        "constant": fit["constant"],
                        "target_metrics_final": binary_rank_metrics(fit["final_scores"], target_final),
                        "selected_by_drop": {},
                    }
                    for scenario_name, max_drop in selection_drops.items():
                        threshold, val_item, reason, floor, val_teacher = choose_probe_global_threshold(
                            tune_records,
                            fit["tune_scores"],
                            threshold_values,
                            max_drop=max_drop,
                        )
                        final_item = evaluate_per_budget_utility_thresholds(
                            scored_final,
                            [float(threshold) for _ in range(cfg.max_correct + 1)],
                        )
                        (
                            per_budget_thresholds,
                            per_budget_val,
                            per_budget_reason,
                            per_budget_floor,
                            per_budget_teacher,
                        ) = choose_per_budget_utility_thresholds(
                            scored_tune,
                            threshold_values,
                            guarded=False,
                            max_drop=max_drop,
                        )
                        per_budget_final = evaluate_per_budget_utility_thresholds(
                            scored_final,
                            per_budget_thresholds,
                        )
                        item["selected_by_drop"][scenario_name] = {
                            "global": {
                                "threshold": float(threshold),
                                "selection_reason": reason,
                                "val_accuracy_floor": floor,
                                "val_full_teacher_acc": val_teacher,
                                "val": val_item,
                                "final": final_item,
                            },
                            "per_budget": {
                                "thresholds": [float(t) for t in per_budget_thresholds],
                                "selection_reason": per_budget_reason,
                                "val_accuracy_floor": per_budget_floor,
                                "val_full_teacher_acc": per_budget_teacher,
                                "val": per_budget_val,
                                "final": per_budget_final,
                            },
                        }
                    probe_summary["probes"][audit_name] = item
                return probe_summary

            def choose_heldout_threshold(
                val_grid,
                policy_name: str,
                max_drop: float | None = None,
            ):
                if max_drop is None:
                    max_drop = cfg.router_val_max_drop
                floor = val_grid["full_teacher_acc"] - max_drop
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
                for policy_name in all_policy_names:
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
                    stability_stack = None
                    if cfg.use_stability_head:
                        stability_stack = torch.stack(
                            [torch.sigmoid(v) for v in out["stability"]],
                            dim=0,
                        )
                    utility_stack = None
                    if cfg.use_utility_router:
                        utility_stack = torch.stack(
                            [torch.sigmoid(v) for v in out["utility"]],
                            dim=0,
                        )
                    consistency_stack = None
                    if cfg.use_consistency_head:
                        consistency_stack = torch.stack(
                            [torch.sigmoid(v) for v in out["consistency"]],
                            dim=0,
                        )
                    next_pred_class = None
                    next_pred_conf = None
                    if cfg.use_next_agreement_head:
                        next_logits_stack = torch.stack(out["next_pred"], dim=0)
                        next_probs = F.softmax(next_logits_stack, dim=-1)
                        next_pred_conf, next_pred_class = next_probs.max(dim=-1)
                    for c, pred in enumerate(preds):
                        metrics[f"jump_c{c}_acc"] += (pred == target).float().mean().item()
                        good = (pred == target).float()
                        update_calibration(calibration["by_budget"][str(c)], verify_stack[c], good)
                        update_calibration(calibration["all"], verify_stack[c], good)
                    correct_stack = pred_stack == target.unsqueeze(0)
                    any_jump_correct = correct_stack.any(dim=0)
                    first_correct_target = first_sufficient_budget(correct_stack)
                    if (
                        cfg.use_budget_controller
                        and cfg.budget_controller_target == "first_acceptable"
                    ):
                        budget_target = first_acceptable_budget(
                            correct_stack,
                            verify_stack,
                            margin_stack,
                            max_prob_stack,
                            cfg.budget_controller_train_threshold,
                        )
                    else:
                        budget_target = first_correct_target
                    fallback = cfg.max_correct + 1
                    first_correct = first_correct_target.clamp(0, cfg.max_correct)
                    budget_choice = None
                    if cfg.use_budget_controller:
                        budget_logits = jumprec.budget_logits(state0, lengths)
                        budget_choice = budget_logits.argmax(dim=-1)
                        pred_fallback = budget_choice == fallback
                        target_fallback = budget_target == fallback
                        pred_tail = budget_tail_loops(budget_choice)
                        target_tail = budget_tail_loops(budget_target)
                        metrics["budget_controller_target_acc"] += (
                            budget_choice == budget_target
                        ).float().mean().item()
                        metrics["budget_controller_under_rate"] += (pred_tail < target_tail).float().mean().item()
                        metrics["budget_controller_over_rate"] += (pred_tail > target_tail).float().mean().item()
                        metrics["budget_controller_within_one_rate"] += (
                            (pred_tail - target_tail).abs() <= 1.0
                        ).float().mean().item()
                        metrics["budget_controller_avg_abs_tail_error"] += (
                            pred_tail - target_tail
                        ).abs().mean().item()
                        metrics["budget_controller_pred_fallback_rate"] += pred_fallback.float().mean().item()
                        metrics["budget_controller_target_fallback_rate"] += target_fallback.float().mean().item()
                        metrics["budget_controller_avg_pred_tail_loops"] += pred_tail.float().mean().item()
                        metrics["budget_controller_avg_target_tail_loops"] += target_tail.float().mean().item()
                        confusion_flat = (
                            budget_target.detach().cpu() * (cfg.max_correct + 2)
                            + budget_choice.detach().cpu()
                        )
                        budget_confusion += torch.bincount(
                            confusion_flat,
                            minlength=(cfg.max_correct + 2) * (cfg.max_correct + 2),
                        ).view(cfg.max_correct + 2, cfg.max_correct + 2)
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
                        if cfg.use_stability_head:
                            for policy_name, stability_threshold in zip(
                                stability_policy_names,
                                stability_policy_thresholds,
                            ):
                                pred_stable, trusted_stable, chosen_stable, stable_core_layers = (
                                    route_stability_predictions(
                                        pred_stack,
                                        verify_stack,
                                        margin_stack,
                                        max_prob_stack,
                                        stability_stack,
                                        full_pred,
                                        target,
                                        threshold,
                                        stability_threshold,
                                    )
                                )
                                prefix = f"router_{suffix}_{policy_name}"
                                metrics[f"{prefix}_acc"] += (pred_stable == target).float().mean().item()
                                metrics[f"{prefix}_full_loop_rate"] += (~trusted_stable).float().mean().item()
                                metrics[f"{prefix}_avg_tail_loops"] += torch.where(
                                    trusted_stable,
                                    chosen_stable.float(),
                                    torch.full_like(chosen_stable.float(), float(cfg.loop_steps)),
                                ).mean().item()
                                metrics[f"{prefix}_avg_core_layers"] += stable_core_layers.mean().item()
                        if cfg.use_utility_router:
                            for policy_name, guarded in utility_policy_defs:
                                pred_utility, trusted_utility, chosen_utility, utility_core_layers = (
                                    route_utility_predictions(
                                        pred_stack,
                                        utility_stack,
                                        margin_stack,
                                        max_prob_stack,
                                        full_pred,
                                        target,
                                        threshold,
                                        guarded,
                                    )
                                )
                                prefix = f"router_{suffix}_{policy_name}"
                                metrics[f"{prefix}_acc"] += (pred_utility == target).float().mean().item()
                                metrics[f"{prefix}_full_loop_rate"] += (
                                    ~trusted_utility
                                ).float().mean().item()
                                metrics[f"{prefix}_avg_tail_loops"] += torch.where(
                                    trusted_utility,
                                    chosen_utility.float(),
                                    torch.full_like(chosen_utility.float(), float(cfg.loop_steps)),
                                ).mean().item()
                                metrics[f"{prefix}_avg_core_layers"] += utility_core_layers.mean().item()
                            for policy_name, confirm_floor in zip(
                                utility_then_agree_policy_names,
                                utility_then_agree_floors,
                            ):
                                pred_confirmed, trusted_confirmed, chosen_confirmed, confirmed_core_layers = (
                                    route_utility_then_agree_predictions(
                                        pred_stack,
                                        verify_stack,
                                        utility_stack,
                                        margin_stack,
                                        max_prob_stack,
                                        agree_stack,
                                        full_pred,
                                        target,
                                        threshold,
                                        confirm_floor,
                                    )
                                )
                                prefix = f"router_{suffix}_{policy_name}"
                                metrics[f"{prefix}_acc"] += (pred_confirmed == target).float().mean().item()
                                metrics[f"{prefix}_full_loop_rate"] += (
                                    ~trusted_confirmed
                                ).float().mean().item()
                                metrics[f"{prefix}_avg_tail_loops"] += torch.where(
                                    trusted_confirmed,
                                    chosen_confirmed.float(),
                                    torch.full_like(chosen_confirmed.float(), float(cfg.loop_steps)),
                                ).mean().item()
                                metrics[f"{prefix}_avg_core_layers"] += confirmed_core_layers.mean().item()
                            for policy_name, utility_floor in zip(
                                agree_then_utility_policy_names,
                                agree_then_utility_floors,
                            ):
                                (
                                    pred_agree_utility,
                                    trusted_agree_utility,
                                    chosen_agree_utility,
                                    agree_utility_core_layers,
                                ) = route_agree_then_utility_predictions(
                                    pred_stack,
                                    verify_stack,
                                    utility_stack,
                                    margin_stack,
                                    max_prob_stack,
                                    agree_stack,
                                    full_pred,
                                    target,
                                    threshold,
                                    utility_floor,
                                )
                                prefix = f"router_{suffix}_{policy_name}"
                                metrics[f"{prefix}_acc"] += (
                                    pred_agree_utility == target
                                ).float().mean().item()
                                metrics[f"{prefix}_full_loop_rate"] += (
                                    ~trusted_agree_utility
                                ).float().mean().item()
                                metrics[f"{prefix}_avg_tail_loops"] += torch.where(
                                    trusted_agree_utility,
                                    chosen_agree_utility.float(),
                                    torch.full_like(chosen_agree_utility.float(), float(cfg.loop_steps)),
                                ).mean().item()
                                metrics[f"{prefix}_avg_core_layers"] += agree_utility_core_layers.mean().item()
                        if cfg.use_consistency_head and cfg.use_utility_router:
                            for policy_name, consistency_threshold in zip(
                                consistency_policy_names,
                                consistency_policy_thresholds,
                            ):
                                (
                                    pred_cats,
                                    trusted_cats,
                                    chosen_cats,
                                    cats_core_layers,
                                ) = route_utility_consistency_predictions(
                                    pred_stack,
                                    utility_stack,
                                    consistency_stack,
                                    full_pred,
                                    target,
                                    threshold,
                                    consistency_threshold,
                                )
                                prefix = f"router_{suffix}_{policy_name}"
                                metrics[f"{prefix}_acc"] += (pred_cats == target).float().mean().item()
                                metrics[f"{prefix}_full_loop_rate"] += (
                                    ~trusted_cats
                                ).float().mean().item()
                                metrics[f"{prefix}_avg_tail_loops"] += torch.where(
                                    trusted_cats,
                                    chosen_cats.float(),
                                    torch.full_like(chosen_cats.float(), float(cfg.loop_steps)),
                                ).mean().item()
                                metrics[f"{prefix}_avg_core_layers"] += cats_core_layers.mean().item()
                        if cfg.use_next_agreement_head:
                            for policy_name, next_threshold in zip(
                                next_agreement_policy_names,
                                next_agreement_thresholds,
                            ):
                                pred_nextagree, trusted_nextagree, chosen_nextagree, nextagree_core_layers = (
                                    route_nextagreement_predictions(
                                        pred_stack,
                                        verify_stack,
                                        margin_stack,
                                        max_prob_stack,
                                        next_pred_class,
                                        next_pred_conf,
                                        full_pred,
                                        target,
                                        threshold,
                                        next_threshold,
                                    )
                                )
                                prefix = f"router_{suffix}_{policy_name}"
                                metrics[f"{prefix}_acc"] += (pred_nextagree == target).float().mean().item()
                                metrics[f"{prefix}_full_loop_rate"] += (
                                    ~trusted_nextagree
                                ).float().mean().item()
                                metrics[f"{prefix}_avg_tail_loops"] += torch.where(
                                    trusted_nextagree,
                                    chosen_nextagree.float(),
                                    torch.full_like(chosen_nextagree.float(), float(cfg.loop_steps)),
                                ).mean().item()
                                metrics[f"{prefix}_avg_core_layers"] += nextagree_core_layers.mean().item()
                        if cfg.use_budget_controller:
                            pred_budget, trusted_budget, chosen_budget, budget_core_layers = route_budget_predictions(
                                pred_stack,
                                verify_stack,
                                margin_stack,
                                max_prob_stack,
                                full_pred,
                                threshold,
                                budget_choice,
                            )
                            prefix = f"router_{suffix}_budget"
                            metrics[f"{prefix}_acc"] += (pred_budget == target).float().mean().item()
                            metrics[f"{prefix}_full_loop_rate"] += (~trusted_budget).float().mean().item()
                            metrics[f"{prefix}_avg_tail_loops"] += torch.where(
                                trusted_budget,
                                chosen_budget.float(),
                                torch.full_like(chosen_budget.float(), float(cfg.loop_steps)),
                            ).mean().item()
                            metrics[f"{prefix}_avg_core_layers"] += budget_core_layers.mean().item()
                            (
                                pred_budget_escalate,
                                trusted_budget_escalate,
                                chosen_budget_escalate,
                                budget_escalate_core_layers,
                            ) = route_budget_escalate_predictions(
                                pred_stack,
                                verify_stack,
                                margin_stack,
                                max_prob_stack,
                                full_pred,
                                threshold,
                                budget_choice,
                            )
                            prefix = f"router_{suffix}_budget_escalate"
                            metrics[f"{prefix}_acc"] += (
                                pred_budget_escalate == target
                            ).float().mean().item()
                            metrics[f"{prefix}_full_loop_rate"] += (
                                ~trusted_budget_escalate
                            ).float().mean().item()
                            metrics[f"{prefix}_avg_tail_loops"] += torch.where(
                                trusted_budget_escalate,
                                chosen_budget_escalate.float(),
                                torch.full_like(chosen_budget_escalate.float(), float(cfg.loop_steps)),
                            ).mean().item()
                            metrics[f"{prefix}_avg_core_layers"] += budget_escalate_core_layers.mean().item()
                            pred_budget_open, trusted_budget_open, chosen_budget_open, budget_open_core_layers = (
                                route_budget_open_predictions(
                                    pred_stack,
                                    full_pred,
                                    budget_choice,
                                )
                            )
                            prefix = f"router_{suffix}_budget_open"
                            metrics[f"{prefix}_acc"] += (pred_budget_open == target).float().mean().item()
                            metrics[f"{prefix}_full_loop_rate"] += (~trusted_budget_open).float().mean().item()
                            metrics[f"{prefix}_avg_tail_loops"] += torch.where(
                                trusted_budget_open,
                                chosen_budget_open.float(),
                                torch.full_like(chosen_budget_open.float(), float(cfg.loop_steps)),
                            ).mean().item()
                            metrics[f"{prefix}_avg_core_layers"] += budget_open_core_layers.mean().item()
                            pred_budget_scan, trusted_budget_scan, chosen_budget_scan, budget_scan_core_layers = (
                                route_budget_scan_up_predictions(
                                    pred_stack,
                                    verify_stack,
                                    margin_stack,
                                    max_prob_stack,
                                    full_pred,
                                    threshold,
                                    budget_choice,
                                )
                            )
                            prefix = f"router_{suffix}_budget_scan_up"
                            metrics[f"{prefix}_acc"] += (pred_budget_scan == target).float().mean().item()
                            metrics[f"{prefix}_full_loop_rate"] += (~trusted_budget_scan).float().mean().item()
                            metrics[f"{prefix}_avg_tail_loops"] += torch.where(
                                trusted_budget_scan,
                                chosen_budget_scan.float(),
                                torch.full_like(chosen_budget_scan.float(), float(cfg.loop_steps)),
                            ).mean().item()
                            metrics[f"{prefix}_avg_core_layers"] += budget_scan_core_layers.mean().item()
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
            if cfg.use_budget_controller:
                metrics["budget_controller_confusion_rows_target_cols_pred"] = budget_confusion.tolist()
            for _, suffix in thresholds:
                metrics[f"strict_{suffix}_fallback_core_savings_vs_full_pct"] = (
                    1.0 - metrics[f"strict_{suffix}_fallback_avg_core_layers"] / full_core_layers
                ) * 100.0
                for policy_name in all_policy_names:
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
                for policy_name in all_policy_names:
                    key, val_item, reason, floor = choose_heldout_threshold(val_grid, policy_name)
                    selected[policy_name] = {
                        "threshold": val_item["threshold"],
                        "threshold_key": key,
                        "selection_reason": reason,
                        "val_accuracy_floor": floor,
                        "val": val_item,
                    }
                final_grid = collect_router_grid(cfg.eval_batches, candidates)
                for policy_name, item in selected.items():
                    item["final"] = final_grid["policies"][policy_name][item["threshold_key"]]
                selected_by_drop = {}
                selection_drops = {
                    "speed": cfg.router_val_max_drop,
                    "tight_001": 0.001,
                    "teacher_floor": 0.0,
                    "teacher_plus_001": -0.001,
                    "teacher_plus_002": -0.002,
                }
                for scenario_name, max_drop in selection_drops.items():
                    scenario = {}
                    for policy_name in all_policy_names:
                        key, val_item, reason, floor = choose_heldout_threshold(
                            val_grid,
                            policy_name,
                            max_drop=max_drop,
                        )
                        scenario[policy_name] = {
                            "threshold": val_item["threshold"],
                            "threshold_key": key,
                            "selection_reason": reason,
                            "val_accuracy_floor": floor,
                            "val": val_item,
                            "final": final_grid["policies"][policy_name][key],
                        }
                    selected_by_drop[scenario_name] = scenario
                heldout_audit = {
                    "threshold_candidates": candidates,
                    "val_batches": cfg.router_val_batches,
                    "final_batches": cfg.eval_batches,
                    "max_val_drop_vs_teacher": cfg.router_val_max_drop,
                    "val_full_teacher_acc": val_grid["full_teacher_acc"],
                    "final_full_teacher_acc": final_grid["full_teacher_acc"],
                    "selected": selected,
                    "selected_by_drop": selected_by_drop,
                    "val_policies": val_grid["policies"],
                    "final_policies": final_grid["policies"],
                }
                val_records = None
                final_records = None

                def ensure_router_records():
                    nonlocal val_records, final_records
                    if val_records is None:
                        val_records = collect_router_records(cfg.router_val_batches)
                    if final_records is None:
                        final_records = collect_router_records(cfg.eval_batches)
                    return val_records, final_records

                if cfg.use_utility_router:
                    val_records, final_records = ensure_router_records()
                    for audit_name, guarded in (
                        ("utility_per_budget", False),
                        ("utility_guarded_per_budget", True),
                    ):
                        (
                            utility_thresholds,
                            utility_val,
                            utility_reason,
                            utility_floor,
                            utility_val_teacher,
                        ) = choose_per_budget_utility_thresholds(
                            val_records,
                            candidates,
                            guarded=guarded,
                        )
                        utility_final = evaluate_per_budget_utility_thresholds(
                            final_records,
                            utility_thresholds,
                            guarded=guarded,
                        )
                        heldout_audit[audit_name] = {
                            "selection_reason": utility_reason,
                            "val_accuracy_floor": utility_floor,
                            "val_full_teacher_acc": utility_val_teacher,
                            "final_full_teacher_acc": final_records["full_correct"].float().mean().item(),
                            "val": utility_val,
                            "final": utility_final,
                        }
                    (
                        utility_monotone_thresholds,
                        utility_monotone_val,
                        utility_monotone_reason,
                        utility_monotone_floor,
                        utility_monotone_val_teacher,
                    ) = choose_per_budget_utility_thresholds(
                        val_records,
                        candidates,
                        monotone=True,
                    )
                    utility_monotone_final = evaluate_per_budget_utility_thresholds(
                        final_records,
                        utility_monotone_thresholds,
                    )
                    heldout_audit["utility_per_budget_monotone"] = {
                        "selection_reason": utility_monotone_reason,
                        "val_accuracy_floor": utility_monotone_floor,
                        "val_full_teacher_acc": utility_monotone_val_teacher,
                        "final_full_teacher_acc": final_records["full_correct"].float().mean().item(),
                        "val": utility_monotone_val,
                        "final": utility_monotone_final,
                    }
                if cfg.router_selective_agree_audit and cfg.use_utility_router:
                    val_records, final_records = ensure_router_records()
                    selective = {}
                    check_penalties = {
                        "core_only": 0.0,
                        "check_025": 0.25,
                        "check_050": 0.50,
                        "check_100": 1.00,
                    }
                    for scenario_name, max_drop in selection_drops.items():
                        scenario = {}
                        for penalty_name, check_penalty in check_penalties.items():
                            (
                                verify_threshold,
                                direct_floor,
                                agree_floor,
                                selective_val,
                                selective_reason,
                                selective_floor,
                                selective_val_teacher,
                            ) = choose_selective_agree_thresholds(
                                val_records,
                                candidates,
                                max_drop=max_drop,
                                check_penalty=check_penalty,
                            )
                            selective_final = evaluate_selective_agree(
                                final_records,
                                verify_threshold,
                                direct_floor,
                                agree_floor,
                            )
                            scenario[penalty_name] = {
                                "selection_reason": selective_reason,
                                "val_accuracy_floor": selective_floor,
                                "val_full_teacher_acc": selective_val_teacher,
                                "verify_threshold": float(verify_threshold),
                                "direct_floor": float(direct_floor),
                                "agree_floor": float(agree_floor),
                                "check_penalty": float(check_penalty),
                                "val": selective_val,
                                "final": selective_final,
                            }
                        selective[scenario_name] = scenario
                    heldout_audit["selective_agreement"] = selective
                if cfg.router_per_budget_audit:
                    val_records, final_records = ensure_router_records()
                    (
                        per_budget_thresholds,
                        per_budget_val,
                        per_budget_reason,
                        per_budget_floor,
                        per_budget_val_teacher,
                    ) = choose_per_budget_thresholds(val_records, candidates)
                    per_budget_final = evaluate_per_budget_thresholds(
                        final_records,
                        per_budget_thresholds,
                    )
                    heldout_audit["per_budget_no_agree"] = {
                        "selection_reason": per_budget_reason,
                        "val_accuracy_floor": per_budget_floor,
                        "val_full_teacher_acc": per_budget_val_teacher,
                        "final_full_teacher_acc": final_records["full_correct"].float().mean().item(),
                        "val": per_budget_val,
                        "final": per_budget_final,
                    }
                    (
                        monotone_thresholds,
                        monotone_val,
                        monotone_reason,
                        monotone_floor,
                        monotone_val_teacher,
                    ) = choose_per_budget_thresholds(val_records, candidates, monotone=True)
                    monotone_final = evaluate_per_budget_thresholds(
                        final_records,
                        monotone_thresholds,
                    )
                    heldout_audit["per_budget_no_agree_monotone"] = {
                        "selection_reason": monotone_reason,
                        "val_accuracy_floor": monotone_floor,
                        "val_full_teacher_acc": monotone_val_teacher,
                        "final_full_teacher_acc": final_records["full_correct"].float().mean().item(),
                        "val": monotone_val,
                        "final": monotone_final,
                    }
                if cfg.router_probe_audit:
                    val_records, final_records = ensure_router_records()
                    heldout_audit["probe_upper_bound"] = run_router_probe_audit(
                        val_records,
                        final_records,
                        candidates,
                        selection_drops,
                    )
                metrics["heldout_threshold_audit"] = heldout_audit
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
                    result = jumprec.forward_budget_encoded(
                        sub_state0,
                        sub_mask,
                        sub_pos,
                        sub_emb,
                        sub_lens,
                        corrections,
                    )
                    logits, verify = result[:2]
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

            out["jumprec_serial_070_ms_per_batch"] = time_fn(
                lambda ids, mask, lens: serial_jumprec(ids, mask, lens, 0.70)
            )
            out["jumprec_serial_080_ms_per_batch"] = time_fn(serial_jumprec)
            out["jumprec_serial_085_ms_per_batch"] = time_fn(
                lambda ids, mask, lens: serial_jumprec(ids, mask, lens, 0.85)
            )
            out["jumprec_serial_090_ms_per_batch"] = time_fn(
                lambda ids, mask, lens: serial_jumprec(ids, mask, lens, 0.90)
            )
            out["jumprec_serial_095_ms_per_batch"] = time_fn(
                lambda ids, mask, lens: serial_jumprec(ids, mask, lens, 0.95)
            )
            out["jumprec_serial_099_ms_per_batch"] = time_fn(
                lambda ids, mask, lens: serial_jumprec(ids, mask, lens, 0.99)
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
                    result = jumprec.forward_budget_encoded(
                        sub_state0,
                        sub_mask,
                        sub_pos,
                        sub_emb,
                        sub_lens,
                        corrections,
                    )
                    logits, verify = result[:2]
                    probs = F.softmax(logits, dim=-1)
                    top2 = probs.topk(2, dim=-1).values
                    accept = (
                        (torch.sigmoid(verify) >= threshold)
                        & ((top2[:, 0] - top2[:, 1]) >= 0.05)
                        & (top2[:, 0] >= 0.45)
                    )
                    if corrections < cfg.max_correct:
                        next_result = jumprec.forward_budget_encoded(
                            sub_state0,
                            sub_mask,
                            sub_pos,
                            sub_emb,
                            sub_lens,
                            corrections + 1,
                        )
                        next_logits = next_result[0]
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

            out["jumprec_serial_agree_010_ms_per_batch"] = time_fn(
                lambda ids, mask, lens: serial_jumprec_agree(ids, mask, lens, 0.10)
            )
            out["jumprec_serial_agree_050_ms_per_batch"] = time_fn(
                lambda ids, mask, lens: serial_jumprec_agree(ids, mask, lens, 0.50)
            )
            out["jumprec_serial_agree_080_ms_per_batch"] = time_fn(serial_jumprec_agree)
            if cfg.use_stability_head:
                def serial_jumprec_stable(
                    ids,
                    mask,
                    lens,
                    threshold: float = 0.80,
                    stability_threshold: float = 0.50,
                ):
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
                        result = jumprec.forward_budget_encoded(
                            sub_state0,
                            sub_mask,
                            sub_pos,
                            sub_emb,
                            sub_lens,
                            corrections,
                        )
                        logits, verify, stability = result[:3]
                        probs = F.softmax(logits, dim=-1)
                        top2 = probs.topk(2, dim=-1).values
                        accept = (
                            (corrections < cfg.max_correct)
                            & (torch.sigmoid(verify) >= threshold)
                            & ((top2[:, 0] - top2[:, 1]) >= 0.05)
                            & (top2[:, 0] >= 0.45)
                            & (torch.sigmoid(stability) >= stability_threshold)
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

                out["jumprec_serial_stable050_080_ms_per_batch"] = time_fn(
                    lambda ids, mask, lens: serial_jumprec_stable(ids, mask, lens, 0.80, 0.50)
                )
                out["jumprec_serial_stable050_090_ms_per_batch"] = time_fn(
                    lambda ids, mask, lens: serial_jumprec_stable(ids, mask, lens, 0.90, 0.50)
                )
                out["jumprec_serial_stable070_090_ms_per_batch"] = time_fn(
                    lambda ids, mask, lens: serial_jumprec_stable(ids, mask, lens, 0.90, 0.70)
                )
            if cfg.use_utility_router:
                def budget_result_utility(result):
                    idx = 2
                    if cfg.use_stability_head:
                        idx += 1
                    if cfg.use_consistency_head:
                        idx += 1
                    return result[idx]

                def serial_jumprec_utility(ids, mask, lens, threshold: float = 0.50):
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
                        result = jumprec.forward_budget_encoded(
                            sub_state0,
                            sub_mask,
                            sub_pos,
                            sub_emb,
                            sub_lens,
                            corrections,
                        )
                        logits = result[0]
                        utility = budget_result_utility(result)
                        accept = torch.sigmoid(utility) >= threshold
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

                out["jumprec_serial_utility_010_ms_per_batch"] = time_fn(
                    lambda ids, mask, lens: serial_jumprec_utility(ids, mask, lens, 0.10)
                )
                out["jumprec_serial_utility_050_ms_per_batch"] = time_fn(serial_jumprec_utility)
                out["jumprec_serial_utility_080_ms_per_batch"] = time_fn(
                    lambda ids, mask, lens: serial_jumprec_utility(ids, mask, lens, 0.80)
                )
                out["jumprec_serial_utility_090_ms_per_batch"] = time_fn(
                    lambda ids, mask, lens: serial_jumprec_utility(ids, mask, lens, 0.90)
                )

                def serial_jumprec_selective_agree(
                    ids,
                    mask,
                    lens,
                    verify_threshold: float = 0.05,
                    direct_floor: float = 0.90,
                    agree_floor: float = 0.00,
                ):
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
                        result = jumprec.forward_budget_encoded(
                            sub_state0,
                            sub_mask,
                            sub_pos,
                            sub_emb,
                            sub_lens,
                            corrections,
                        )
                        logits, verify = result[:2]
                        utility = torch.sigmoid(budget_result_utility(result))
                        probs = F.softmax(logits, dim=-1)
                        top2 = probs.topk(2, dim=-1).values
                        direct = utility >= direct_floor
                        accept = direct.clone()

                        if corrections < cfg.max_correct:
                            check = (
                                (~direct)
                                & (utility >= agree_floor)
                                & (torch.sigmoid(verify) >= verify_threshold)
                                & ((top2[:, 0] - top2[:, 1]) >= 0.05)
                                & (top2[:, 0] >= 0.45)
                            )
                            if check.any():
                                check_idx = torch.nonzero(check, as_tuple=False).flatten()
                                check_global = open_idx.index_select(0, check_idx)
                                check_state0 = state0.index_select(0, check_global)
                                check_mask = take_batch(layer_mask, check_global)
                                check_pos = take_batch(position_ids, check_global)
                                check_emb = (
                                    tuple(take_batch(t, check_global) for t in position_embeddings)
                                    if position_embeddings
                                    else None
                                )
                                check_lens = lens.index_select(0, check_global)
                                next_result = jumprec.forward_budget_encoded(
                                    check_state0,
                                    check_mask,
                                    check_pos,
                                    check_emb,
                                    check_lens,
                                    corrections + 1,
                                )
                                next_logits = next_result[0]
                                agree = logits.index_select(0, check_idx).argmax(dim=-1) == next_logits.argmax(dim=-1)
                                accept[check_idx] = agree

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

                out["jumprec_serial_selective_agree_speed_ms_per_batch"] = time_fn(
                    lambda ids, mask, lens: serial_jumprec_selective_agree(ids, mask, lens, 0.05, 0.90, 0.00)
                )
                out["jumprec_serial_selective_agree_quality002_ms_per_batch"] = time_fn(
                    lambda ids, mask, lens: serial_jumprec_selective_agree(ids, mask, lens, 0.35, 0.93, 0.30)
                )
                if cfg.use_consistency_head:
                    def serial_jumprec_utility_cats(
                        ids,
                        mask,
                        lens,
                        utility_threshold: float = 0.10,
                        consistency_threshold: float = 0.70,
                    ):
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
                            sub_emb = (
                                tuple(take_batch(t, open_idx) for t in position_embeddings)
                                if position_embeddings
                                else None
                            )
                            sub_lens = lens.index_select(0, open_idx)
                            result = jumprec.forward_budget_encoded(
                                sub_state0,
                                sub_mask,
                                sub_pos,
                                sub_emb,
                                sub_lens,
                                corrections,
                            )
                            logits = result[0]
                            consistency_idx = 3 if cfg.use_stability_head else 2
                            consistency = result[consistency_idx]
                            utility = result[consistency_idx + 1]
                            accept = (
                                (torch.sigmoid(utility) >= utility_threshold)
                                & (torch.sigmoid(consistency) >= consistency_threshold)
                            )
                            last = logits
                            open_idx = open_idx[~accept]
                        if open_idx.numel() > 0:
                            sub_state0 = state0.index_select(0, open_idx)
                            sub_mask = take_batch(layer_mask, open_idx)
                            sub_pos = take_batch(position_ids, open_idx)
                            sub_emb = (
                                tuple(take_batch(t, open_idx) for t in position_embeddings)
                                if position_embeddings
                                else None
                            )
                            sub_lens = lens.index_select(0, open_idx)
                            full = model.run_steps_from(
                                sub_state0,
                                sub_state0,
                                sub_mask,
                                sub_pos,
                                sub_emb,
                                0,
                                cfg.loop_steps,
                            )
                            last = model.classify_state(full, sub_lens, sub_mask, sub_pos, sub_emb)
                        return last

                    out["jumprec_serial_utility_cats050_010_ms_per_batch"] = time_fn(
                        lambda ids, mask, lens: serial_jumprec_utility_cats(ids, mask, lens, 0.10, 0.50)
                    )
                    out["jumprec_serial_utility_cats070_010_ms_per_batch"] = time_fn(
                        lambda ids, mask, lens: serial_jumprec_utility_cats(ids, mask, lens, 0.10, 0.70)
                    )
                    out["jumprec_serial_utility_cats090_010_ms_per_batch"] = time_fn(
                        lambda ids, mask, lens: serial_jumprec_utility_cats(ids, mask, lens, 0.10, 0.90)
                    )
            if cfg.use_next_agreement_head:
                def serial_jumprec_nextagree(
                    ids,
                    mask,
                    lens,
                    threshold: float = 0.80,
                    next_threshold: float = 0.50,
                ):
                    state0, layer_mask, position_ids, position_embeddings = model.encode(ids, mask)
                    open_idx = torch.arange(ids.size(0), device=ids.device)
                    last = None

                    def take_batch(x, idx):
                        if x is None:
                            return None
                        if x.size(0) == ids.size(0):
                            return x.index_select(0, idx)
                        return x

                    for corrections in range(cfg.max_correct):
                        if open_idx.numel() == 0:
                            break
                        sub_state0 = state0.index_select(0, open_idx)
                        sub_mask = take_batch(layer_mask, open_idx)
                        sub_pos = take_batch(position_ids, open_idx)
                        sub_emb = tuple(take_batch(t, open_idx) for t in position_embeddings) if position_embeddings else None
                        sub_lens = lens.index_select(0, open_idx)
                        result = jumprec.forward_budget_encoded(
                            sub_state0,
                            sub_mask,
                            sub_pos,
                            sub_emb,
                            sub_lens,
                            corrections,
                        )
                        logits, verify = result[:2]
                        next_logits = result[-1]
                        probs = F.softmax(logits, dim=-1)
                        top2 = probs.topk(2, dim=-1).values
                        next_probs = F.softmax(next_logits, dim=-1)
                        next_conf, next_pred = next_probs.max(dim=-1)
                        accept = (
                            (torch.sigmoid(verify) >= threshold)
                            & ((top2[:, 0] - top2[:, 1]) >= 0.05)
                            & (top2[:, 0] >= 0.45)
                            & (next_conf >= next_threshold)
                            & (next_pred == logits.argmax(dim=-1))
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

                out["jumprec_serial_nextagree050_080_ms_per_batch"] = time_fn(
                    lambda ids, mask, lens: serial_jumprec_nextagree(ids, mask, lens, 0.80, 0.50)
                )
                out["jumprec_serial_nextagree050_090_ms_per_batch"] = time_fn(
                    lambda ids, mask, lens: serial_jumprec_nextagree(ids, mask, lens, 0.90, 0.50)
                )
                out["jumprec_serial_nextagree070_090_ms_per_batch"] = time_fn(
                    lambda ids, mask, lens: serial_jumprec_nextagree(ids, mask, lens, 0.90, 0.70)
                )
                out["jumprec_serial_nextagree000_090_ms_per_batch"] = time_fn(
                    lambda ids, mask, lens: serial_jumprec_nextagree(ids, mask, lens, 0.90, 0.00)
                )
                out["jumprec_serial_nextagree015_090_ms_per_batch"] = time_fn(
                    lambda ids, mask, lens: serial_jumprec_nextagree(ids, mask, lens, 0.90, 0.15)
                )
            if cfg.use_budget_controller:
                def budget_jumprec(
                    ids,
                    mask,
                    lens,
                    threshold: float = 0.80,
                    escalate: bool = False,
                    open_loop: bool = False,
                    scan_up: bool = False,
                ):
                    state0, layer_mask, position_ids, position_embeddings = model.encode(ids, mask)
                    budget_choice = jumprec.budget_logits(state0, lens).argmax(dim=-1)
                    accepted = torch.zeros(ids.size(0), dtype=torch.bool, device=ids.device)
                    last = None

                    def take_batch(x, idx):
                        if x is None:
                            return None
                        if x.size(0) == ids.size(0):
                            return x.index_select(0, idx)
                        return x

                    for corrections in range(cfg.max_correct + 1):
                        if scan_up:
                            idx = torch.nonzero(
                                (~accepted)
                                & (budget_choice < cfg.max_correct + 1)
                                & (budget_choice <= corrections),
                                as_tuple=False,
                            ).flatten()
                        else:
                            idx = torch.nonzero(budget_choice == corrections, as_tuple=False).flatten()
                        if idx.numel() == 0:
                            continue
                        sub_state0 = state0.index_select(0, idx)
                        sub_mask = take_batch(layer_mask, idx)
                        sub_pos = take_batch(position_ids, idx)
                        sub_emb = tuple(take_batch(t, idx) for t in position_embeddings) if position_embeddings else None
                        sub_lens = lens.index_select(0, idx)
                        result = jumprec.forward_budget_encoded(
                            sub_state0,
                            sub_mask,
                            sub_pos,
                            sub_emb,
                            sub_lens,
                            corrections,
                        )
                        logits, verify = result[:2]
                        if open_loop:
                            accept = torch.ones(idx.numel(), dtype=torch.bool, device=ids.device)
                        else:
                            probs = F.softmax(logits, dim=-1)
                            top2 = probs.topk(2, dim=-1).values
                            accept = (
                                (torch.sigmoid(verify) >= threshold)
                                & ((top2[:, 0] - top2[:, 1]) >= 0.05)
                                & (top2[:, 0] >= 0.45)
                            )
                        accepted[idx] = accept
                        last = logits
                    if escalate and not scan_up and not open_loop:
                        next_choice = (budget_choice + 1).clamp(max=cfg.max_correct)
                        can_escalate = (~accepted) & (budget_choice < cfg.max_correct)
                        for corrections in range(1, cfg.max_correct + 1):
                            idx = torch.nonzero(can_escalate & (next_choice == corrections), as_tuple=False).flatten()
                            if idx.numel() == 0:
                                continue
                            sub_state0 = state0.index_select(0, idx)
                            sub_mask = take_batch(layer_mask, idx)
                            sub_pos = take_batch(position_ids, idx)
                            sub_emb = tuple(take_batch(t, idx) for t in position_embeddings) if position_embeddings else None
                            sub_lens = lens.index_select(0, idx)
                            result = jumprec.forward_budget_encoded(
                                sub_state0,
                                sub_mask,
                                sub_pos,
                                sub_emb,
                                sub_lens,
                                corrections,
                            )
                            logits, verify = result[:2]
                            probs = F.softmax(logits, dim=-1)
                            top2 = probs.topk(2, dim=-1).values
                            accept = (
                                (torch.sigmoid(verify) >= threshold)
                                & ((top2[:, 0] - top2[:, 1]) >= 0.05)
                                & (top2[:, 0] >= 0.45)
                            )
                            accepted[idx] = accept
                            last = logits
                    fallback_idx = torch.nonzero(~accepted, as_tuple=False).flatten()
                    if fallback_idx.numel() > 0:
                        sub_state0 = state0.index_select(0, fallback_idx)
                        sub_mask = take_batch(layer_mask, fallback_idx)
                        sub_pos = take_batch(position_ids, fallback_idx)
                        sub_emb = tuple(take_batch(t, fallback_idx) for t in position_embeddings) if position_embeddings else None
                        sub_lens = lens.index_select(0, fallback_idx)
                        full = model.run_steps_from(sub_state0, sub_state0, sub_mask, sub_pos, sub_emb, 0, cfg.loop_steps)
                        last = model.classify_state(full, sub_lens, sub_mask, sub_pos, sub_emb)
                    return last

                out["jumprec_budget_080_ms_per_batch"] = time_fn(budget_jumprec)
                out["jumprec_budget_090_ms_per_batch"] = time_fn(
                    lambda ids, mask, lens: budget_jumprec(ids, mask, lens, 0.90)
                )
                out["jumprec_budget_095_ms_per_batch"] = time_fn(
                    lambda ids, mask, lens: budget_jumprec(ids, mask, lens, 0.95)
                )
                out["jumprec_budget_escalate_080_ms_per_batch"] = time_fn(
                    lambda ids, mask, lens: budget_jumprec(ids, mask, lens, 0.80, True)
                )
                out["jumprec_budget_escalate_090_ms_per_batch"] = time_fn(
                    lambda ids, mask, lens: budget_jumprec(ids, mask, lens, 0.90, True)
                )
                out["jumprec_budget_escalate_095_ms_per_batch"] = time_fn(
                    lambda ids, mask, lens: budget_jumprec(ids, mask, lens, 0.95, True)
                )
                out["jumprec_budget_open_ms_per_batch"] = time_fn(
                    lambda ids, mask, lens: budget_jumprec(ids, mask, lens, open_loop=True)
                )
                out["jumprec_budget_scan_up_090_ms_per_batch"] = time_fn(
                    lambda ids, mask, lens: budget_jumprec(ids, mask, lens, 0.90, scan_up=True)
                )
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
                    item["jumprec_serial_070_ms_per_batch"] = time_fn(
                        lambda ids, mask, lens: serial_jumprec(ids, mask, lens, 0.70),
                        timing_bsz,
                    )
                    item["jumprec_serial_080_ms_per_batch"] = time_fn(serial_jumprec, timing_bsz)
                    item["jumprec_serial_085_ms_per_batch"] = time_fn(
                        lambda ids, mask, lens: serial_jumprec(ids, mask, lens, 0.85),
                        timing_bsz,
                    )
                    item["jumprec_serial_090_ms_per_batch"] = time_fn(
                        lambda ids, mask, lens: serial_jumprec(ids, mask, lens, 0.90),
                        timing_bsz,
                    )
                    item["jumprec_serial_095_ms_per_batch"] = time_fn(
                        lambda ids, mask, lens: serial_jumprec(ids, mask, lens, 0.95),
                        timing_bsz,
                    )
                    item["jumprec_serial_099_ms_per_batch"] = time_fn(
                        lambda ids, mask, lens: serial_jumprec(ids, mask, lens, 0.99),
                        timing_bsz,
                    )
                    item["jumprec_serial_agree_010_ms_per_batch"] = time_fn(
                        lambda ids, mask, lens: serial_jumprec_agree(ids, mask, lens, 0.10),
                        timing_bsz,
                    )
                    item["jumprec_serial_agree_050_ms_per_batch"] = time_fn(
                        lambda ids, mask, lens: serial_jumprec_agree(ids, mask, lens, 0.50),
                        timing_bsz,
                    )
                    item["jumprec_serial_agree_080_ms_per_batch"] = time_fn(
                        serial_jumprec_agree,
                        timing_bsz,
                    )
                    if cfg.use_stability_head:
                        item["jumprec_serial_stable050_080_ms_per_batch"] = time_fn(
                            lambda ids, mask, lens: serial_jumprec_stable(ids, mask, lens, 0.80, 0.50),
                            timing_bsz,
                        )
                        item["jumprec_serial_stable050_090_ms_per_batch"] = time_fn(
                            lambda ids, mask, lens: serial_jumprec_stable(ids, mask, lens, 0.90, 0.50),
                            timing_bsz,
                        )
                        item["jumprec_serial_stable070_090_ms_per_batch"] = time_fn(
                            lambda ids, mask, lens: serial_jumprec_stable(ids, mask, lens, 0.90, 0.70),
                            timing_bsz,
                        )
                    if cfg.use_utility_router:
                        item["jumprec_serial_utility_010_ms_per_batch"] = time_fn(
                            lambda ids, mask, lens: serial_jumprec_utility(ids, mask, lens, 0.10),
                            timing_bsz,
                        )
                        item["jumprec_serial_utility_050_ms_per_batch"] = time_fn(
                            serial_jumprec_utility,
                            timing_bsz,
                        )
                        item["jumprec_serial_utility_080_ms_per_batch"] = time_fn(
                            lambda ids, mask, lens: serial_jumprec_utility(ids, mask, lens, 0.80),
                            timing_bsz,
                        )
                        item["jumprec_serial_utility_090_ms_per_batch"] = time_fn(
                            lambda ids, mask, lens: serial_jumprec_utility(ids, mask, lens, 0.90),
                            timing_bsz,
                        )
                        item["jumprec_serial_selective_agree_speed_ms_per_batch"] = time_fn(
                            lambda ids, mask, lens: serial_jumprec_selective_agree(ids, mask, lens, 0.05, 0.90, 0.00),
                            timing_bsz,
                        )
                        item["jumprec_serial_selective_agree_quality002_ms_per_batch"] = time_fn(
                            lambda ids, mask, lens: serial_jumprec_selective_agree(ids, mask, lens, 0.35, 0.93, 0.30),
                            timing_bsz,
                        )
                        if cfg.use_consistency_head:
                            item["jumprec_serial_utility_cats050_010_ms_per_batch"] = time_fn(
                                lambda ids, mask, lens: serial_jumprec_utility_cats(ids, mask, lens, 0.10, 0.50),
                                timing_bsz,
                            )
                            item["jumprec_serial_utility_cats070_010_ms_per_batch"] = time_fn(
                                lambda ids, mask, lens: serial_jumprec_utility_cats(ids, mask, lens, 0.10, 0.70),
                                timing_bsz,
                            )
                            item["jumprec_serial_utility_cats090_010_ms_per_batch"] = time_fn(
                                lambda ids, mask, lens: serial_jumprec_utility_cats(ids, mask, lens, 0.10, 0.90),
                                timing_bsz,
                            )
                    if cfg.use_next_agreement_head:
                        item["jumprec_serial_nextagree050_080_ms_per_batch"] = time_fn(
                            lambda ids, mask, lens: serial_jumprec_nextagree(ids, mask, lens, 0.80, 0.50),
                            timing_bsz,
                        )
                        item["jumprec_serial_nextagree050_090_ms_per_batch"] = time_fn(
                            lambda ids, mask, lens: serial_jumprec_nextagree(ids, mask, lens, 0.90, 0.50),
                            timing_bsz,
                        )
                        item["jumprec_serial_nextagree070_090_ms_per_batch"] = time_fn(
                            lambda ids, mask, lens: serial_jumprec_nextagree(ids, mask, lens, 0.90, 0.70),
                            timing_bsz,
                        )
                        item["jumprec_serial_nextagree000_090_ms_per_batch"] = time_fn(
                            lambda ids, mask, lens: serial_jumprec_nextagree(ids, mask, lens, 0.90, 0.00),
                            timing_bsz,
                        )
                        item["jumprec_serial_nextagree015_090_ms_per_batch"] = time_fn(
                            lambda ids, mask, lens: serial_jumprec_nextagree(ids, mask, lens, 0.90, 0.15),
                            timing_bsz,
                        )
                    if cfg.use_budget_controller:
                        item["jumprec_budget_080_ms_per_batch"] = time_fn(budget_jumprec, timing_bsz)
                        item["jumprec_budget_090_ms_per_batch"] = time_fn(
                            lambda ids, mask, lens: budget_jumprec(ids, mask, lens, 0.90),
                            timing_bsz,
                        )
                        item["jumprec_budget_095_ms_per_batch"] = time_fn(
                            lambda ids, mask, lens: budget_jumprec(ids, mask, lens, 0.95),
                            timing_bsz,
                        )
                        item["jumprec_budget_escalate_080_ms_per_batch"] = time_fn(
                            lambda ids, mask, lens: budget_jumprec(ids, mask, lens, 0.80, True),
                            timing_bsz,
                        )
                        item["jumprec_budget_escalate_090_ms_per_batch"] = time_fn(
                            lambda ids, mask, lens: budget_jumprec(ids, mask, lens, 0.90, True),
                            timing_bsz,
                        )
                        item["jumprec_budget_escalate_095_ms_per_batch"] = time_fn(
                            lambda ids, mask, lens: budget_jumprec(ids, mask, lens, 0.95, True),
                            timing_bsz,
                        )
                        item["jumprec_budget_open_ms_per_batch"] = time_fn(
                            lambda ids, mask, lens: budget_jumprec(ids, mask, lens, open_loop=True),
                            timing_bsz,
                        )
                        item["jumprec_budget_scan_up_090_ms_per_batch"] = time_fn(
                            lambda ids, mask, lens: budget_jumprec(ids, mask, lens, 0.90, scan_up=True),
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
            "dry_natgraph_teacher",
            "dry_natgraph_joint_halt_quality_stability",
            "dry_natgraph_joint_halt_quality_stability_reuse",
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
            "dry_strathop_polish2_cost_controller",
            "dry_strathop_polish2_calib_controller",
            "dry_strathop_polish2_budget_controller",
            "dry_strathop_polish2_budget_controller_reuse",
            "dry_strathop_polish2_budget_verifytarget",
            "dry_strathop_polish2_budget_verifytarget_reuse",
            "dry_strathop_polish2_stability_router",
            "dry_strathop_polish2_stability_router_reuse",
            "dry_strathop_polish2_utility_router",
            "dry_strathop_polish2_utility_router_reuse",
            "dry_strathop_polish2_utility_stability_router",
            "dry_strathop_polish2_utility_stability_router_reuse",
            "dry_strathop_polish2_nextagree_router",
            "dry_strathop_polish2_nextagree_router_reuse",
            "dry_strathop_polish2_joint_halt",
            "dry_strathop_polish2_joint_halt_reuse",
            "dry_strathop_polish2_joint_halt_stability",
            "dry_strathop_polish2_joint_halt_stability_reuse",
            "dry_strathop_polish2_joint_halt_quality",
            "dry_strathop_polish2_joint_halt_quality_reuse",
            "dry_strathop_polish2_joint_halt_quality_agdistill",
            "dry_strathop_polish2_joint_halt_quality_agdistill_reuse",
            "dry_strathop_polish2_joint_halt_quality_cats",
            "dry_strathop_polish2_joint_halt_quality_cats_reuse",
            "dry_strathop_polish2_joint_halt_slo",
            "dry_strathop_polish2_joint_halt_slo_reuse",
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
            "core3_8n4h_natgraph_teacher",
            "core3_8n4h_natgraph_teacher_resume",
            "core3_8n4h_natgraph_polish_teacher",
            "core3_8n4h_natgraph_polish2_teacher",
            "core3_8n4h_strathop_gate_teacher",
            "core3_8n4h_strathop_polish_teacher",
            "core3_8n4h_strathop_polish2_teacher",
            "core3_8n4h_strathop_eval_teacher",
            "core3_8n4h_natgraph_eval_teacher",
            "core3_8n4h_natgraph_polish_eval_teacher",
            "core3_8n4h_natgraph_polish2_eval_teacher",
            "core3_8n4h_natgraph_polish2_audit_teacher",
            "core3_8n4h_strathop_polish2_eval_teacher",
            "core3_8n4h_strathop_audit_teacher",
            "core3_8n4h_strathop_polish2_audit_teacher",
            "core3_8n4h_strathop_verifier_audit",
            "core3_8n4h_strathop_polish2_verifier_audit",
            "core3_8n4h_strathop_jumprec",
            "core3_8n4h_strathop_gate_jumprec",
            "core3_8n4h_strathop_polish_jumprec",
            "core3_8n4h_strathop_polish2_jumprec",
            "core3_8n4h_strathop_cost_controller",
            "core3_8n4h_strathop_polish2_cost_controller",
            "core3_8n4h_strathop_calib_controller",
            "core3_8n4h_strathop_polish2_calib_controller",
            "core3_8n4h_strathop_budget_controller",
            "core3_8n4h_strathop_polish2_budget_controller",
            "core3_8n4h_strathop_budget_controller_reuse",
            "core3_8n4h_strathop_polish2_budget_controller_reuse",
            "core3_8n4h_strathop_budget_verifytarget",
            "core3_8n4h_strathop_polish2_budget_verifytarget",
            "core3_8n4h_strathop_budget_verifytarget_reuse",
            "core3_8n4h_strathop_polish2_budget_verifytarget_reuse",
            "core3_8n4h_strathop_stability_router",
            "core3_8n4h_strathop_polish2_stability_router",
            "core3_8n4h_strathop_stability_router_reuse",
            "core3_8n4h_strathop_polish2_stability_router_reuse",
            "core3_8n4h_strathop_utility_router",
            "core3_8n4h_strathop_polish2_utility_router",
            "core3_8n4h_strathop_utility_router_reuse",
            "core3_8n4h_strathop_polish2_utility_router_reuse",
            "core3_8n4h_strathop_utility_stability_router",
            "core3_8n4h_strathop_polish2_utility_stability_router",
            "core3_8n4h_strathop_utility_stability_router_reuse",
            "core3_8n4h_strathop_polish2_utility_stability_router_reuse",
            "core3_8n4h_strathop_nextagree_router",
            "core3_8n4h_strathop_polish2_nextagree_router",
            "core3_8n4h_strathop_nextagree_router_reuse",
            "core3_8n4h_strathop_polish2_nextagree_router_reuse",
            "core3_8n4h_strathop_joint_halt",
            "core3_8n4h_strathop_polish2_joint_halt",
            "core3_8n4h_strathop_joint_halt_reuse",
            "core3_8n4h_strathop_polish2_joint_halt_reuse",
            "core3_8n4h_strathop_joint_halt_stability",
            "core3_8n4h_strathop_polish2_joint_halt_stability",
            "core3_8n4h_strathop_joint_halt_stability_reuse",
            "core3_8n4h_strathop_polish2_joint_halt_stability_reuse",
            "core3_8n4h_strathop_joint_halt_reuse_highval",
            "core3_8n4h_strathop_polish2_joint_halt_reuse_highval",
            "core3_8n4h_strathop_joint_halt_stability_reuse_highval",
            "core3_8n4h_strathop_polish2_joint_halt_stability_reuse_highval",
            "core3_8n4h_strathop_joint_halt_quality",
            "core3_8n4h_strathop_polish2_joint_halt_quality",
            "core3_8n4h_strathop_joint_halt_quality_reuse",
            "core3_8n4h_strathop_polish2_joint_halt_quality_reuse",
            "core3_8n4h_strathop_joint_halt_quality_reuse_highval",
            "core3_8n4h_strathop_polish2_joint_halt_quality_reuse_highval",
            "core3_8n4h_strathop_joint_halt_quality_agdistill",
            "core3_8n4h_strathop_polish2_joint_halt_quality_agdistill",
            "core3_8n4h_strathop_joint_halt_quality_agdistill_reuse",
            "core3_8n4h_strathop_polish2_joint_halt_quality_agdistill_reuse",
            "core3_8n4h_strathop_joint_halt_quality_agdistill_reuse_highval",
            "core3_8n4h_strathop_polish2_joint_halt_quality_agdistill_reuse_highval",
            "core3_8n4h_strathop_joint_halt_quality_cats",
            "core3_8n4h_strathop_polish2_joint_halt_quality_cats",
            "core3_8n4h_strathop_joint_halt_quality_cats_reuse",
            "core3_8n4h_strathop_polish2_joint_halt_quality_cats_reuse",
            "core3_8n4h_strathop_joint_halt_quality_cats_reuse_highval",
            "core3_8n4h_strathop_polish2_joint_halt_quality_cats_reuse_highval",
            "core3_8n4h_natgraph_joint_halt_quality_stability",
            "core3_8n4h_natgraph_joint_halt_quality_stability_reuse",
            "core3_8n4h_natgraph_joint_halt_quality_stability_reuse_audit",
            "core3_8n4h_natgraph_joint_halt_quality_stability_reuse_highval",
            "core3_8n4h_natgraph_polish2_joint_halt_quality_stability",
            "core3_8n4h_natgraph_polish2_joint_halt_quality_stability_reuse",
            "core3_8n4h_natgraph_polish2_joint_halt_quality_stability_reuse_audit",
            "core3_8n4h_natgraph_polish2_joint_halt_quality_stability_reuse_highval",
            "core3_8n4h_strathop_joint_halt_quality_stability",
            "core3_8n4h_strathop_polish2_joint_halt_quality_stability",
            "core3_8n4h_strathop_joint_halt_quality_stability_reuse",
            "core3_8n4h_strathop_polish2_joint_halt_quality_stability_reuse",
            "core3_8n4h_strathop_joint_halt_quality_stability_reuse_highval",
            "core3_8n4h_strathop_polish2_joint_halt_quality_stability_reuse_highval",
            "core3_8n4h_strathop_joint_halt_slo",
            "core3_8n4h_strathop_polish2_joint_halt_slo",
            "core3_8n4h_strathop_joint_halt_slo_reuse",
            "core3_8n4h_strathop_polish2_joint_halt_slo_reuse",
            "core3_8n4h_strathop_joint_halt_slo_reuse_highval",
            "core3_8n4h_strathop_polish2_joint_halt_slo_reuse_highval",
            "core3_8n4h_strathop_joint_halt_slo_stability",
            "core3_8n4h_strathop_polish2_joint_halt_slo_stability",
            "core3_8n4h_strathop_joint_halt_slo_stability_reuse",
            "core3_8n4h_strathop_polish2_joint_halt_slo_stability_reuse",
            "core3_8n4h_strathop_joint_halt_slo_stability_reuse_highval",
            "core3_8n4h_strathop_polish2_joint_halt_slo_stability_reuse_highval",
            "core3_8n4h_strathop_ablate_no_adapter",
            "core3_8n4h_strathop_ablate_no_distill",
            "core3_8n4h_strathop_ablate_no_verifier",
            "core3_8n4h_strathop_polish2_ablate_no_adapter",
            "core3_8n4h_strathop_polish2_ablate_no_distill",
            "core3_8n4h_strathop_polish2_ablate_no_verifier",
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
        "dry_natgraph_teacher",
        "dry_natgraph_joint_halt_quality_stability",
        "dry_natgraph_joint_halt_quality_stability_reuse",
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
        "dry_strathop_polish2_cost_controller",
        "dry_strathop_polish2_calib_controller",
        "dry_strathop_polish2_budget_controller",
        "dry_strathop_polish2_budget_controller_reuse",
        "dry_strathop_polish2_budget_verifytarget",
        "dry_strathop_polish2_budget_verifytarget_reuse",
        "dry_strathop_polish2_stability_router",
        "dry_strathop_polish2_stability_router_reuse",
        "dry_strathop_polish2_utility_router",
        "dry_strathop_polish2_utility_router_reuse",
        "dry_strathop_polish2_utility_stability_router",
        "dry_strathop_polish2_utility_stability_router_reuse",
        "dry_strathop_polish2_nextagree_router",
        "dry_strathop_polish2_nextagree_router_reuse",
        "dry_strathop_polish2_joint_halt",
        "dry_strathop_polish2_joint_halt_reuse",
        "dry_strathop_polish2_joint_halt_stability",
        "dry_strathop_polish2_joint_halt_stability_reuse",
        "dry_strathop_polish2_joint_halt_quality",
        "dry_strathop_polish2_joint_halt_quality_reuse",
        "dry_strathop_polish2_joint_halt_quality_agdistill",
        "dry_strathop_polish2_joint_halt_quality_agdistill_reuse",
        "dry_strathop_polish2_joint_halt_quality_cats",
        "dry_strathop_polish2_joint_halt_quality_cats_reuse",
        "dry_strathop_polish2_joint_halt_slo",
        "dry_strathop_polish2_joint_halt_slo_reuse",
        "dry_sweep",
        "dry_sweep_reuse",
    ):
        raise SystemExit("Local CPU guard: only dry modes are allowed locally.")
    cfg = config_for_mode(args.mode)
    if args.seed is not None:
        cfg.seed = args.seed
    result = run_experiment(cfg, "cpu")
    print(json.dumps(result, indent=2))
