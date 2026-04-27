# Checkpoint Manifest

This file documents the current checkpoint dependencies for the active
recurrent SmolLM2 JumpRec experiments. It is not a complete historical ledger;
use `JUMPREC_RESULTS.md` for run history and interpretation.

## Storage

Modal runs save summaries and checkpoints in the `/results` volume. Local runs
use `checkpoints/` by default. Checkpoint tags interpolate `{seed}`.

## Teacher Families

These are the required predecessors for current JumpRec and joint-halting work:

| Family | Teacher checkpoint tag | Notes |
|---|---|---|
| Base 8-node / 4-hop | `core3_8n4h_strathop_seed{seed}` | Seeds 101 and 202 in the current cross-seed set. |
| Repaired polish2 | `core3_8n4h_strathop_polish2_seed{seed}` | Seed 42 in the current cross-seed set. |

## Active Joint-Halt Modes

Training modes load the teacher family and save a reusable joint-halt checkpoint:

| Mode pattern | Loads | Saves |
|---|---|---|
| `core3_8n4h_strathop_joint_halt` | `core3_8n4h_strathop_seed{seed}` | `core3_8n4h_strathop_joint_halt_seed{seed}` |
| `core3_8n4h_strathop_polish2_joint_halt` | `core3_8n4h_strathop_polish2_seed{seed}` | `core3_8n4h_strathop_polish2_joint_halt_seed{seed}` |
| `*_joint_halt_stability` | matching teacher | `*_joint_halt_stability_seed{seed}` |
| `*_joint_halt_quality` | matching teacher | `*_joint_halt_quality_seed{seed}` |
| `*_joint_halt_slo` | matching teacher | `*_joint_halt_slo_seed{seed}` |
| `*_joint_halt_quality_cats` | `*_joint_halt_quality_seed{seed}` | `*_joint_halt_quality_cats_seed{seed}` |

Reuse modes load the corresponding joint-halt checkpoint, skip additional
joint-halt training, and run evaluation/audit only:

| Mode suffix | Behavior |
|---|---|
| `_reuse` | Loads the matching checkpoint and runs the standard held-out audit. |
| `_reuse_highval` | Loads the matching checkpoint and runs the larger validation/final audit with the finer threshold grid and selector scenarios. |

The CATS-style quality modes are post-hoc consistency-head runs. They load a
corrected quality joint-halt checkpoint, freeze the existing teacher/JumpRec
path, train only `consistency_heads`, and save a reusable
`*_joint_halt_quality_cats_seed{seed}` checkpoint. Their reuse modes load that
CATS checkpoint directly.

## Current Cross-Seed Mapping

| Seed | Mode prefix | Teacher |
|---:|---|---|
| 42 | `core3_8n4h_strathop_polish2` | repaired polish2 teacher |
| 101 | `core3_8n4h_strathop` | base teacher |
| 202 | `core3_8n4h_strathop` | base teacher |

If a dry or reuse path fails immediately with a missing checkpoint, run the
matching predecessor in the tables above first.
