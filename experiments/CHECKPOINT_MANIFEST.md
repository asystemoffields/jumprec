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
| Natgraph base bridge | `core3_8n4h_natgraph_seed{seed}` | Natural-language route-card bridge; seed 101 base teacher was hop-4 weak. |
| Natgraph polish | `core3_8n4h_natgraph_polish_seed{seed}` | Max-hop-focused polish from the base natgraph teacher. |
| Natgraph polish2 | `core3_8n4h_natgraph_polish2_seed{seed}` | Active seed-101 natgraph predecessor for joint halt/selective agreement. |

## Active Joint-Halt Modes

Training modes load the teacher family and save a reusable joint-halt checkpoint:

| Mode pattern | Loads | Saves |
|---|---|---|
| `core3_8n4h_strathop_joint_halt` | `core3_8n4h_strathop_seed{seed}` | `core3_8n4h_strathop_joint_halt_seed{seed}` |
| `core3_8n4h_strathop_polish2_joint_halt` | `core3_8n4h_strathop_polish2_seed{seed}` | `core3_8n4h_strathop_polish2_joint_halt_seed{seed}` |
| `*_joint_halt_stability` | matching teacher | `*_joint_halt_stability_seed{seed}` |
| `*_joint_halt_quality` | matching teacher | `*_joint_halt_quality_seed{seed}` |
| `*_joint_halt_slo` | matching teacher | `*_joint_halt_slo_seed{seed}` |
| `*_joint_halt_quality_agdistill` | matching teacher | `*_joint_halt_quality_agdistill_seed{seed}` |
| `*_joint_halt_quality_stability` | matching teacher | `*_joint_halt_quality_stability_seed{seed}` |
| `*_joint_halt_slo_stability` | matching teacher | `*_joint_halt_slo_stability_seed{seed}` |
| `*_joint_halt_quality_cats` | `*_joint_halt_quality_seed{seed}` | `*_joint_halt_quality_cats_seed{seed}` |
| `core3_8n4h_natgraph_joint_halt_quality_stability` | `core3_8n4h_natgraph_seed{seed}` | `core3_8n4h_natgraph_joint_halt_quality_stability_seed{seed}` |
| `core3_8n4h_natgraph_polish2_joint_halt_quality_stability` | `core3_8n4h_natgraph_polish2_seed{seed}` | `core3_8n4h_natgraph_polish2_joint_halt_quality_stability_seed{seed}` |

## Current Promoted Checkpoint Family

Selective agreement currently uses the quality-plus-stability checkpoints:

| Seed | Mode prefix | Checkpoint tag |
|---:|---|---|
| 101 | `core3_8n4h_strathop` | `core3_8n4h_strathop_joint_halt_quality_stability_seed101` |
| 202 | `core3_8n4h_strathop` | `core3_8n4h_strathop_joint_halt_quality_stability_seed202` |
| 42 | `core3_8n4h_strathop_polish2` | `core3_8n4h_strathop_polish2_joint_halt_quality_stability_seed42` |

The active evaluation modes are:

```text
core3_8n4h_strathop_joint_halt_quality_stability_reuse_highval
core3_8n4h_strathop_polish2_joint_halt_quality_stability_reuse_highval
```

These modes run the high-validation held-out audit, the selective-agreement
audit, the probe upper-bound audit, and the batch-size timing sweep.

The natgraph bridge reuse mode is:

```text
core3_8n4h_natgraph_joint_halt_quality_stability_reuse_highval
core3_8n4h_natgraph_polish2_joint_halt_quality_stability_reuse_highval
core3_8n4h_natgraph_polish2_joint_halt_quality_stability_reuse_audit
```

The active seed-101 bridge uses the `natgraph_polish2` variants. The highval
mode runs the selective-agreement held-out selector audit on the route-card
prompt distribution. The reuse-audit mode skips held-out threshold grids and
runs teacher plus JumpRec prompt variants: `normal`, `relabel`, `map_scramble`,
and `hop_random`.

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

The agreement-distilled quality modes are full joint-halt training runs. They
load the matching teacher, train candidate/halting paths with adjacent-budget
and full-teacher agreement distillation, and save
`*_joint_halt_quality_agdistill_seed{seed}`.

The quality/SLO stability modes are also full joint-halt training runs. They
load the matching teacher, jointly train the stability head with candidate and
utility objectives, and save either `*_joint_halt_quality_stability_seed{seed}`
or `*_joint_halt_slo_stability_seed{seed}`.

## Current Cross-Seed Mapping

| Seed | Mode prefix | Teacher |
|---:|---|---|
| 42 | `core3_8n4h_strathop_polish2` | repaired polish2 teacher |
| 101 | `core3_8n4h_strathop` | base teacher |
| 202 | `core3_8n4h_strathop` | base teacher |

If a dry or reuse path fails immediately with a missing checkpoint, run the
matching predecessor in the tables above first.
