# JumpRec Current State

Last updated: 2026-04-28.

This is the first file to read after `README.md`. It is intentionally short:
use it to orient yourself, then use `JUMPREC_RESULTS.md` for evidence and
`JUMPREC_NEXT_STEPS.md` for the work queue.

## Current Answer

Selective agreement is the active adaptive-inference contract.

The current result says:

1. A one-candidate utility router has useful signal but does not fully recover
   the agreement frontier.
2. Always running adjacent-budget agreement is high quality but too expensive.
3. Selective agreement is the first policy that matches the agreement frontier
   while reducing adjacent checks.

The contract is:

1. Accept high-utility candidates directly.
2. Run adjacent-budget agreement only for ambiguous candidates.
3. Fall back to the full recurrent teacher when neither path is trusted.

## Headline Numbers

High-validation means over seeds 101, 202, and repaired polish2 seed 42:

| Policy | Accuracy | Counted Core | Adjacent Check Rate |
|---|---:|---:|---:|
| Utility speed selector | 99.628% | 2.87 / 18 | 0% |
| True agreement | 99.711% | 2.71 / 18 | Broad reference check |
| Selective agreement | 99.713% | 2.70 / 18 | 29.9% |

Timing means:

| Batch size | Full loop | Utility 0.90 | True agreement 0.50 | Selective agreement |
|---:|---:|---:|---:|---:|
| 1 | 22.64 ms | 11.98 ms | 16.59 ms | 12.24 ms |
| 64 | 34.96 ms | 40.15 ms | 48.03 ms | 38.94 ms |

Interpretation:

- Small-batch and interactive use is unblocked for general looped-LLM
  application testing.
- Batched throughput is not solved. Dynamic subset routing is still slower
  than full-loop or all-budgets parallel execution at batch size 64.

## Active Checkpoints

The current promoted family is:

```text
*_joint_halt_quality_stability_seed{seed}
```

Cross-seed mapping:

| Seed | Mode prefix | Checkpoint family |
|---:|---|---|
| 101 | `core3_8n4h_strathop` | `core3_8n4h_strathop_joint_halt_quality_stability_seed101` |
| 202 | `core3_8n4h_strathop` | `core3_8n4h_strathop_joint_halt_quality_stability_seed202` |
| 42 | `core3_8n4h_strathop_polish2` | `core3_8n4h_strathop_polish2_joint_halt_quality_stability_seed42` |

Reuse/high-validation modes load those checkpoints and run evaluation only.

## Reproduce The Current Result

```powershell
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_quality_stability_reuse_highval --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_quality_stability_reuse_highval --seed 202
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_polish2_joint_halt_quality_stability_reuse_highval --seed 42
```

Expected high-level output:

- `heldout_threshold_audit.selective_agreement` reports the selector result.
- `timing.jumprec_serial_selective_agree_speed_ms_per_batch` reports the
  representative speed-point serial timing.
- `timing.batch_size_sweep["1"]` and `["64"]` must be interpreted separately.

## Local Guardrails

Run these before pushing code or mode changes:

```powershell
python -m py_compile .\run_recurrent_smol.py
python -m unittest discover -s tests
python .\run_recurrent_smol.py --mode dry_strathop_polish2_joint_halt_stability_reuse --local
```

The dry run is not a quality result. It only checks wiring for the fake-model
path, held-out audit, and timing keys.

## What Is Current

- Current promoted controller: selective agreement.
- Current checkpoint family: quality plus jointly trained stability.
- Current claim regime: synthetic recurrence benchmark, small-batch adaptive
  inference.
- Current next application step: build the first small-batch general looped-LLM
  application probe around selective agreement.
- Current throughput step: test grouped or cached adjacent checks so selective
  agreement does not fragment batch-64 execution.

## What Is Historical Or Diagnostic

- `no_agree` and plain utility are speed-shape references, not current winners.
- True agreement is the quality reference, not the final scalable contract.
- CATS, next-agreement heads, agreement distillation, probe upper bounds, and
  per-budget thresholding are useful negative or diagnostic branches.
- Broad language-model acceleration is not yet established.

## Claim Boundary

Acceptable current claim:

> On a synthetic textual recurrence benchmark, selective agreement matches the
> true-agreement quality frontier while reducing adjacent checks, and it gives
> favorable small-batch latency.

Not acceptable yet:

> JumpRec accelerates language models generally.

The next phase should earn that broader claim through a non-synthetic
small-batch looped-LLM application test.
