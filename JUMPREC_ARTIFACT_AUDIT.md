# JumpRec Artifact Audit

This is the firewall for avoiding a Lighthouse-style false positive.

JumpRec's current evidence is not vulnerable to the exact Lighthouse failure:
there is no non-causal chunk summary broadcast back into earlier token
positions, and the main metric is classification on generated recurrence
targets rather than teacher-forced next-token BPC. The remaining risks are
different: prompt shortcuts, verifier privilege, teacher-quality selection,
synthetic-task overfitting, and timing artifacts.

## Promotion Rule

Do not promote a JumpRec result unless it passes all four gates:

1. **Information gate**: the accepted jump path uses only state available at
   deployment time.
2. **Control gate**: JumpRec beats direct, early-exit, and equal-compute
   controls on the same seeds and data distribution.
3. **Robustness gate**: the full recurrent teacher is strong by hop/task, and
   JumpRec remains positive on held-out task variants.
4. **Execution gate**: counted core-layer savings convert to wall-clock savings
   in the stated batch-size regime.

Any result that fails one gate can still be logged as diagnostic, but it should
not become a headline.

## Current Promotion Status

The active promoted contract is selective agreement on the synthetic
8-node / 4-hop SmolLM2 hard case.

It passes the current small-batch claim boundary:

- information gate: JumpRec and router features use deployment-available prompt
  and candidate signals;
- control gate: direct, early-exit, full-loop, no-agreement, true-agreement,
  and all-budget references have been reported;
- robustness gate: three teacher seeds pass the widened hop/task checks and
  prompt artifact audits;
- execution gate: batch-1 timing is favorable for selective agreement.

It does not yet pass a high-throughput serving claim. At batch size 64,
selective agreement is faster than true agreement but still slower than the
full-loop/all-budgets parallel baselines because dynamic subset execution
fragments the GPU workload.

## Information Gate

### What is currently OK

- `run_recurrent_smol.py` builds prompts as:
  `Task`, `Map`, `Start`, `Hops`, then `Answer:`.
- The classifier reads the final prompt position through `read_last`, not a
  pooled window that includes answer tokens.
- SmolLM path builds a causal attention mask for the pretrained backbone.
- JumpRec receives `state0`, the frozen teacher's encoded prompt state, plus
  the same masks/positions the teacher uses.
- Verifier features are derived from the proposed state and proposed logits:
  hidden readout, entropy, margin, and max probability.

### Required checks

- Done: generated prompts assert that nothing appears after `Answer:`.
- Done: permutation relabeling audit preserves accuracy.
- Done: map scrambling audit collapses accuracy near chance.
- Done: hop randomization collapses accuracy near chance.
- Still needed before broader LM claims: task-random/blind audits on a mixed
  non-synthetic application and a verifier-input audit after package extraction.

## Verifier Gate

The verifier is allowed to see only deployment-time uncertainty features from
the proposed path. It is not allowed to see whether the proposal is correct,
except as a training label.

Current verifier training uses:

```text
good = proposed_argmax == target
```

That is acceptable as supervision. It would become invalid only if `target`,
teacher correctness, or full-loop logits were used as verifier inputs or as
threshold-selection data on the same eval set being reported.

Required checks:

- Report verifier calibration on held-out eval batches, not just routed
  accuracy.
- Freeze thresholds before final eval, or tune thresholds on a separate
  validation split and report once on a final split.
- Report no-agreement and agreement policies separately. Agreement routing
  spends extra JumpRec budgets and must be counted in timing.
- Include an oracle-router upper bound, clearly labeled as oracle, to show how
  much headroom exists without mixing it into real claims.

## Control Gate

Every promoted table needs these rows:

- Full recurrent teacher.
- One-loop or zero-loop teacher.
- Early-exit teacher using confidence thresholds.
- Direct control with the same starting state and comparable trainable budget.
- JumpRec fixed budgets.
- JumpRec router with full-loop fallback.

Required ablations:

- No temporary adapter.
- No distillation loss.
- No verifier loss.
- Train/eval on held-out seeds.
- Train on one hop/task distribution, evaluate on another.
- Equal wall-clock control when possible, not only equal counted layers.

## Robustness Gate

Teacher quality is part of the architecture, not a nuisance variable.

Required checks:

- Report full teacher accuracy by hop and task.
- Report worst-hop accuracy and restore the best worst-hop checkpoint.
- Do not train JumpRec on hard-case checkpoints that fail the teacher gate
  unless the run is explicitly diagnostic.
- For hard-hop results, require at least three seed-confirmed teachers before
  making architecture claims.
- Keep synthetic recurrence language in the claim until a non-synthetic LM or
  reasoning evaluation is positive.

## Execution Gate

Counted layer savings are not the same as speedup.

Required checks:

- Report batch-size sweep from 1 through 64.
- Separate local-latency claims from throughput-serving claims.
- Include timing for full teacher, all-budget JumpRec, serial router,
  agreement-aware router, and direct control.
- Count agreement checks as real work.
- Use checkpoint reuse for timing-only probes so retraining noise does not
  blur the execution result.
- State the hardware and batch size in every speedup claim.

## Lighthouse-Specific Regression Tests

These are the direct lessons from Lighthouse:

- No operation may summarize hidden states from positions later than the
  prediction/readout position and feed that summary back into the readout path.
- `detach()` is not an information barrier. Detached tensors are still forward
  inputs and must be treated as potentially privileged.
- A removal/collapse test does not prove mechanism. It can also prove the model
  became dependent on a shortcut.
- Cross-seed and cross-model replication does not validate causality if the
  same access-pattern bug is present in every run.
- Teacher-forced metrics are not enough for any generative LM extension. Any
  future language-modeling JumpRec claim needs autoregressive evaluation.

## Current Risk Register

| Risk | Status | Mitigation |
|---|---|---|
| Non-causal token leakage | Low in current classification runner | Add prompt/attention assertions anyway |
| Prompt shortcut / symbolic memorization | Medium | Relabeling, scrambling, hop/task blind audits |
| Verifier privilege | Medium | Audit verifier inputs and split threshold tuning |
| Teacher selection bias | Medium | Preserve teacher gates and report failed teachers |
| Synthetic-only generality | High | Keep claims narrow until non-synthetic eval |
| Counted-compute vs wall-clock gap | High | Batch sweep and hardware-specific claims; selective agreement currently passes batch-1 but not batch-64 throughput |

## Current Claim Wording

Acceptable:

> On a synthetic textual recurrence benchmark, selective agreement matches the
> true-agreement quality frontier while reducing adjacent-budget checks, and it
> gives favorable small-batch latency.

Not acceptable yet:

> JumpRec accelerates language models generally.

Not acceptable without more audits:

> JumpRec learns reasoning.
