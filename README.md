# JumpRec

JumpRec is an experiment in adaptive inference for looped language models.
The central question is whether a model can spend extra computation only when a
problem needs it, while preserving the performance of a deeper looped model.

The current codebase studies that question with synthetic text tasks and a
recurrent SmolLM2-135M retrofit. Results, run history, caveats, and tables live
in `JUMPREC_RESULTS.md`; this README is meant to explain the architecture and
how the repo is currently set up.

## Start Here

- `JUMPREC_CURRENT_STATE.md`: the short current-state handoff. Read this first
  if you are a new instance.
- `JUMPREC_RESULTS.md`: full experiment ledger, tables, and interpretation.
- `JUMPREC_NEXT_STEPS.md`: current queue plus older handoff notes.
- `experiments/CHECKPOINT_MANIFEST.md`: active checkpoint dependencies.
- `JUMPREC_ARTIFACT_AUDIT.md`: claim-safety and audit gates.

Current headline: selective agreement is the active adaptive contract. It
matches the true-agreement quality frontier on the synthetic hard case while
checking adjacent budgets on only the ambiguous slice. It is promoted for
small-batch/interactive application testing, not yet for high-throughput
serving.

## Core Idea

A recurrent teacher runs part of a transformer multiple times. That gives the
model extra depth for problems that require sequential refinement, but the full
loop is expensive if every input pays the maximum cost.

JumpRec searches for a good learned shortcut:

1. Train a strong full-loop recurrent teacher.
2. Train a small jump module that maps the initial encoded state to a state
   near a later point in the teacher's recurrent computation.
3. Run only a short remaining tail of the recurrent loop.
4. Use a verifier or halting policy to decide whether the shortcut is good
   enough.
5. Fall back to the full recurrent teacher when the shortcut is uncertain.

## Architecture

The main implementation is `run_recurrent_smol.py`.

The recurrent teacher is a pretrained SmolLM2-style decoder split into three
parts:

- `prelude`: early transformer blocks that encode the prompt.
- `core`: a shared block group that is looped several times.
- `coda`: final transformer blocks and the classifier head.

The loop uses input reinjection and loop/time conditioning so that repeated
passes aren't just identical block calls. The teacher is trained on textual
algorithmic tasks where the answer should improve with recurrent depth.

JumpRec sits on top of the trained teacher. For each correction budget `c`, it:

- adds a learned landing embedding for that budget;
- runs a small copied stack of jump blocks;
- optionally applies a temporary low-rank adapter;
- runs the remaining teacher tail from that landing state;
- emits candidate logits plus verifier features.

The verifier features are deployment-available signals only: candidate entropy,
top-class margin, max probability, budget id, and hidden-state readout features.
Routers are not allowed to inspect privileged labels or teacher internals that
would be unavailable at inference time.

## Routing And Control

Routing is the controller layer between the jump candidates and the full-loop
fallback. It decides how much recurrent compute an example should spend without
using labels, future teacher states, or any signal that would be missing at
deployment time.

The controller surface is split into deployable one-candidate policies and
audit-only reference policies:

- `no_agree`: accepts the first candidate whose verifier and confidence gates
  pass.
- `agree`: additionally checks whether adjacent correction budgets predict the
  same answer. This is a strong quality reference, but it is expensive because
  it runs extra budget candidates.
- `selective_agreement`: accepts high-utility candidates directly, runs the
  adjacent-budget agreement check only for ambiguous cases, and falls back to
  the full teacher otherwise. This is the current best small-batch adaptive
  contract because it preserves agreement-level quality while reducing the
  adjacent-check surface.
- `budget_controller`: predicts a sufficient correction budget from the initial
  state before candidate evaluation. It is the cheapest controller shape, but it
  has to infer difficulty before seeing candidate confidence.
- `stability_router`: learns to approximate adjacent-budget agreement.
- `utility_router`: learns a post-hoc accept/fallback policy from route utility.
- `joint_halt`: trains the candidate path and utility halting head together.
- `utility_cats`: adds a cheap consistency head to a utility router. The head
  predicts whether the current candidate is stable against the next correction
  budget, with the final budget trained against the full teacher prediction.
  It is deployment-available because inference reads one candidate plus the
  consistency/utility heads, rather than running the adjacent budget.
- `utility_per_budget`: an audit-only held-out selector that learns separate
  utility thresholds for each correction budget without additional training.
- `utility_then_agree` and `agree_then_utility`: hybrid audit policies that
  test whether utility confidence can reduce or condition adjacent-budget
  agreement. These are useful diagnostics, but any policy that depends on true
  agreement still has the agreement path's extra candidate cost.

`joint_halt` is the main deployable controller family. Earlier post-hoc routers
score candidates after the jump model is already trained. Joint halting changes
the training problem: candidate logits and halting probabilities are optimized
together against route utility, so the candidate path can learn states that are
easier to accept safely. Its active variants differ mainly in what they ask the
controller to optimize:

- `joint_halt_quality`: emphasizes avoiding false accepts, even at higher
  fallback cost.
- `joint_halt_slo`: samples cost and false-accept weights so a single
  checkpoint can serve multiple operating points.
- `joint_halt_quality_cats`: freezes a quality checkpoint and trains a cheap
  consistency head as a one-candidate proxy for agreement.
- `joint_halt_quality_agdistill`: distills adjacent-budget and full-teacher
  agreement into the candidate logits and route-risk term during joint training.
- `joint_halt_quality_stability` / `joint_halt_slo_stability`: jointly train a
  stability head and feed that feature back into the utility router.

The budget controller therefore fits as the lowest-overhead member of a broader
control stack: it chooses a compute budget up front, while utility and
joint-halting controllers can inspect candidate confidence before deciding to
accept or fall back. At small scale the difference is mostly counted core
layers. At larger scale it becomes an execution-planning problem: every extra
candidate, dynamic branch, or fallback split can create latency and batching
overhead even when the counted model FLOPs look favorable.

## Training And Evaluation Flow

Most serious runs follow this sequence:

1. Train or load a recurrent teacher checkpoint.
2. Train or load JumpRec candidate heads.
3. Train an optional router/controller/halting head.
4. Evaluate candidate accuracy, fallback routing, oracle routing, calibration,
   held-out threshold selection, and wall-clock timing.
5. Save checkpoints into the Modal volume for reuse.

Thresholds are selected on a held-out validation split and then reported on a
separate final split. Timing is reported separately from counted core-layer
compute because dynamic routing overhead can dominate at larger batch sizes.
Held-out audits also retain the validation and final threshold curves so a run
can be reinterpreted at different quality/cost operating points. The runner
records speed-biased, tighter-drop, teacher-floor, and teacher-plus selector
views, plus per-policy acceptance precision and acceptance share by correction
budget. Current high-validation audits also report selective-agreement check
rate and an effective-core proxy that charges for the adjacent checks actually
run.

Mode names encode this flow. A mode ending in `_reuse` or `_reuse_highval`
loads an existing checkpoint and performs evaluation only. A mode with
`quality`, `slo`, `cats`, `agdistill`, or `stability` selects the corresponding
controller objective described above. Full Modal runs use real SmolLM2
checkpoints from the `/results` volume; dry modes use tiny fake-model schedules
to catch configuration and training-loop wiring errors quickly.

## Current Experiment Setup

The strongest current benchmark family is a synthetic natural-language
transition task. Prompts describe graph/permutation transitions over small node
sets, and the model must answer after one or more hops. Variants include
forward, inverse, alternate, and square transitions.

The main hard-case configuration uses:

- SmolLM2-135M as the base model.
- 8 graph nodes.
- Up to 4 hops.
- 2 preserve steps, giving 6 recurrent loop steps.
- A 3-layer recurrent core.
- JumpRec correction budgets `0..3`.
- Full-loop fallback as the safety policy.

This is still a synthetic benchmark. It is useful because recurrence genuinely
matters, controls are easy to define, and shortcut/fallback behavior can be
measured precisely. It's not really evidence for open-ended chat or broad
reasoning performance yet.

The next bridge family is `core3_8n4h_natgraph_*`. It keeps the graph traversal
target controlled but changes the interface to natural-language route-card
prompts, so selective agreement can be tested outside the compact symbolic
prompt format before any broader LLM claim.

## Artifact Discipline

`JUMPREC_ARTIFACT_AUDIT.md` is a standing design constraint. Before promoting a
result, check the audit gates:

- no privileged state in the router;
- no threshold tuning on the reported final split;
- no headline without direct, early-exit, and full-loop controls;
- no speedup claim without batch-size timing;
- no general LLM claim from synthetic-only evidence.

`JUMPREC_RESULTS.md` is the source of record for run history and interpretation.
`JUMPREC_NEXT_STEPS.md` tracks the current research queue.
`experiments/CHECKPOINT_MANIFEST.md` records the active predecessor checkpoint
dependencies so reuse/audit modes are reproducible from a cleaner checkout.

## Local Checks

The runner is still a research script, but the repo now has a small standard
library test suite for active mode resolution and joint-halting guardrails:

```powershell
python -m py_compile .\run_recurrent_smol.py
python -m unittest discover -s tests
```

Dry modes use a fake model and short training schedules to catch wiring errors
before launching Modal jobs. The current promoted-path guard is:

```powershell
python .\run_recurrent_smol.py --mode dry_strathop_polish2_joint_halt_stability_reuse --local
```

Additional dry modes exist for quality, SLO, CATS, and agreement-distillation
branches, but those are historical or diagnostic unless a new experiment
explicitly revives them.

## Scaling Constraint

JumpRec, if it works, should be a model-size-independent interface.
The recurrent core, jump module, verifier, halting policy, and fallback path
need to preserve the same conceptual contract from small models through larger
local models.

The execution strategy differs by scale:

- `135M`: serial subset routing is acceptable for fast iteration and exposes
  the latency/quality tradeoff.
- `2B` to `9B`: routing overhead must be low, checkpoint reuse must be
  standard, and the jump path should avoid many tiny dynamic batches.
- very large dense or MoE models: the router must skip expensive blocks or
  experts cleanly, avoid extra cross-device communication, and preferably use a
  fused or static-enough execution path.

This is why the agreement path is treated as a reference rather than the final
architecture. It often gives the best quality frontier on the synthetic task,
but it does so by evaluating neighboring candidates. Selective agreement is the
current compromise: it keeps that reliability signal for ambiguous cases while
skipping it for obvious accepts. For local and interactive use this is enough to
justify general looped-LLM application tests. For high-throughput batching, the
same policy still needs a better execution plan because many small dynamic
subsets can erase the counted-compute saving.

## Files

- `run_recurrent_smol.py`: primary recurrent SmolLM2 and JumpRec runner.
- `run_jumprec_v0.py`: older pure synthetic JumpRec runner.
- `run_jumprec_smol.py`: earlier SmolLM2 crash-test runner.
- `JUMPREC_SPEC.md`: architecture sketch and experimental framing.
- `JUMPREC_CURRENT_STATE.md`: short current-state handoff and active claims.
- `JUMPREC_RESULTS.md`: run history, tables, and interpretation.
- `JUMPREC_NEXT_STEPS.md`: current research queue.
- `JUMPREC_ARTIFACT_AUDIT.md`: artifact and claim-safety checklist.
- `experiments/`: launch helpers and orchestration notes.
- `experiments/CHECKPOINT_MANIFEST.md`: active checkpoint prerequisites.
- `jumprec/`: staging area for a future general-use package.
- `tests/`: lightweight configuration and guardrail tests.
- `requirements.txt`: Python package dependencies.
