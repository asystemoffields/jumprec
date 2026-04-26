# JumpRec results

## 2026-04-25 - v0 smoke probe

Command:

```text
modal run run_jumprec_v0.py --mode smoke
```

Note: this initial probe used the original H200 setting. Future runs are
configured for H100 in `run_jumprec_v0.py`.

Config:

- Pointer task: 8 nodes, 1-4 hops
- Full loop depth: 6 blocks
- JumpRec path: 1 jump block plus 0-2 corrective loop blocks
- Teacher train steps: 900
- Jump train steps: 900
- Temporary window adapter: enabled

Result:

- Teacher one-loop accuracy: 41.1%
- Teacher full-loop accuracy: 100.0%
- Jump only accuracy: 50.7%
- Jump + 1 correction accuracy: 65.5%
- Jump + 2 corrections accuracy: 86.5%
- Adaptive accuracy at threshold 0.80: 86.1%
- Adaptive average corrective loops: 1.00
- Adaptive loop-equivalent blocks: 2.00 vs 6.00 full-loop blocks
- Compute savings vs full loop: 66.6%

By hop for adaptive threshold 0.80:

- 1 hop: 99.4% accuracy, 0.00 corrective loops
- 2 hop: 99.6% accuracy, 0.73 corrective loops
- 3 hop: 97.5% accuracy, 1.61 corrective loops
- 4 hop: 47.8% accuracy, 1.69 corrective loops

Interpretation:

This is a real signal but not a pass yet. The teacher learned the iterative
task perfectly, and the adaptive path spent more compute as hop count rose.
However, the jump path fails badly on max-depth examples. Next tests should
either train the jump longer, allow a third correction loop, or train separate
jump heads for different correction budgets.

## 2026-04-25 - v0 probe: multi-landing jumps on H100

Command:

```text
modal run run_jumprec_v0.py --mode probe
```

Config:

- GPU: H100 80GB
- Pointer task: 8 nodes, 1-4 hops
- Full loop depth: 6 blocks
- JumpRec path: 1 learned jump block, then 0-4 real tail loop blocks
- Teacher train steps: 1400
- Jump train steps: 5000
- Temporary window adapter: enabled

Change from smoke probe:

Instead of one final-state jump plus cleanup loops, this test trained a separate
landing condition for each correction budget. For `c` remaining loops, the jump
module learns to land on the teacher trajectory at `full_depth - c`, then the
frozen trained loop runs the final `c` steps. This is a fairer "go through the
loop" test because corrections are ordinary trained tail loops.

Teacher:

- 1 block: 40.3%
- 2 blocks: 59.3%
- 3 blocks: 78.2%
- 4 blocks: 100.0%
- 5 blocks: 100.0%
- 6 blocks: 100.0%

JumpRec fixed budgets:

- Jump + 0 tail loops: 85.7%
- Jump + 1 tail loop: 98.2%
- Jump + 2 tail loops: 99.98%
- Jump + 3 tail loops: 100.0%
- Jump + 4 tail loops: 99.99%

Adaptive verifier:

- Threshold 0.70: 98.2% accuracy, 1.22 block-equivalents, 79.7% savings vs full loop
- Threshold 0.80: 98.8% accuracy, 1.25 block-equivalents, 79.2% savings vs full loop
- Threshold 0.90: 99.4% accuracy, 1.28 block-equivalents, 78.6% savings vs full loop
- Threshold 0.95: 99.7% accuracy, 1.32 block-equivalents, 78.0% savings vs full loop

By hop at threshold 0.80:

- 1 hop: 100.0% accuracy, 0.00 tail loops
- 2 hop: 99.9% accuracy, 0.00 tail loops
- 3 hop: 97.8% accuracy, 0.15 tail loops
- 4 hop: 97.6% accuracy, 0.84 tail loops

Interpretation:

This is the first strong signal. The ruthless comparison is the fixed-loop
teacher: at roughly the same compute as adaptive JumpRec, the teacher's 1-block
baseline is 40.3%, and even 2 real loop blocks are only 59.3%. Adaptive JumpRec
gets 98.8-99.7% with about 1.25-1.32 block-equivalents. That suggests the jump
is doing real amortized recurrent work, not merely buying accuracy with a few
ordinary loop steps.

The unfair part in our favor is that this is still a tiny synthetic task and
JumpRec has more trainable parameters than the teacher loop block. The next
honest test should increase the task size or add an equal-parameter non-jump
baseline.

## 2026-04-25 - v0 quick scale: 12 nodes / 6 hops on H100

Command:

```text
modal run run_jumprec_v0.py --mode quick
```

Config:

- GPU: H100 80GB
- Pointer task: 12 nodes, 1-6 hops
- Full loop depth: 8 blocks
- JumpRec path: 1 learned jump block, then 0-4 real tail loop blocks
- Teacher train steps: 3500
- Jump train steps: 3500
- Temporary window adapter: enabled

Teacher:

- 1 block: 30.8%
- 2 blocks: 44.9%
- 3 blocks: 57.0%
- 4 blocks: 71.0%
- 5 blocks: 84.5%
- 6 blocks: 100.0%
- 7 blocks: 100.0%
- 8 blocks: 100.0%

JumpRec fixed budgets:

- Jump + 0 tail loops: 37.1%
- Jump + 1 tail loop: 62.3%
- Jump + 2 tail loops: 75.0%
- Jump + 3 tail loops: 88.1%
- Jump + 4 tail loops: 98.4%

Adaptive verifier:

- Threshold 0.70: 97.9% accuracy, 2.65 block-equivalents, 66.8% savings vs full loop
- Threshold 0.80: 98.0% accuracy, 2.67 block-equivalents, 66.7% savings vs full loop
- Threshold 0.90: 98.2% accuracy, 2.68 block-equivalents, 66.4% savings vs full loop
- Threshold 0.95: 98.3% accuracy, 2.70 block-equivalents, 66.2% savings vs full loop

By hop at threshold 0.80:

- 1 hop: 100.0% accuracy, 0.00 tail loops
- 2 hop: 100.0% accuracy, 0.84 tail loops
- 3 hop: 99.6% accuracy, 1.00 tail loops
- 4 hop: 99.1% accuracy, 1.87 tail loops
- 5 hop: 99.1% accuracy, 2.81 tail loops
- 6 hop: 90.7% accuracy, 3.43 tail loops

Interpretation:

The effect survives the first scale-up. Adaptive JumpRec still beats ordinary
looping at similar compute by a lot: around 98.0% at 2.67 block-equivalents,
while the teacher with 3 real loop blocks is 57.0% and 4 real loop blocks is
71.0%. The cost of scaling is visible: harder 6-hop cases are only 90.7% even
when the verifier spends most of the allowed 4 tail loops. This suggests the
method is not collapsing, but max-depth cases need either a larger correction
budget, longer jump training, or a better hard-case router.

## 2026-04-25 - parallel ablation suite on H100

Commands:

```text
modal run run_jumprec_v0.py --mode quick_no_adapter
modal run run_jumprec_v0.py --mode quick_c6
modal run run_jumprec_v0.py --mode quick_c6_no_adapter
modal run run_jumprec_v0.py --mode quick_ood_hops
```

All use the 12-node / 1-6 hop pointer task unless noted.

### Summary at adaptive threshold 0.80

| Mode | Adapter | Max Tail | Train Hops | Accuracy | Block-Equiv | Savings | 6-Hop Acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| quick | yes | 4 | 1-6 | 98.0% | 2.67 | 66.7% | 90.7% |
| quick_no_adapter | no | 4 | 1-6 | 88.0% | 2.67 | 66.6% | 33.2% |
| quick_c6 | yes | 6 | 1-6 | 99.3% | 2.98 | 62.8% | 99.8% |
| quick_c6_no_adapter | no | 6 | 1-6 | 99.5% | 3.26 | 59.2% | 99.8% |
| quick_ood_hops | yes | 6 | 1-4 | 72.8% | 1.86 | 76.8% | 19.0% |

### Interpretation

The temporary window adapter is important for efficient jumping. With the same
4-tail budget, removing it drops overall accuracy from 98.0% to 88.0%, and
6-hop accuracy from 90.7% to 33.2%, at basically identical average compute.

Increasing the tail budget from 4 to 6 fixes the hard edge. With adapter enabled,
accuracy rises to 99.3% while average compute rises from 2.67 to 2.98
block-equivalents. That is a clean trade: +1.3 accuracy points, hard 6-hop cases
almost solved, and still ~63% cheaper than full looping.

No-adapter with max tail 6 catches up in accuracy, but only by spending more
tail loops: 3.26 block-equivalents vs 2.98 with adapter. This suggests the
adapter is not strictly necessary for eventual correctness when the router can
fall back hard enough, but it makes the jump land closer and saves compute.

The OOD hop test is a hard fail for extrapolation. Training on 1-4 hops and
evaluating on 1-6 hops leaves both teacher and JumpRec poor on unseen 5-6 hop
chains. JumpRec does not discover longer recursion by itself; it amortizes the
recurrence distribution it has seen.

### Current Best Setting

For this task, `quick_c6` is the best operating point:

- 99.3% accuracy at threshold 0.80
- 2.98 block-equivalents vs 8 full-loop blocks
- 62.8% compute savings
- 99.8% 6-hop accuracy

The next fair test should add an equal-parameter non-jump baseline and a
hard-case router that can explicitly fall back to the full loop when verifier
confidence remains low.

## 2026-04-25 - fairness controls on H100

Commands:

```text
modal run run_jumprec_v0.py --mode quick_c6_no_hidden
modal run run_jumprec_v0.py --mode quick_direct3
```

Both use the 12-node / 1-6 hop pointer task.

### Results

| Mode | What It Tests | Trainable Params | Accuracy | Block-Equiv | Savings | 6-Hop Acc |
|---|---|---:|---:|---:|---:|---:|
| quick_c6 | JumpRec with hidden trajectory loss | 0.599M | 99.3% | 2.98 | 62.8% | 99.8% |
| quick_c6_no_hidden | JumpRec without hidden trajectory loss | 0.599M | 99.6% | 2.31 | 71.2% | 99.2% |
| quick_direct3 | 3 trainable direct blocks, no jump/tail/verifier | 0.593M | 88.6% | 3.00 | 62.5% | 38.4% |

### Interpretation

The equal-parameter direct baseline is the key fairness result. It has nearly
the same trainable parameter count as JumpRec and roughly the same compute as
adaptive JumpRec, but reaches only 88.6% overall and collapses to 38.4% on
6-hop cases. So the current evidence says the win is not merely "extra params."
The structured jump-plus-tail mechanism matters.

The hidden trajectory MSE is not helping here. Removing it improves the adaptive
operating point: 99.6% accuracy at 2.31 block-equivalents, saving 71.2% compute.
This suggests that forcing the jump state to match the teacher's exact hidden
trajectory may over-constrain it. The better objective is likely: land anywhere
from which the frozen loop tail can finish correctly.

Current best setting is now `quick_c6_no_hidden`:

- 99.6% accuracy
- 2.31 block-equivalents vs 8 full-loop blocks
- 71.2% compute savings
- 99.2% 6-hop accuracy

Next tests should confirm this with seeds, then add a full-loop fallback for
the small fraction of cases where all verifier thresholds remain uncertain.

## 2026-04-25 - seed confirmation and full-loop fallback

Commands:

```text
modal run run_jumprec_v0.py --mode quick_c6_no_hidden --seed 101
modal run run_jumprec_v0.py --mode quick_c6_no_hidden --seed 202
modal run run_jumprec_v0.py --mode quick_c6_no_hidden --seed 303
```

Baseline seed 42 is the prior `quick_c6_no_hidden` run. All results below use
threshold 0.80 on the 12-node / 1-6 hop pointer task.

### Results

| Seed | Accuracy | Block-Equiv | Savings | 6-Hop Acc | Full-Loop Fallback Rate |
|---:|---:|---:|---:|---:|---:|
| 42 | 99.64% | 2.31 | 71.15% | 99.21% | 0.00% |
| 101 | 99.69% | 2.30 | 71.31% | 99.49% | 0.00% |
| 202 | 99.85% | 2.31 | 71.07% | 99.82% | 0.00% |
| 303 | 99.50% | 2.15 | 73.11% | 99.43% | 0.00% |
| Mean | 99.67% | 2.27 | 71.66% | 99.49% | 0.00% |
| Stdev | 0.14% | 0.08 | 0.97% | 0.25% | 0.00% |

### Interpretation

The best result is seed-stable. The current strongest claim is now:
JumpRec without hidden trajectory loss reaches 99.67% +/- 0.14% accuracy while
using 2.27 +/- 0.08 block-equivalents instead of 8, saving 71.66% +/- 0.97%
compute on this synthetic recurrence task.

The full-loop fallback path is implemented in the evaluator: if no verifier
threshold accepts any candidate budget, the sample is charged as a full teacher
loop and receives the full-loop prediction. In these seed runs the fallback
rate is 0.00% at thresholds 0.70, 0.80, 0.90, and 0.95, so fallback is harmless
but not yet exercised. That means the current verifier is permissive on this
in-distribution task. The next fallback test needs either stricter calibration,
an OOD split, or a reject rule based on confidence margin/disagreement rather
than only the raw verifier threshold.

## 2026-04-25 - mixed recurrence suite and strict fallback

Code changes:

- Added `dry_mixed`, `quick_mix`, and `quick_mix_strict` modes.
- Added a 4-task recurrence mix over the same permutation table:
  `forward`, `inverse`, `alternate`, and `square`.
- Added strict accept rules for fallback evaluation:
  verifier threshold, answer margin, max answer probability, and agreement with
  the next-larger correction budget.

Commands:

```text
python run_jumprec_v0.py --local --mode dry_mixed
modal run run_jumprec_v0.py --mode quick_mix
modal run run_jumprec_v0.py --mode quick_mix_strict
```

Both H100 runs use 12 nodes, 1-6 hops, 8 full-loop blocks, 6 correction budgets,
and no hidden trajectory loss.

### Results at Threshold 0.80

| Mode | Accept Policy | Teacher Full | Jump/Fallback Acc | Block-Equiv | Savings | Full-Loop Fallback |
|---|---|---:|---:|---:|---:|---:|
| quick_mix | verifier only | 97.87% | 99.18% | 2.18 | 72.71% | 0.03% |
| quick_mix_strict | verifier + margin + confidence + budget agreement | 99.97% | 99.87% | 2.55 | 68.11% | 5.38% |

`quick_mix_strict` by task at threshold 0.80:

| Task | Fallback Acc | Block-Equiv | Full-Loop Fallback |
|---|---:|---:|---:|
| forward | 100.00% | 2.07 | 0.00% |
| inverse | 99.71% | 3.54 | 10.87% |
| alternate | 100.00% | 1.00 | 0.00% |
| square | 99.77% | 3.58 | 10.54% |

### Interpretation

This is the first broader result that is not just single-family pointer chasing.
The strict fallback policy is doing real work: it sends about 5.4% of examples
to the full teacher loop, mostly the inverse and square tasks, and still saves
about 68% block-equivalent compute while keeping accuracy near the full teacher.

The non-strict mixed run is useful but not a clean policy-only ablation because
the teacher did not train to the same quality despite the same nominal seed;
CUDA attention nondeterminism is enough to make these parallel runs diverge.
The safer headline is therefore the strict run by itself: mixed recurrence,
99.87% fallback accuracy, 2.55/8 block-equivalent compute, and an actually
exercised full-loop fallback.

The next best test should add a same-run policy sweep so strict and non-strict
acceptance are evaluated against the exact same trained weights, then repeat
`quick_mix_strict` across seeds.

## 2026-04-25 - mixed strict seed confirmation

Commands:

```text
modal run run_jumprec_v0.py --mode quick_mix_strict --seed 101
modal run run_jumprec_v0.py --mode quick_mix_strict --seed 202
modal run run_jumprec_v0.py --mode quick_mix_strict --seed 303
```

Baseline seed 42 is the prior `quick_mix_strict` run. Results below use
threshold 0.80 and the full-loop fallback policy.

### Results

| Seed | Teacher Full | Fallback Acc | Block-Equiv | Savings | Full-Loop Fallback |
|---:|---:|---:|---:|---:|---:|
| 42 | 99.97% | 99.87% | 2.55 | 68.11% | 5.38% |
| 101 | 99.97% | 99.93% | 2.63 | 67.10% | 5.61% |
| 202 | 96.68% | 97.30% | 2.19 | 72.67% | 4.06% |
| 303 | 99.63% | 99.87% | 2.28 | 71.47% | 2.19% |
| Mean | 99.06% | 99.24% | 2.41 | 69.84% | 4.31% |
| Stdev | 1.60% | 1.30% | 0.21 | 2.65% | 1.57% |

Teacher-solved seeds only (`42`, `101`, `303`):

| Mean Fallback Acc | Mean Block-Equiv | Mean Savings | Mean Full-Loop Fallback |
|---:|---:|---:|---:|
| 99.89% +/- 0.03% | 2.49 +/- 0.18 | 68.89% +/- 2.29% | 4.39% +/- 1.91% |

### Interpretation

The mixed strict result mostly holds up, but seed 202 is important. The teacher
only reached 96.68% full-loop accuracy because the `square` task stalled at
86.76%, and JumpRec inherited that weakness. This is not evidence against the
jump mechanism; it is evidence that the mixed teacher training recipe is still
somewhat brittle.

When the teacher solves the mixed recurrence family, JumpRec strict fallback
stays near 99.9% accuracy while saving roughly 69% block-equivalent compute.
When the teacher does not solve a subtask, fallback cannot magically repair the
teacher. The next implementation change should add a teacher-quality gate:
continue training, lower LR, or mark the run invalid when full-loop accuracy is
below a target such as 99.5%.

## 2026-04-26 - pre-LLM decision round

Code changes:

- Added `quick_mix_round`: same trained weights, same-run policy sweep,
  teacher-quality gate, mixed direct baseline, and H100 timing.
- Teacher gate requires full-loop accuracy >= 99.5% and each mixed task >= 98%.
  If the gate fails, the teacher receives 4,000 extra lower-LR steps.
- Policy sweep evaluates verifier-only, margin+confidence, and strict
  margin+confidence+next-budget-agreement policies on the exact same weights.

Commands:

```text
modal run run_jumprec_v0.py --mode quick_mix_round --seed 42
modal run run_jumprec_v0.py --mode quick_mix_round --seed 202
```

Seed 202 is intentionally included because it previously exposed a square-task
teacher failure.

### Results at Threshold 0.80

| Seed | Gate Extra Steps | Teacher Full | Strict Fallback Acc | Block-Equiv | Savings | Full-Loop Fallback | Direct Acc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0 | 99.90% | 99.93% | 2.43 | 69.57% | 2.83% | 94.32% |
| 202 | 4,000 | 99.91% | 99.89% | 2.27 | 71.66% | 2.10% | 91.02% |
| Mean | 2,000 | 99.90% | 99.91% | 2.35 | 70.61% | 2.46% | 92.67% |

### Policy Sweep at Threshold 0.80

| Seed | Verifier-Only Fallback Acc | Margin+Confidence Fallback Acc | Strict Fallback Acc |
|---:|---:|---:|---:|
| 42 | 99.53% | 99.53% | 99.93% |
| 202 | 99.41% | 99.44% | 99.89% |

Strict routing costs slightly more fallback, but clearly improves reliability on
the exact same trained weights. This is the first clean policy comparison.

### Timing

Measured on H100, batch size 384, 32 timing batches.

| Metric | Seed 42 | Seed 202 | Mean |
|---|---:|---:|---:|
| Teacher full loop | 4.00 ms | 3.90 ms | 3.95 ms |
| JumpRec c0 only | 2.84 ms | 2.45 ms | 2.64 ms |
| JumpRec c6 only | 3.84 ms | 3.68 ms | 3.76 ms |
| All budgets evaluated | 11.19 ms | 10.99 ms | 11.09 ms |
| Direct baseline | 2.95 ms | 2.53 ms | 2.74 ms |

### Interpretation

This round clears the bar for moving to a small pretrained LM crash test. The
teacher-quality gate fixed the known seed-202 failure, strict policy routing
beat verifier-only routing in a same-run comparison, and the direct baseline
remained far below JumpRec despite comparable trainable parameter count.

The timing result is a warning label: evaluating all budgets is slower than the
full teacher loop. The compute-savings story only turns into a wall-clock story
if inference uses early-exit/serial routing, not parallel evaluation of every
budget. The correct implementation target for an LM is therefore: try a cheap
jump, verify, then run only the needed tail or full fallback.

Next target: SmolLM2-135M as the first pretrained local-LM crash test dummy.

## 2026-04-26 - SmolLM2-135M first crash test

Code changes:

- Added `run_jumprec_smol.py`.
- Uses frozen `HuggingFaceTB/SmolLM2-135M` as a text encoder.
- Trains a looped refinement teacher over frozen LM hidden states.
- Trains JumpRec over the teacher states with verifier-controlled fallback.
- Keeps a direct 3-block baseline over the same frozen encoder states.

Commands:

```text
python run_jumprec_smol.py --local --mode dry
modal run run_jumprec_smol.py --mode smol_pointer
modal run run_jumprec_smol.py --mode smol_pointer_easy
```

The first real mode used 8 nodes / 4 hops. The easier diagnostic used
6 nodes / 3 hops and more teacher training.

### Results

| Mode | Teacher Full | JumpRec Best | Strict Fallback Acc | Direct Acc | Notes |
|---|---:|---:|---:|---:|---|
| smol_pointer | 14.91% | 24.61% | 16.93% | 23.89% | Teacher failed completely |
| smol_pointer_easy | 44.27% | 58.33% | 54.17% | 54.07% | Partial learning, still no competent teacher |

Timing on H100:

| Mode | Teacher Full | All JumpRec Budgets |
|---|---:|---:|
| smol_pointer | 21.98 ms/batch | 28.30 ms/batch |
| smol_pointer_easy | 22.92 ms/batch | 27.68 ms/batch |

### Interpretation

This is a real negative result for the first naive LM wrapper, not a negative
result for JumpRec itself. The blocker is upstream: the looped teacher over
frozen SmolLM2 hidden states did not solve the textual pointer task. Without a
competent full-loop teacher, JumpRec has no valid trajectory to amortize.

The easier task is informative. JumpRec and the direct baseline can extract some
signal from frozen SmolLM2 states, but the recurrent teacher objective is not yet
forming a reliable algorithmic state machine. The current wrapper asks too much
of a randomly initialized recurrent block on top of frozen natural-language
features.

Next SmolLM2 attempt should change the interface before scaling runs:

- Add trainable input/task adapters before the recurrent teacher, not only after
  the frozen LM.
- Train the teacher with a final-answer objective first, then add per-step
  recurrence supervision once final accuracy is high.
- Consider using SmolLM2 intermediate layer states or a small LoRA on the last
  few LM layers, while still keeping most of the LM frozen.
- Keep the synthetic JumpRec suite as the regression target; do not evaluate
  JumpRec claims on SmolLM2 until the full-loop teacher is strong.

## 2026-04-26 - SmolLM2 adapter/curriculum repair attempt

Code changes:

- Added `smol_pointer_adapter`.
- Adds two trainable contextual input-adapter transformer blocks before the
  looped teacher.
- Trains the teacher with a final-answer-only warmup first, then switches to
  recurrence supervision.

Command:

```text
modal run run_jumprec_smol.py --mode smol_pointer_adapter
```

Configuration:

- Frozen `HuggingFaceTB/SmolLM2-135M`
- 6 nodes / 3 hops
- 5 full-loop blocks
- 5,000 final-answer teacher warmup steps
- 3,000 recurrent teacher steps
- 3,000 JumpRec steps
- 3,000 direct baseline steps

### Results

| Mode | Teacher Full | JumpRec Best | Strict Fallback Acc | Direct Acc | Runtime |
|---|---:|---:|---:|---:|---:|
| smol_pointer_easy | 44.27% | 58.33% | 54.17% | 54.07% | ~5 min |
| smol_pointer_adapter | 31.84% | 35.89% | 31.79% | 36.21% | ~12 min |

Timing on H100:

| Metric | smol_pointer_adapter |
|---|---:|
| Teacher full loop | 32.78 ms/batch |
| All JumpRec budgets | 37.09 ms/batch |

### Interpretation

The adapter/curriculum repair did not help. It made the model larger and slower,
but the full-loop teacher got worse than the simpler `smol_pointer_easy` run.
That suggests the issue is not just "needs a little trainable adapter." Frozen
SmolLM2 final hidden states are not currently presenting the synthetic pointer
state in a form our small recurrent teacher can reliably use.

The right next move is to change the representation interface more radically:

- Use a structured/token-level probe task where the answer is read from a
  dedicated answer/query token instead of the final natural-language token.
- Compare frozen SmolLM2 states against learned token embeddings on the exact
  same textualized task, so we know whether the LM is helping or hurting.
- Try intermediate SmolLM2 layers instead of only final hidden states.
- Only then consider a tiny LoRA on the last few LM layers.

This result reinforces the boundary: JumpRec is promising on a clean recurrent
state space, but porting it to pretrained LM hidden states requires first
building a competent looped teacher interface.

## 2026-04-26 - SmolLM2 workspace sidecar attempt

Code changes:

- Added `smol_workspace`.
- Prepends trainable latent workspace tokens to frozen SmolLM2 hidden states.
- Runs trainable input-adapter blocks over `[workspace, text]`.
- Keeps only the workspace tokens for the recurrent teacher, JumpRec, and
  direct baseline.

Commands:

```text
python run_jumprec_smol.py --local --mode dry
modal run run_jumprec_smol.py --mode smol_workspace
```

Configuration:

- Frozen `HuggingFaceTB/SmolLM2-135M`
- 6 nodes / 3 hops
- 8 workspace tokens
- 5 full-loop blocks
- 5,000 final-answer teacher warmup steps
- 3,000 recurrent teacher steps
- 3,000 JumpRec steps
- 3,000 direct baseline steps

### Results

| Mode | Teacher Full | JumpRec Best | Strict Fallback Acc | Full-Loop Fallback | Direct Acc |
|---|---:|---:|---:|---:|---:|
| smol_pointer_adapter | 31.84% | 35.89% | 31.79% | n/a | 36.21% |
| smol_workspace | 21.66% | 30.62% | 30.13% | 94.36% | 33.40% |

Timing on H100:

| Metric | smol_workspace |
|---|---:|
| Teacher full loop | 29.56 ms/batch |
| All JumpRec budgets | 34.39 ms/batch |

### Interpretation

This is a second negative result for the frozen-SmollLM2-wrapper family. The
latent workspace did not become a competent recurrent scratchpad. In fact, the
full-loop teacher was worse than the prior adapter/curriculum run, and the
direct baseline beat both the teacher and strict JumpRec fallback.

The result is especially useful in light of current looped-transformer recipes:
successful recurrent-depth language models usually place recurrence inside the
model path itself, commonly using a prelude/recurrent/coda split, recurrent
input injection, loop or time conditioning, recurrence curricula, and sometimes
adaptive exit objectives. Our sidecar compressed frozen LM outputs once, then
looped only the workspace. That is still too disconnected from how looped LMs
are normally trained.

The next implementation direction should move away from frozen sidecar
scratchpads and toward an actual recurrent-depth LM retrofit:

- Split a pretrained small LM into prelude, recurrent core, and coda.
- Reuse the core for multiple recurrent steps with explicit input reinjection.
- Train with a recurrence curriculum rather than jumping immediately to a fixed
  loop depth.
- Add JumpRec only after the recurrent full-depth model is competent.

The synthetic JumpRec results remain promising; the SmolLM2 results say the LM
interface is still wrong.

## 2026-04-26 - recurrent-depth SmolLM2 retrofit

Code changes:

- Added `run_recurrent_smol.py`.
- Splits SmolLM2-135M into frozen/trainable prelude, shared recurrent core, and
  coda blocks.
- Adds loop-step embeddings and explicit input-state reinjection.
- Trains with final-answer warmup followed by per-loop recurrence supervision.
- Adds optional JumpRec over the recurrent SmolLM2 state space.

Commands:

```text
python run_recurrent_smol.py --local --mode dry
modal run run_recurrent_smol.py --mode retrofit_probe --seed 42
modal run run_recurrent_smol.py --mode retrofit_probe --seed 101
modal run run_recurrent_smol.py --mode retrofit_probe --seed 202
modal run run_recurrent_smol.py --mode retrofit_probe --seed 303
modal run run_recurrent_smol.py --mode retrofit_8n4h
modal run run_recurrent_smol.py --mode retrofit_8n4h_unfreeze
```

### 6 nodes / 3 hops: recurrent teacher

Configuration:

- Frozen SmolLM2 embeddings and most blocks.
- Prelude: first 4 SmolLM2 decoder blocks, frozen.
- Recurrent core: blocks 4-5, trainable and shared across loops.
- Coda: blocks 28-29, trainable.
- 5 recurrent loops: 3 task hops plus 2 preserve steps.
- Trainable parameters: 14.17M.

Results:

| Seed | 0 Loops | 1 Loop | 2 Loops | Full 5 Loops | Gain vs 1 Loop |
|---:|---:|---:|---:|---:|---:|
| 42 | 19.70% | 50.54% | 72.14% | 99.34% | 48.80% |
| 101 | 20.58% | 50.61% | 72.71% | 99.76% | 49.15% |
| 202 | 19.75% | 49.93% | 72.56% | 99.68% | 49.76% |
| 303 | 19.19% | 50.10% | 72.00% | 99.63% | 49.54% |
| Mean | 19.81% | 50.29% | 72.35% | 99.60% | 49.31% |
| Stdev | 0.58% | 0.33% | 0.34% | 0.18% | 0.43% |

Interpretation:

This is the first positive pretrained-LM result. The task was deliberately
chosen so recurrent depth should matter: each loop can execute one transition
through the map. The learned model reflects that structure. One-loop and
two-loop accuracy remain far below full-loop accuracy, while three or more
loops solve nearly all examples.

This validates the architectural pivot. Recurrence had to be inside the model
path; external loops over frozen final hidden states and sidecar workspaces did
not work.

### 8 nodes / 4 hops: first scale-up

Command:

```text
modal run run_recurrent_smol.py --mode retrofit_8n4h
```

Result:

| Metric | Value |
|---|---:|
| 0-loop accuracy | 13.53% |
| 1-loop accuracy | 39.77% |
| 2-loop accuracy | 58.86% |
| 3-loop accuracy | 74.61% |
| 4-loop accuracy | 84.16% |
| Full 6-loop accuracy | 84.47% |

By hop:

| Hop | Full-Loop Accuracy |
|---:|---:|
| 1 | 100.00% |
| 2 | 99.88% |
| 3 | 85.18% |
| 4 | 54.01% |

Interpretation:

The scale-up preserves the recurrent-depth curve but does not solve the harder
task. The core weakness is max-depth 4-hop cases. This is the current boundary:
the small recurrent core learns the iterative algorithm on 6/3, partly scales
to 8/4, but needs either a stronger training recipe, more recurrence-aware
regularization, or a better benchmark curriculum before moving to 12/6.

An attempted `retrofit_8n4h_unfreeze` run with 35.41M trainable parameters was
stopped early after it underperformed the smaller-core run at comparable
progress. Simply unfreezing more SmolLM2 layers is not an obvious fix.

## 2026-04-26 - JumpRec on recurrent SmolLM2

Commands:

```text
modal run run_recurrent_smol.py --mode jumprec_probe --seed 42
modal run run_recurrent_smol.py --mode jumprec_probe --seed 101
modal run run_recurrent_smol.py --mode jumprec_probe --seed 202
```

Configuration:

- Same 6-node / 3-hop recurrent SmolLM2 teacher as `retrofit_probe`.
- Teacher full recurrent core cost: 5 loops x 2 core layers = 10 core layers.
- JumpRec: one trainable copied decoder layer, landing embeddings, temporary
  adapter, verifiers, and 0-3 frozen teacher tail loops.
- JumpRec trainable parameters: 6.99M.

Fixed-budget results:

| Seed | Jump + 0 Tail | Jump + 1 Tail | Jump + 2 Tail | Jump + 3 Tail |
|---:|---:|---:|---:|---:|
| 42 | 37.01% | 58.72% | 91.77% | 97.61% |
| 101 | 40.70% | 59.57% | 95.29% | 98.83% |
| 202 | 42.70% | 59.20% | 93.99% | 98.90% |

Strict fallback at threshold 0.80:

| Seed | Accuracy | Avg Core Layers | Savings vs Full | Full-Loop Fallback |
|---:|---:|---:|---:|---:|
| 42 | 98.80% | 4.92 / 10 | 50.77% | 12.77% |
| 101 | 99.24% | 4.51 / 10 | 54.91% | 10.35% |
| 202 | 99.29% | 4.40 / 10 | 55.98% | 9.08% |
| Mean | 99.11% | 4.61 / 10 | 53.88% | 10.73% |
| Stdev | 0.27% | 0.28 | 2.75% | 1.87% |

Stricter threshold 0.95:

| Mean Accuracy | Mean Core Layers | Mean Savings |
|---:|---:|---:|
| 99.50% +/- 0.23% | 5.41 / 10 | 45.86% +/- 2.98% |

Timing on H100, batch size 64:

| Seed | Full Teacher | All JumpRec Budgets |
|---:|---:|---:|
| 42 | 23.77 ms | 27.65 ms |
| 101 | 21.40 ms | 25.80 ms |
| 202 | 21.69 ms | 25.04 ms |

Interpretation:

This is the first positive result for the original LM-facing JumpRec idea. Once
SmolLM2 is made recurrent inside its own model path, JumpRec can recover nearly
the full teacher accuracy while using roughly half the recurrent core-layer
compute on this 6/3 textual pointer task.

The caveat is important: current timing evaluates all JumpRec budgets in
parallel, which is slower wall-clock than the full recurrent teacher. The
compute-savings result only becomes an inference-speed result with serial
early-exit routing: try the cheap jump, verify, and run only the necessary tail
or full fallback. That is the next engineering target.

## 2026-04-26 - recurrent SmolLM2 controls and ablations

This sweep uses the updated `run_recurrent_smol.py` path where copied JumpRec
and direct-control decoder blocks are explicitly trainable. That makes this
section the better reference for the current implementation.

Commands:

```text
modal run run_recurrent_smol.py --mode retrofit_probe
modal run run_recurrent_smol.py --mode jumprec_probe
modal run run_recurrent_smol.py --mode jumprec_probe --seed 101
modal run run_recurrent_smol.py --mode jumprec_probe --seed 202
modal run run_recurrent_smol.py --mode direct_probe
modal run run_recurrent_smol.py --mode direct_probe --seed 101
modal run run_recurrent_smol.py --mode direct_probe --seed 202
modal run run_recurrent_smol.py --mode jumprec_no_adapter
modal run run_recurrent_smol.py --mode jumprec_no_agree
modal run run_recurrent_smol.py --mode retrofit_no_reinject
modal run run_recurrent_smol.py --mode retrofit_core1
modal run run_recurrent_smol.py --mode retrofit_core3
modal run run_recurrent_smol.py --mode mixed_probe
modal run run_recurrent_smol.py --mode retrofit_8n4h_curriculum
```

### Trainable JumpRec vs direct control

All rows use the 6-node / 3-hop textual pointer task. Full recurrent teacher
cost is 10 core layers: 5 loops x 2 shared core layers.

| Seed | Teacher Full | Jump + 0 | Jump + 1 | Jump + 2 | Jump + 3 | Strict 0.80 | Avg Core Layers | Savings | Full Fallback | Direct 3-Layer |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 97.73% | 87.30% | 98.68% | 99.44% | 99.34% | 99.44% | 1.59 / 10 | 84.14% | 1.00% | 98.36% |
| 101 | 99.32% | 88.16% | 98.95% | 99.76% | 99.76% | 99.85% | 1.49 / 10 | 85.05% | 0.10% | 99.49% |
| 202 | 99.83% | 95.21% | 99.98% | 100.00% | 99.95% | 100.00% | 1.17 / 10 | 88.26% | 0.00% | 99.44% |
| Mean | 98.96% | 90.23% | 99.20% | 99.73% | 99.68% | 99.76% | 1.42 / 10 | 85.82% | 0.37% | 99.10% |
| Stdev | 1.09% | 4.34% | 0.68% | 0.28% | 0.31% | 0.29% | 0.22 | 2.16% | 0.55% | 0.63% |

Timing, H100 batch size 64:

| Seed | Full Teacher | All JumpRec Budgets | Serial JumpRec 0.80 |
|---:|---:|---:|---:|
| 42 | 21.06 ms | 24.89 ms | 21.57 ms |
| 101 | 21.70 ms | 26.07 ms | 17.58 ms |
| 202 | 21.01 ms | 24.49 ms | 13.01 ms |
| Mean | 21.25 ms | 25.15 ms | 17.39 ms |

The serial timing is a first implementation, not a final router. It omits the
agreement check used by strict accuracy evaluation and uses subset routing
inside a small batch, so the wall-clock numbers are noisy. Still, it is the
first run where the adaptive path is sometimes faster than the full recurrent
teacher rather than only cheaper in counted core layers.

Interpretation:

Trainable JumpRec is a much stronger operating point than the earlier
adapter-only path: strict fallback reaches 99.76% mean accuracy while using
only 1.42 of 10 recurrent core layers on average. That is about 85.8% recurrent
core-layer savings.

The direct baseline is the main warning label. A simple 3-copied-layer
non-recurrent control reaches 99.10% mean accuracy at 3 of 10 core layers. That
does not erase the JumpRec result, because JumpRec is slightly more accurate
and uses less than half the direct control's core layers. But it does mean the
6/3 task is now too easy to be the headline. Future claims need harder tasks
where the direct control cannot nearly solve the problem.

### Vanilla recurrent early exit

Across the 6/3 teacher runs, naive confidence-based early exit is badly
miscalibrated. Thresholds 0.80, 0.90, and 0.95 all exit after one loop on all
examples, giving only about 49-51% accuracy while reporting 80% core-layer
savings. This confirms that the verifier/fallback machinery is not optional:
raw softmax confidence is not a reliable router here.

### Ablations and harder probes

| Mode | Full Teacher | Strict 0.80 | Avg Core Layers | Savings | Key Result |
|---|---:|---:|---:|---:|---|
| `jumprec_probe` | 97.73% | 99.44% | 1.59 / 10 | 84.14% | Baseline trainable JumpRec. |
| `jumprec_no_adapter` | 97.73% | 99.68% | 1.55 / 10 | 84.52% | Temp adapter is not needed on this easy 6/3 task. |
| `jumprec_no_agree` | 97.73% | 99.22% | 1.55 / 10 | 84.52% | Agreement check slightly improves accuracy. |
| `retrofit_no_reinject` | 99.37% | n/a | n/a | n/a | Input reinjection is not required for easy 6/3. |
| `retrofit_core1` | 96.00% | n/a | n/a | n/a | One core layer is cheaper but loses accuracy. |
| `retrofit_core3` | 99.98% | n/a | n/a | n/a | Three core layers nearly solve 6/3, at higher full-loop cost. |
| `mixed_probe` | 86.08% | n/a | n/a | n/a | Four-task robustness is not solved. |
| `retrofit_8n4h_curriculum` | 73.51% | n/a | n/a | n/a | Simple hop curriculum made 8/4 worse. |

Mixed task breakdown:

| Task | Accuracy |
|---|---:|
| forward | 88.02% |
| inverse | 84.61% |
| alternate | 100.00% |
| square | 71.97% |

8-node / 4-hop curriculum breakdown:

| Hop | Accuracy |
|---:|---:|
| 1 | 100.00% |
| 2 | 99.55% |
| 3 | 56.67% |
| 4 | 34.93% |

Interpretation:

The strongest positive signal is now clear: JumpRec can learn an amortized
transition over a recurrent SmolLM2 state space, and a verifier can route most
examples through a very short path. The strongest negative signal is just as
clear: this has not yet become a robust local-LLM architecture. Mixed recurrence
families and 8/4 scaling both expose brittleness, and the direct control is
strong enough that easy pointer tasks are no longer sufficient evidence.

The next research move should make the benchmark harder and more architecture
honest before scaling claims: use mixed recurrence tasks as the default, keep
the direct 3-layer control in every table, try stronger recurrent cores on the
harder probes, and improve routing with agreement-aware serial inference.

## 2026-04-26 - harder mixed/core3 confirmation

Commands:

```text
modal run run_recurrent_smol.py --mode mixed_jumprec_direct
modal run run_recurrent_smol.py --mode mixed_core3_jumprec_direct
modal run run_recurrent_smol.py --mode jumprec_8n4h_direct
modal run run_recurrent_smol.py --mode core3_8n4h_jumprec_direct
modal run run_recurrent_smol.py --mode mixed_core3_jumprec_direct --seed 101
modal run run_recurrent_smol.py --mode mixed_core3_jumprec_direct --seed 202
```

This round tested the encouraging JumpRec result on harder LM-facing settings:
mixed transition families, explicit direct controls, a stronger 3-layer
recurrent core, and the 8-node / 4-hop scale-up. It also added per-hop and
per-task breakdowns for JumpRec and the direct control, plus two serial timing
paths: a fast router without agreement, and an agreement-aware router matching
strict evaluation more closely.

Headline results:

| Mode | Teacher Full | JumpRec Strict 0.80 | Avg Core Layers | Savings | Direct Control | Notes |
|---|---:|---:|---:|---:|---:|---|
| `mixed_jumprec_direct` | 86.08% | 87.94% | 2.99 / 10 | 70.13% | 87.43% | Core2 mixed is still weak; JumpRec only modestly helps. |
| `mixed_core3_jumprec_direct`, seeds 42/101/202 | 97.92% +/- 1.18% | 98.74% +/- 0.93% | 2.31 / 15 | 84.60% +/- 2.51% | 95.97% +/- 0.73% | Strongest LM-facing result so far. |
| `jumprec_8n4h_direct` | 76.37% | 77.98% | 4.65 / 12 | 61.24% | 83.13% | Weak teacher; direct beats JumpRec. |
| `core3_8n4h_jumprec_direct` | 84.42% | 85.03% | 5.45 / 18 | 69.72% | 80.64% | Better than direct, but 4-hop cases remain poor. |

Mixed/core3 seed breakdown:

| Seed | Teacher Full | JumpRec Strict 0.80 | Avg Core Layers | Savings | Full Fallback | Direct Control |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 98.71% | 99.56% | 2.08 / 15 | 86.16% | 0.44% | 96.73% |
| 101 | 98.49% | 98.93% | 2.11 / 15 | 85.94% | 1.78% | 95.92% |
| 202 | 96.56% | 97.73% | 2.74 / 15 | 81.71% | 2.44% | 95.26% |
| Mean | 97.92% | 98.74% | 2.31 / 15 | 84.60% | 1.55% | 95.97% |

Mixed/core3 by task, mean over seeds:

| Task | Teacher Full | JumpRec Strict 0.80 | Direct Control |
|---|---:|---:|---:|
| forward | 99.09% | 99.56% | 96.85% |
| inverse | 98.91% | 99.46% | 94.67% |
| alternate | 100.00% | 100.00% | 100.00% |
| square | 93.66% | 96.17% | 92.21% |

Mixed/core3 timing on H100, mean over seeds:

| Path | Mean ms/batch | Interpretation |
|---|---:|---|
| Full recurrent teacher | 28.23 | Real baseline. |
| All JumpRec budgets | 32.17 | Diagnostic only; computes every budget in parallel. |
| Fast serial JumpRec 0.80 | 27.83 | Slightly faster than full, but omits strict agreement. |
| Agreement-aware serial JumpRec 0.80 | 49.19 | More faithful to strict eval, but too slow. |

Per-hop hard-case notes:

- Mixed/core3 stays strong through 3 hops. Seed-42 JumpRec reaches 100.00%,
  99.20%, and 99.49% on hops 1, 2, and 3 respectively.
- The weaker seed-202 result is mostly the `square` transition family:
  teacher 89.09%, JumpRec 92.62%, direct 86.01%.
- On 8/4, both core2 and core3 still fail many 4-hop cases. Core3 JumpRec is
  48.65% on hop 4, close to the teacher's 48.35%. This is a teacher-quality
  bottleneck, not a routing success.

Interpretation:

The mixed/core3 result is the first robust LM-facing evidence that JumpRec is
not just matching a shallow direct model on an easy pointer task. It beats the
teacher and the 3-layer direct control across three seeds while using about
2.31 of 15 recurrent core layers on average. The gain over direct is about
2.77 percentage points, and the gain over the full teacher is about 0.82
points, with the largest relative help on the harder `square` family.

The result is conditional, though. JumpRec only looks good when the underlying
recurrent teacher is already competent. In the 8/4 setting, adding JumpRec
mostly exposes the same 4-hop weakness the full teacher has. That makes the
next split clear: use mixed/core3 to solve routing and real inference speed,
and separately improve hard-hop training before treating 8/4 as a publishable
scaling benchmark.

Raw softmax early exits remain unusable. They exit too early and over-trust bad
predictions, so the verifier remains part of the architecture, not a cosmetic
add-on.

Next direction:

1. Train or calibrate a router that can replace the expensive agreement check.
   The target is mixed/core3 strict accuracy near 98.7% while keeping serial
   timing near the fast 27.8 ms/batch path.
2. Keep the direct control in every run. The claim is only interesting when
   JumpRec beats direct at comparable or lower counted core layers.
3. For 8/4, stop treating JumpRec as the fix until the recurrent teacher can
   handle 4-hop cases. Try balanced hard-hop replay or a more recurrence-aware
   curriculum next.

## 2026-04-26 - preliminary agreement-free router test

Commands:

```text
modal run run_recurrent_smol.py --mode mixed_core3_router_no_agree
modal run run_recurrent_smol.py --mode mixed_core3_router_verifier1
```

These runs reused the mixed/core3 setting, skipped the direct-control training
to save time, and evaluated both agreement-free and agreement-filtered router
policies from the same JumpRec model. The goal was to see whether the verifier
can replace the extra next-budget agreement pass, because the previous strict
agreement serial path was too slow to count as a real inference win.

Seed-42 router results:

| Mode | Router | Threshold | Accuracy | Avg Core Layers | Savings | Full Fallback | Serial Timing |
|---|---|---:|---:|---:|---:|---:|---:|
| `mixed_core3_router_no_agree` | no agreement | 0.80 | 98.71% | 2.02 / 15 | 86.55% | 0.10% | 22.86 ms |
| `mixed_core3_router_no_agree` | no agreement | 0.90 | 99.17% | 2.11 / 15 | 85.94% | 0.32% | 26.63 ms |
| `mixed_core3_router_no_agree` | no agreement | 0.95 | 99.32% | 2.18 / 15 | 85.44% | 0.49% | 34.01 ms |
| `mixed_core3_router_verifier1` | no agreement | 0.80 | 98.49% | 2.04 / 15 | 86.42% | 0.22% | 22.32 ms |
| `mixed_core3_router_verifier1` | no agreement | 0.90 | 98.97% | 2.16 / 15 | 85.60% | 0.42% | 27.74 ms |
| `mixed_core3_router_verifier1` | no agreement | 0.95 | 99.27% | 2.28 / 15 | 84.81% | 0.76% | 28.02 ms |

Baselines for this seed:

| Path | Accuracy | Timing |
|---|---:|---:|
| Full recurrent teacher | 98.71% | 28.82 ms in `router_no_agree`; 28.43 ms in `verifier1` |
| Prior direct 3-layer control | 96.73% | not timed in this router-only run |
| Prior agreement-filtered JumpRec 0.80 | 99.56% | 46.11 ms in the previous full mixed/core3 run |

Interpretation:

This is encouraging but not seed-confirmed yet. The plain verifier at threshold
0.90 is the cleanest trade-off: it beats the full teacher and the direct
control on seed 42, keeps about 85.9% counted core-layer savings, and is
modestly faster wall-clock than the full teacher on H100. Threshold 0.95 is
more accurate but loses the timing win in the plain-verifier run. The heavier
verifier loss does not help enough to justify itself; it is slightly worse than
the plain verifier at comparable thresholds.

A small instrumentation bug was also found: in modes with
`strict_need_agreement = False`, the `jumprec_serial_agree_080_ms_per_batch`
timing path was not forcing agreement, even though agreement-filtered eval
metrics were still computed correctly. The fast no-agreement serial timings
above are the relevant values from this run; agreement serial timing should be
re-measured after the fix.

Next direction:

1. Seed-confirm `mixed_core3_router_no_agree` at seeds 101 and 202, with the
   threshold-0.90 router as the main candidate.
2. If seed-confirmed, update the headline claim from counted core-layer savings
   to a narrow but real H100 wall-clock win on the harder mixed/core3 benchmark.
3. If not seed-stable, train the verifier on the whole accept/reject rule or
   add a learned uncertainty objective instead of simply increasing verifier
   loss weight.

## 2026-04-26 - agreement-free router seed confirmation

Commands:

```text
modal run run_recurrent_smol.py --mode mixed_core3_router_no_agree --seed 101
modal run run_recurrent_smol.py --mode mixed_core3_router_no_agree --seed 202
```

Together with the seed-42 preliminary run, this confirms the agreement-free
router as an accuracy result but not yet as a batch-64 wall-clock result.

Agreement-free router, threshold 0.90:

| Seed | Teacher Full | Router Acc | Avg Core Layers | Savings | Full Timing | Serial Timing |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 98.71% | 99.17% | 2.11 / 15 | 85.94% | 28.82 ms | 26.63 ms |
| 101 | 98.49% | 98.58% | 2.25 / 15 | 85.01% | 28.28 ms | 39.75 ms |
| 202 | 96.56% | 97.12% | 2.85 / 15 | 81.03% | 28.07 ms | 41.46 ms |
| Mean | 97.92% | 98.29% | 2.40 / 15 | 84.00% | 28.39 ms | 35.95 ms |

Threshold sweep, mean over seeds:

| Threshold | Router Acc | Avg Core Layers | Savings | Serial Timing |
|---:|---:|---:|---:|---:|
| 0.80 | 97.88% | 2.22 / 15 | 85.23% | 31.50 ms |
| 0.90 | 98.29% | 2.40 / 15 | 84.00% | 35.95 ms |
| 0.95 | 98.40% | 2.60 / 15 | 82.70% | 39.89 ms |

Agreement-filtered eval is still more accurate: threshold 0.80 gives the same
98.74% mean accuracy as the earlier strict mixed/core3 result, at about 2.31
of 15 core layers. But the corrected agreement-aware serial path is around 60
ms/batch in these router-only runs, so agreement is not acceptable as the final
speed path.

Interpretation:

The verifier is good enough to route without agreement in the accuracy sense:
threshold 0.90 beats the full teacher mean and the prior direct control mean
while using only about 2.40 of 15 counted recurrent core layers. The weaker
seed 202 remains the limiting case, but even there the no-agreement router
beats the weak full teacher by about 0.56 points.

The wall-clock result did not seed-confirm at batch size 64. Seed 42 was
faster than full, but seeds 101 and 202 were much slower despite lower counted
core usage. That points to implementation overhead and GPU utilization:
serial subset routing launches several small dynamic batches, which can erase
the theoretical compute savings on an H100 batch-64 benchmark.

Next direction:

1. Measure timing at batch size 1, which is closer to local interactive LLM
   use and may reward adaptive exits more than batch-64 throughput does.
2. If batch-1 timing is still poor, the next work is a fused or static router
   implementation rather than more verifier calibration.
3. Keep the no-agreement verifier as the main research path; increasing
   verifier loss did not improve the trade-off.

## 2026-04-26 - batch-1 timing probe

Command:

```text
modal run run_recurrent_smol.py --mode mixed_core3_router_no_agree_b1
```

This is the same seed-42 mixed/core3 agreement-free router as above, but with
benchmark timing batch size set to 1 while leaving the training batch size at
64. This is closer to the local interactive LLM use case than the batch-64 H100
throughput benchmark.

Seed-42 batch-1 timing:

| Path | Timing |
|---|---:|
| One recurrent loop | 5.76 ms |
| Full recurrent teacher | 17.51 ms |
| All JumpRec budgets | 21.56 ms |
| No-agreement serial 0.80 | 7.41 ms |
| No-agreement serial 0.90 | 7.37 ms |
| No-agreement serial 0.95 | 7.84 ms |
| Agreement serial 0.80 | 12.72 ms |

Accuracy is unchanged from the seed-42 router run because only the timing batch
size changed: no-agreement threshold 0.90 gives 99.17% accuracy at 2.11 of 15
core layers, with 85.94% counted core-layer savings.

Interpretation:

This is the first clean wall-clock result in the intended local direction.
Batch size 64 punished dynamic routing because subset routing launched several
small GPU calls inside a throughput benchmark. Batch size 1 reverses that: the
serial no-agreement router is about 2.37x faster than the full recurrent
teacher on seed 42 while also beating its accuracy.

Seed confirmation:

```text
modal run run_recurrent_smol.py --mode mixed_core3_router_no_agree_b1 --seed 101
modal run run_recurrent_smol.py --mode mixed_core3_router_no_agree_b1 --seed 202
```

Agreement-free router, threshold 0.90, batch-1 timing:

| Seed | Teacher Full | Router Acc | Avg Core Layers | Savings | Full Timing | Serial Timing | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 98.71% | 99.17% | 2.11 / 15 | 85.94% | 17.51 ms | 7.37 ms | 2.37x |
| 101 | 98.49% | 98.58% | 2.25 / 15 | 85.01% | 23.33 ms | 10.80 ms | 2.16x |
| 202 | 96.56% | 97.12% | 2.85 / 15 | 81.03% | 15.60 ms | 8.26 ms | 1.89x |
| Mean | 97.92% | 98.29% | 2.40 / 15 | 84.00% | 18.81 ms | 8.81 ms | 2.13x |

Threshold sweep, mean batch-1 timing over seeds:

| Threshold | Router Acc | Avg Core Layers | Savings | Serial Timing | Speedup vs Full |
|---:|---:|---:|---:|---:|---:|
| 0.80 | 97.88% | 2.22 / 15 | 85.23% | 7.84 ms | 2.40x |
| 0.90 | 98.29% | 2.40 / 15 | 84.00% | 8.81 ms | 2.13x |
| 0.95 | 98.40% | 2.60 / 15 | 82.70% | 9.53 ms | 1.97x |

Agreement-filtered threshold 0.80 also becomes viable at batch size 1: it
averages about 98.74% accuracy and 15.22 ms, still faster than the 18.81 ms
full-loop mean. The no-agreement threshold-0.90 path remains the better
latency/accuracy trade-off for local inference.

Interpretation:

This is now the strongest result in the project. The batch-64 throughput result
did not convert counted savings into speed, but the local-style batch-1 result
does. On a harder mixed textual recurrence task, recurrent SmolLM2 plus JumpRec
beats the full recurrent teacher's accuracy, beats the shallow direct control,
uses about 2.40 of 15 recurrent core layers, and runs about 2.13x faster than
the full recurrent teacher at batch size 1 on H100.

The claim should stay narrow. This is still a synthetic textual recurrence
benchmark, not general language modeling, and it is measured on H100 rather
than a consumer local GPU. But it is exactly the first shape we wanted: a small
pretrained-LM retrofit with a recurrent state space where a learned jump/router
can spend much less compute on easier cases while preserving or improving hard
task accuracy.

## 2026-04-26 - checkpointed batch-size timing sweep

Command:

```text
modal run run_recurrent_smol.py --mode mixed_core3_router_bsize_sweep --seed 42
modal run run_recurrent_smol.py --mode mixed_core3_router_bsize_sweep --seed 101
modal run run_recurrent_smol.py --mode mixed_core3_router_bsize_sweep --seed 202
```

This reruns the mixed/core3 agreement-free router with checkpoint saving and
a single-run timing sweep over batch sizes 1, 2, 4, 8, 16, 32, and 64. The
checkpoints were saved to Modal volume paths like
`/results/checkpoints/mixed_core3_router_seed42.pt`.

Agreement-free router, threshold 0.90:

| Seed | Teacher Full | Router Acc | Avg Core Layers | Savings | B1 Full | B1 Router | B1 Speedup | B64 Full | B64 Router | B64 Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 98.71% | 99.17% | 2.11 / 15 | 85.94% | 23.58 ms | 10.05 ms | 2.35x | 28.46 ms | 27.14 ms | 1.05x |
| 101 | 98.49% | 98.58% | 2.25 / 15 | 85.01% | 20.57 ms | 8.22 ms | 2.50x | 28.32 ms | 36.25 ms | 0.78x |
| 202 | 96.56% | 97.12% | 2.85 / 15 | 81.03% | 22.71 ms | 11.65 ms | 1.95x | 28.85 ms | 44.27 ms | 0.65x |
| Mean | 97.92% | 98.29% | 2.40 / 15 | 84.00% | 22.29 ms | 9.97 ms | 2.24x | 28.54 ms | 35.89 ms | 0.80x |

Mean batch-size sweep for threshold 0.90:

| Batch Size | Full Teacher | Router Serial | Speedup |
|---:|---:|---:|---:|
| 1 | 22.29 ms | 9.97 ms | 2.24x |
| 2 | 22.89 ms | 12.76 ms | 1.79x |
| 4 | 23.32 ms | 16.63 ms | 1.40x |
| 8 | 23.65 ms | 19.56 ms | 1.21x |
| 16 | 24.33 ms | 24.21 ms | 1.00x |
| 32 | 24.76 ms | 29.59 ms | 0.84x |
| 64 | 28.54 ms | 35.89 ms | 0.80x |

Interpretation:

The timing sweep confirms the previous batch-1 result and draws the line more
clearly. In the current unfused serial-router implementation, JumpRec is a
local/small-batch latency win through roughly batch size 8, breaks even around
batch size 16, and loses for larger throughput batches. This is consistent with
the implementation: dynamic subset routing creates multiple small GPU launches,
so counted layer savings convert to wall-clock speed only when per-request
latency matters more than large-batch utilization.

This is still encouraging for the intended local-model direction. The measured
regime is exactly where a single-user local assistant, agent, or interactive
tool would usually live. But it also means we should not claim a generic
production-serving throughput improvement without a fused/static routing
implementation.

## 2026-04-26 - checkpoint reuse verification

Command:

```text
modal run run_recurrent_smol.py --mode mixed_core3_router_bsize_sweep_reuse --seed 42
```

The first Windows launcher attempt failed before the Modal job due to redirected
console encoding of a checkmark character. Rerunning with UTF-8 console settings
worked.

The reuse mode loaded `/results/checkpoints/mixed_core3_router_seed42.pt`,
skipped teacher training, loaded JumpRec, and reran eval/timing. This confirms
that future threshold or timing-only probes can avoid the full teacher and
JumpRec training cost.

Representative seed-42 reuse numbers:

| Metric | Value |
|---|---:|
| Full recurrent teacher | 98.61% |
| Router 0.90 no-agreement | 99.17% |
| Router 0.90 avg core layers | 2.10 / 15 |
| Router 0.90 core savings | 85.99% |
| Batch-1 full timing | 21.54 ms |
| Batch-1 router 0.90 timing | 8.49 ms |
| Batch-1 speedup | 2.54x |
| Batch-64 full timing | 28.25 ms |
| Batch-64 router 0.90 timing | 26.52 ms |
| Batch-64 speedup | 1.07x |

These numbers should not replace the seed-mean table above because this was a
reuse validation run, not a new seed-confirmation sweep. The important result is
workflow: checkpoint reuse now works on Modal.

## 2026-04-26 - 8-node / 4-hop hard-hop teacher repair

Command:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_hardhop_teacher --seed 42
```

This run tests whether the prior 8/4 weakness was a fundamental recurrent-LM
failure or a training recipe failure. Changes from the earlier
`core3_8n4h_jumprec_direct` setup:

- 70% of training examples use the current maximum hop count.
- max-hop examples receive 2.5x loss weight.
- the final recurrent loop receives 4x loop-loss weight instead of 2x.
- teacher checkpoints are saved for a separate JumpRec run.
- evaluation remains uniform over hops, so the training bias does not inflate
  the reported per-hop table.

Seed-42 teacher result:

| Model | Accuracy | Hop 1 | Hop 2 | Hop 3 | Hop 4 |
|---|---:|---:|---:|---:|---:|
| Prior core3 8/4 full teacher | 84.42% | 99.93% | 100.00% | 89.17% | 48.35% |
| Hard-hop core3 full teacher | 94.87% | 99.89% | 95.02% | 88.68% | 95.74% |
| Hard-hop 3-layer direct control | 77.28% | 98.72% | 76.33% | 55.03% | 78.00% |

Loop-depth profile:

| Loops | Final-target Accuracy |
|---:|---:|
| 0 | 15.48% |
| 1 | 39.14% |
| 2 | 58.58% |
| 3 | 75.55% |
| 4 | 95.62% |
| 5 | 95.35% |
| 6 | 94.87% |

Interpretation:

This is the first clean evidence that the 8/4 teacher weakness was not an
architectural ceiling. The old full teacher failed mainly on hop-4 examples;
the hard-hop teacher reaches 95.74% on hop 4 under uniform eval. The remaining
weakness moved to hop 3, at 88.68%, which suggests the biased training recipe
may slightly over-specialize to the maximum-depth case.

The direct control result is important: it reaches only 77.28%, with hop 3 at
55.03%. So the full recurrent teacher is not merely learning a shallow shortcut.
This justifies the next run:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_hardhop_jumprec --seed 42
```

That run loads the saved teacher checkpoint and asks whether JumpRec can retain
the hard-hop teacher's improvement while spending fewer recurrent core layers.

## 2026-04-26 - JumpRec on repaired 8/4 teacher

Command:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_hardhop_jumprec --seed 42
```

This run loaded `/results/checkpoints/core3_8n4h_hardhop_seed42.pt`, skipped
teacher training, trained JumpRec, and ran a batch-size timing sweep. The loaded
teacher eval was 94.71%, close to the 94.87% teacher-only run above.

JumpRec quality/compute trade-off:

| Path | Accuracy | Avg Core Layers | Savings |
|---|---:|---:|---:|
| Full recurrent teacher | 94.71% | 18.00 / 18 | 0.00% |
| Jump budget 0 | 46.78% | jump only | n/a |
| Jump budget 1 | 86.36% | jump + 1 tail | n/a |
| Jump budget 2 | 94.51% | jump + 2 tails | n/a |
| Jump budget 3 | 95.80% | jump + 3 tails | n/a |
| No-agree router 0.80 | 93.77% | 4.22 / 18 | 76.57% |
| No-agree router 0.90 | 94.73% | 4.70 / 18 | 73.91% |
| No-agree router 0.95 | 95.23% | 5.14 / 18 | 71.44% |
| Agreement router 0.80 | 96.09% | 4.72 / 18 | 73.78% |

No-agreement threshold 0.90 essentially matches the loaded full teacher while
using 26.09% of the counted recurrent core layers. Threshold 0.95 slightly
beats the teacher while still saving 71.44% counted core compute. Agreement at
0.80 is the most accurate path, but its extra pass is slower.

Hop breakdown for strict/no-agreement threshold 0.80:

| Hop | Accuracy |
|---:|---:|
| 1 | 98.90% |
| 2 | 94.84% |
| 3 | 88.45% |
| 4 | 93.22% |

Batch-size timing for no-agreement threshold 0.90:

| Batch Size | Full Teacher | Router 0.90 | Speedup |
|---:|---:|---:|---:|
| 1 | 19.64 ms | 10.32 ms | 1.90x |
| 2 | 20.47 ms | 15.10 ms | 1.36x |
| 4 | 20.83 ms | 20.23 ms | 1.03x |
| 8 | 20.97 ms | 23.91 ms | 0.88x |
| 16 | 22.15 ms | 30.01 ms | 0.74x |
| 32 | 22.75 ms | 34.59 ms | 0.66x |
| 64 | 34.33 ms | 39.12 ms | 0.88x |

Interpretation:

This is the strongest "hard problem" result so far. The repaired teacher turns
8/4 from a weak-teacher case into a competent recurrent-depth case, and JumpRec
then recovers the teacher's accuracy with about 74% counted core-layer savings.
Unlike the easier mixed/core3 benchmark, the useful wall-clock regime is
narrower: batch size 1 is clearly faster, batch size 2 is faster, batch size 4
is roughly break-even, and larger batches lose with the current serial router.

The main caveat is seeds. This is one seed. It is strong enough to justify
seed-confirming the hard-hop teacher plus JumpRec path, but not yet enough for a
paper-level claim.
