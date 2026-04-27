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

## 2026-04-26 - hard-hop teacher seed confirmation

Commands:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_hardhop_teacher --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_hardhop_teacher --seed 202
```

Teacher results under uniform hop eval:

| Seed | Full Teacher | Hop 1 | Hop 2 | Hop 3 | Hop 4 | Direct Control |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 94.87% | 99.89% | 95.02% | 88.68% | 95.74% | 77.28% |
| 101 | 96.14% | 99.80% | 98.08% | 93.30% | 93.25% | 61.80% |
| 202 | 76.87% | 99.17% | 73.08% | 50.41% | 86.37% | 58.85% |
| Mean | 89.30% | 99.62% | 88.73% | 77.46% | 91.79% | 65.31% |

Interpretation:

This partially confirms the hard-hop teacher repair but exposes a stability
problem. Seeds 42 and 101 are strong and show the intended behavior: a competent
recurrent teacher, high max-hop accuracy, and a large gap over direct control.
Seed 202 does not confirm; it improves hop 4 compared with the old teacher but
collapses badly on hop 3 and is not a good JumpRec teacher.

So the updated claim is narrower:

- Validated: hard-hop replay/loss weighting can repair the 8/4 teacher weakness.
- Not yet validated: the recipe is seed-robust.
- Next action: run JumpRec only on strong teacher seed 101, and treat seed 202
  as a teacher-stability failure rather than a JumpRec target.

Likely stability fixes to test next:

- lower max-hop replay from 70% to around 50% so hop 3 is less starved;
- use stratified hard replay instead of only max-hop replay, e.g. explicit hop
  weights for hops 2/3/4;
- checkpoint by uniform validation accuracy rather than final training step;
- consider a short uniform fine-tune after hard-hop training;
- reset seeds before direct-control training so the direct baseline is less
  dependent on whether JumpRec trained before it.

## 2026-04-26 - JumpRec on second repaired 8/4 teacher

Command:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_hardhop_jumprec --seed 101
```

Seed 101 loaded the repaired teacher checkpoint and confirmed the JumpRec
behavior seen on seed 42. Seed 202 was not run through JumpRec because its
teacher was weak.

Strong-teacher JumpRec summary:

| Seed | Loaded Teacher | Router 0.90 Acc | Router 0.90 Core | Router 0.90 Savings | Router 0.95 Acc | Router 0.95 Core | Router 0.95 Savings |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 94.71% | 94.73% | 4.70 / 18 | 73.91% | 95.23% | 5.14 / 18 | 71.44% |
| 101 | 96.06% | 96.55% | 4.18 / 18 | 76.79% | 97.04% | 4.50 / 18 | 75.01% |
| Mean | 95.39% | 95.64% | 4.44 / 18 | 75.35% | 96.13% | 4.82 / 18 | 73.22% |

Agreement-filtered router:

| Seed | Agreement 0.80 Acc | Avg Core Layers | Savings |
|---:|---:|---:|---:|
| 42 | 96.09% | 4.72 / 18 | 73.78% |
| 101 | 97.51% | 4.14 / 18 | 77.00% |
| Mean | 96.80% | 4.43 / 18 | 75.39% |

Batch-1 timing for no-agreement threshold 0.90:

| Seed | Full Teacher | Router 0.90 | Speedup |
|---:|---:|---:|---:|
| 42 | 19.64 ms | 10.32 ms | 1.90x |
| 101 | 21.10 ms | 10.68 ms | 1.98x |
| Mean | 20.37 ms | 10.50 ms | 1.94x |

Seed-101 batch-size timing for no-agreement threshold 0.90:

| Batch Size | Full Teacher | Router 0.90 | Speedup |
|---:|---:|---:|---:|
| 1 | 21.10 ms | 10.68 ms | 1.98x |
| 2 | 22.46 ms | 13.63 ms | 1.65x |
| 4 | 21.78 ms | 16.61 ms | 1.31x |
| 8 | 22.68 ms | 22.90 ms | 0.99x |
| 16 | 23.20 ms | 27.01 ms | 0.86x |
| 32 | 23.80 ms | 34.45 ms | 0.69x |
| 64 | 33.73 ms | 39.71 ms | 0.85x |

Interpretation:

On strong repaired 8/4 teachers, JumpRec now has a two-seed confirmation:
it matches or beats the full teacher, saves roughly three quarters of counted
recurrent core compute, and gives a roughly 2x batch-1 speedup. This is a much
more meaningful hard-case result than the earlier easy/mixed result.

The caveat is equally important: only two of three teacher seeds became strong.
The bottleneck is now teacher stability, not JumpRec on a good teacher.

## 2026-04-26 - stratified-hop stability repair

Command:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_teacher --seed 202
```

This was the follow-up to the weak max-hop-only seed 202 teacher. Instead of
sampling 70% max-hop examples, the stratified run sampled hops with weights
`0.10,0.20,0.35,0.35` and weighted hop losses as `1.0,1.2,2.0,2.0`.

Seed-202 teacher comparison:

| Recipe | Full Teacher | Hop 1 | Hop 2 | Hop 3 | Hop 4 | Direct Control |
|---|---:|---:|---:|---:|---:|---:|
| Max-hop hard replay | 76.87% | 99.17% | 73.08% | 50.41% | 86.37% | 58.85% |
| Stratified hard replay | 99.51% | 100.00% | 99.83% | 99.94% | 98.27% | 91.62% |

Loop-depth profile for the stratified teacher:

| Loops | Final-target Accuracy |
|---:|---:|
| 0 | 15.01% |
| 1 | 41.02% |
| 2 | 59.31% |
| 3 | 78.16% |
| 4 | 99.46% |
| 5 | 99.59% |
| 6 | 99.51% |

Interpretation:

This is a major correction to the prior caveat. The seed-202 failure appears
to have been caused by max-hop-only pressure starving intermediate hard hops,
especially hop 3. Stratified hard replay repaired the same seed from 76.87% to
99.51% uniform full-loop accuracy. It also raised the direct baseline to
91.62%, but the direct model still fails the hardest hop relative to recurrence
at hop 4: 69.08% direct versus 98.27% recurrent.

The current best teacher recipe for 8/4 is therefore stratified hard replay, not
70% max-hop replay. The next best experiment is:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_jumprec --seed 202
```

Run JumpRec on this repaired stratified seed-202 checkpoint. If JumpRec retains
the teacher while saving compute, then the hard-case story becomes much cleaner:
stratified recurrence training creates a robust teacher, and JumpRec compresses
the recurrent compute.

## 2026-04-26 - JumpRec on stratified 8/4 teacher

Command:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_jumprec --seed 202
```

This loaded `/results/checkpoints/core3_8n4h_strathop_seed202.pt`, skipped
teacher training, trained JumpRec, trained the direct control, and ran the
batch-size timing sweep.

The loaded teacher eval reproduced the teacher-only result:

| Path | Accuracy | Avg Core Layers | Savings |
|---|---:|---:|---:|
| Full recurrent teacher | 99.53% | 18.00 / 18 | 0.00% |
| Direct 3-layer control | 97.56% | 3.00 / 18 | 83.33% |
| No-agree router 0.80 | 99.15% | 3.26 / 18 | 81.88% |
| No-agree router 0.90 | 99.45% | 3.33 / 18 | 81.49% |
| No-agree router 0.95 | 99.64% | 3.42 / 18 | 80.98% |
| Agreement router 0.80 | 99.79% | 3.33 / 18 | 81.49% |

Fixed JumpRec correction budgets:

| Budget | Accuracy |
|---:|---:|
| Jump + 0 tail loops | 48.42% |
| Jump + 1 tail loop | 94.42% |
| Jump + 2 tail loops | 99.37% |
| Jump + 3 tail loops | 99.74% |

Hop breakdown for no-agreement threshold 0.80:

| Hop | Accuracy |
|---:|---:|
| 1 | 99.91% |
| 2 | 99.85% |
| 3 | 99.30% |
| 4 | 97.58% |

The direct control is much stronger under the stratified recipe than it was in
the max-hop runs, but it still leaves a clear hard-hop gap: direct hop-4
accuracy is 90.79%, while the recurrent teacher is 98.08% on hop 4 and JumpRec
threshold 0.80 is 97.58%.

Batch-size timing for no-agreement threshold 0.90:

| Batch Size | Full Teacher | Router 0.90 | Speedup |
|---:|---:|---:|---:|
| 1 | 20.61 ms | 9.20 ms | 2.24x |
| 2 | 21.54 ms | 12.13 ms | 1.78x |
| 4 | 22.00 ms | 13.95 ms | 1.58x |
| 8 | 22.29 ms | 16.03 ms | 1.39x |
| 16 | 23.14 ms | 18.56 ms | 1.25x |
| 32 | 23.60 ms | 20.42 ms | 1.16x |
| 64 | 34.18 ms | 24.45 ms | 1.40x |

Threshold 0.95 is the cleanest quality point on this seed: it slightly beats
the full teacher, 99.64% vs 99.53%, while using only 3.42 of 18 counted
recurrent core layers. It also remains faster than the full teacher in the
timing sweep: 9.48 ms vs 20.61 ms at batch size 1, and 26.79 ms vs 34.18 ms at
batch size 64.

Interpretation:

This is the cleanest hard-case result so far. The stratified recipe repaired
the previously weak seed-202 teacher, and JumpRec then retained or slightly
improved that teacher while saving about 81% counted recurrent core compute.
The direct baseline is now genuinely competitive on easy/intermediate hops,
which makes the comparison more meaningful: the remaining advantage is
concentrated where recurrence should matter most.

This run is also the first hard-case timing sweep where the current unfused
serial no-agreement router is faster than the full recurrent teacher across
all measured batch sizes. That should be treated as promising rather than
settled: it is one seed, one synthetic recurrence family, and one H100 timing
setup. But it is a much better shape than the earlier max-hop hard-hop runs,
where the router lost at moderate and large batch sizes.

Next:

1. Seed-confirm the stratified teacher recipe on seeds 42 and 101.
2. If those teachers are strong, run stratified JumpRec on the same seeds.
3. Compare stratified-vs-max-hop across matched seeds, including direct
   control, router accuracy, counted layers, and timing.
4. Start preparing the paper-style table around three tiers: toy pointer
   proof, mixed SmolLM2 seed-confirmed local latency, and stratified 8/4
   hard-case robustness.

## 2026-04-26 - stratified hard-case seed follow-up

Commands:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_teacher --seed 42
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_teacher --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_jumprec --seed 101
```

The stratified teacher recipe is better than the max-hop-only recipe on seeds
101 and 202, but not a complete stability solution. Seed 42 improved on hop 3
but shifted weakness back onto hop 4.

Teacher comparison:

| Seed | Recipe | Full Teacher | Hop 1 | Hop 2 | Hop 3 | Hop 4 | Direct Control |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | max-hop | 94.87% | 99.89% | 95.02% | 88.68% | 95.74% | 77.28% |
| 42 | stratified | 96.32% | 100.00% | 100.00% | 99.03% | 86.10% | 94.34% |
| 101 | max-hop | 96.14% | 99.80% | 98.08% | 93.30% | 93.25% | 61.80% |
| 101 | stratified | 99.74% | 100.00% | 99.86% | 100.00% | 99.16% | 96.16% |
| 202 | max-hop | 76.87% | 99.17% | 73.08% | 50.41% | 86.37% | 58.85% |
| 202 | stratified | 99.51% | 100.00% | 99.83% | 99.94% | 98.27% | 91.62% |

Interpretation of the teacher runs:

Stratified hard replay fixed the seed-202 collapse and made seed 101 excellent,
but seed 42 shows a new failure mode. The recipe no longer starves hop 3, but
it can under-serve the max-hop edge on some seeds. This means the teacher story
is improved but not solved. The next teacher hygiene step should checkpoint by
uniform validation, especially worst-hop accuracy, rather than only saving the
final training step. A short late-stage uniform or max-hop polish may also be
needed.

Because seed 101 was a clean strong stratified teacher, we ran JumpRec on it.
Seed 42 was not run through JumpRec in this round because its teacher is not a
clean hard-hop target.

Strong-stratified JumpRec summary:

| Seed | Loaded Teacher | Same-Run Direct | Router 0.90 Acc | Router 0.90 Core | Router 0.90 Savings | Router 0.95 Acc | Router 0.95 Core | Router 0.95 Savings |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 99.79% | 90.32% | 99.43% | 2.36 / 18 | 86.90% | 99.54% | 2.40 / 18 | 86.69% |
| 202 | 99.53% | 97.56% | 99.45% | 3.33 / 18 | 81.49% | 99.64% | 3.42 / 18 | 80.98% |
| Mean | 99.66% | 93.44% | 99.44% | 2.84 / 18 | 84.20% | 99.59% | 2.91 / 18 | 83.83% |

Agreement-filtered router:

| Seed | Agreement 0.80 Acc | Avg Core Layers | Savings |
|---:|---:|---:|---:|
| 101 | 99.76% | 2.36 / 18 | 86.87% |
| 202 | 99.79% | 3.33 / 18 | 81.49% |
| Mean | 99.77% | 2.85 / 18 | 84.18% |

Seed-101 no-agreement timing at threshold 0.90:

| Batch Size | Full Teacher | Router 0.90 | Speedup |
|---:|---:|---:|---:|
| 1 | 19.59 ms | 7.75 ms | 2.53x |
| 2 | 20.35 ms | 10.22 ms | 1.99x |
| 4 | 20.72 ms | 11.05 ms | 1.87x |
| 8 | 20.56 ms | 11.94 ms | 1.72x |
| 16 | 21.32 ms | 14.00 ms | 1.52x |
| 32 | 22.03 ms | 16.79 ms | 1.31x |
| 64 | 33.51 ms | 21.11 ms | 1.59x |

For the two strong stratified JumpRec seeds, threshold 0.95 averages 99.59%
accuracy while using 2.91 of 18 counted recurrent core layers. Batch-1 speedup
is about 2.29x on average, and batch-64 speedup is about 1.39x, both versus the
full recurrent teacher in the current serial no-agreement implementation.

Interpretation:

On strong stratified teachers, JumpRec has now confirmed the desirable hard-case
shape twice: recurrent depth matters, direct control is much weaker on hard
hops, and the router retains high accuracy while skipping most recurrent core
layers. This is stronger than the max-hop result because the stratified seed
101 and seed 202 teachers both exceed 99.5% full-loop accuracy, and the serial
router is faster across the measured batch-size sweep.

The uncomfortable but useful caveat is seed 42. Stratified replay is not a
universal teacher recipe; it changes which hop can fail. The next research step
should not be more blind JumpRec runs. It should be a teacher-quality gate:
track uniform validation by hop during teacher training, save the best
worst-hop checkpoint, and test a blended schedule such as stratified training
followed by a short uniform or max-hop polish. Once seed 42 can be repaired
without harming seeds 101/202, rerun JumpRec on the gated checkpoints.

## 2026-04-26 - worst-hop teacher gate on stratified seed 42

Code changes:

- Added `teacher_val_every`, `teacher_val_batches`,
  `teacher_gate_min_full`, and `teacher_gate_min_worst_hop`.
- Added `core3_8n4h_strathop_gate_teacher` and
  `core3_8n4h_strathop_gate_jumprec` modes.
- Teacher validation samples uniformly over hops, reports per-hop accuracy, and
  saves/restores the checkpoint with the best worst-hop validation accuracy.
- Validation preserves the Python RNG state so validation examples do not
  perturb the subsequent training sequence.

Command:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_gate_teacher --seed 42
```

Gate settings:

| Setting | Value |
|---|---:|
| Validation interval | 500 recurrent steps |
| Validation batches | 16 |
| Full-accuracy gate | 99.5% |
| Worst-hop gate | 98.0% |

Best validation checkpoint:

| Step | Full Val Acc | Hop 1 | Hop 2 | Hop 3 | Hop 4 / Worst Hop | Gate Passed |
|---:|---:|---:|---:|---:|---:|---|
| 10000 | 96.68% | 100.00% | 100.00% | 98.78% | 88.08% | no |

Final uniform eval after restoring the best validation checkpoint:

| Model | Full Acc | Hop 1 | Hop 2 | Hop 3 | Hop 4 |
|---|---:|---:|---:|---:|---:|
| Stratified gated teacher | 96.32% | 100.00% | 100.00% | 99.03% | 86.10% |
| Direct control | 94.34% | 98.86% | 99.73% | 99.39% | 79.63% |

Interpretation:

The gate implementation works, but checkpoint selection alone does not repair
seed 42. It confirms the failure mode instead: stratified training solves hops
1-3 and leaves hop 4 far below the target. The restored best checkpoint is the
final checkpoint, and the final eval matches the earlier ungated stratified
seed-42 run, which is a useful sanity check that validation did not perturb the
training stream.

This changes the next experiment. We should not run JumpRec on this checkpoint.
The next teacher-side test should add a short late-stage polish after
stratified training, probably uniform or max-hop-heavy, while keeping the same
worst-hop validation gate. The goal is to preserve the seed-202/101 stratified
fix while recovering seed-42 hop-4 accuracy.

## 2026-04-26 - max-hop polish on stratified seed 42

Command:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_polish_teacher --seed 42
```

Polish settings:

| Setting | Value |
|---|---:|
| Loaded checkpoint | `core3_8n4h_strathop_gate_seed42` |
| Extra recurrent steps | 3000 |
| Max-hop sample probability | 70% |
| Max-hop loss weight | 2.5 |
| Final-loop loss weight | 4.0 |
| Validation interval | 250 steps |
| Validation batches | 16 |
| Full-accuracy gate | 99.5% |
| Worst-hop gate | 98.0% |

Best validation checkpoint:

| Step | Full Val Acc | Hop 1 | Hop 2 | Hop 3 | Hop 4 / Worst Hop | Gate Passed |
|---:|---:|---:|---:|---:|---:|---|
| 3000 | 97.66% | 100.00% | 99.59% | 97.35% | 93.89% | no |

Final eval after restoring the best validation checkpoint:

| Model | Full Acc | Hop 1 | Hop 2 | Hop 3 | Hop 4 |
|---|---:|---:|---:|---:|---:|
| Stratified gated teacher before polish | 96.32% | 100.00% | 100.00% | 99.03% | 86.10% |
| After max-hop polish | 98.21% | 100.00% | 99.70% | 98.49% | 94.82% |

Accuracy by loop after polish:

| Loops | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| Accuracy | 15.43% | 40.90% | 58.69% | 78.06% | 98.55% | 98.45% | 98.21% |

Timing sanity:

| Batch Size | One Loop | Full Loop |
|---:|---:|---:|
| 64 | 11.70 ms | 33.86 ms |

Interpretation:

The first polish is useful but not a pass. It lifts the exact weak cell, seed-42
hop 4, by 8.72 points on final eval and raises full-loop accuracy by 1.89
points. That argues the failure is trainable rather than a hard architectural
ceiling.

The gate still fails, so this checkpoint should not be used for a headline
JumpRec run. The loop profile also hints that part of the remaining problem is
answer preservation: accuracy peaks at loop 4 and drifts down through loops 5
and 6. The next test should continue from this checkpoint with gentler learning
rates and a stronger final-loop loss, still under the worst-hop gate. If that
passes, run JumpRec on the repaired checkpoint; if it does not, revisit the
teacher objective rather than spending router compute.

## 2026-04-26 - second-stage polish on stratified seed 42

Command:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_polish2_teacher --seed 42
```

Polish settings:

| Setting | Value |
|---|---:|
| Loaded checkpoint | `core3_8n4h_strathop_polish_seed42` |
| Extra recurrent steps | 4000 |
| Block LR | 2e-5 |
| Head LR | 1.5e-4 |
| Max-hop sample probability | 70% |
| Max-hop loss weight | 2.5 |
| Final-loop loss weight | 8.0 |
| Validation interval | 250 steps |
| Validation batches | 16 |
| Full-accuracy gate | 99.5% |
| Worst-hop gate | 98.0% |

Best validation checkpoint:

| Step | Full Val Acc | Hop 1 | Hop 2 | Hop 3 / Worst Hop | Hop 4 | Gate Passed |
|---:|---:|---:|---:|---:|---:|---|
| 3250 | 99.90% | 100.00% | 100.00% | 99.62% | 100.00% | yes |

Final 96-batch eval after restoring the best validation checkpoint:

| Model | Full Acc | Hop 1 | Hop 2 | Hop 3 | Hop 4 |
|---|---:|---:|---:|---:|---:|
| Stratified gated teacher before polish | 96.32% | 100.00% | 100.00% | 99.03% | 86.10% |
| After first polish | 98.21% | 100.00% | 99.70% | 98.49% | 94.82% |
| After second-stage polish | 99.40% | 100.00% | 100.00% | 99.55% | 98.02% |

Accuracy by loop after second-stage polish:

| Loops | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| Accuracy | 16.36% | 41.24% | 59.78% | 78.40% | 99.56% | 99.50% | 99.40% |

Timing sanity:

| Batch Size | One Loop | Full Loop |
|---:|---:|---:|
| 64 | 12.35 ms | 34.46 ms |

Interpretation:

This is the first seed-42 repair that clears the teacher validation gate. The
broader 96-batch final eval lands slightly below the 99.5% full-accuracy target
but comfortably above the 98% worst-hop target, with hop 4 lifted from 86.10%
before polish to 98.02% after the second stage. The result supports the idea
that seed 42 was a trainable teacher-objective failure, not a fatal recurrent
architecture failure.

The rigorous reading is not "solved forever." The gate was only 16 batches, and
the final eval shows some sampling variance around the strict 99.5% full target.
For headline use, repeat or widen the gate. For the next diagnostic experiment,
this checkpoint is strong enough to run JumpRec and test whether the repaired
teacher also gives the expected compute-saving router behavior.

## 2026-04-26 - JumpRec on repaired stratified seed 42

Command:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_polish2_jumprec --seed 42
```

Loaded checkpoint:

```text
core3_8n4h_strathop_polish2_seed42
```

Teacher and direct control:

| Model | Accuracy | Hop 1 | Hop 2 | Hop 3 | Hop 4 | Counted Core Layers |
|---|---:|---:|---:|---:|---:|---:|
| Full recurrent teacher | 99.53% | 100.00% | 100.00% | 99.49% | 98.68% | 18 |
| 3-layer direct control | 59.13% | 99.35% | 53.16% | 38.10% | 46.64% | 3 |

Fixed JumpRec budgets:

| Budget | Accuracy |
|---:|---:|
| c0 | 47.95% |
| c1 | 96.18% |
| c2 | 99.35% |
| c3 | 99.54% |

Router results:

| Policy | Threshold | Accuracy | Full-Loop Rate | Avg Core Layers | Core Savings |
|---|---:|---:|---:|---:|---:|
| No agreement | 0.80 | 97.87% | 0.15% | 3.08 / 18 | 82.87% |
| No agreement | 0.90 | 98.47% | 0.28% | 3.18 / 18 | 82.34% |
| No agreement | 0.95 | 98.89% | 0.75% | 3.32 / 18 | 81.58% |
| Agreement | 0.80 | 99.51% | 1.06% | 3.23 / 18 | 82.05% |
| Agreement | 0.90 | 99.56% | 1.53% | 3.33 / 18 | 81.47% |
| Agreement | 0.95 | 99.59% | 2.33% | 3.48 / 18 | 80.65% |

Timing sweep:

| Batch Size | Full Teacher | Serial 0.90 | Serial 0.95 | Agreement 0.80 |
|---:|---:|---:|---:|---:|
| 1 | 25.94 ms | 9.45 ms | 9.83 ms | 18.95 ms |
| 2 | 25.39 ms | 14.10 ms | 14.75 ms | 23.94 ms |
| 4 | 25.58 ms | 15.51 ms | 15.37 ms | 27.88 ms |
| 8 | 24.82 ms | 17.58 ms | 19.37 ms | 29.79 ms |
| 16 | 26.25 ms | 20.75 ms | 22.89 ms | 38.66 ms |
| 32 | 26.58 ms | 24.53 ms | 28.02 ms | 44.81 ms |
| 64 | 34.45 ms | 31.19 ms | 33.77 ms | 50.44 ms |

Interpretation:

This is the cleanest seed-42 JumpRec result so far. The repaired teacher is
strong on a fresh 96-batch eval, and the direct 3-layer control is far too weak
to explain the result away as shallow shortcut learning. The fixed-budget
profile is also the expected recurrent shape: c0 is poor, c1 is useful, and c2
or c3 reaches teacher-like accuracy.

The no-agreement router is fast and compute-light but still leaves quality on
the table relative to the teacher. The agreement-filtered router reaches
teacher-level accuracy while using about 3.2-3.5 of 18 counted recurrent core
layers, but its current serial implementation is slower in the timing sweep.
So the result strengthens the quality/compute story, while keeping the
hardware-aware execution caveat intact.

Across the three strong hard-case teachers now available, the result is broadly
consistent: seeds 101 and 202 were solved by stratified replay, and seed 42
needed the second-stage polish. Before turning this into a headline, the next
teacher robustness step should use a wider or repeated validation gate, and the
next artifact step should run the audit checks before moving toward a
general-use looped LLM bridge.

## 2026-04-26 - 256-batch teacher robustness check

Commands:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_polish2_eval_teacher --seed 42
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_eval_teacher --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_eval_teacher --seed 202
```

These eval-only modes load existing checkpoints, do no training, and evaluate
the full recurrent teacher over 256 uniform-hop batches.

| Seed | Checkpoint | Full Acc | Hop 1 | Hop 2 | Hop 3 | Hop 4 | Worst Hop |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | `core3_8n4h_strathop_polish2_seed42` | 99.56% | 100.00% | 99.95% | 99.57% | 98.72% | 98.72% |
| 101 | `core3_8n4h_strathop_seed101` | 99.70% | 100.00% | 99.84% | 99.98% | 98.90% | 98.90% |
| 202 | `core3_8n4h_strathop_seed202` | 99.60% | 100.00% | 99.93% | 99.87% | 98.54% | 98.54% |

Loop profile:

| Seed | 0 Loops | 1 Loop | 2 Loops | 3 Loops | 4 Loops | 5 Loops | 6 Loops |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 15.96% | 40.81% | 59.32% | 77.55% | 99.57% | 99.63% | 99.56% |
| 101 | 12.55% | 41.26% | 59.59% | 78.14% | 99.66% | 99.72% | 99.70% |
| 202 | 14.67% | 41.14% | 60.03% | 78.64% | 99.50% | 99.58% | 99.60% |

Interpretation:

This closes the immediate teacher-robustness loop for the 8-node/4-hop hard
case. All three seeds clear 99.5% full accuracy on a wider uniform eval, and
all worst-hop scores are above 98.5%. The seed-42 repair no longer looks like a
small validation-slice artifact.

This does not rule out synthetic-task artifacts or shortcut learning. It does
mean the next blocker is no longer "can we get a robust recurrent teacher on
this hard case?" The next blocker is the artifact audit: information-leakage
checks, no-answer-token assertions, relabeling/scrambling controls, threshold
tuning hygiene, and execution claims tied to the stated batch regime.

## 2026-04-26 - prompt artifact audit, corrected relabel probe

Commands:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_polish2_audit_teacher --seed 42
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_audit_teacher --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_audit_teacher --seed 202
```

These eval-only modes load the three robust 8-node/4-hop teachers and run
prompt-level information probes. The first relabel attempt was superseded after
we noticed the relabeled map entries were not sorted by displayed source label.
Commit `7817ae4` fixes that by sorting relabeled map entries before printing
the prompt.

Prompt audit definitions:

- `normal`: standard prompt.
- `relabel`: consistently rename node symbols while preserving the same graph
  and target. Accuracy should stay stable.
- `map_scramble`: show a scrambled map while keeping the original target.
  Accuracy should collapse.
- `hop_random`: show the wrong hop count while keeping the original target.
  Accuracy should collapse.

| Seed | Checkpoint | Variant | Full Acc | Hop 1 | Hop 2 | Hop 3 | Hop 4 | Worst Hop |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 42 | `core3_8n4h_strathop_polish2_seed42` | normal | 99.61% | 100.00% | 99.94% | 99.60% | 98.88% | 98.88% |
| 42 | `core3_8n4h_strathop_polish2_seed42` | relabel | 99.50% | 100.00% | 100.00% | 99.43% | 98.62% | 98.62% |
| 42 | `core3_8n4h_strathop_polish2_seed42` | map_scramble | 14.50% | 11.45% | 13.77% | 13.52% | 18.84% | 11.45% |
| 42 | `core3_8n4h_strathop_polish2_seed42` | hop_random | 18.85% | 20.31% | 16.51% | 17.43% | 20.64% | 16.51% |
| 101 | `core3_8n4h_strathop_seed101` | normal | 99.62% | 100.00% | 99.83% | 100.00% | 98.60% | 98.60% |
| 101 | `core3_8n4h_strathop_seed101` | relabel | 99.77% | 100.00% | 99.80% | 100.00% | 99.31% | 99.31% |
| 101 | `core3_8n4h_strathop_seed101` | map_scramble | 15.15% | 11.53% | 14.00% | 13.87% | 20.50% | 11.53% |
| 101 | `core3_8n4h_strathop_seed101` | hop_random | 18.69% | 19.61% | 16.88% | 16.39% | 21.49% | 16.39% |
| 202 | `core3_8n4h_strathop_seed202` | normal | 99.60% | 100.00% | 99.93% | 99.73% | 98.76% | 98.76% |
| 202 | `core3_8n4h_strathop_seed202` | relabel | 99.49% | 100.00% | 100.00% | 99.91% | 98.03% | 98.03% |
| 202 | `core3_8n4h_strathop_seed202` | map_scramble | 15.14% | 11.86% | 14.33% | 15.26% | 19.50% | 11.86% |
| 202 | `core3_8n4h_strathop_seed202` | hop_random | 19.21% | 19.79% | 18.09% | 17.15% | 21.82% | 17.15% |

Interpretation:

The prompt shortcut check passes. Consistent symbol relabeling preserves
teacher performance across all three seeds, so the teachers are not merely
memorizing fixed node names or a brittle display order. Scrambling the displayed
map and randomizing the displayed hop count both collapse performance near
chance, which is the desired destructive-control behavior: the teacher is using
the map and hop fields causally.

This strengthens the information gate, but it is still a teacher-level audit.
The next audit step should apply the same spirit to JumpRec routing and verifier
hygiene: held-out threshold selection, verifier calibration, oracle-router
headroom, and an explicit statement that verifier inputs contain only proposed
state/logit uncertainty features available at deployment time.

## 2026-04-26 - verifier audit with held-out threshold selection

Commands:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_polish2_verifier_audit --seed 42
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_verifier_audit --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_verifier_audit --seed 202
```

Commit `72bcc88` adds this audit. It keeps the existing fixed thresholds, then
adds three verifier-gate checks:

- verifier calibration against held-out correctness labels;
- an oracle router upper bound, clearly labeled as non-deployable;
- a validation/final split for threshold selection. Thresholds are chosen on
  64 validation batches, then reported on a separate 128-batch final split. The
  validation selector allows at most a 0.25-point drop from validation teacher
  accuracy and then chooses the lowest average core-layer cost.

Fixed threshold sanity:

| Seed | Fixed Full Teacher | Fixed No-Agree 0.90 | No-Agree Layers | Fixed Agree 0.90 | Agree Layers |
|---:|---:|---:|---:|---:|---:|
| 42 | 99.51% | 98.75% | 3.16 / 18 | 99.51% | 3.29 / 18 |
| 101 | 99.78% | 99.30% | 2.35 / 18 | 99.78% | 2.40 / 18 |
| 202 | 99.60% | 99.37% | 3.28 / 18 | 99.83% | 3.36 / 18 |

Held-out threshold results:

| Seed | Policy | Selected Threshold | Final Teacher | Final Router | Avg Core Layers | Savings | Coverage | Accepted Precision |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | No agreement | 0.99 | 99.49% | 99.32% | 6.01 / 18 | 66.64% | 97.14% | 99.65% |
| 42 | Agreement | 0.50 | 99.49% | 99.30% | 3.09 / 18 | 82.85% | 99.37% | 99.39% |
| 101 | No agreement | 0.90 | 99.69% | 99.48% | 2.37 / 18 | 86.83% | 99.93% | 99.50% |
| 101 | Agreement | 0.50 | 99.69% | 99.79% | 2.33 / 18 | 87.07% | 99.87% | 99.80% |
| 202 | No agreement | 0.80 | 99.60% | 99.01% | 3.28 / 18 | 81.80% | 99.94% | 99.05% |
| 202 | Agreement | 0.50 | 99.60% | 99.66% | 3.24 / 18 | 81.99% | 99.55% | 99.69% |

Verifier calibration:

| Seed | Mean Verifier Conf | Empirical Correctness | Brier | ECE-10 |
|---:|---:|---:|---:|---:|
| 42 | 85.5% | 85.9% | 0.0308 | 0.0040 |
| 101 | 91.1% | 91.4% | 0.0216 | 0.0038 |
| 202 | 85.4% | 85.7% | 0.0331 | 0.0050 |

By-budget ECE-10:

| Seed | c0 | c1 | c2 | c3 |
|---:|---:|---:|---:|---:|
| 42 | 0.0103 | 0.0047 | 0.0067 | 0.0015 |
| 101 | 0.0128 | 0.0073 | 0.0037 | 0.0019 |
| 202 | 0.0147 | 0.0241 | 0.0034 | 0.0016 |

Oracle router upper bound:

| Seed | Oracle Acc | Avg Core Layers | Savings | Full-Loop Fallback Rate |
|---:|---:|---:|---:|---:|
| 42 | 99.98% | 2.67 / 18 | 85.19% | 0.24% |
| 101 | 99.96% | 2.01 / 18 | 88.82% | 0.12% |
| 202 | 99.95% | 2.69 / 18 | 85.07% | 0.07% |

Interpretation:

The verifier gate mostly passes. The verifier is well calibrated on held-out
examples, and the agreement router survives held-out threshold selection across
all three seeds while retaining about 82-87% counted recurrent-core savings.
That is a much cleaner claim than tuning a threshold on the reported eval set.

The caveat is also clear: no-agreement routing is less robust. Seed 202 passes
the validation selector but drops to 99.01% on the final split, about 0.59
points below the final teacher. That is still strong, but it should remain a
speed-oriented diagnostic policy rather than the promoted quality-preserving
policy.

The oracle router shows remaining headroom. If we could pick the first correct
JumpRec budget with oracle knowledge, all three seeds would land around
99.95-99.98% at 2.01-2.69 / 18 core layers. The deployable agreement router is
close, but not at the oracle ceiling; better verifier/routing objectives are
still worth exploring.

This does not prove general LLM transfer. It does make the synthetic JumpRec
claim substantially harder to dismiss: the teacher is robust, prompt shortcuts
are checked, verifier thresholds are held out, verifier calibration is measured,
and the best deployable policy remains dramatically cheaper in counted core
layers.

## 2026-04-26 - JumpRec prompt artifact audit

Commands:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_polish2_audit_teacher --seed 42
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_audit_teacher --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_audit_teacher --seed 202
```

Commit `c6ad7fe` extends the prompt artifact audit to the loaded JumpRec path.
This uses the same prompt variants as the teacher audit and reports the
deployable agreement router at threshold 0.90. The teacher column is included
as a sanity reference from the same run.

| Seed | Variant | Teacher Full Acc | JumpRec Agree 0.90 | Avg Core Layers | JumpRec No-Agree 0.90 | Fixed c3 |
|---:|---|---:|---:|---:|---:|---:|
| 42 | normal | 99.61% | 99.28% | 3.38 / 18 | 98.18% | 99.33% |
| 42 | relabel | 99.50% | 99.49% | 3.37 / 18 | 98.57% | 99.39% |
| 42 | map_scramble | 14.50% | 14.55% | 3.37 / 18 | 14.56% | 14.51% |
| 42 | hop_random | 18.85% | 18.93% | 3.29 / 18 | 19.12% | 18.90% |
| 101 | normal | 99.62% | 99.74% | 2.43 / 18 | 99.41% | 99.72% |
| 101 | relabel | 99.77% | 99.80% | 2.41 / 18 | 99.35% | 99.69% |
| 101 | map_scramble | 15.15% | 14.40% | 2.44 / 18 | 14.47% | 14.40% |
| 101 | hop_random | 18.69% | 19.49% | 2.41 / 18 | 19.49% | 19.49% |
| 202 | normal | 99.60% | 99.74% | 3.41 / 18 | 99.41% | 99.74% |
| 202 | relabel | 99.49% | 99.83% | 3.39 / 18 | 99.38% | 99.71% |
| 202 | map_scramble | 15.14% | 15.93% | 3.36 / 18 | 15.98% | 15.95% |
| 202 | hop_random | 19.21% | 18.66% | 3.46 / 18 | 18.66% | 18.69% |

Interpretation:

The deployable JumpRec path passes the prompt artifact audit. Consistent symbol
relabeling preserves routed JumpRec performance, while scrambling the displayed
map or lying about the hop count collapses both fixed-budget and routed JumpRec
near chance. That makes the information-gate story stronger than the
teacher-only audit: the shortcut checks now cover the accelerated path we would
actually claim.

The seed-42 no-agreement router is still visibly weaker than agreement routing
on normal/relabel prompts, which matches the held-out verifier audit. The
promotable policy remains agreement routing; no-agreement remains a speed
diagnostic.

## 2026-04-26 - seed-101 JumpRec component ablations

Commands:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_ablate_no_adapter --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_ablate_no_distill --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_ablate_no_verifier --seed 101
```

Commit `762c7cb` adds these modes. Each ablation loads the same seed-101
stratified teacher checkpoint but forces fresh JumpRec training instead of
reusing saved JumpRec weights.

| Mode | Temp Adapter | Distill Loss | Verifier Loss | Teacher | c0 | c1 | c2 | c3 | Agree 0.90 | Avg Core Layers | Savings | No-Agree 0.90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | yes | 0.2 | 0.2 | 99.79% | 68.67% | 97.64% | 99.74% | 99.77% | 99.77% | 2.40 / 18 | 86.66% | 99.43% |
| No adapter | no | 0.2 | 0.2 | 99.79% | 71.39% | 97.40% | 99.71% | 99.79% | 99.80% | 2.34 / 18 | 86.98% | 99.41% |
| No distill | yes | 0.0 | 0.2 | 99.79% | 69.29% | 97.48% | 99.69% | 99.72% | 99.79% | 2.41 / 18 | 86.63% | 99.24% |
| No verifier | yes | 0.2 | 0.0 | 99.79% | 69.50% | 96.81% | 99.76% | 99.71% | 99.58% | 18.00 / 18 | 0.00% | 99.58% |

Timing:

| Mode | Full Teacher | Serial No-Agree 0.90 | Serial Agreement 0.80 |
|---|---:|---:|---:|
| Baseline | 33.55 ms | 21.56 ms | 41.24 ms |
| No adapter | 34.57 ms | 24.05 ms | 45.28 ms |
| No distill | 34.94 ms | 27.68 ms | 58.40 ms |
| No verifier | 34.70 ms | 52.05 ms | 76.09 ms |

Interpretation:

On this seed-101 8-node/4-hop SmolLM hardcase, the verifier loss is the
essential ablation for adaptive savings. Without verifier supervision, the
fixed JumpRec budgets still learn the task, but the learned verifier never
routes confidently, so the deployable policies fall back to the full loop and
save no counted core layers.

The temporary adapter and distillation loss are not essential on this specific
setup: removing either one preserves fixed-budget accuracy and agreement-router
quality. This should be interpreted narrowly. Earlier 12-node/6-hop toy-runner
results showed the adapter mattered a lot under a harder max-hop edge and a
smaller correction budget. The current result says the adapter/distillation
benefits are task-regime dependent, not that they are useless.

For the paper-facing claim, this is useful: JumpRec is not just a pile of
ingredients where every piece must work by faith. The core jump-plus-tail
mechanism learns well even under removals, while verifier supervision is
clearly required for the adaptive compute-saving behavior.

## 2026-04-26 - seed-42/202 JumpRec component ablation replication

Commands:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_polish2_ablate_no_verifier --seed 42
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_ablate_no_verifier --seed 202
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_polish2_ablate_no_adapter --seed 42
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_ablate_no_adapter --seed 202
```

The runner now has explicit `core3_8n4h_strathop_polish2_ablate_*` modes so
seed 42 loads the repaired teacher checkpoint
`core3_8n4h_strathop_polish2_seed42` instead of the weak original stratified
checkpoint. Seeds 101 and 202 continue to use the original strong
`core3_8n4h_strathop_seed{seed}` checkpoints.

| Seed | Mode | Teacher | c0 | c1 | c2 | c3 | Agree 0.90 | Avg Core Layers | Savings | No-Agree 0.90 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | No verifier | 99.53% | 48.00% | 96.47% | 99.30% | 99.61% | 99.56% | 18.00 / 18 | 0.00% | 99.56% |
| 202 | No verifier | 99.53% | 47.98% | 94.30% | 99.37% | 99.76% | 99.54% | 18.00 / 18 | 0.00% | 99.54% |
| 42 | No adapter | 99.53% | 47.88% | 95.83% | 99.17% | 99.41% | 99.50% | 3.41 / 18 | 81.04% | 98.70% |
| 202 | No adapter | 99.53% | 48.08% | 93.95% | 99.33% | 99.67% | 99.80% | 3.46 / 18 | 80.78% | 99.17% |

Timing:

| Seed | Mode | Full Teacher | Serial No-Agree 0.90 | Serial Agreement 0.80 |
|---:|---|---:|---:|---:|
| 42 | No verifier | 34.45 ms | 52.66 ms | 76.80 ms |
| 202 | No verifier | 34.80 ms | 50.31 ms | 73.12 ms |
| 42 | No adapter | 33.74 ms | 31.93 ms | 48.15 ms |
| 202 | No adapter | 34.57 ms | 31.11 ms | 61.69 ms |

Interpretation:

The seed-101 ablation story replicates on seeds 42 and 202. Removing verifier
supervision does not prevent fixed-budget JumpRec from learning: c2/c3 still
reach teacher-level accuracy. It does break adaptive compute, because the
deployable verifier/router falls back to the full recurrent teacher at every
reported threshold, yielding 18.00 / 18 core layers and no counted savings.

Removing the temporary adapter does not break this current SmolLM 8-node/4-hop
regime. Agreement routing at threshold 0.90 remains teacher-level on both
additional seeds while using about 3.4 / 18 counted core layers. No-agreement
routing is weaker, especially on seed 42, which is consistent with the earlier
held-out verifier audit.

The narrow conclusion is now stronger: in the current hardcase setup, verifier
loss is load-bearing for adaptive savings, while the temporary adapter is not.
This should not be overgeneralized to harder task regimes; older 12-node/6-hop
toy-runner evidence still suggests the adapter can matter when the correction
budget is tighter.

## 2026-04-26 - seed-101 controller objective first pass

Commands:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_cost_controller --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_calib_controller --seed 101
```

Commit/worktree note: this was run from a local worktree adding two controller
variants. `cost_controller` adds false-accept weighting, positive weighting for
cheaper correct budgets, and a first-good ranking loss. `calib_controller`
keeps the correctness BCE unweighted and adds only a small ranking loss against
earlier wrong budgets.

Held-out threshold audit on seed 101:

| Mode | Policy | Selected Threshold | Final Teacher | Final Router | Avg Core Layers | Savings | Accepted Precision | ECE-10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline verifier | No agreement | 0.90 | 99.69% | 99.48% | 2.37 / 18 | 86.83% | 99.50% | 0.0038 |
| Baseline verifier | Agreement | 0.50 | 99.69% | 99.79% | 2.33 / 18 | 87.07% | 99.80% | 0.0038 |
| Cost-weighted controller | No agreement | 0.97 | 99.66% | 99.38% | 3.41 / 18 | 81.06% | 99.60% | 0.0305 |
| Cost-weighted controller | Agreement | 0.50 | 99.66% | 99.66% | 2.32 / 18 | 87.11% | 99.68% | 0.0305 |
| Calibrated ranking controller | No agreement | 0.97 | 99.66% | 99.38% | 2.60 / 18 | 85.53% | 99.50% | 0.0094 |
| Calibrated ranking controller | Agreement | 0.50 | 99.66% | 99.61% | 2.33 / 18 | 87.04% | 99.66% | 0.0094 |

Fixed-budget JumpRec still learned in both variants. For the calibrated
ranking controller, c0/c1/c2/c3 were 71.35%, 96.08%, 99.46%, and 99.67%.

Interpretation:

This first controller-objective pass is diagnostic, not an improvement. The
cost-weighted objective pushed the scalar verifier away from well-calibrated
correctness estimates. The held-out selector responded by raising the
no-agreement threshold to 0.97, which preserved accepted precision but spent
more compute and lost accuracy versus the baseline verifier. The calibrated
ranking-only variant was less damaging to calibration and recovered most of the
compute profile, but still did not beat the baseline on seed 101.

The lesson is that the existing scalar verifier is already strong on this seed,
and simply reshaping its BCE objective is probably not the main path to the
oracle-router gap. The next controller attempt should be a distinct budget
controller trained on oracle-router traces, or another architecture that
predicts the first sufficient budget while keeping the correctness verifier
calibrated for fallback decisions.

## 2026-04-26 - seed-101 separate budget controller

Command:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_budget_controller --seed 101
```

This mode loads the existing seed-101 stratified teacher plus baseline JumpRec
checkpoint, freezes the teacher/JumpRec/verifiers, and trains only a small
budget controller from deployment-time `state0` features. The budget label is
the first JumpRec correction budget whose prediction is correct, or fallback if
no JumpRec budget is correct. At inference, the controller selects one budget;
the existing calibrated verifier accepts that proposed path or falls back to
the full recurrent teacher.

Held-out threshold audit:

| Policy | Selected Threshold | Final Teacher | Final Router | Avg Core Layers | Savings | Coverage | Accepted Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| No agreement scan | 0.90 | 99.76% | 99.57% | 2.32 / 18 | 87.10% | - | - |
| Agreement scan | 0.50 | 99.76% | 99.82% | 2.30 / 18 | 87.23% | - | - |
| Budget controller | 0.85 | 99.76% | 99.55% | 4.23 / 18 | 76.48% | 87.00% | 99.69% |

Budget-controller diagnostics:

| Metric | Value |
|---|---:|
| Budget target accuracy | 76.73% |
| Predicted fallback rate | 0.00% |
| Target fallback rate | 0.06% |
| Avg predicted tail loops | 0.37 |
| Avg target tail loops | 0.34 |
| Oracle router avg core layers | 2.02 / 18 |
| Verifier ECE-10 | 0.0064 |

Timing highlights:

| Batch Size | Full Teacher | Serial No-Agree 0.90 | Budget 0.90 | Agreement 0.80 |
|---:|---:|---:|---:|---:|
| 1 | 18.61 ms | 6.28 ms | 6.32 ms | 11.53 ms |
| 64 | 33.83 ms | 23.01 ms | 24.97 ms | 39.64 ms |

Interpretation:

The separate budget controller is architecturally cleaner than reshaping the
verifier, but this first version is not an improvement. It preserves verifier
calibration and avoids agreement-scan overhead, but the one-shot policy has a
bad failure mode: if it predicts an early budget that the verifier rejects, it
falls directly to the full loop instead of trying the next likely sufficient
budget. That raises counted core layers to 4.23 / 18, behind the baseline scan
policies at about 2.3 / 18.

The controller did learn a meaningful target: predicted average tail loops
0.37 versus target 0.34, with 76.7% exact budget accuracy. The problem is not
that `state0` contains no budget signal. The problem is that exact first-budget
classification is too brittle for a one-shot router; underprediction is much
more expensive than overprediction because rejection goes to full fallback.

Next budget-controller variant should either:

- train with an asymmetric ordinal loss that penalizes underestimating the
  needed budget more than overestimating it; or
- allow one cheap escalation from predicted budget to predicted budget + 1
  before full fallback, and count that as a distinct deployable policy.

Do not seed-confirm this exact budget-controller variant unless it is needed as
a negative baseline.

## 2026-04-27 - seed-101 controller policy sweep

Commands:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_budget_controller_reuse --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_budget_verifytarget --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_budget_verifytarget_reuse --seed 101
```

This pass tested three follow-ups to the first budget-controller result:

- one-step escalation from predicted budget to predicted budget + 1 before full fallback;
- verifier-aware budget targets, where the controller predicts the first budget
  that is both correct and acceptable to the existing verifier gate;
- per-budget verifier thresholds, including a monotone conservative variant.

Held-out final results on seed 101:

| Variant | Policy | Selected Threshold(s) | Final Router | Avg Core Layers | Coverage | Accepted Precision |
|---|---|---:|---:|---:|---:|---:|
| Baseline verifier | No agreement scan | 0.90 | 99.48% | 2.37 / 18 | 99.93% | 99.50% |
| Baseline verifier | Agreement scan | 0.50 | 99.79% | 2.33 / 18 | 99.87% | 99.80% |
| First-correct controller | One-shot budget | 0.90 | 99.48% | 4.26 / 18 | 86.94% | 99.65% |
| First-correct controller | One-step escalation | 0.97 | 99.68% | 2.84 / 18 | 98.11% | 99.78% |
| First-correct controller | Open-loop budget | n/a | 90.88% | 2.14 / 18 | 100.00% | 90.88% |
| First-correct controller | Scan upward from prediction | 0.90 | 99.46% | 2.58 / 18 | 99.93% | 99.49% |
| Verifier-target controller | One-shot budget | 0.97 | 99.62% | 3.46 / 18 | 92.96% | 99.74% |
| Verifier-target controller | One-step escalation | 0.97 | 99.67% | 2.75 / 18 | 99.80% | 99.74% |
| Verifier-target controller | Open-loop budget | n/a | 97.66% | 2.50 / 18 | 100.00% | 97.66% |
| Verifier-target controller | Scan upward from prediction | 0.90 | 99.46% | 2.65 / 18 | 99.93% | 99.49% |
| Per-budget thresholds | Unconstrained no-agree | 0.5,0.9,0.85,0.5 | 99.27% | 2.35 / 18 | 99.99% | 99.28% |
| Per-budget thresholds | Monotone no-agree | 0.9,0.9,0.6,0.5 | 99.43% | 2.36 / 18 | 99.99% | 99.44% |

Budget-controller diagnostics:

| Controller Target | Exact Target Acc | Under Rate | Over Rate | Avg Pred Tail | Avg Target Tail |
|---|---:|---:|---:|---:|---:|
| First correct | 76.65% | 9.40% | 13.95% | 0.37 | 0.34 |
| First verifier-acceptable | 85.28% | 4.94% | 9.78% | 0.50 | 0.46 |

Interpretation:

The state0 budget controller is learning real routing signal, and the
verifier-aware target is clearly better as a supervised objective than "first
correct." However, the deployable policies still do not beat the simple
calibrated serial verifier scan. Open-loop budget routing is too inaccurate.
One-step escalation and scan-up recover quality but spend more counted compute
than the global-threshold serial scan. Per-budget thresholds looked attractive
on the validation split but overfit: the unconstrained search chose an unsafe
low budget-0 threshold, and the monotone constrained search still lost final
accuracy for only a tiny cost gain.

On seed 101, the current best wall-clock-oriented deployable policy remains the
calibrated no-agreement serial verifier at threshold 0.90, with agreement
routing retained as the higher-quality diagnostic/scientific policy. The
budget-controller work should be treated as a negative/diagnostic branch unless
a later architecture can use richer deployment-time features than `state0`
alone or train the verifier/controller jointly against the actual routing
policy. Cross-seed timing below makes the broader scaling story more nuanced.

## 2026-04-27 - cross-seed verifier timing hygiene

Commands:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_polish2_verifier_audit --seed 42
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_verifier_audit --seed 202
```

This reran the verifier audit modes after adding batch-size timing for the
actually selected families: no-agree threshold 0.99 and agreement threshold
0.50. This matters because the earlier timing only included agreement 0.80 and
no-agree up to 0.95.

Held-out final routing:

| Seed | Policy | Selected Threshold | Final Teacher | Final Router | Avg Core Layers | Savings | Accepted Precision |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | No agreement | 0.99 | 99.49% | 99.32% | 6.01 / 18 | 66.64% | 99.65% |
| 42 | Agreement | 0.50 | 99.49% | 99.30% | 3.09 / 18 | 82.85% | 99.39% |
| 202 | No agreement | 0.80 | 99.60% | 99.01% | 3.28 / 18 | 81.80% | 99.05% |
| 202 | Agreement | 0.50 | 99.60% | 99.66% | 3.24 / 18 | 81.99% | 99.69% |

Wall-clock timing highlights:

| Seed | Batch | Full Teacher | No-Agree Selected-ish | Agreement 0.50 |
|---:|---:|---:|---:|---:|
| 42 | 1 | 26.39 ms | 20.16 ms at 0.99 | 25.85 ms |
| 42 | 64 | 34.97 ms | 46.14 ms at 0.99 | 57.16 ms |
| 202 | 1 | 20.43 ms | 7.58 ms at 0.80 | 18.38 ms |
| 202 | 64 | 33.80 ms | 25.05 ms at 0.80 | 44.82 ms |

Interpretation:

The scaling story is more nuanced than "no-agree is the keeper." On seed 202,
no-agreement routing is the clear wall-clock path but loses more accuracy than
agreement. On seed 42, held-out selection must push no-agreement to threshold
0.99 to stay near teacher quality; that makes counted compute and batch-64
wall-clock much worse. Agreement remains the most robust quality-preserving
policy across seeds, but its current implementation has poor wall-clock scaling
because it evaluates adjacent budgets.

The practical keeper is therefore split:

- agreement routing is the robust scientific quality reference;
- no-agreement routing is the scalable candidate when calibration is strong
  enough for the seed/task regime;
- the unsolved target is still a quality-preserving router with no-agreement-like
  wall-clock behavior.

The negative budget-controller results narrow the search: the next router
should not be another `state0`-only exact-budget classifier. It should either
improve the verifier features available after each cheap candidate, train
against the actual accept/fallback policy, or change the execution path so
agreement-style stability can be checked without a second full budget pass.

## 2026-04-27 - learned stability router

Commands:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_stability_router --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_polish2_stability_router --seed 42
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_stability_router --seed 202
```

This pass tested a cheap learned approximation to adjacent-budget agreement.
Each JumpRec correction budget gets a frozen-post-hoc stability head. The head
sees the same deployment-time verifier features for the current candidate and
is trained to predict whether the current candidate prediction matches the next
correction budget prediction. At inference, the `stable_*` policies require the
normal verifier gate plus predicted stability, but do not execute the adjacent
budget to check agreement.

Stability-head training telemetry at step 2000:

| Seed | Mode | Stability Acc | Pred Stable | Target Stable | Pred-Stable Precision |
|---:|---|---:|---:|---:|---:|
| 101 | `strathop_stability_router` | 96.5% | 59.4% | 61.3% | 98.7% |
| 42 | `strathop_polish2_stability_router` | 93.0% | 53.5% | 55.1% | 94.9% |
| 202 | `strathop_stability_router` | 94.1% | 54.7% | 57.4% | 97.1% |

Held-out final routing:

| Seed | Policy | Selected Gate | Final Teacher | Final Router | Avg Core Layers | Savings | Coverage | Accepted Precision |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 101 | No agreement | verifier 0.90 | 99.76% | 99.57% | 2.32 / 18 | 87.10% | 99.99% | 99.57% |
| 101 | Agreement | verifier 0.50 | 99.76% | 99.82% | 2.30 / 18 | 87.23% | 99.85% | 99.83% |
| 101 | Stable 0.50/0.70 | verifier 0.95 | 99.76% | 99.69% | 2.38 / 18 | 86.79% | 99.72% | 99.76% |
| 42 | No agreement | verifier 0.99 | 99.40% | 99.23% | 6.00 / 18 | 66.64% | 97.14% | 99.62% |
| 42 | Agreement | verifier 0.50 | 99.40% | 99.29% | 3.09 / 18 | 82.85% | 99.30% | 99.34% |
| 42 | Stable 0.50/0.70/0.90 | verifier 0.99 | 99.40% | 99.19% | 7.78 / 18 | 56.80% | 75.00% | 99.66% |
| 202 | No agreement | verifier 0.90 | 99.65% | 99.27% | 3.36 / 18 | 81.35% | 99.94% | 99.27% |
| 202 | Agreement | verifier 0.50 | 99.65% | 99.66% | 3.27 / 18 | 81.85% | 99.63% | 99.67% |
| 202 | Stable 0.70 | verifier 0.95 | 99.65% | 99.45% | 3.49 / 18 | 80.62% | 99.27% | 99.48% |

Batch-64 wall-clock timing highlights:

| Seed | Full Teacher | No-Agree Selected | Agreement 0.50 | Stable Timing Highlight |
|---:|---:|---:|---:|---:|
| 101 | 35.01 ms | 26.54 ms at 0.90 | 43.57 ms | 27.27 ms for stable0.50/verifier0.90 |
| 42 | 34.19 ms | 44.97 ms at 0.99 | 55.37 ms | 30.62 ms for stable0.50/verifier0.80 |
| 202 | 43.28 ms | 37.42 ms at 0.90 | 50.76 ms | 31.85 ms for stable0.70/verifier0.90 |

Interpretation:

The learned stability signal is real: the heads learn adjacent-budget agreement
with high predicted-stable precision, and the stable timing path can be close to
no-agreement latency because it avoids the second budget pass. However, as a
deployable router it is not good enough yet. On seed 101 it lands between
no-agreement and true agreement. On seed 202, stable 0.70 improves over
no-agreement quality but still trails agreement while spending more counted
compute. On seed 42, the learned stability gate is a regression: validation
selection drives it to verifier 0.99, coverage collapses to 75%, and average
core layers rise to 7.78 / 18.

Conclusion:

Learned stability is a useful diagnostic and a promising feature, but not a
standalone gate. The next router should not simply stack verifier confidence and
stability as independent thresholds. Better next bets are:

- feed predicted stability into a unified verifier/halting head;
- train the halting decision against actual route utility, including false
  accepts, fallbacks, and counted cost;
- use agreement-style labels as supervision, but calibrate the deployed policy
  for correctness and wall-clock, not just adjacent-budget consistency.

## 2026-04-27 - utility and next-agreement router probes

Commands:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_utility_router --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_utility_stability_router --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_nextagree_router --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_nextagree_router_reuse --seed 101
```

This pass tested two post-hoc learned alternatives to the hand-built router:

- a utility router trained with a differentiable expected route loss over
  accept/fallback outcomes, false accepts, and counted core cost;
- a next-agreement proxy head trained to predict the next correction budget's
  predicted class from the current budget features, then accept when that
  predicted next class matches the current class.

Seed-101 held-out final results:

| Variant | Policy | Selected Gate | Final Router | Avg Core Layers | Coverage | Accepted Precision |
|---|---|---:|---:|---:|---:|---:|
| Utility only | No agreement | verifier 0.90 | 99.57% | 2.32 / 18 | 99.99% | 99.57% |
| Utility only | Agreement | verifier 0.10 | 99.78% | 2.26 / 18 | 99.89% | 99.78% |
| Utility only | Utility | utility 0.20 | 99.57% | 2.34 / 18 | 99.83% | 99.63% |
| Utility + stability feature | Stable 0.70 | verifier 0.95 | 99.69% | 2.38 / 18 | 99.72% | 99.76% |
| Utility + stability feature | Utility | utility 0.10 | 99.55% | 2.32 / 18 | 99.91% | 99.58% |
| Next-agreement proxy | Nextagree 0.50 | verifier 0.10 | 99.76% | 17.90 / 18 | 0.95% | 100.00% |

The utility router did not improve the Pareto frontier. It learned a plausible
accept policy, but it landed near the no-agreement verifier rather than the
agreement router. Feeding the learned stability logit into the utility router
also did not help; it slightly worsened the utility policy and still trailed the
true agreement reference.

The next-agreement proxy initially looked over-conservative because its learned
next-class confidence was low. A reuse audit expanded the proxy confidence grid
downward:

| Next-Agreement Confidence | Final Router | Avg Core Layers | Coverage | Accepted Precision |
|---:|---:|---:|---:|---:|
| 0.00 / 0.10 | 99.43% | 13.29 / 18 | 32.80% | 99.00% |
| 0.15 | 99.49% | 14.19 / 18 | 28.50% | 99.06% |
| 0.20 | 99.60% | 15.44 / 18 | 20.85% | 99.24% |
| 0.30 | 99.68% | 17.19 / 18 | 7.30% | 99.33% |
| 0.50 | 99.69% | 17.89 / 18 | 0.99% | 100.00% |

Lowering the next-agreement confidence threshold recovers coverage, but only by
accepting too many weak candidates and spending most of the full compute anyway.
Keeping the threshold high preserves quality, but falls back almost always. That
makes the proxy useful diagnostically, not deployable.

Timing highlights:

| Run | Batch | Full Teacher | No-Agree 0.90 | Agreement 0.50 | Learned Router Highlight |
|---|---:|---:|---:|---:|---:|
| Utility only | 64 | 33.92 ms | 22.77 ms | 41.96 ms | Utility 0.50: 29.07 ms |
| Utility + stability | 64 | 34.43 ms | 23.75 ms | 45.10 ms | Utility 0.50: 25.94 ms |
| Nextagree high-conf | 64 | 33.97 ms | 23.53 ms | 41.72 ms | Nextagree 0.50/0.90: 39.97 ms |
| Nextagree low-conf reuse | 64 | 45.57 ms | 37.84 ms | 65.60 ms | Nextagree 0.00/0.90: 55.41 ms |

Do not over-read cross-run timing deltas because Modal performance varied, but
the direction is enough: none of the learned post-hoc routers combines
agreement-level quality with no-agreement-like wall-clock behavior.

One positive incidental finding: widening the verifier threshold grid let true
agreement select verifier threshold 0.10 on seed 101, improving the counted
quality/cost point to about 99.78% at 2.26 / 18 core layers. This reinforces
that adjacent-budget stability is a strong signal. The failure is not the
stability idea; the failure is trying to approximate it after the fact with
small frozen heads and independent thresholds.

Conclusion:

Treat utility, utility-plus-stability, and next-agreement proxy routing as
negative post-hoc branches. The next serious attempt should move the halting
signal into training, closer to early-exit / LayerSkip / PonderNet style
supervision: train the candidate predictor and halting/verifier together for
deployment utility, rather than freezing JumpRec and learning another small
gate afterward.

## 2026-04-27 - joint halting seed-101 probe

Command:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt --seed 101
```

This mode loads the existing seed-101 stratified teacher plus baseline JumpRec
checkpoint, then unfreezes the JumpRec candidate path, verifier features, and
utility halting head together. The joint loss combines:

- differentiable expected route utility over accept/fallback outcomes;
- per-candidate cross-entropy to the task label;
- distillation toward the full recurrent teacher;
- verifier correctness BCE;
- optional stability supervision when a stability head is enabled.

This is the first router attempt that changes the candidates themselves instead
of scoring a frozen JumpRec path after the fact.

Joint-halt training telemetry:

| Step | Loss | Route | CE | Distill | Verifier | Aux | Expected Wrong | Expected Core | 0.5-Gate Avg Core |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 0.3826 | 0.0609 | 0.3311 | 0.3308 | 0.1340 | 0.0269 | 0.0197 | 0.1815 | 3.02 |
| 1000 | 0.3516 | 0.0711 | 0.2744 | 0.2724 | 0.1466 | 0.0328 | 0.0309 | 0.1432 | 2.41 |
| 1500 | 0.3008 | 0.0381 | 0.2912 | 0.2917 | 0.0597 | 0.0092 | 0.0042 | 0.1528 | 2.59 |
| 2000 | 0.3749 | 0.0456 | 0.3496 | 0.3495 | 0.1128 | 0.0209 | 0.0071 | 0.1659 | 2.97 |

The training minibatch 0.5-gate metrics stayed at 100% accuracy/precision in
these logs, but those are only online training diagnostics. The held-out router
selection below is the relevant result.

Held-out final routing:

| Policy | Selected Gate | Final Teacher | Final Router | Avg Core Layers | Savings | Coverage | Accepted Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| No agreement | verifier 0.70 | 99.76% | 99.68% | 2.21 / 18 | 87.71% | 99.98% | 99.68% |
| Agreement | verifier 0.10 | 99.76% | 99.90% | 2.20 / 18 | 87.79% | 99.89% | 99.94% |
| Joint utility | utility 0.10 | 99.76% | 99.84% | 2.24 / 18 | 87.55% | 99.96% | 99.85% |
| Joint utility guarded | utility 0.10 | 99.76% | 99.84% | 2.24 / 18 | 87.55% | 99.96% | 99.85% |

The important movement is the utility policy. The previous post-hoc utility
router on this seed selected utility 0.20 and landed at 99.57% final accuracy
using 2.34 / 18 core layers. Joint halting moves the utility route to 99.84%
while using 2.24 / 18 core layers. It still trails true agreement slightly, but
it now sits much closer to the desired quality/cost frontier while preserving a
one-candidate utility timing path.

Timing highlights from the batch-size sweep:

| Batch | Full Teacher | No-Agree 0.90 | Agreement 0.50 | Utility 0.50 | Utility 0.80 | Utility 0.90 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 41.70 ms | 12.04 ms | 29.16 ms | 13.32 ms | 13.28 ms | 15.36 ms |
| 64 | 47.84 ms | 31.67 ms | 61.92 ms | 36.88 ms | 34.84 ms | 38.37 ms |

Interpretation:

This is a positive first probe. It supports the hypothesis from the post-hoc
failures: the halting policy should be trained with the candidate path, not
bolted onto a frozen JumpRec model. The current result is not seed-confirmed
yet, and Modal timing varied enough that quality should be weighted more than
single-run timing. The next credibility gate is cross-seed confirmation on seed
202 and repaired polish2 seed 42. If those hold, joint halting becomes the main
router path.

## 2026-04-27 - joint halting cross-seed check

Commands:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt --seed 202
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_polish2_joint_halt --seed 42
```

After the seed-101 run, the same joint-halting recipe was checked on seed 202
and repaired polish2 seed 42. The quality result seed-confirms the direction,
but with an important caveat: joint utility is now strong and often
teacher-level, yet it still does not replace true agreement on quality.

Held-out final routing from the training runs:

| Seed | Policy | Selected Gate | Final Teacher | Final Router | Avg Core Layers | Savings | Coverage | Accepted Precision |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 101 | No agreement | verifier 0.70 | 99.76% | 99.68% | 2.21 / 18 | 87.71% | 99.98% | 99.68% |
| 101 | Agreement | verifier 0.10 | 99.76% | 99.90% | 2.20 / 18 | 87.79% | 99.89% | 99.94% |
| 101 | Joint utility | utility 0.10 | 99.76% | 99.84% | 2.24 / 18 | 87.55% | 99.96% | 99.85% |
| 42 | No agreement | verifier 0.95 | 99.40% | 99.18% | 2.95 / 18 | 83.61% | 99.83% | 99.27% |
| 42 | Agreement | verifier 0.10 | 99.40% | 99.52% | 2.89 / 18 | 83.95% | 99.88% | 99.56% |
| 42 | Joint utility | utility 0.10 | 99.40% | 99.41% | 3.06 / 18 | 83.00% | 99.50% | 99.58% |
| 202 | No agreement | verifier 0.85 | 99.65% | 99.17% | 3.20 / 18 | 82.25% | 99.79% | 99.22% |
| 202 | Agreement | verifier 0.10 | 99.65% | 99.78% | 3.09 / 18 | 82.84% | 99.46% | 99.83% |
| 202 | Joint utility | utility 0.10 | 99.65% | 99.62% | 3.31 / 18 | 81.63% | 99.71% | 99.69% |

Mean over seeds:

| Policy | Mean Router Acc | Mean Avg Core Layers | Mean Coverage | Mean Accepted Precision |
|---|---:|---:|---:|---:|
| Full teacher | 99.60% | 18.00 / 18 | - | - |
| No agreement | 99.34% | 2.79 / 18 | 99.87% | 99.39% |
| Agreement | 99.74% | 2.73 / 18 | 99.77% | 99.89% |
| Joint utility | 99.63% | 2.87 / 18 | 99.72% | 99.71% |

Interpretation:

Joint halting is real, but the current utility head is not the final router.
It lifted the learned utility policy from the post-hoc failure mode into the
teacher-quality neighborhood, and it beats the mean full teacher by a small
amount while using about 2.87 / 18 counted core layers. True agreement still
wins the quality/cost table at 99.74% and 2.73 / 18. The architectural target is
therefore sharper now: keep the joint-trained candidate improvements, then make
the one-candidate utility head learn more of the agreement signal without
running the adjacent budget.

Selected-threshold timing audit:

After the cross-seed runs, the timing harness was patched to include the actual
selected gates: no-agree 0.70/0.85, agreement 0.10, and utility 0.10. Checkpoint
reuse runs were then launched:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_reuse --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_polish2_joint_halt_reuse --seed 42
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_reuse --seed 202
```

These reuse runs skip training, so their held-out quality streams differ from
the training runs above. Use them as timing-only evidence.

| Seed | Batch | Full Teacher | Agreement 0.10 | Utility 0.10 |
|---:|---:|---:|---:|---:|
| 101 | 1 | 74.73 ms | 32.85 ms | 20.06 ms |
| 101 | 64 | 90.66 ms | 151.04 ms | 86.78 ms |
| 42 | 1 | 26.97 ms | 22.96 ms | 10.13 ms |
| 42 | 64 | 34.23 ms | 42.25 ms | 35.52 ms |
| 202 | 1 | 25.08 ms | 18.20 ms | 10.26 ms |
| 202 | 64 | 34.93 ms | 53.17 ms | 33.03 ms |

Timing remains noisy, especially the seed-101 reuse run. The stable directional
finding is still useful: selected agreement is expensive because it evaluates
adjacent budgets, while selected utility keeps the intended one-candidate shape
and is the plausible wall-clock path if its quality can be tightened.

Next probe:

Run a stability-augmented joint-halting variant where the stability head is
trained jointly and its logit is fed into the utility head. Earlier post-hoc
stability was not enough, but the cross-seed joint-halting result suggests
agreement-style information may be more useful when it is part of the joint
candidate/halting objective.

## 2026-04-27 - stability-augmented joint halting seed-101 probe

Command:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_stability --seed 101
```

This variant keeps the joint candidate/verifier/utility training loop, adds the
stability head to the joint loss, and feeds the predicted stability logit into
the utility router. The goal is to learn some of true agreement's safety signal
without paying for an adjacent-budget candidate at inference.

Held-out final routing:

| Policy | Selected Gate | Final Teacher | Final Router | Avg Core Layers | Savings | Coverage | Accepted Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| No agreement | verifier 0.50 | 99.76% | 99.62% | 2.20 / 18 | 87.75% | 99.98% | 99.63% |
| Agreement | verifier 0.10 | 99.76% | 99.91% | 2.20 / 18 | 87.77% | 99.93% | 99.95% |
| Stable 0.50 | stability 0.10 | 99.76% | 99.58% | 2.19 / 18 | 87.81% | 100.00% | 99.58% |
| Stable 0.70 | stability 0.10 | 99.76% | 99.68% | 2.21 / 18 | 87.72% | 99.99% | 99.69% |
| Stable 0.90 | stability 0.10 | 99.76% | 99.78% | 2.23 / 18 | 87.62% | 99.96% | 99.80% |
| Joint stability utility | utility 0.10 | 99.76% | 99.89% | 2.26 / 18 | 87.47% | 99.91% | 99.93% |
| Joint stability utility guarded | utility 0.10 | 99.76% | 99.89% | 2.26 / 18 | 87.47% | 99.91% | 99.93% |

Compared with plain joint utility on seed 101, the stability-fed utility route
improves final accuracy from 99.84% to 99.89%, while counted core rises only
from 2.24 / 18 to 2.26 / 18. It remains just below true agreement on quality,
but the gap is now about 0.024 percentage points on this seed and the router
keeps the one-candidate inference shape.

Selected timing from the batch-size sweep:

| Batch | Full Teacher | Agreement 0.10 | Utility 0.10 | Stable 0.70/0.90 |
|---:|---:|---:|---:|---:|
| 1 | 26.06 ms | 17.19 ms | 9.76 ms | 10.30 ms |
| 2 | 25.91 ms | 22.75 ms | 12.28 ms | 11.45 ms |
| 4 | 26.39 ms | 27.92 ms | 14.59 ms | 14.00 ms |
| 8 | 28.41 ms | 29.59 ms | 14.10 ms | 14.46 ms |
| 16 | 27.24 ms | 36.27 ms | 18.55 ms | 18.26 ms |
| 32 | 30.94 ms | 30.85 ms | 18.58 ms | 18.58 ms |
| 64 | 34.62 ms | 41.33 ms | 23.79 ms | 24.07 ms |

Interpretation:

This is the best single-seed deployable router result so far. True agreement
still defines the quality reference, but stability-fed joint utility nearly
matches it while avoiding agreement's adjacent-budget pass. This is strong
enough to justify cross-seed stability runs on repaired polish2 seed 42 and
seed 202 before changing the router design again.

## 2026-04-27 - stability-augmented joint halting cross-seed check

Commands:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_polish2_joint_halt_stability --seed 42
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_stability --seed 202
```

The seed-101 improvement did not fully seed-confirm. Stability-fed utility is
still viable, but it is a small average improvement rather than a new frontier.

Held-out final routing from the stability runs:

| Seed | Policy | Selected Gate | Final Teacher | Final Router | Avg Core Layers | Savings | Coverage | Accepted Precision |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 101 | No agreement | verifier 0.50 | 99.76% | 99.62% | 2.20 / 18 | 87.75% | 99.98% | 99.63% |
| 101 | Agreement | verifier 0.10 | 99.76% | 99.91% | 2.20 / 18 | 87.77% | 99.93% | 99.95% |
| 101 | Stability utility | utility 0.10 | 99.76% | 99.89% | 2.26 / 18 | 87.47% | 99.91% | 99.93% |
| 42 | No agreement | verifier 0.99 | 99.40% | 99.23% | 2.99 / 18 | 83.38% | 99.78% | 99.34% |
| 42 | Agreement | verifier 0.10 | 99.40% | 99.55% | 2.90 / 18 | 83.90% | 99.85% | 99.56% |
| 42 | Stability utility | utility 0.10 | 99.40% | 99.38% | 3.05 / 18 | 83.05% | 99.63% | 99.50% |
| 202 | No agreement | verifier 0.95 | 99.65% | 99.41% | 3.24 / 18 | 82.01% | 99.83% | 99.45% |
| 202 | Agreement | verifier 0.10 | 99.65% | 99.74% | 3.06 / 18 | 82.98% | 99.63% | 99.79% |
| 202 | Stability utility | utility 0.10 | 99.65% | 99.63% | 3.31 / 18 | 81.63% | 99.77% | 99.68% |

Mean over seeds:

| Policy | Mean Router Acc | Mean Avg Core Layers | Mean Coverage | Mean Accepted Precision |
|---|---:|---:|---:|---:|
| Full teacher | 99.60% | 18.00 / 18 | - | - |
| No agreement | 99.42% | 2.81 / 18 | 99.86% | 99.47% |
| Agreement | 99.74% | 2.72 / 18 | 99.80% | 99.77% |
| Stability utility | 99.63% | 2.87 / 18 | 99.77% | 99.70% |

Compared with plain joint utility, stability-fed utility changes the mean from
99.63% at 2.87 / 18 to 99.63% at 2.87 / 18 after rounding. More precisely, it
is about +0.008 percentage points in accuracy and +0.002 counted core layers.
The benefit is concentrated on seed 101, tiny on seed 202, and negative on the
repaired seed-42 branch. Treat the stability feature as useful auxiliary
supervision, not as a decisive router improvement yet.

Selected-threshold timing:

| Seed | Batch | Full Teacher | Agreement 0.10 | Utility 0.10 |
|---:|---:|---:|---:|---:|
| 101 | 1 | 26.06 ms | 17.19 ms | 9.76 ms |
| 101 | 64 | 34.62 ms | 41.33 ms | 23.79 ms |
| 42 | 1 | 20.68 ms | 16.78 ms | 9.00 ms |
| 42 | 64 | 33.60 ms | 37.99 ms | 30.32 ms |
| 202 | 1 | 26.93 ms | 22.63 ms | 11.60 ms |
| 202 | 64 | 35.20 ms | 49.74 ms | 33.67 ms |

The one-candidate utility path remains the only plausible deployment-speed
route. Agreement is still the best quality reference, but its adjacent-budget
pass is a real wall-clock cost.

Additional threshold observation:

The fixed 0.80/0.90/0.95 evaluation points show that a stricter utility gate can
buy more final accuracy at higher cost. For example, stability utility at 0.80
lands at 99.88%, 99.54%, and 99.87% on seeds 101, 42, and 202 respectively, but
uses more counted core than the selected 0.10 gate. This suggests the next
router work should expose a quality/cost operating point rather than treating
one selected threshold as the whole story.

Instrumentation update:

`run_recurrent_smol.py` now stores full validation and final threshold curves in
the held-out audit as `val_policies` and `final_policies`, instead of recording
only the selected threshold. That should make future calibration and
quality-SLO analysis much less lossy. It also records `selected_by_drop`
scenarios for the normal speed-biased selector, a tighter 0.1 percentage-point
drop selector, a teacher-floor selector, and teacher-plus selectors.

## 2026-04-27 - full-curve calibration audit

Commands:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_reuse --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_polish2_joint_halt_reuse --seed 42
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_reuse --seed 202
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_stability_reuse --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_polish2_joint_halt_stability_reuse --seed 42
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_stability_reuse --seed 202
```

These runs used the upgraded held-out audit to record complete validation and
final threshold curves. The first finding is that the utility score is useful
and mostly monotonic: stricter utility thresholds usually buy more final
accuracy at higher fallback cost. The normal speed-biased selector still chooses
the lowest threshold, because it optimizes minimal counted core inside the
validation accuracy floor.

Diagnostic final-curve examples for utility:

| Variant | Seed | Teacher | Best Final Utility | Best Utility Core | Lowest Utility Core At Or Above Teacher |
|---|---:|---:|---:|---:|---:|
| Plain | 42 | 99.49% | 99.55% | 3.56 / 18 | 3.15 / 18 |
| Plain | 101 | 99.69% | 99.83% | 2.35 / 18 | 2.29 / 18 |
| Plain | 202 | 99.60% | 99.83% | 4.10 / 18 | 3.26 / 18 |
| Stability | 42 | 99.49% | 99.54% | 3.40 / 18 | 3.29 / 18 |
| Stability | 101 | 99.69% | 99.85% | 2.35 / 18 | 2.31 / 18 |
| Stability | 202 | 99.60% | 99.84% | 3.65 / 18 | 3.28 / 18 |

This shows the bottleneck is not that utility lacks a usable ranking signal.
It has a quality/cost curve. The problem is that the selected speed point is not
the same thing as a production quality point, and agreement still gets better
quality for less counted core.

## 2026-04-27 - high-validation selector audit

Commands:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_reuse_highval --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_polish2_joint_halt_reuse_highval --seed 42
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_reuse_highval --seed 202
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_stability_reuse_highval --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_polish2_joint_halt_stability_reuse_highval --seed 42
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_stability_reuse_highval --seed 202
```

High-validation reuse modes use 256 validation batches, 256 final batches, a
finer threshold grid from 0.05 to 0.99, and selector scenarios for speed,
teacher floor, teacher plus 0.1 percentage points, and teacher plus 0.2
percentage points. Mean full-teacher accuracy in these runs is 99.56%.

Plain joint utility:

| Selector | Utility Acc | Utility Core | Agreement Acc | Agreement Core | Utility Gap vs Agreement |
|---|---:|---:|---:|---:|---:|
| Speed | 99.54% | 2.84 / 18 | 99.74% | 2.71 / 18 | -0.20 pp |
| Teacher floor | 99.57% | 2.86 / 18 | 99.74% | 2.71 / 18 | -0.16 pp |
| Teacher +0.1 pp | 99.65% | 3.02 / 18 | 99.75% | 2.75 / 18 | -0.11 pp |
| Teacher +0.2 pp | 99.69% | 3.07 / 18 | 99.76% | 2.79 / 18 | -0.06 pp |

Stability-fed joint utility:

| Selector | Utility Acc | Utility Core | Agreement Acc | Agreement Core | Utility Gap vs Agreement |
|---|---:|---:|---:|---:|---:|
| Speed | 99.54% | 2.85 / 18 | 99.72% | 2.71 / 18 | -0.19 pp |
| Teacher floor | 99.58% | 2.90 / 18 | 99.72% | 2.71 / 18 | -0.14 pp |
| Teacher +0.1 pp | 99.63% | 2.93 / 18 | 99.74% | 2.74 / 18 | -0.11 pp |
| Teacher +0.2 pp | 99.68% | 2.97 / 18 | 99.75% | 2.77 / 18 | -0.06 pp |

Conclusion:

Calibration answers the first bottleneck but not the whole bottleneck.
With enough validation and a stricter selector, the one-candidate utility route
can be made teacher-level. It does not yet match the true agreement frontier:
even the teacher-plus selectors trail agreement by about 0.06 percentage points
and use about 0.2 to 0.3 more counted core layers. Stability remains an
auxiliary feature, not a promoted route; it is slightly better on some
teacher-plus operating points but not decisively better than plain utility.

The next training move should therefore target the joint objective itself. The
utility router should be trained to support quality-SLO operating points, not
just the cheapest acceptable route. Candidate next variants:

- sweep higher false-accept penalties and lower cost weights;
- train with sampled quality/cost lambdas so the utility score becomes a
  smoother operating-point knob;
- use agreement labels as auxiliary supervision, but keep deployment to one
  candidate plus fallback;
- continue reporting full threshold curves, because a single selected threshold
  hides the actual quality/cost tradeoff.

## 2026-04-27 - fixed agreement-aux objective audit

An external audit pointed out a real objective bug in the first quality/SLO
joint-halt sweep: the agreement auxiliary target treated the last correction
budget as unsafe because it has no adjacent-budget partner. The fix masks the
last budget out of the agreement auxiliary loss instead of labeling it false.
The route utility loss still decides whether the last budget should be accepted.

The held-out audit now also records acceptance count, accepted precision, and
acceptance share by correction budget for each policy/threshold. This was added
to directly test whether the utility router was learning to avoid the highest
budget.

Commands for the corrected training pass:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_quality --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_polish2_joint_halt_quality --seed 42
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_quality --seed 202
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_slo --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_polish2_joint_halt_slo --seed 42
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_slo --seed 202
```

Standard held-out audit means:

| Variant | Policy | Final Acc | Avg Core | Coverage | Accepted Precision | Accepted Share c0/c1/c2/c3 |
|---|---|---:|---:|---:|---:|---|
| Quality | agreement | 99.740% | 2.72 / 18 | 99.77% | 99.780% | 46.1% / 51.7% / 2.2% / 0.0% |
| Quality | utility | 99.675% | 2.93 / 18 | 99.51% | 99.791% | 44.3% / 50.3% / 4.7% / 0.7% |
| SLO | agreement | 99.723% | 2.74 / 18 | 99.69% | 99.775% | 45.7% / 52.1% / 2.2% / 0.0% |
| SLO | utility | 99.597% | 2.91 / 18 | 99.50% | 99.730% | 44.3% / 50.9% / 4.4% / 0.5% |

The fix slightly improves the quality objective relative to the pre-fix run,
especially on seeds 101 and 202, but it does not remove the agreement frontier
gap. It also falsifies the suspected failure mode: utility is not simply
refusing the highest correction budget. It accepts c3 rarely but nonzero, and
the accepted c3 examples are usually high precision. The deeper issue is that
the one-candidate utility policy still needs more fallback/deeper-budget usage
to approach the agreement policy's quality.

Commands for high-validation reuse on the corrected checkpoints:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_quality_reuse_highval --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_polish2_joint_halt_quality_reuse_highval --seed 42
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_quality_reuse_highval --seed 202
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_slo_reuse_highval --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_polish2_joint_halt_slo_reuse_highval --seed 42
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_slo_reuse_highval --seed 202
```

High-validation selector means:

| Variant | Selector | Utility Acc | Utility Core | Agreement Acc | Agreement Core | Utility Gap |
|---|---|---:|---:|---:|---:|---:|
| Quality | Speed | 99.632% | 2.89 / 18 | 99.740% | 2.70 / 18 | -0.108 pp |
| Quality | Teacher floor | 99.658% | 2.92 / 18 | 99.746% | 2.71 / 18 | -0.088 pp |
| Quality | Teacher +0.1 pp | 99.681% | 3.03 / 18 | 99.762% | 2.75 / 18 | -0.081 pp |
| Quality | Teacher +0.2 pp | 99.699% | 3.06 / 18 | 99.766% | 2.79 / 18 | -0.067 pp |
| SLO | Speed | 99.569% | 2.87 / 18 | 99.744% | 2.72 / 18 | -0.175 pp |
| SLO | Teacher floor | 99.605% | 2.90 / 18 | 99.756% | 2.74 / 18 | -0.151 pp |
| SLO | Teacher +0.1 pp | 99.683% | 3.07 / 18 | 99.774% | 2.78 / 18 | -0.092 pp |
| SLO | Teacher +0.2 pp | 99.691% | 3.13 / 18 | 99.780% | 2.79 / 18 | -0.089 pp |

Conclusion:

The corrected quality objective is the best deployable utility variant so far,
but it still does not justify general LLM application testing as the next step.
It can be tuned to teacher-level performance on this synthetic benchmark, but
it does not match the agreement quality/cost frontier. The current bottleneck is
therefore not calibration and not the final-budget auxiliary bug. It is the
missing deployable substitute for adjacent-budget agreement.

## 2026-04-27 - CATS-style consistency-head audit

Literature pass motivated a CATS-style deployable agreement surrogate:
train a cheap consistency head to predict whether the current JumpRec candidate
is stable against more recurrence, then gate the quality utility router with
that consistency probability. This keeps inference to one candidate plus heads,
instead of true agreement's adjacent-budget pass.

Implementation:

- Added `*_joint_halt_quality_cats` modes.
- Training modes load corrected `*_joint_halt_quality_seed{seed}` checkpoints,
  freeze the existing teacher/JumpRec/utility path, train only
  `consistency_heads`, and save `*_joint_halt_quality_cats_seed{seed}`.
- Reuse/highval modes load the CATS checkpoint and evaluate `utility_cats_050`,
  `utility_cats_070`, and `utility_cats_090`.
- Consistency target:
  - budgets before the last correction: current prediction equals the next
    correction-budget prediction;
  - final correction budget: current prediction equals the full teacher
    prediction.

Commands:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_quality_cats --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_quality_cats --seed 202
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_polish2_joint_halt_quality_cats --seed 42
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_quality_cats_reuse_highval --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_quality_cats_reuse_highval --seed 202
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_polish2_joint_halt_quality_cats_reuse_highval --seed 42
```

Training diagnostics at step 2000:

| Seed | Target Stable | Pred Stable | Precision | Final Target | Final Pred |
|---:|---:|---:|---:|---:|---:|
| 101 | 89.1% | 85.2% | 100.0% | 100.0% | 100.0% |
| 202 | 85.2% | 78.1% | 100.0% | 98.4% | 98.4% |
| 42 | 83.6% | 77.3% | 99.0% | 98.4% | 95.3% |

High-validation means over seeds 101, 202, and repaired polish2 42:

| Selector | Plain Utility Acc | Plain Utility Core | Best CATS Acc | Best CATS Core | Agreement Acc | Agreement Core |
|---|---:|---:|---:|---:|---:|---:|
| Speed | 99.632% | 2.88 / 18 | 99.636% | 2.90 / 18 | 99.740% | 2.70 / 18 |
| Teacher floor | 99.658% | 2.92 / 18 | 99.662% | 2.93 / 18 | 99.746% | 2.71 / 18 |
| Teacher +0.1 pp | 99.681% | 3.03 / 18 | 99.687% | 3.04 / 18 | 99.762% | 2.75 / 18 |
| Teacher +0.2 pp | 99.699% | 3.06 / 18 | 99.703% | 3.07 / 18 | 99.766% | 2.79 / 18 |

Conclusion:

The consistency head learned the intended signal, including the corrected
final-budget target, but it did not materially improve routing. `utility_cats`
mostly shadows plain utility, with tiny accuracy changes and slightly higher
counted core at the best CATS operating point. True agreement remains ahead by
roughly 0.06 to 0.10 percentage points while also using less counted core.

This falsifies the simplest CATS-style post-hoc surrogate as the missing
deployable agreement replacement. The remaining bottleneck is likely in the
training objective or candidate trajectory itself, not in adding a separate
stable/unstable head after the quality checkpoint.

## 2026-04-27 - no-training controller selection audit

After CATS failed to move the frontier, the next check was whether the existing
corrected quality checkpoints already contain enough signal for better
held-out controller selection without retraining. The runner now evaluates:

- `utility_per_budget`: separate utility thresholds by correction budget.
- `utility_guarded_per_budget`: the same search with margin/max-prob guards.
- `utility_per_budget_monotone`: a monotone version that keeps earlier budgets
  at least as strict as later budgets.
- `utility_then_agree_*`: direct utility accept at the selected threshold,
  otherwise true agreement confirmation above a fixed utility floor.
- `agree_then_utility_*`: true agreement with a separate high utility direct
  accept floor at 0.90, 0.95, or 0.99.

Commands:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_quality_reuse_highval --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_quality_reuse_highval --seed 202
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_polish2_joint_halt_quality_reuse_highval --seed 42
```

High-validation means over seeds 101, 202, and repaired polish2 42:

| Selector | Policy | Final Acc | Avg Core |
|---|---|---:|---:|
| Speed | Agreement | 99.740% | 2.70 / 18 |
| Speed | Utility | 99.632% | 2.89 / 18 |
| Speed | Utility per-budget | 99.646% | 2.88 / 18 |
| Speed | Utility then agreement | 99.585% | 2.69 / 18 |
| Speed | Agreement then utility 0.99 | 99.717% | 2.70 / 18 |
| Teacher floor | Agreement | 99.746% | 2.71 / 18 |
| Teacher floor | Utility | 99.658% | 2.92 / 18 |
| Teacher floor | Utility per-budget | 99.646% | 2.88 / 18 |
| Teacher floor | Utility then agreement | 99.646% | 2.71 / 18 |
| Teacher floor | Agreement then utility 0.99 | 99.721% | 2.71 / 18 |
| Teacher +0.1 pp | Agreement | 99.762% | 2.75 / 18 |
| Teacher +0.1 pp | Utility | 99.681% | 3.03 / 18 |
| Teacher +0.1 pp | Agreement then utility 0.99 | 99.740% | 2.75 / 18 |
| Teacher +0.2 pp | Agreement | 99.766% | 2.79 / 18 |
| Teacher +0.2 pp | Utility | 99.699% | 3.06 / 18 |
| Teacher +0.2 pp | Agreement then utility 0.99 | 99.746% | 2.77 / 18 |

Findings:

- Per-budget utility threshold search chose `[0.05, 0.05, 0.05, 0.05]` on all
  three seeds. This falsifies the idea that the current gap is mostly a
  per-budget operating-point issue.
- Guarded and monotone per-budget variants collapsed to the same choice.
- `utility_then_agree_000` can save counted core by over-accepting low-threshold
  utility candidates, but it loses accuracy and is not a viable promoted route.
- Decoupling the thresholds is better: `agree_then_utility_099` tracks true
  agreement closely, often within 0.02 to 0.03 percentage points at similar
  counted core.
- That hybrid still depends on true adjacent-budget agreement, so it is not the
  deployable one-candidate substitute needed for scaling. It is a useful
  diagnostic/reference policy, not the road out.

Conclusion:

The road is now unblocked conceptually, but not by a new promoted controller.
The evidence points away from more no-training threshold tricks. True agreement
is still the frontier; cheap utility has useful ranking signal but does not
separate safe accepts sharply enough; cheap post-hoc stability/consistency did
not repair that. The next viable path is objective-level agreement
distillation: train the candidate trajectory and halting score so the
one-candidate path internalizes what adjacent-budget agreement is currently
checking.

## 2026-04-27 - agreement-distilled quality objective audit

The next attempt moved agreement supervision into the joint objective instead
of adding another post-hoc selector. `*_joint_halt_quality_agdistill` adds two
terms to the corrected quality branch:

- adjacent-budget/full-teacher distribution distillation for candidate logits;
- an agreement-shaped route risk that penalizes accepting target-probable but
  distributionally unstable candidates.

Commands:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_quality_agdistill --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_quality_agdistill --seed 202
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_polish2_joint_halt_quality_agdistill --seed 42
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_quality_agdistill_reuse_highval --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_quality_agdistill_reuse_highval --seed 202
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_polish2_joint_halt_quality_agdistill_reuse_highval --seed 42
```

High-validation means over seeds 101, 202, and repaired polish2 42:

| Selector | Utility Acc | Utility Core | Agreement Acc | Agreement Core | Agreement-then-Utility 0.99 Acc | Agreement-then-Utility 0.99 Core |
|---|---:|---:|---:|---:|---:|---:|
| Speed | 99.618% | 2.94 / 18 | 99.668% | 2.71 / 18 | 99.664% | 2.71 / 18 |

Selected speed-point utility by seed:

| Seed | Teacher | Utility Acc | Utility Core | Agreement Acc | Agreement Core | Utility accepted share c0/c1/c2/c3 |
|---:|---:|---:|---:|---:|---:|---|
| 101 | 99.713% | 99.707% | 2.32 / 18 | 99.835% | 2.25 / 18 | 59.2% / 38.9% / 1.8% / 0.1% |
| 202 | 99.530% | 99.652% | 3.32 / 18 | 99.670% | 2.99 / 18 | 35.4% / 54.9% / 9.4% / 0.3% |
| 42 | 99.438% | 99.493% | 3.18 / 18 | 99.500% | 2.90 / 18 | 39.1% / 56.8% / 3.4% / 0.7% |

Conclusion:

Agreement distillation is a clean negative result for the current objective.
It did not improve the deployable one-candidate utility frontier. It also
falsifies a simple "final budget is still being avoided" story: the selected
utility route accepts the final correction budget rarely but not never, and the
remaining false accepts come mostly from budget-1/budget-2 candidates that true
agreement filters more sharply. The useful lesson is that adjacent-budget
agreement is checking something not captured by this KL-style trajectory
matching plus scalar route-risk shaping.

## 2026-04-27 - quality plus jointly trained stability audit

After agreement distillation failed, the next check was the already-scaffolded
`*_joint_halt_quality_stability` branch. This combines the corrected quality
joint objective with a jointly trained stability head and feeds the stability
logit into the utility router. Unlike CATS, the stability feature is trained in
the joint candidate/halting loop rather than added afterward.

Commands:

```text
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_quality_stability --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_quality_stability --seed 202
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_polish2_joint_halt_quality_stability --seed 42
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_quality_stability_reuse_highval --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_quality_stability_reuse_highval --seed 202
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_polish2_joint_halt_quality_stability_reuse_highval --seed 42
```

High-validation selector means:

| Selector | Utility Acc | Utility Core | Agreement Acc | Agreement Core | Utility Gap |
|---|---:|---:|---:|---:|---:|
| Speed | 99.628% | 2.87 / 18 | 99.711% | 2.71 / 18 | -0.083 pp |
| Teacher floor | 99.648% | 2.88 / 18 | 99.711% | 2.71 / 18 | -0.063 pp |
| Teacher +0.1 pp | 99.687% | 2.99 / 18 | 99.742% | 2.76 / 18 | -0.055 pp |
| Teacher +0.2 pp | 99.705% | 3.03 / 18 | 99.748% | 2.79 / 18 | -0.043 pp |

Selected speed-point utility by seed:

| Seed | Teacher | Utility Acc | Utility Core | Agreement Acc | Agreement Core | Utility accepted share c0/c1/c2/c3 |
|---:|---:|---:|---:|---:|---:|---|
| 101 | 99.713% | 99.762% | 2.31 / 18 | 99.872% | 2.24 / 18 | 59.3% / 38.4% / 2.1% / 0.2% |
| 202 | 99.530% | 99.695% | 3.26 / 18 | 99.725% | 3.00 / 18 | 35.5% / 56.0% / 7.9% / 0.6% |
| 42 | 99.438% | 99.426% | 3.04 / 18 | 99.536% | 2.88 / 18 | 38.7% / 58.0% / 2.7% / 0.5% |

Conclusion:

Quality-stability is the best high-quality one-candidate variant by a small
margin at the teacher-plus selectors, but it is not a road-unblocking result.
It roughly ties corrected quality at the speed point and narrows the
teacher-plus gap, yet true agreement still wins on both accuracy and counted
core. The current answer to the bottleneck is therefore negative: neither
post-hoc consistency, no-training selector search, agreement-shaped objective
distillation, nor jointly trained stability has replaced adjacent-budget
agreement.

Do not promote general LLM application testing yet. The synthetic benchmark
still supports the JumpRec idea and a teacher-level deployable utility route,
but the scalable one-candidate controller has not matched the agreement
frontier. The next serious work should either harden/refactor the experiment
surface or test a stronger mechanism that changes the candidate trajectory or
controller supervision more radically than the current auxiliary heads.
