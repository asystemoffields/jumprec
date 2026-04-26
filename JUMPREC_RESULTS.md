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
