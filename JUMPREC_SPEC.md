# JumpRec v0 spec

## Mission

Test whether a looped transformer can be accelerated by a learned hidden-state jump:

```text
state_before_loop -> predicted_state_after_many_loops -> optional corrective loops
```

The goal is not to replace looping everywhere. The goal is to amortize common
recursive computation so a local model can usually "go through" the loop and
only pay real recurrence when the jump is uncertain.

## Core idea

Train a shared-loop transformer teacher on an iterative task. Then freeze the
teacher and train a small jump module that predicts the teacher's final recurrent
state from the initial encoded state.

At inference:

1. Encode the prompt/window once.
2. Run the jump module once.
3. Ask a verifier whether the jump state is good enough.
4. If accepted, decode from the jump state.
5. If not, apply one or two corrective loop steps.
6. Persist the final hidden/KV state for future turns.

This is speculative decoding moved inward: speculation over recurrent hidden
states rather than output tokens.

## v0 test bed

Synthetic pointer chasing:

```text
[BOS] [MAP] 0 f(0) 1 f(1) ... 15 f(15) [Q] start [H] hop_count [OUT]
```

The answer is `f^hop_count(start)`.

This task is useful because it has an obvious iterative solution. A looped model
can learn "one hop per loop." A jump module must learn to approximate the final
multi-hop state without replaying every hop.

## Conditions

- `one_loop`: teacher after one loop step.
- `full_loop`: teacher after all recurrent steps.
- `jump0`: jump only, no corrective recurrence.
- `jump1`: jump plus one corrective loop.
- `jump2`: jump plus two corrective loops.
- `adaptive`: verifier chooses jump0, jump1, or jump2.

## Pass criteria

Strong v0 signal:

- `full_loop` reaches high validation accuracy.
- `jump1` or `adaptive` is within 1-2 accuracy points of `full_loop`.
- Average corrective loops for `adaptive` is much lower than full-loop depth.
- The verifier spends more compute on higher-hop examples.

Weak signal:

- Jump improves over one-loop but needs two corrective loops often.

Fail:

- Jump does not beat one-loop, or verifier cannot identify bad jumps.

## Where SMOKE fits

v0 includes a window-level temporary adapter inside the jump module. A hypernet
reads the query/window state and emits a low-rank adapter applied across the
whole reasoning window. This is intentionally not per-token synthesis: the
adapter is reused for the local reasoning episode and discarded after the final
state is written.

## Why this is publishable if it works

The novelty claim is not "looped transformers" or "adaptive compute." The claim
is:

> Speculative recursive-state refinement: a small local model predicts the
> post-recursion hidden state of a looped transformer, verifies the jump, and
> falls back to corrective recurrence only when needed.

The important ablation is compute-quality frontier against full looping,
early-exit looping, and jump-without-verifier.
