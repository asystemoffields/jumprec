# JumpRec

JumpRec is an experiment in speculative recursive-state refinement: train a
looped transformer teacher, then train a small jump module to land near a later
recursive state so only a short frozen loop tail is needed.

The goal is to test whether a small/local model can spend extra computation only
when a problem needs it, while preserving most of the performance of a deeper
looped model.

## Current Status

The strongest current result is on a mixed synthetic recurrence suite with four
transition families over a permutation table: `forward`, `inverse`, `alternate`,
and `square`.

On teacher-solved mixed strict seeds, JumpRec reaches about 99.9% accuracy while
using about 2.5 block-equivalents instead of 8, saving roughly 69% compute. The
strict fallback policy sends uncertain examples back through the full loop and
is exercised in practice.

See `JUMPREC_RESULTS.md` for the experimental log and caveats.

## Files

- `run_jumprec_v0.py`: Modal/local experiment runner.
- `JUMPREC_SPEC.md`: architecture sketch and experimental framing.
- `JUMPREC_RESULTS.md`: run history, tables, and interpretation.
- `requirements.txt`: Python package dependencies.

## Quick Sanity Test

```bash
python run_jumprec_v0.py --local --mode dry
python run_jumprec_v0.py --local --mode dry_mixed
```

The script guards against accidental heavy local runs. Use Modal for real tests:

```bash
modal run run_jumprec_v0.py --mode quick_c6_no_hidden
modal run run_jumprec_v0.py --mode quick_mix_strict
```

## Current Next Steps

1. Repair the SmolLM2-135M wrapper. The first frozen-encoder crash test ran, but
   the looped teacher did not solve the textual pointer task.
2. Add a trainable input/task adapter and train the recurrent teacher with a
   final-answer objective before adding per-step recurrence supervision.
3. Consider exposing intermediate SmolLM2 layer states or a tiny LoRA on the
   last few LM layers if frozen final hidden states remain too brittle.
4. Keep the synthetic suite as the regression test; do not make JumpRec claims
   on SmolLM2 until the full-loop teacher is strong.
