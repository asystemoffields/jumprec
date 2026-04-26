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

The pretrained-LM crash tests are currently negative. Frozen SmolLM2 final
states, input adapters, curriculum warmup, and a latent workspace sidecar have
not produced a competent looped teacher. The evidence now points toward a true
recurrent-depth LM retrofit rather than another frozen hidden-state wrapper.

Current looped-transformer recipes usually apply recurrence inside the model
path itself: a prelude block encodes the input, a shared recurrent block is
looped, and a coda block produces logits. Recent work also emphasizes input
reinjection, loop/time conditioning, recurrence curricula, adaptive exits, and
stability constraints. JumpRec should be layered on top only after that
full-loop recurrent model is strong.

## Files

- `run_jumprec_v0.py`: Modal/local experiment runner.
- `run_jumprec_smol.py`: SmolLM2 crash-test runner.
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

1. Build a real recurrent-depth SmolLM2 retrofit: prelude, shared recurrent
   core, coda, and explicit input reinjection.
2. Train it with a recurrence curriculum and verify that increasing loops helps
   before adding JumpRec.
3. Add JumpRec only after the recurrent full-loop teacher is competent.
4. Keep the synthetic suite as the regression test; do not make JumpRec claims
   on SmolLM2 until the full-loop teacher is strong.
