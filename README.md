# JumpRec

JumpRec is an experiment in speculative recursive-state refinement: train a
looped transformer teacher, then train a small jump module to land near a later
recursive state so only a short frozen loop tail is needed.

The goal is to test whether a small/local model can spend extra computation only
when a problem needs it, while preserving most of the performance of a deeper
looped model.

## Current Status

The strongest pure JumpRec result is on a mixed synthetic recurrence suite with
four transition families over a permutation table: `forward`, `inverse`,
`alternate`, and `square`.

On teacher-solved mixed strict seeds, JumpRec reaches about 99.9% accuracy while
using about 2.5 block-equivalents instead of 8, saving roughly 69% compute. The
strict fallback policy sends uncertain examples back through the full loop and
is exercised in practice.

The strongest pretrained-LM result is now positive but bounded: a
recurrent-depth SmolLM2-135M retrofit gives JumpRec a real state space to
accelerate. With trainable copied JumpRec blocks on the 6-node / 3-hop textual
pointer task, JumpRec reaches 99.76% +/- 0.29% strict-fallback accuracy at
threshold 0.80 while using 1.42 +/- 0.22 recurrent core layers instead of 10,
saving 85.82% +/- 2.16% core-layer compute across three seeds. A first serial
router averages 17.39 ms/batch versus 21.25 ms/batch for the full teacher on
H100, though the timing path is still rough.

The main caution is that a simple 3-layer direct control is also very strong on
the same easy task: 99.10% +/- 0.63% accuracy at 3 of 10 core layers. Harder
probes show the current boundary: the mixed recurrence task reaches only 86.08%
full-loop accuracy, and a simple 8-node / 4-hop curriculum drops to 73.51%.
The next credible result needs to beat the direct control on harder mixed or
scaled recurrence tasks.

See `JUMPREC_RESULTS.md` for the experimental log and caveats.

The first pretrained-LM crash tests were negative: frozen SmolLM2 final states,
input adapters, curriculum warmup, and a latent workspace sidecar did not
produce a competent looped teacher. The successful path was a true
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
- `run_recurrent_smol.py`: recurrent-depth SmolLM2 retrofit and JumpRec runner.
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
modal run run_recurrent_smol.py --mode retrofit_probe
modal run run_recurrent_smol.py --mode jumprec_probe
modal run run_recurrent_smol.py --mode direct_probe
modal run run_recurrent_smol.py --mode mixed_probe
```

## Current Next Steps

1. Move the default LM benchmark to harder mixed recurrence tasks and keep the
   3-layer direct control in every table.
2. Improve the 8-node / 4-hop recurrent retrofit; the naive hop curriculum was
   worse than the first scale-up and likely caused forgetting.
3. Make the serial JumpRec router agreement-aware and benchmark it as a real
   inference path, not all budgets in parallel.
4. Try stronger recurrent cores, especially the 3-core-layer setting, on mixed
   and 8/4 tasks before making broader architecture claims.
