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

The strongest pretrained-LM result is now the mixed recurrent SmolLM2-135M
setting. With a 3-layer recurrent core, four textual transition families, and a
3-layer direct control in the table, JumpRec reaches 98.74% +/- 0.93%
strict-fallback accuracy at threshold 0.80 while using 2.31 +/- 0.38 recurrent
core layers instead of 15. That saves 84.60% +/- 2.51% counted core-layer
compute across three seeds. The full recurrent teacher reaches 97.92% +/- 1.18%
and the direct 3-layer control reaches 95.97% +/- 0.73%.

That is the best evidence so far that JumpRec is doing something more than
compressing an easy task into a shallow direct head. The wall-clock story is
not solved yet: on H100, the mixed/core3 full teacher averages 28.23 ms/batch,
the fast serial router averages 27.83 ms/batch but omits the strict agreement
check, and the agreement-aware serial router averages 49.19 ms/batch. The
current claim is therefore compute-layer efficiency plus promising routing
behavior, not finished inference speed.

The main remaining caution is robustness. The 8-node / 4-hop setting is still
limited by teacher quality and max-depth failures; JumpRec cannot reliably
recover a weak full-loop teacher. The next credible result needs either a
better strict router that does not need an extra agreement pass, or a stronger
hard-case training recipe for 8/4 and beyond.

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

1. Make the strict router agreement-free but calibrated, so the measured
   core-layer savings can become real wall-clock savings.
2. Improve the 8-node / 4-hop recurrent retrofit with hard-hop replay or a
   better balanced curriculum; core depth alone did not solve 4-hop cases.
3. Keep mixed/core3 as the default LM benchmark and keep the 3-layer direct
   control in every table.
4. Seed-confirm any router or hard-case training improvement before making
   broader architecture claims.
