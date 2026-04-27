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

A seed-confirmed agreement-free router is positive on accuracy but not yet on
batch-64 wall-clock. At threshold 0.90 it reaches 98.29% mean accuracy while
using 2.40 of 15 recurrent core layers, beating the 97.92% full teacher mean
and prior 95.97% direct-control mean. But serial timing averages 35.95 ms/batch
versus 28.39 ms/batch for the full teacher. The next question is whether this
becomes favorable at batch size 1, which is closer to local interactive use.

The batch-1 timing result is seed-confirmed and now the strongest local-target
result. At threshold 0.90, agreement-free JumpRec reaches 98.29% mean accuracy
while using 2.40 of 15 recurrent core layers. Two batch-1 timing passes both
showed about a 2x speedup over the full recurrent teacher while also beating
the full teacher and prior direct-control accuracies.

A checkpointed batch-size sweep now shows the useful latency regime directly:
the current serial-router implementation is faster through roughly batch size
8, breaks even around batch size 16, and loses for larger throughput batches.
That supports the local/small-batch target while cautioning against broad
production-serving throughput claims without fused routing.

The runner now supports checkpoint save/load and batch-size timing sweeps, so
future timing and threshold probes do not need to retrain from scratch.

The main remaining caution is robustness. JumpRec does not reliably recover a
weak full-loop teacher, so hard-case teacher quality is now a first-class part
of the architecture story.

The first max-hop repair was positive on seeds 42 and 101 but failed on seed
202, mainly from hop-3 collapse. On the two strong max-hop teachers, JumpRec
matched or beat the full teacher while using about 4.4 of 18 recurrent core
layers at threshold 0.90, with about a 1.9x batch-1 speedup.

The better hard-case recipe is stratified hard-hop replay, but it is not yet a
complete teacher solution. On the previously weak seed 202, sampling hops with
weights `0.10,0.20,0.35,0.35` and loss weights `1.0,1.2,2.0,2.0` raised
full-loop accuracy from 76.9% to 99.5%. Seed 101 also becomes very strong at
99.7%. Seed 42 improves overall versus max-hop-only, but shifts the remaining
weakness onto hop 4: 86.1% hop-4 accuracy.

On the two strong stratified teachers, seeds 101 and 202, JumpRec confirms the
hard-case shape. At threshold 0.95, the no-agreement router averages 99.59%
accuracy while using 2.91 of 18 recurrent core layers, saving 83.83% counted
core compute. The same serial router is faster than the full teacher in the
current timing sweep, averaging about 2.29x speedup at batch size 1 and 1.39x
at batch size 64 across those two seeds. Agreement routing recovers teacher
parity at about 99.77% mean accuracy, but costs more wall-clock.

That makes the hard-case result real but still gated by teacher quality. The
teacher-quality gate is now implemented: it tracks uniform validation by hop
during teacher training and restores the best worst-hop checkpoint. On seed 42,
the gate confirmed the issue rather than fixing it: the best checkpoint still
only reached 88.1% worst-hop validation accuracy and 86.1% hop-4 final eval.
The next credibility step is therefore a short late uniform or max-hop polish
under the same gate, so seed 42 can be repaired without harming seeds 101/202.

See `JUMPREC_RESULTS.md` for the experimental log and caveats.

## Scaling Constraint

JumpRec should be designed as a model-size-independent interface, not as a
135M-specific patch. The recurrent core, jump module, verifier, and fallback
policy need to preserve the same conceptual contract from small models through
larger local models.

The execution strategy will probably differ by scale:

- `135M`: serial subset routing is acceptable for fast iteration and exposes
  the basic latency/quality trade-off.
- `2B` to `9B`: routing overhead must be low, checkpoint reuse must be standard,
  and the jump path should avoid launching many tiny dynamic batches.
- very large dense or MoE models: the router must skip expensive block/experts
  cleanly, avoid extra cross-device communication, and preferably be fused,
  static, or block-level enough that saved recurrence is larger than routing
  overhead.

This makes hard-case robustness and hardware-aware routing both first-class
requirements. A result that only works because the toy runner overhead is small
does not count as the end goal.

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
- `experiments/`: launch helpers and experiment orchestration notes.
- `jumprec/`: staging area for the future general-use package.
- `requirements.txt`: Python package dependencies.

## Quick Sanity Test

```bash
python run_jumprec_v0.py --local --mode dry
python run_jumprec_v0.py --local --mode dry_mixed
python run_recurrent_smol.py --local --mode dry_hardhop
python run_recurrent_smol.py --local --mode dry_strathop
python run_recurrent_smol.py --local --mode dry_strathop_gate
python run_recurrent_smol.py --local --mode dry_sweep
python run_recurrent_smol.py --local --mode dry_sweep_reuse
```

The script guards against accidental heavy local runs. Use Modal for real tests:

```bash
modal run run_jumprec_v0.py --mode quick_c6_no_hidden
modal run run_jumprec_v0.py --mode quick_mix_strict
modal run run_recurrent_smol.py --mode retrofit_probe
modal run run_recurrent_smol.py --mode jumprec_probe
modal run run_recurrent_smol.py --mode direct_probe
modal run run_recurrent_smol.py --mode mixed_probe
modal run run_recurrent_smol.py --mode mixed_core3_router_bsize_sweep
modal run run_recurrent_smol.py --mode mixed_core3_router_bsize_sweep_reuse
modal run run_recurrent_smol.py --mode core3_8n4h_hardhop_teacher
modal run run_recurrent_smol.py --mode core3_8n4h_hardhop_jumprec
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_teacher
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_gate_teacher
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_jumprec
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_gate_jumprec
```

## Current Next Steps

1. Make the mixed/core3 small-batch result the current local-inference
   headline, while keeping the synthetic-task caveat explicit.
2. Add a teacher-quality gate for hard cases: validate uniformly by hop during
   training, save the best worst-hop checkpoint, and report the gate result.
3. Test a blended teacher schedule for 8/4, likely stratified training followed
   by a short uniform or max-hop polish, then rerun seed 42.
4. Run JumpRec only on teacher-gated checkpoints; avoid spending GPU on weak
   teachers unless the purpose is explicitly diagnostic.
5. Begin staging paper-style result tables around three tiers: toy pointer
   proof, mixed SmolLM2 seed-confirmed local latency, and stratified/gated 8/4
   hard-case robustness.
6. Keep scale portability in the design loop: favor block-level mechanisms that
   can survive 2B, 9B, and larger serving economics.
7. Keep mixed/core3 as the default LM benchmark and keep the 3-layer direct
   control in every table.
8. Seed-confirm any router or hard-case training improvement before making
   broader architecture claims.
