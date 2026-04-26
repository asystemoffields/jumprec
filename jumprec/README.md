# JumpRec Package Staging

This directory is reserved for the future general-use implementation.

Near-term rule: keep research code in `run_recurrent_smol.py` until the
interfaces stop moving. Move code here only when it is useful outside a single
experiment.

Planned package surfaces:

- `RecurrentBackbone`: wrapper for a pretrained decoder with prelude, recurrent core, and coda sections.
- `JumpModule`: learned state jump plus optional temporary adapter.
- `VerifierRouter`: acceptance policy over jump budgets, thresholds, and fallback.
- `AdaptiveGenerateMixin`: generation-time integration once the approach leaves synthetic classification tasks.

Promotion checklist:

- Seed-confirmed result on the current mixed/core3 benchmark.
- Timing sweep documents where adaptive routing helps and where batching hurts.
- Checkpoint save/load path is stable.
- At least one non-synthetic language-modeling or reasoning evaluation shows a real benefit.
