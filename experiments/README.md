# Experiments

This folder is for reproducible experiment orchestration around JumpRec.
`CHECKPOINT_MANIFEST.md` records the active predecessor checkpoints for reuse
and high-validation modes.

Current cost policy:

- Prefer H100 while the research loop is still exploratory and latency matters.
- Avoid retraining for timing-only or threshold-only probes once checkpoints are available.
- Keep every headline run paired with the direct control or a previously seed-confirmed direct-control table.
- Treat batch-size-1 and batch-size-64 as different claims: local latency versus throughput serving.

Current primary benchmark:

- `mixed_core3_router_no_agree`: mixed textual recurrence with SmolLM2-135M, 3 recurrent core layers, no-agreement verifier routing.
- Best local claim so far: threshold 0.90, batch size 1, 98.29% mean accuracy and 2.13x speedup over the full recurrent teacher across seeds 42, 101, and 202.

Current hard-case benchmark:

- `core3_8n4h_hardhop_teacher`: 8 nodes / 4 hops, 3 recurrent core layers, max-hop replay, max-hop loss weighting, stronger final-loop weighting, and checkpoint save.
- `core3_8n4h_hardhop_jumprec`: loads the hard-hop teacher checkpoint, trains JumpRec/direct control, and runs the batch-size timing sweep.
- `core3_8n4h_strathop_teacher`: stability probe that samples hops with weights `0.10,0.20,0.35,0.35` and weights losses `1.0,1.2,2.0,2.0` to protect hop 3 while still emphasizing hard hops.
- `core3_8n4h_strathop_jumprec`: loads the stratified-hop teacher checkpoint and tests JumpRec only if the teacher is strong.
- Do not treat JumpRec results on 8/4 as meaningful until the full recurrent teacher is competent on hop-4 examples.

Useful Modal modes:

```powershell
modal run run_recurrent_smol.py --mode mixed_core3_router_bsize_sweep --seed 42
modal run run_recurrent_smol.py --mode mixed_core3_router_bsize_sweep_reuse --seed 42
modal run run_recurrent_smol.py --mode mixed_core3_router_no_agree --seed 101
modal run run_recurrent_smol.py --mode mixed_core3_router_no_agree_b1 --seed 202
modal run run_recurrent_smol.py --mode core3_8n4h_hardhop_teacher --seed 42
modal run run_recurrent_smol.py --mode core3_8n4h_hardhop_jumprec --seed 42
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_teacher --seed 202
```

`mixed_core3_router_bsize_sweep` trains once, saves a checkpoint, and reports timing from batch size 1 through 64. The `_reuse` mode loads the checkpoint and reruns eval/timing without paying the training cost.
