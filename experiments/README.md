# Experiments

This folder is for reproducible experiment orchestration around JumpRec.
Read `../JUMPREC_CURRENT_STATE.md` first for the current research answer.
`CHECKPOINT_MANIFEST.md` records the active predecessor checkpoints for reuse
and high-validation modes.

## Active Path

The current active controller is selective agreement on top of the
quality-plus-stability joint-halt checkpoint family.

Active mode family:

```text
*_joint_halt_quality_stability_reuse_highval
```

Current cross-seed set:

| Seed | Mode |
|---:|---|
| 101 | `core3_8n4h_strathop_joint_halt_quality_stability_reuse_highval` |
| 202 | `core3_8n4h_strathop_joint_halt_quality_stability_reuse_highval` |
| 42 | `core3_8n4h_strathop_polish2_joint_halt_quality_stability_reuse_highval` |

The reuse/highval modes load existing checkpoints, skip training, and run the
large held-out audit plus timing sweep. The selective-agreement result is in
`heldout_threshold_audit.selective_agreement`.

## Application Bridge

The next live path is the natural-language graph traversal bridge. It uses
route-card prose instead of compact `Task/Map/Start/Hops` prompts, while keeping
the recurrence target controlled enough for teacher gates and artifact audits.

Current seed-101 run order:

```powershell
modal run run_recurrent_smol.py --mode core3_8n4h_natgraph_teacher --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_natgraph_teacher_resume --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_natgraph_polish_teacher --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_natgraph_polish2_teacher --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_natgraph_polish2_joint_halt_quality_stability --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_natgraph_polish2_joint_halt_quality_stability_reuse_highval --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_natgraph_polish2_joint_halt_quality_stability_reuse_audit --seed 101
```

The base natgraph teacher was not strong enough on hop 4. The promoted bridge
path is therefore the `polish2` family. Seed 101 is positive: the teacher and
prompt shortcut audits pass, and selective agreement reaches teacher-level
quality at small-batch speed. The caveat is adjacent-check rate: natgraph needs
roughly 70-90% checks in the current selectors, so batch-64 throughput is not
unblocked.

## Reproduce Current Result

```powershell
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_quality_stability_reuse_highval --seed 101
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_joint_halt_quality_stability_reuse_highval --seed 202
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_polish2_joint_halt_quality_stability_reuse_highval --seed 42
```

For Windows background launches with UTF-8 logs:

```powershell
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$jobs = @(
  @{ mode='core3_8n4h_strathop_joint_halt_quality_stability_reuse_highval'; seed=101; name='seed101' },
  @{ mode='core3_8n4h_strathop_joint_halt_quality_stability_reuse_highval'; seed=202; name='seed202' },
  @{ mode='core3_8n4h_strathop_polish2_joint_halt_quality_stability_reuse_highval'; seed=42; name='polish2_seed42' }
)
foreach ($job in $jobs) {
  $log = "modal_selective_agreement_$($job.name)_$stamp.log"
  $cmd = "`$env:PYTHONIOENCODING='utf-8'; `$env:PYTHONUTF8='1'; [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new(); modal run run_recurrent_smol.py --mode $($job.mode) --seed $($job.seed) *> '$log'"
  Start-Process -FilePath 'powershell' -ArgumentList @('-NoProfile','-Command',$cmd) -WorkingDirectory (Get-Location) -PassThru -WindowStyle Hidden
}
```

Logs are ignored by git. Temporary job-state JSONs should be deleted after use.

## Local Guards

```powershell
python -m py_compile .\run_recurrent_smol.py
python -m unittest discover -s tests
python .\run_recurrent_smol.py --mode dry_strathop_polish2_joint_halt_stability_reuse --local
python .\run_recurrent_smol.py --mode dry_natgraph_teacher --local
python .\run_recurrent_smol.py --mode dry_natgraph_joint_halt_quality_stability --local
python .\run_recurrent_smol.py --mode dry_natgraph_joint_halt_quality_stability_reuse --local
```

The dry mode is a wiring guard only. It does not validate quality.

## Claim Policy

Treat batch-size-1 and batch-size-64 as different claims:

- Batch size 1: local/interactive latency. Selective agreement is currently
  positive here.
- Batch size 64: throughput serving. Selective agreement is faster than true
  agreement but still slower than full-loop/all-budgets parallel execution.

Every headline should state:

- seed set;
- mode names;
- checkpoint family;
- selected policy;
- counted core;
- wall-clock batch size;
- whether adjacent checks were charged.

## Historical Branches

These are useful background but are not the current promoted path:

- `mixed_core3_router_no_agree`: earlier mixed textual recurrence result.
- `*_budget_controller*`: up-front budget prediction studies.
- `*_utility_router*`: post-hoc one-candidate utility routing.
- `*_nextagree_router*`: learned next-agreement proxy.
- `*_joint_halt_quality_cats*`: cheap consistency head.
- `*_joint_halt_quality_agdistill*`: agreement distillation branch.

Use `../JUMPREC_RESULTS.md` for historical numbers before reviving any branch.
