# JumpRec Next Steps Handoff

This file is meant for a fresh Codex instance. Read this before loading long
logs or compaction notes.

## Current Thesis

JumpRec is testing a local-LLM-oriented architecture:

1. Train or adapt a small transformer into a recurrent/looped teacher.
2. Train a JumpRec module to skip most recurrent depth by landing near a later
   point on the teacher trajectory.
3. Run a small number of ordinary tail loop steps after the jump.
4. Use a verifier/controller to accept the cheap path when it is reliable, or
   fall back to the full recurrent teacher when needed.

The important conceptual update is that the jump path and the adaptive compute
policy are separate objects. The jump can learn good fixed-budget solutions,
but the verifier/controller is what turns it into "small unless hard."

Current active focus:

> The concept has enough legs in the synthetic SmolLM hard case. Stop spending
> the main research loop on proving that the jump can learn. The next important
> work is to design the best routing/controller possible: quality-preserving,
> calibrated, cheap, and robust enough to replace agreement routing where
> possible.

## Repo State To Respect

- Workspace: `C:\Users\power\Documents\SMOKE`
- Main runner: `run_recurrent_smol.py`
- Results ledger: `JUMPREC_RESULTS.md`
- Audit checklist from another Codex instance: `JUMPREC_ARTIFACT_AUDIT.md`
- Remote: `https://github.com/asystemoffields/jumprec`
- Recent pushed commits before this update:
  - `7817ae4` Sort relabeled audit mappings
  - `abc4958` Record corrected prompt audit
  - `72bcc88` Add held-out verifier audit
  - `d518104` Record held-out verifier audit
  - `c6ad7fe` Add JumpRec prompt audit
  - `098399a` Record JumpRec prompt audit
  - `762c7cb` Add JumpRec hardcase ablation modes
  - `caa078b` Record seed-101 JumpRec ablations
  - `7f5f17a` Add JumpRec next steps handoff

Check `git log --oneline -5` for the final commit hash of this update.

Current working tree, before this handoff file, had unrelated changes:

- `README.md` modified by another instance.
- `JUMPREC_ARTIFACT_AUDIT.md` untracked but important.
- old untracked smoke files: `SPEC.md`, `run_smoke_v0*.py`, `run_smoke_v1.py`.

Do not revert or casually commit those unless the user explicitly asks.

## Main Results So Far

### Teacher Robustness

The robust 8-node / 4-hop SmolLM2-135M looped teachers are now seed-confirmed.

256-batch uniform-hop eval:

| Seed | Checkpoint | Full Acc | Worst Hop |
|---:|---|---:|---:|
| 42 | `core3_8n4h_strathop_polish2_seed42` | 99.56% | 98.72% |
| 101 | `core3_8n4h_strathop_seed101` | 99.70% | 98.90% |
| 202 | `core3_8n4h_strathop_seed202` | 99.60% | 98.54% |

Seed 42 originally underperformed because the teacher objective/optimization
left hop 4 weak. It was repaired by max-hop-focused polish, then second-stage
polish with lower LR and stronger final-loop loss.

### Main JumpRec Results

On the 8-node / 4-hop hard case:

- Full recurrent teacher uses 18 counted recurrent core layers.
- JumpRec agreement routing reaches teacher-level accuracy using roughly
  2.3 to 3.5 / 18 counted core layers depending on seed and threshold.
- No-agreement routing is faster but less reliable, especially on seed 42 and
  in held-out threshold audits.

Seed 42 repaired JumpRec:

| Policy | Threshold | Accuracy | Avg Core Layers | Savings |
|---|---:|---:|---:|---:|
| No agreement | 0.90 | 98.47% | 3.18 / 18 | 82.34% |
| Agreement | 0.90 | 99.56% | 3.33 / 18 | 81.47% |

Seed 101 baseline JumpRec:

| Policy | Threshold | Accuracy | Avg Core Layers | Savings |
|---|---:|---:|---:|---:|
| No agreement | 0.90 | 99.43% | 2.36 / 18 | 86.90% |
| Agreement | 0.90 | 99.77% | 2.40 / 18 | 86.66% |

Seed 202 baseline JumpRec:

Agreement routing was also teacher-level; see `JUMPREC_RESULTS.md` for exact
seed-202 tables.

### Prompt Artifact Audits

Corrected teacher prompt audit:

- Consistent node relabeling stays high across all three teachers.
- Map scrambling collapses near chance.
- Wrong-hop prompts collapse near chance.

JumpRec prompt audit:

- Same pattern holds for the deployable JumpRec path.
- Relabel stays high.
- Map scrambling and wrong-hop prompts collapse near chance.

Implication: current synthetic results are not explained by fixed node symbols,
map display order, or ignoring the hop field.

### Verifier Audit

Held-out threshold audit added in `72bcc88`:

- Thresholds selected on 64 validation batches.
- Final metrics reported on a separate 128-batch split.
- Verifier calibration measured.
- Oracle router upper bound reported and clearly labeled non-deployable.

Held-out agreement router:

| Seed | Final Teacher | Final Router | Avg Core Layers | Savings |
|---:|---:|---:|---:|---:|
| 42 | 99.49% | 99.30% | 3.09 / 18 | 82.85% |
| 101 | 99.69% | 99.79% | 2.33 / 18 | 87.07% |
| 202 | 99.60% | 99.66% | 3.24 / 18 | 81.99% |

Verifier calibration was strong:

| Seed | Mean Conf | Empirical Correctness | ECE-10 |
|---:|---:|---:|---:|
| 42 | 85.5% | 85.9% | 0.0040 |
| 101 | 91.1% | 91.4% | 0.0038 |
| 202 | 85.4% | 85.7% | 0.0050 |

Oracle router upper bound:

| Seed | Oracle Acc | Avg Core Layers | Savings |
|---:|---:|---:|---:|
| 42 | 99.98% | 2.67 / 18 | 85.19% |
| 101 | 99.96% | 2.01 / 18 | 88.82% |
| 202 | 99.95% | 2.69 / 18 | 85.07% |

Implication: the deployable agreement router is strong, but there is still
headroom in better budget selection.

### Component Ablation

Seed-101 ablations were run on the current SmolLM hard case, then the key
ablation story was replicated on seeds 42 and 202. Seed 42 must use the
repaired polish2 teacher checkpoint; `run_recurrent_smol.py` now includes
explicit `core3_8n4h_strathop_polish2_ablate_*` modes for that.

| Mode | Temp Adapter | Distill Loss | Verifier Loss | Agree 0.90 | Avg Core Layers | Savings |
|---|---|---:|---:|---:|---:|---:|
| Baseline | yes | 0.2 | 0.2 | 99.77% | 2.40 / 18 | 86.66% |
| No adapter | no | 0.2 | 0.2 | 99.80% | 2.34 / 18 | 86.98% |
| No distill | yes | 0.0 | 0.2 | 99.79% | 2.41 / 18 | 86.63% |
| No verifier | yes | 0.2 | 0.0 | 99.58% | 18.00 / 18 | 0.00% |

Replication on seeds 42 and 202:

| Seed | Mode | Teacher | c2 | c3 | Agree 0.90 | Avg Core Layers | Savings | No-Agree 0.90 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | No verifier | 99.53% | 99.30% | 99.61% | 99.56% | 18.00 / 18 | 0.00% | 99.56% |
| 202 | No verifier | 99.53% | 99.37% | 99.76% | 99.54% | 18.00 / 18 | 0.00% | 99.54% |
| 42 | No adapter | 99.53% | 99.17% | 99.41% | 99.50% | 3.41 / 18 | 81.04% | 98.70% |
| 202 | No adapter | 99.53% | 99.33% | 99.67% | 99.80% | 3.46 / 18 | 80.78% | 99.17% |

Implication:

- Fixed-budget jumping still learns without verifier loss.
- Adaptive compute savings collapse without verifier loss across seeds 42, 101,
  and 202.
- In the current 8-node / 4-hop SmolLM hard case, temp adapter is not
  load-bearing across checked seeds.
- Distillation was only checked on seed 101 and was not load-bearing there.
  Seeds 42/202 no-distill can be run later for table completeness, but it is
  not a blocker.
- This does not prove adapter/distillation are useless. Earlier 12-node/6-hop
  toy-runner results suggested the adapter mattered under harder max-hop and
  smaller correction-budget regimes.

## Core Interpretation

The concept looks valid in its current synthetic form:

- A looped/recurrent teacher can solve an iterative task robustly.
- A learned jump path can amortize recurrent depth.
- Ordinary tail loops after the jump preserve the recurrent computation story.
- A verifier/controller can make this adaptive and mostly teacher-quality.

The real architectural lesson is:

> JumpRec is not just "add a jump adapter." The architecture is a looped
> backbone plus a jump/landing module plus a verifier/controller.

For a general LLM version, the verifier/controller may be as important as the
jump itself.

## Main Caveats

1. Current strongest results are still synthetic pointer/graph traversal tasks.
2. Counted core-layer savings do not always equal wall-clock savings.
3. Agreement routing is quality-preserving but has serial overhead.
4. No-agreement routing is faster but less reliable.
5. Adapter/distillation importance appears task-regime dependent.
6. We have not yet shown transfer to broad natural-language reasoning.
7. We have not yet shown clean scaling beyond SmolLM2-135M.

## Immediate Next Steps

### 1. Improve Verifier/Controller

The verifier/controller is the key research object now. The project has moved
from "does the jump work?" to "can we route as well as possible?" The current
agreement router is good scientifically but expensive in wall-clock terms. The
next goal is a controller that preserves agreement-router quality while
approaching no-agreement or oracle-router cost.

Promising changes:

- Cost-aware verifier loss: reward early correct acceptance, penalize false
  acceptance more than fallback.
- Separate false-positive penalty for accepted wrong jumps.
- Calibrated confidence targets instead of only binary correctness.
- Budget-ranking loss: teach the router to choose the first sufficient budget.
- Stability features beyond adjacent-budget agreement.
- Train a small controller that predicts budget directly, with fallback.
- Use oracle-router traces as supervision for budget choice, while keeping
  oracle metrics clearly separate from deployable results.
- Report accepted precision, false-accept rate, coverage, average core layers,
  calibration, and held-out final accuracy for every controller variant.

Success criteria:

- Agreement-level quality with less agreement overhead.
- No-agreement policy becomes safer, especially on seed 42 and seed 202.
- Held-out threshold audit still passes.
- Calibration remains good.
- Wall-clock improves in the intended local/small-batch regime, not only counted
  core layers.

### 2. Equal Wall-Clock Controls

The strongest scientific claim is counted core-layer savings. The engineering
claim needs more wall-clock hygiene.

Needed controls:

- Compare full teacher vs serial JumpRec at batch size 1, 2, 4, 8, 16, 32, 64.
- Separate no-agreement vs agreement timing.
- Add an equal-wall-clock direct/shallow control where possible.
- Be explicit that local interactive inference is the target regime, not high
  throughput serving.

Known current pattern:

- Batch-1 and small-batch serial JumpRec can be faster.
- Agreement routing often has too much overhead in the current implementation.
- Counted savings are more dramatic than wall-clock savings.

### 3. Harder Synthetic / Algorithmic Bridge

We need tasks where recurrent depth should genuinely shine but which are closer
to LLM behavior.

Good bridge tasks:

- Natural-language pointer chasing.
- Multi-hop graph traversal expressed in text.
- Symbolic rewriting with multiple sequential transformations.
- Small proof/search tasks with explicit intermediate state.
- Program execution traces or simple algorithm simulation.

Keep the rule: do not jump to open-ended chat yet. First show a natural-language
reasoning benchmark where recurrence actually matters.

### 4. General LLM Architecture Direction

The likely publishable/general architecture is:

1. Use a modern small model backbone, initially SmolLM2-135M.
2. Insert or train a recurrent/looped segment.
3. Train a JumpRec landing module to skip recurrent depth.
4. Train a verifier/controller on deployment-available uncertainty features.
5. Preserve full-loop fallback.

Important design constraint:

- If the architecture is meant to be looped, it may need to be trained that way
  from early in the pipeline. Retrofitting a nearly finished pretrained model
  can work for synthetic tasks, but a cleaner general LLM may require recurrent
  training as a core part of adaptation, not a bolt-on at the end.

### 5. Paper Hygiene

Before any strong paper claim:

- Read and use `JUMPREC_ARTIFACT_AUDIT.md`.
- Keep `JUMPREC_RESULTS.md` as the source of record.
- Label oracle results clearly as oracle.
- Separate deployable routing from diagnostic routing.
- Always report teacher quality by hop/task.
- Always include direct and early-exit controls.
- Always report counted compute and wall-clock separately.
- Keep synthetic language in the claims until a non-synthetic task is positive.

## Suggested Next Run Order

1. Add a cost-aware verifier/controller objective and test it on seed 101.
2. Compare against current agreement routing, no-agreement routing, and oracle
   routing on held-out validation/final splits.
3. If controller quality/cost improves, confirm on seeds 42 and 202.
4. Run no-distill seeds 42/202 only as background table cleanup, not as a
   blocker.
5. Add a wall-clock-focused controller benchmark for batch sizes 1, 2, 4, 8,
   16, 32, and 64.
6. Start a natural-language graph traversal bridge task.
7. Only then move toward a general-use looped LLM artifact.

## Useful Commands

Run a Modal job:

```powershell
modal run run_recurrent_smol.py --mode core3_8n4h_strathop_verifier_audit --seed 101
```

Safer background launch on Windows with UTF-8 logs:

```powershell
$mode = "core3_8n4h_strathop_verifier_audit"
$seed = 101
$log = "modal_recurrent_smol_${mode}_seed${seed}_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
$cmd = "Set-Location -LiteralPath 'C:\Users\power\Documents\SMOKE'; `$env:PYTHONIOENCODING='utf-8'; `$env:PYTHONUTF8='1'; [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new(); modal run run_recurrent_smol.py --mode $mode --seed $seed *> '$log'"
Start-Process -FilePath powershell -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-Command',$cmd) -WindowStyle Hidden -PassThru
```

Check running Modal processes:

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match 'modal run run_recurrent_smol.py' } |
  Select-Object ProcessId, CommandLine |
  Format-List
```

Compile check:

```powershell
python -m py_compile .\run_recurrent_smol.py
```

Local dry guard:

```powershell
python .\run_recurrent_smol.py --mode dry_strathop_polish2_verifier_audit --local
```

## Bottom Line

The current state is encouraging:

- The synthetic JumpRec concept has real legs so far.
- The teacher robustness issue is mostly closed for the 8-node / 4-hop case.
- Prompt shortcut audits pass for teacher and JumpRec.
- Held-out verifier audits pass for agreement routing.
- No-verifier ablations across seeds show the jump learns but adaptive savings
  require verifier supervision.
- No-adapter ablations show the temp adapter is not load-bearing in the current
  SmolLM hard case.
- The verifier/controller is now clearly the most important next research
  target.

The next phase should be ruthless but constructive:

1. Design the best deployable router/controller possible.
2. Preserve agreement-router quality while reducing overhead.
3. Tie compute savings to wall-clock in the intended local-inference regime.
4. Bridge from synthetic pointer chasing to natural-language reasoning tasks.
