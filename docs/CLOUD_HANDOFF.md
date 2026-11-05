# Authorized cloud execution handoff

## Latest user instruction: two independent GitHub projects

As of 2026-11-05 the user explicitly requires separate code and report projects.
Code: https://github.com/UandUfindU/aiaa5034-hvac-ppo .
Report: https://github.com/UandUfindU/aiaa5034-hvac-ppo-report .
This supersedes instructions below to commit final exports or findings in the code
repository. Put final figures, bounded tables, manifests, analysis and LaTeX manuscript
in the report repository; retain runnable source/config/tests in the code repository.
Report initial scaffold SHA: 96d274c (resolve full SHA through GitHub before checkout).
The report will be uploaded to Overleaf: `main.tex`, standard packages, relative asset
paths and bibliography; compile and visually validate final PDF, target 10 total
pages required by the course. Include exact executed code commit and artifact hashes.
Do not invent author/team identity. Do not configure/upload to Overleaf without access.
Keep existing experiment jobs and singleton supervisor; adjust destinations without
restarting completed training. Completion mail should link both final commits.

User request (2026-11-05, Asia/Shanghai): create a homework subdirectory, implement
the task; prefer cloud execution; submit code to cloud, which owns making it work
and returning real results, notify by email, use GitHub as exchange medium.

Repository: https://github.com/UandUfindU/aiaa5034-hvac-ppo (private).
The dispatch prompt supplies the immutable starting SHA. Fetch that exact commit
on authorized HPC and verify HEAD. Read TASK_SPEC.md and README.md completely.
Do not clone experiment code or install training dependencies on GreenCloud VPS.

## Scope and execution

1. Inspect existing HPC access, scheduler, available devices, environment and jobs.
   Read BASE-STATION.md and USER-PREFERENCES.md first. Do not expose credentials.
2. In a separate HPC worktree run tests, a 2048-step smoke and reward sanity checks
   (thermostat versus always-off). Benchmark CPU versus GPU if useful: SB3 small
   MLP PPO is often CPU-bound. Prefer available cloud GPU when useful; record the
   measured choice. Do not silently switch a CUDA request to CPU.
3. Fix bugs, commit and push tested changes before starting the full matrix. Initial
   implementation is intentionally reviewable; cloud owns verification and repair.
   Any scientifically needed observation scaling or reward change must be justified
   before freezing a protocol; rerun affected experiments consistently. Never tune
   settings using final test results. Diagnose failure using training/validation.
4. Execute seven unique conditions x three seeds, 100000 requested steps each,
   20 identical held-out scenarios, with prescribed validation curves. Preserve raw
   data, checkpoint hashes, environment versions, hardware, job IDs, timing and
   actual steps. Use scheduler arrays or bounded parallel jobs without oversubscription.
5. Aggregate, inspect five PNG/PDF figures, document actual metrics and uncertainty,
   observations for clip/GAE/epochs, limitations and reproduction commands.
   Record completeness: 21 models, 420 final PPO scenario rows plus references.
6. Commit bounded exports under `results/` (CSV tables, five PNG/PDF figures, manifests,
   tests/smoke evidence, artifact index with HPC paths and checksums). Full progress
   logs and models stay on HPC. Update docs/experimental_findings.md and README.
   GitHub is the durable exchange channel; push a final results commit and update
   a dedicated GitHub issue with exact run status, commit and artifact paths.

## Limits and continuation

Maximum 4 simultaneous experiment jobs, 8 GPU-hours and 32 CPU-hours total, 48-hour
wall-clock deadline from dispatch; use materially less where possible. No paid
resource purchase. No budget resets or unrelated job cancellation. Do not rerun
completed conditions unless a documented validity bug requires it. All full
conditions must share the frozen settings and actual rollout count.

Own execution through completion. If scheduler work outlives a Codex process, use
a **singleton bounded project-specific server-side supervisor**, persisting state,
remaining budget, run SHA/job IDs, and deadline. It may resume repair and execution
within this scope; stop at completion, deadline, budget exhaustion or human blocker.
Check for an existing supervisor before creating one. Do not create a laptop
automation. Recoverable job failures are work to repair, not terminal success.

## Authorized email

Use existing `/home/researcher/control/bin/research-mail.py` and its fixed configured
recipient; inspect helper status/help without reading SMTP credential files. Send
one deduplicated completion email including repository/results commit and compact
metrics, or one new actionable-blocker email if human intervention is necessary.
No progress spam, arbitrary recipients, attachments, credential disclosure or
automatic resend after unknown/failed delivery. Record delivery status and helper
event ID in durable project status; do not claim delivery without evidence.
