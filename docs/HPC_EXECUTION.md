# Frozen HPC execution protocol (2026-11-05)

The scientific settings in configs/default.json and seven-condition matrix are
unchanged from aa99d381b5af400f728f948bc8b85ff2d9e630de. Train seeds 0,1,2;
100000 nominal / 100352 actual steps; final scenarios 1000--1019; validation
2000--2003. No checkpoint selection or final-test tuning. One torch thread.

Project-only Python 3.12 venv at
/data/user/yfeng044/projects/aiaa5034-hvac-assets/venv accesses existing torch
2.13.0+cu130 read-only through a .pth entry. All requirements.txt dependencies
are installed in the project venv with their pinned versions. This differs from
local Python 3.11 / torch 2.5.1; scheduler tests and explicit CPU/CUDA benchmarks
must pass before full execution. The shared environment is never modified.

cluster/job.sh runs tests, CPU and CUDA 2048-step smokes, full runs split by seed,
and aggregation. Device choice will be recorded from measured smoke timings.
Each full seed writes disjoint raw condition/seed directories and manifests.
Reference outputs are per invocation and deduplicated by the aggregator.

All jobs require explicit --time, --output and --error and use 1 CPU. Preflight
requests one GPU. Full runs use CPU if supported by measured benchmark. Runtime
artifacts (models, logs, raw evaluation files) remain under
/data/user/yfeng044/projects/aiaa5034-hvac-artifacts on HPC3. Bounded exports go
in results/ only after completeness checks. GreenCloud retains metadata only.

The external singleton controller records reservations BEFORE each sbatch,
then records job ID immediately. Total allocated CPU/GPU accounting including
failed runs must remain <=32 CPUh and <=8 GPUh, <=4 concurrent project jobs.
Deadline 2026-11-07T07:21Z. Reservations are replaced only by sacct allocated
TRES elapsed accounting. No unrelated jobs are cancelled. No progress email;
terminal notification is deduplicated by the existing outer supervisor.

## Passed preflight and device decision

Slurm 801894 (debug, ACD1-63): COMPLETED exit 0, 86 allocated seconds,
1 CPU / 1 GPU = 0.023889 CPUh and GPUh. Seven tests passed. Both explicit
CPU/CUDA smoke runs reached 2048 steps, and aggregate/plot emitted five PNG/PDF
pairs. Training/evaluation timer: CPU 1.226535 s, CUDA 3.199766 s (one seed,
small MLP, one CPU thread). Choose CPU for all full conditions. This measurement
excludes interpreter startup and is a short benchmark, not a general hardware claim.
Thermostat smoke return -1.770002 versus off -48.255612 verifies reward sanity;
2048-step PPO is not a scientific performance result. No physics/reward/settings
were adjusted based on these test scenarios. All full jobs use the original
frozen settings and final checkpoints. Full jobs request 30 minutes per seed
(one CPU, no GPU); measured smoke implies ample headroom for seven conditions.
