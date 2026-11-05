# AIAA5034 Topic 16: PPO for Single-Room HVAC

Pure Python implementation of a toy thermal environment, reactive thermostat and
always-off references, and seven one-factor-at-a-time Stable-Baselines3 PPO conditions.
See `TASK_SPEC.md` for the complete assignment. No experimental finding is assumed
before execution. Final cloud results, figures and the Overleaf report belong in the
separate private repository https://github.com/UandUfindU/aiaa5034-hvac-ppo-report.
This repository owns runnable code, configuration, tests and technical documentation.

## Installation and commands

Use Python 3.10 or 3.11. Install PyTorch 2.5.1 with the appropriate CUDA build first,
then `python -m pip install -r requirements.txt`. Local verified GPU interpreter:
`C:\Users\fyy\tools\miniconda\envs\pytorch_learn\python.exe`.
Local optional dependencies were installed into `.deps`; use `$env:PYTHONPATH="$PWD\.deps"`
in PowerShell for that installation. It is excluded from Git.

```bash
python -m pytest -q
python -m src.experiments --smoke --device cuda --output-dir outputs/smoke
python -m src.aggregate --input-dir outputs/smoke --output-dir outputs/smoke/tables
python -m src.plot --input-dir outputs/smoke/tables --output-dir outputs/smoke/figures
python -m src.experiments --full --device cuda --seeds 0 1 2 --total-timesteps 100000 --output-dir outputs/full
python -m src.aggregate --input-dir outputs/full --output-dir outputs/full/tables
python -m src.plot --input-dir outputs/full/tables --output-dir outputs/full/figures
```

CPU execution must be explicitly requested with `--device cpu`. Small MLP PPO may
be faster on CPU; the cloud executor should benchmark before selecting the device,
and document its choice. CUDA requests fail if unavailable. Do not train on a login
node or on the GreenCloud dispatch VPS. Full-matrix execution defaults to sequential;
Slurm jobs may split `--configs` and `--seed` with disjoint run directories.

## Frozen experimental protocol

The configuration lives in `configs/default.json`. Indoor temperature evolves as
`T_next = T + 0.08*(T_out-T) + 2*u + noise`. One step is one hour; episodes last 168
hours. Episode weather has mean uniform [18,32] C, amplitude uniform [2,6] C and phase
uniform [0,24] hours. Noise is Gaussian sigma 0.05 C, clipped to +/-0.25 C to bound
extreme disturbances. Weather and noise are generated at reset before any action.
Initial temperature is uniform [20,28] C. Observations contain current indoor and
outdoor temperatures and daily sine/cosine, without future weather. No normalization
or simulator calibration is claimed. Default SB3 separate 64x64 tanh MLPs are fixed.

Energy per step is `3 kW * abs(u) * 1 hour`. Comfort deviation is distance of the
**post-action temperature** outside [22,26] C. Reward is `-deviation**2 - 0.05*energy`.
Horizon endings are truncations. The thermostat is reactive: off within the band,
otherwise proportional correction `(nearest_band_edge - indoor)/2`, clipped [-1,1].
Always-off is an additional reward sanity reference, not a trained method.

| Condition | Clip range | GAE lambda | Optimization epochs |
|---|---:|---:|---:|
| default | .20 | .95 | 10 |
| clip_010 | .10 | .95 | 10 |
| clip_030 | .30 | .95 | 10 |
| gae_090 | .20 | .90 | 10 |
| gae_099 | .20 | .99 | 10 |
| epochs_3 | .20 | .95 | 3 |
| epochs_20 | .20 | .95 | 20 |

Train seeds: 0,1,2. Final test scenarios: 1000--1019. Validation learning curves:
2000--2003, measured initially, every 10,240 steps and at the final checkpoint.
No checkpoint selection is performed. SB3 completes whole 1024-step rollouts, so
100,000 requested steps yields **100,352 actual steps**, identically for all runs.
Evaluation is deterministic. Disturbance tapes depend only on scenario seed.
The prespecified temperature example is test seed 1000 and PPO training seed 0.

## Output interpretation

`outputs/*/raw/<condition>_seed<seed>/` holds final per-scenario CSV, validation CSV,
training monitor/progress CSV, model and timing; manifests record package versions,
source hashes, commit, hardware and requested budget. `tables/results_summary.csv`
averages scenarios per training seed; `aggregated.csv` reports mean and sample SD
across training seeds, including final return SD as a stability measure. Reference
controllers have one deterministic evaluation and undefined training-seed SD.
Learning curves use a trailing three-checkpoint mean per seed before seed aggregation.
All figures read saved tables only and export PNG at 300 dpi plus vector PDF.
Comfort and energy must be interpreted together. Three seeds do not establish
significance or monotonicity. Toy physics, reward sensitivity, finite seeds and no
real weather validation limit generalization. Smoke outputs are engineering checks,
not scientific conclusions.

## Cloud ownership and results

The user authorizes GreenCloud to execute, debug, commit fixes, push results via this
private GitHub repository, and email completion or actionable blockers using its
existing configured mail helper. See `docs/CLOUD_HANDOFF.md` for bounded execution.
This Git repository stores source; the separate report Git repository stores bounded
CSV/figures/reports and executable-code provenance. Full logs and models remain on HPC.
Never put credentials in Git. Preserve all unrelated cloud jobs and project files.

## Attribution

PPO is provided by Stable-Baselines3; environment and experiment orchestration are
original project code. No third-party repository source was copied.

- Raffin et al. (2021), *Stable-Baselines3: Reliable Reinforcement Learning Implementations*,
  JMLR 22(268):1--8. https://jmlr.org/papers/v22/20-1364.html
- SB3 2.4.1 PPO API and device guidance:
  https://stable-baselines3.readthedocs.io/en/v2.4.1/modules/ppo.html
- Schulman et al. (2017), *Proximal Policy Optimization Algorithms*, arXiv:1707.06347.

The official course also requires a ten-page team report and a presentation; these
are separate from the runnable code-agent deliverable described in `TASK_SPEC.md`.
