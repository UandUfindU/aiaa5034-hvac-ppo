# AIAA5034 Final Project — Topic 16
## Code Agent Implementation Brief: PPO for Single-Room HVAC Control

> **Priority:** Implement, execute, verify, and document a complete **Python-script-based** experiment. No Jupyter Notebook is required or desired. The final artifact should be a reproducible runnable codebase, actual experimental outputs, and publication-ready figures. Do **not** fabricate experimental results.
>
> **Course topic (official):** “Building HVAC Control with PPO: Effect of Clip Range, GAE Lambda, and Epochs.” A controller selects heating/cooling power to balance temperature comfort and energy use in a simple room thermal model. Train PPO; ablate `clip_range`, `gae_lambda`, and number of epochs; report comfort violations, energy use, and training stability. Suggested tools: custom thermal model wrapped as Gymnasium environment and Stable-Baselines3.

## 1. Objective and scope

Build a small, clean, fully reproducible final-project codebase from scratch. Use **Stable-Baselines3 PPO** rather than rewriting PPO. Use Gymnasium to implement a **simple, physically interpretable single-zone thermal model**. The scientific question is how three PPO training hyperparameters affect learning and the comfort–energy tradeoff, **not** how to construct a high-fidelity building simulator.

Deliver:
1. A valid custom HVAC Gymnasium environment and unit/smoke tests.
2. A straightforward rule-based thermostat baseline.
3. Baseline PPO training and one-factor-at-a-time hyperparameter ablations.
4. Evaluations on identical held-out weather scenarios, with multiple training seeds.
5. Plots generated **programmatically from saved raw results**.
6. A machine-readable results table, a compact analysis summary, and exact commands to reproduce everything.

**Constraints:** No `.ipynb` implementation; only `.py` for implementation, training, evaluation and plots. Avoid heavy simulator dependencies such as EnergyPlus, Sinergym or complex external datasets. Keep the code understandable to a student studying PPO/GAE. Use English in plots and project files (code comments may be English or Chinese).

## 2. Environment specification (proposed starting design)

Implement a class `SingleRoomHVACEnv(gymnasium.Env)`.

- **Time interval**: `dt_hours = 1.0` (one step = one hour).
- **Episode horizon**: `168` steps (one week).
- **Action**: single continuous HVAC command `u_t ∈ [-1, 1]`; negative = cooling, positive = heating, zero = off. Use `spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)`.
- **Observation** (at least): current indoor temperature `T_in`, current outdoor temperature `T_out`, and periodic time encoding `sin(2πh/24), cos(2πh/24)`. Optional additional *observable* signal only if justified. Use a correctly shaped `float32` `Box` with sensible limits. Never expose future disturbances.
- **Thermal model** (simple first-order discrete-time model):

  `T_in(t+1) = T_in(t) + α [T_out(t) − T_in(t)] + β u_t + ξ_t`.

  Suggested starting constants: `α = 0.08` per step, `β = 2.0` °C per step at maximum action; modest zero-mean noise `ξ_t` (e.g. normal with σ = 0.05 °C). These are educational model assumptions, not calibrated engineering constants. Put values in a single config file and explain their units. Avoid implausible unbounded dynamics.
- **Outdoor temperature**: day/night sinusoidal profile with a per-episode sampled base temperature, amplitude, and phase; for example `T_out(t) = μ + A sin(2π (h − φ)/24)`. Sample `μ` in roughly `[18, 32]` °C and `A` in `[2, 6]` °C. Sample scenario parameters once per episode, seeded through `self.np_random`; optional small time-step noise. Training and held-out evaluation scenarios must be reproducible.
- **Initial indoor temperature**: choose from a plausible bounded range, e.g. `[20, 28]` °C.
- **Comfort band**: `[22, 26]` °C, with a nominal center of `24` °C.
- **Power/energy**: `P_t = P_max · |u_t|`, with `P_max = 3 kW` (toy approximation); `E_t = P_t · dt_hours` in kWh. Do **not** call the sum of normalized actions “kWh.”
- **Comfort violation**: `d_t = max(22 − T_in(t+1), 0, T_in(t+1) − 26)` in °C. Per-step indicator = `1[d_t > 0]`. Compute metrics from `T_in(t+1)` consistently, and state convention in documentation.
- **Reward**: for instance `r_t = −w_c · d_t² − w_e · E_t`. Choose and document weights so trivial always-off or always-max control does not dominate; do a short sanity check first. Suggested initial weights `w_c = 1`, `w_e = 0.05` are **starting points to validate, not claimed optimum**. Fix reward parameters for all ablations.
- **Return from `step`**: Gymnasium 5-tuple `(obs, reward, terminated, truncated, info)`. The episode ending due to horizon should be `truncated=True`, not falsely a physical terminal state. `info` must contain `indoor_temp`, `outdoor_temp`, `hvac_action`, `energy_kwh`, `comfort_violation`, `comfort_deviation_c` and optionally scenario metadata.
- **Reset**: implement `reset(seed=None, options=None)` properly; call `super().reset(seed=seed)`. A deterministic test scenario may be requested via `options` or a dedicated scenario constructor; do not allow training-time leakage of held-out seeds.

If small refinements are needed to produce a learnable environment, implement them but **do not change the environment between experimental conditions**. Any change should be explained in the README.

## 3. Baselines and model

### 3.1 Thermostat (non-RL reference)
Create a simple deadband or proportional thermostat that observes the **same information** and outputs an action in `[-1,1]`. It should turn heating/cooling on when indoor temperature leaves the band, and avoid gratuitous consumption when already comfortable. This baseline is useful to demonstrate that the RL task is meaningful. Evaluate under exactly the same held-out scenarios. Optional extra reference: always-off HVAC.

### 3.2 PPO
Use `stable_baselines3.PPO('MlpPolicy', env, ...)` with a small default MLP. A standard set of fixed settings:

- `learning_rate=3e-4`
- `gamma=0.99`
- `n_steps=1024`
- `batch_size=256`
- `ent_coef=0.0`
- `clip_range=0.2`
- `gae_lambda=0.95`
- `n_epochs=10`
- `total_timesteps=100_000` per seed/config in the main experiment (allow CLI override).

Respect the **same** fixed training budget, policy architecture, evaluation conditions, reward, and other hyperparameters across all ablations. SB3 `n_epochs` is the number of optimization passes over each sampled rollout buffer, **not** number of full environment passes or a stopping criterion.

### 3.3 Ablation matrix
Use one-factor-at-a-time, with a single shared baseline:

| Group | Name | `clip_range` | `gae_lambda` | `n_epochs` |
|---|---|---:|---:|---:|
| Baseline | `default` | 0.20 | 0.95 | 10 |
| Clip | `clip_010` | 0.10 | 0.95 | 10 |
| Clip | `clip_030` | 0.30 | 0.95 | 10 |
| GAE | `gae_090` | 0.20 | 0.90 | 10 |
| GAE | `gae_099` | 0.20 | 0.99 | 10 |
| Epochs | `epochs_3` | 0.20 | 0.95 | 3 |
| Epochs | `epochs_20` | 0.20 | 0.95 | 20 |

Seven unique configurations total. Suggested seeds: `[0, 1, 2]` (21 PPO runs). You can optionally run more seeds if computationally affordable. **Never claim a significant or monotonic effect merely because one noisy run looks better**.

Use a modest **smoke configuration** (e.g. one run, 2–5k steps, 2 evaluation scenarios) to test end-to-end execution, then run the full matrix. Write commands for both. Ensure the full experiment does not silently run when invoking tests.

## 4. Fair, reproducible evaluation

Training episodes use sampled scenarios from the training distribution. Final evaluation uses a fixed set of **held-out scenario seeds** (e.g. `[1000, 1001, …, 1019]`) that are independent of training seeds. All trained configurations and thermostat are evaluated on these *same* weather/disturbance sequences. Prefer pre-generating weather/noise tapes per held-out scenario or otherwise guaranteeing that action-dependent branch paths do not alter the disturbances.

- Use the PPO policy with `deterministic=True` at evaluation time.
- Record per-scenario/per-seed metrics (not just a global average).
- Evaluate **the final policy after the same nominal number of training steps**. If checkpoints are used to draw learning curves, do not select one model using the final test set.
- Suggested checkpoint/evaluation cadence: every 10k environment steps; evaluations use fixed non-training scenarios and are clearly distinguished from any final held-out test set if model selection occurs. Simpler: keep final-checkpoint comparison only, with fixed checkpoints for learning curves, and state how the plotted evaluation scenarios were used.
- Seed Python, NumPy, PyTorch, SB3 and the Gymnasium environment, and save full experiment config, package versions, hardware details, and git commit or code version when available. Exact bitwise reproducibility across machines is not promised.

### Required primary metrics

1. **Comfort violation rate** = percentage of evaluation steps with indoor temperature outside `[22,26]` °C. Lower is better.
2. **Energy consumption** = total `Σ E_t` per 168-step episode, in `kWh/episode`. Lower is better, but interpret jointly with comfort.
3. **Training stability** = show smoothed learning curves and seed-to-seed variability; use a defined numerical summary such as standard deviation of evaluation return across training seeds (for each checkpoint or at the final checkpoint). Do not substitute evaluation return for comfort/energy metrics.

### Useful secondary metrics

- Mean temperature violation magnitude (°C outside range).
- Evaluation episode return (with fixed reward definition).
- PPO internal `train/approx_kl`, `train/clip_fraction`, `train/entropy_loss`, `train/value_loss` or `train/explained_variance` where available. Clarify how these relate to `clip_range` and `n_epochs`.
- Optional compute/time comparison across configurations, clearly stating that more epochs usually require more gradient updates per rollout even with fixed environment steps.

Keep comfort, energy and return separately visible: an improvement in cumulative reward **alone** does not establish that both comfort and energy improved.

## 5. Plots and tables: actual saved data only

All figures must be generated by independent Python scripts from CSV/JSON outputs (never hardcoded illustration numbers). Save high-resolution PNG **and** vector PDF when feasible. Label axes, units, legend, tested hyperparameter, number of seeds. Use consistent colors/scales across parameter-group comparisons; show mean ± standard deviation or confidence band over seeds when meaningful. No Seaborn dependency needed; Matplotlib is sufficient.

Minimum outputs:

1. `fig_learning_curves.png` — eval return versus environment steps (overall or panels for Clip, GAE, Epochs), showing multi-seed mean and variability.
2. `fig_comfort_ablation.png` — final comfort violation rate by configuration; uncertainty across seeds.
3. `fig_energy_ablation.png` — final energy kWh/episode by configuration; uncertainty across seeds.
4. `fig_comfort_energy_tradeoff.png` — scatter plot: energy vs comfort violations; points for every PPO configuration and thermostat, annotated. A useful way to reveal tradeoffs.
5. `fig_temperature_trajectory.png` — example held-out scenario for selected PPO model and rule-based thermostat, plotting indoor temperature, outdoor temperature and comfort band; optionally action trace in a second panel. The chosen scenario must be declared, not cherry-picked silently.
6. `results_summary.csv` — one row per config/seed with mean metrics over held-out scenarios, plus an aggregated report with mean and standard deviation across training seeds.

Optional: `fig_ppo_diagnostics.png` with approximation KL and clip fraction versus environment steps by hyperparameter.

## 6. Pure-Python source tree (suggestion, adapt if justified)

```text
hvac-ppo-ablation/
├── README.md                         # installation, model assumptions, run commands, outputs
├── requirements.txt                  # compatible pinned versions, not unbounded latest
├── configs/
│   └── default.yaml                  # physics, reward, PPO settings, eval scenario seeds
├── src/
│   ├── __init__.py
│   ├── env.py                        # Gymnasium HVAC environment
│   ├── thermostat.py                 # non-RL control baseline
│   ├── config.py                     # config loading + validation
│   ├── train.py                      # one PPO configuration and seed
│   ├── evaluate.py                   # fixed-scenario eval; save per-episode metrics
│   ├── experiments.py                # matrix runner, --smoke and --full
│   ├── aggregate.py                  # statistics and CSV result generation
│   └── plot.py                       # all plots from saved results
├── tests/
│   ├── test_env.py
│   └── test_reproducibility.py
├── outputs/
│   ├── models/
│   ├── raw/                         # per-episode, per-checkpoint and PPO logger output
│   ├── tables/
│   ├── figures/
│   └── run_manifest.json
└── docs/
    └── experimental_findings.md     # concise facts, limitations, reproducibility info
```

All entry points should be runnable as `python -m src.experiments ...`, etc. Add `--help`, `--output-dir`, `--seed`, `--total-timesteps` where appropriate. Do not require notebook execution, cloud APIs or manual plotting steps.

## 7. Quality-control checklist

Implement and actually run tests covering:

- Gymnasium API correctness via `stable_baselines3.common.env_checker.check_env`.
- Action and observation dtypes/shapes/bounds; reward finite; horizon `truncated` handling.
- Thermal dynamics sanity: with equal initial conditions, positive HVAC action warms more than zero, negative action cools more than zero; at `u=0`, indoor temperature drifts toward outdoors absent noise.
- Correct energy units and `|u|` dependence; no energy consumed when action is zero.
- Reward/violation calculations on temperature inside and outside the comfort band.
- Resetting with the same seed/scenario yields identical exogenous weather/noise, regardless of the policy's actions.
- Exact ablation matrix: non-varied parameters are constant; single baseline is not run three times under different labels.
- Distinct train and eval seeds; identical evaluation scenarios across all methods.
- Plotting works from saved data alone and does not secretly retrain.
- Smoke mode finishes and emits nonempty output files. Run a full experiment if machine resources allow; if not, report precisely which configurations and seeds ran and which did not. **Never invent metrics or plots representing unexecuted training.**

## 8. Commands and final handoff

Provide executable commands in README, for example (exact CLI may differ but should be comparable):

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.experiments --smoke
python -m src.experiments --full --seeds 0 1 2 --total-timesteps 100000
python -m src.aggregate --input-dir outputs/raw --output-dir outputs/tables
python -m src.plot --input-dir outputs/tables --output-dir outputs/figures
```

At completion, report:

- What was implemented and which commands were actually executed.
- Exact files written and where the plots/results live.
- The seven model settings, training steps, training seeds and evaluation scenario seeds.
- A compact metrics table with real values and uncertainty; thermostat comparison.
- Findings for each of Clip Range, GAE λ and Epochs, explicitly separating observation from plausible theoretical explanation.
- Limitations (toy building physics, reward sensitivity, stochastic variance, omitted real-weather validation, compute limits).
- Citation for Stable-Baselines3 and any external code reused; **do not copy code from repositories with unclear licenses**. If consulting third-party source code, respect its license and attribution. Prioritize new, clean implementation here.

## 9. Suggested working order for the Code Agent

1. Read this task spec and the official Topic 16 instruction; inspect the local repo and available CPU/GPU.
2. Implement and validate `SingleRoomHVACEnv`; run unit tests and dynamic sanity tests.
3. Implement and evaluate the thermostat against a few fixed scenarios; inspect comfort/energy ranges.
4. Train a single PPO baseline in smoke mode and verify that both metrics are logged. Adjust obvious environment/reward defects **before freezing the experimental settings**; then freeze them.
5. Run the complete 7-configuration ablation matrix over three training seeds, with periodic logging, consistent budgets and held-out evaluation.
6. Aggregate and plot the real results; interpret whether learned strategies are reasonable. If baseline PPO fails, diagnose first rather than plotting meaningless comparisons as success.
7. Write concise README and `docs/experimental_findings.md`, enumerate any unexecuted steps, and provide all generated source/data/figures.

**Definition of done:** Another student can clone/open the folder, run the installation/test/train/eval/plot commands, and reproduce the tables and figures for the course project without opening a single Notebook.
