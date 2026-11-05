# Actual HPC experimental findings (2026-11-05)

All seven frozen conditions completed for training seeds 0,1,2: 21 models,
420 final PPO scenario rows plus 40 distinct reference rows. Every policy reached
100352 actual steps (100000 nominal). Final scenarios 1000--1019; separate
validation scenarios 2000--2003. Final checkpoints only; no model selection or
test-set tuning. All models were successfully reloaded on CPU during verification.
Executed training commit: `c63e6bae6305ad875c9bc0c0724b6593f06a5388`.

## Held-out results

Entries are mean +/- sample SD across three training-seed means, each evaluated
on the same 20 scenarios. SD is training-seed variation, not a standard error or
scenario SD. References have no training-seed SD. Energy is per 168-hour episode.

| Condition | Return | Energy (kWh) | Comfort violations (%) |
|---|---:|---:|---:|
| default | -4.884 +/- 0.960 | 78.952 +/- 4.121 | 1.151 +/- 0.981 |
| clip_010 | -5.726 +/- 1.718 | 79.361 +/- 4.906 | 1.667 +/- 0.438 |
| clip_030 | -4.639 +/- 1.038 | 77.540 +/- 1.875 | 1.349 +/- 0.467 |
| gae_090 | -3.724 +/- 0.153 | 74.201 +/- 3.187 | 0.079 +/- 0.017 |
| gae_099 | -321.139 +/- 58.065 | 49.449 +/- 13.835 | 66.300 +/- 8.035 |
| epochs_3 | -733.622 +/- 239.872 | 52.232 +/- 9.503 | 67.500 +/- 3.960 |
| epochs_20 | -3.455 +/- 0.139 | 68.600 +/- 2.756 | 0.278 +/- 0.253 |
| thermostat | -9.852 | 31.751 | 42.768 |
| always_off | -749.993 | 0.000 | 65.357 |

## Observations and interpretation

- Clip range: 0.10, 0.20 and 0.30 all learned low-violation policies. Default
  comfort was 1.151 +/- 0.981%, versus 1.667 +/- 0.438% at 0.10 and
  1.349 +/- 0.467% at 0.30. Energy means were similar (77.54--79.36 kWh).
  These three seeds do not establish a monotonic clip effect or significance.
- GAE lambda: 0.90 had 0.079 +/- 0.017% violations and return -3.724 +/-
  0.153, while 0.99 had 66.300 +/- 8.035% violations and return -321.139 +/-
  58.065 at this fixed budget. Validation curves also show slower, variable
  learning at 0.99. Longer-horizon advantage estimation may affect optimization
  variance, but these observations do not establish that causal mechanism.
- Epochs: 20 reached 0.278 +/- 0.253% violations and 68.600 +/- 2.756 kWh;
  3 epochs yielded 67.500 +/- 3.960% and 52.232 +/- 9.503 kWh. The latter is
  an observed poor-learning condition, not missing data or a successful energy
  saving strategy. Fewer updates may underfit this rollout budget; this remains
  an interpretation. More epochs also take more compute.
- Thermostat consumes less energy (31.751 kWh) than the successful PPO policies,
  but violates comfort on 42.768% of steps. Always-off uses zero energy and
  violates comfort on 65.357%. Successful PPO improves comfort and reward here,
  while using more energy than these references. Reward alone does not imply
  simultaneous improvement of both metrics.

## Stability and compute

Validation return is measured initially, every 10240 steps, and finally. Curves
use a trailing three-checkpoint mean within seed then mean +/- sample SD across
seeds; smoothing does not choose a checkpoint. Final return SD ranges from
0.139 (epochs_20) to 239.872 (epochs_3). Final return SD is one stability summary;
it is kept separate from comfort and energy. No hypothesis test was performed.

Full per-run wall times, including validation/final evaluation and saving, are in
results/timings.csv. All conditions used one CPU thread and identical environment
interaction counts; optimization work differs with epochs.

| Condition | Run time, mean +/- sample SD (s) |
|---|---:|
| default | 35.251 +/- 0.269 |
| clip_010 | 34.228 +/- 0.140 |
| clip_030 | 34.587 +/- 0.189 |
| gae_090 | 34.677 +/- 0.317 |
| gae_099 | 35.248 +/- 0.142 |
| epochs_3 | 28.722 +/- 0.218 |
| epochs_20 | 43.372 +/- 0.307 |

CPU/CUDA 2048-step timers were 1.226535/3.199766 seconds on preflight node
ACD1-63; CPU was explicitly chosen. Training jobs 801907/801908/801909 ran on
ACD1-28, allocated 298/300/297 seconds respectively, one CPU and no GPU each.
Preflight 801894 allocated one CPU and GPU for 86 seconds. Initial verifier/plot
job 801918 allocated one CPU for 27 seconds. Additional figure-only checks are
recorded in results/resource_accounting.json; no policies were retrained.

## Runtime, artifacts and limitations

Cloud runtime: Python 3.12, torch 2.13.0+cu130; all seven requirements.txt packages
retain pinned versions. This differs from local Python 3.11 / torch 2.5.1.
Seven tests passed on HPC; CPU/CUDA smokes, aggregation/plotting, and all saved
model reloads passed. Small dependency wheels were SHA256-verified after direct
HPC2-to-HPC3 transfer. Existing shared torch environment remained unmodified.
Manifests record versions and source hashes. Exact bitwise reproduction across
software/hardware versions is not promised.

The five PNG/PDF figures come only from saved tables. Temperature example is the
prespecified scenario 1000/default seed 0, not a selected best episode. Visual
inspection confirmed axes, uncertainty and trajectory; tradeoff layout was
improved to separate low-violation labels. Bounded tables/figures/manifests and
SHA256 provenance are in results/. Full model, raw evaluation and logging
artifacts remain at /data/user/yfeng044/projects/aiaa5034-hvac-artifacts/full
on HPC3; results/completeness.json indexes their sizes and hashes.

This is an educational, uncalibrated first-order room model with synthetic weather,
a fixed reward, unscaled temperature observations and a reactive thermostat.
Three seeds, 20 scenarios and a fixed short training budget limit generalization.
No real weather validation, building deployment, statistical significance or
uniform PPO success is claimed. Negative conditions were preserved without tuning
or retries. The ten-page team manuscript/presentation is a separate deliverable;
this execution follows the current controller request for code-repository exports.
