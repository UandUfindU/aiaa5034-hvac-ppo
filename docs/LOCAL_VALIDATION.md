# Local validation: 2026-11-05

Interpreter: `C:\Users\fyy\tools\miniconda\envs\pytorch_learn\python.exe`.
PyTorch 2.5.1 CUDA 11.8, CUDA available. Project dependencies isolated in `.deps`.

Executed successfully:

```
python -m pytest -q
# 7 passed in 2.22 seconds
python -m src.experiments --smoke --device cuda --output-dir outputs/smoke
# Completed default seed=0, actual_steps=2048
python -m src.aggregate --input-dir outputs/smoke --output-dir outputs/smoke/tables
python -m src.plot --input-dir outputs/smoke/tables --output-dir outputs/smoke/figures
```

All five PNG and PDF figures were generated. Learning and temperature plots were
visually inspected. Smoke uses two test scenarios (1000,1001), one training seed
and 2048 steps only, so the numbers below are **not full-experiment findings**.

| Method | Return | Energy (kWh/episode) | Comfort violation (%) |
|---|---:|---:|---:|
| default, seed 0 | -3.706664 | 35.049424 | 4.166667 |
| thermostat | -1.770002 | 10.673687 | 27.380952 |
| always-off | -48.255612 | 0.000000 | 52.083333 |

The sanity check distinguishes energy versus comfort and avoids assuming that
higher return improves both metrics. Full 21-run cloud validation is outstanding.
Smoke began before initial Git initialization; its manifest truthfully has no
commit SHA and includes source hashes. Cloud runs must record a clean pinned SHA.
