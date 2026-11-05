import json
from pathlib import Path

MATRIX = {
    'default': dict(clip_range=.2, gae_lambda=.95, n_epochs=10),
    'clip_010': dict(clip_range=.1, gae_lambda=.95, n_epochs=10),
    'clip_030': dict(clip_range=.3, gae_lambda=.95, n_epochs=10),
    'gae_090': dict(clip_range=.2, gae_lambda=.90, n_epochs=10),
    'gae_099': dict(clip_range=.2, gae_lambda=.99, n_epochs=10),
    'epochs_3': dict(clip_range=.2, gae_lambda=.95, n_epochs=3),
    'epochs_20': dict(clip_range=.2, gae_lambda=.95, n_epochs=20),
}

def load():
    c = json.loads((Path(__file__).resolve().parents[1] / 'configs/default.json').read_text())
    assert not (set(c['test_seeds']) & set(c['validation_seeds']))
    assert not (set(c['seeds']) & set(c['test_seeds']))
    return c
