import numpy as np

def thermostat(obs):
    """Reactive proportional deadband controller, using indoor observation only."""
    t = float(obs[0])
    target = 22 if t < 22 else 26 if t > 26 else t
    return np.array([np.clip((target-t)/2, -1, 1)], np.float32)

def always_off(obs):
    return np.zeros(1, np.float32)
