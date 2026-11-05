import pandas as pd
from .env import SingleRoomHVACEnv

def evaluate(policy, seeds, config, name, train_seed, steps, trajectory=False):
    rows, traces = [], []
    for scenario in seeds:
        env = SingleRoomHVACEnv(config)
        obs, _ = env.reset(seed=int(scenario))
        ret = energy = violations = deviation = 0.
        for t in range(config['horizon']):
            action = policy.predict(obs, deterministic=True)[0] if hasattr(policy,'predict') else policy(obs)
            obs, reward, _, _, info = env.step(action)
            ret += reward; energy += info['energy_kwh']
            violations += info['comfort_violation']; deviation += info['comfort_deviation_c']
            if trajectory:
                traces.append(dict(config=name, seed=train_seed, scenario=scenario, hour=t+1, **info))
        rows.append(dict(config=name, seed=train_seed, scenario=scenario, steps=steps,
                         episode_return=ret, energy_kwh=energy, comfort_pct=100*violations/config['horizon'],
                         deviation_c=deviation/config['horizon']))
        env.close()
    return pd.DataFrame(rows), pd.DataFrame(traces)
