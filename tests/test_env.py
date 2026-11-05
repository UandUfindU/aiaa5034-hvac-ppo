import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env
from src.config import load, MATRIX
from src.env import SingleRoomHVACEnv

def test_api_horizon():
    e=SingleRoomHVACEnv(); check_env(e)
    obs,_=e.reset(seed=0); assert obs.dtype==np.float32
    for t in range(168):
        obs,r,term,trunc,info=e.step(np.array([0],np.float32))
        assert e.observation_space.contains(obs) and np.isfinite(r) and not term
        assert trunc==(t==167)
        assert info['energy_kwh']==0

def test_dynamics():
    temps=[]
    for u in [-1,0,1]:
        e=SingleRoomHVACEnv(); e.reset(seed=5,options=dict(mean=30,amplitude=0,initial_temp=24))
        e.noise[:]=0
        obs,r,_,_,info=e.step([u]); temps.append(float(obs[0]))
        assert info['energy_kwh']==3*abs(u)
        d=max(22-info['indoor_temp'],0,info['indoor_temp']-26)
        assert r==pytest.approx(-d*d-.05*3*abs(u))
    assert temps[0]<temps[1]<temps[2] and temps[1]>24

@pytest.mark.parametrize('temp',[18,24,30])
def test_reward(temp):
    e=SingleRoomHVACEnv(); e.reset(seed=1,options=dict(mean=temp,amplitude=0,initial_temp=temp)); e.noise[:]=0
    _,r,_,_,info=e.step([0]); d=max(22-temp,0,temp-26)
    assert info['comfort_deviation_c']==d and info['comfort_violation']==int(d>0)
    assert r==-d*d

def test_tapes():
    a,b=SingleRoomHVACEnv(),SingleRoomHVACEnv(); a.reset(seed=1000); b.reset(seed=1000)
    for _ in range(168): a.step([-1]); b.step([1])
    np.testing.assert_array_equal(a.weather,b.weather); np.testing.assert_array_equal(a.noise,b.noise)
    assert a.temp!=b.temp

def test_matrix_and_seeds():
    c=load(); assert len(MATRIX)==7
    for n,v in MATRIX.items():
        assert sum(v[k]!=MATRIX['default'][k] for k in v)==(0 if n=='default' else 1)
    assert not set(c['seeds'])&set(c['test_seeds'])
