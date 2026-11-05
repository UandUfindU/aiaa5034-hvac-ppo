"""Validate completeness and provenance without selecting or modifying policies."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from stable_baselines3 import PPO
from src.config import MATRIX, load

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); a=p.parse_args()
    root=Path(a.root); c=load(); artifacts=[]; times=[]
    manifests=[json.loads(f.read_text()) for f in root.glob('manifest_*.json')]
    assert len(manifests)==3
    assert len({m['commit'] for m in manifests})==1
    assert all(m['dirty'] is False and m['device']=='cpu' and m['settings']==c for m in manifests)
    for name in MATRIX:
        for seed in c['seeds']:
            run=root/'raw'/f'{name}_seed{seed}'; df=pd.read_csv(run/'final.csv')
            assert len(df)==20 and sorted(df.scenario)==c['test_seeds']
            assert set(df.config)=={name} and set(df.seed)=={seed} and set(df.steps)=={100352}
            assert not df.duplicated(['config','seed','scenario']).any()
            assert df.select_dtypes('number').notna().all().all()
            v=pd.read_csv(run/'validation.csv')
            assert set(v.scenario)==set(c['validation_seeds'])
            assert sorted(v.steps.unique())==[0,*range(10240,100000,10240),100352]
            assert len(v)==44
            t=json.loads((run/'timing.json').read_text()); assert t['actual_steps']==100352
            times.append(dict(config=name,seed=seed,**t))
            model=PPO.load(run/'model.zip',device='cpu')
            assert model.num_timesteps==100352
            for f in sorted(run.iterdir()):
                if f.is_file():
                    artifacts.append(dict(path=str(f),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    for name in ['thermostat','always_off']:
        frames=[pd.read_csv(f) for f in root.glob(f'baselines/*/{name}/final.csv')]
        assert len(frames)==3
        for f in frames[1:]: pd.testing.assert_frame_equal(frames[0],f)
        assert len(frames[0])==20 and sorted(frames[0].scenario)==c['test_seeds']
    result=dict(models=21,ppo_episode_rows=420,reference_episode_rows=40,
                training_sha=manifests[0]['commit'],root=str(root),artifact_hashes=artifacts)
    (root/'completeness.json').write_text(json.dumps(result,indent=2))
    pd.DataFrame(times).to_csv(root/'timings.csv',index=False)
    print(json.dumps({k:v for k,v in result.items() if k!='artifact_hashes'}))
if __name__=='__main__': main()
