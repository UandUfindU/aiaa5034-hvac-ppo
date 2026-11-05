"""Run real experiments. Test scenarios are never used for model selection."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from .config import load, MATRIX
from .env import SingleRoomHVACEnv
from .evaluate import evaluate
from .thermostat import thermostat, always_off

class ValidationCallback(BaseCallback):
    def __init__(self, c, path, name, seed):
        super().__init__(); self.c=c; self.path=path; self.name=name; self.seed=seed; self.rows=[]
    def measure(self):
        df,_=evaluate(self.model,self.c['validation_seeds'],self.c['physics'],self.name,self.seed,self.num_timesteps)
        self.rows.extend(df.to_dict('records'))
        pd.DataFrame(self.rows).to_csv(self.path,index=False)
    def _on_training_start(self):
        self.measure()
    def _on_step(self):
        return True
    def _on_rollout_start(self):
        if self.num_timesteps and self.num_timesteps % 10240 == 0:
            self.measure()
    def _on_training_end(self):
        self.measure()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke',action='store_true'); mode.add_argument('--full',action='store_true')
    p.add_argument('--output-dir',default='outputs')
    p.add_argument('--seeds',type=int,nargs='+'); p.add_argument('--seed',type=int)
    p.add_argument('--configs',nargs='+',choices=list(MATRIX)); p.add_argument('--total-timesteps',type=int)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=p.parse_args(); c=load()
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no silent CPU fallback')
    torch.set_num_threads(1)
    seeds=args.seeds or ([args.seed] if args.seed is not None else ([0] if args.smoke else c['seeds']))
    if set(seeds) & (set(c['test_seeds'])|set(c['validation_seeds'])):
        p.error('Training seeds overlap evaluation seeds')
    names=args.configs or (['default'] if args.smoke else list(MATRIX))
    steps=args.total_timesteps or (2048 if args.smoke else c['total_timesteps'])
    if args.smoke:
        c['test_seeds']=c['test_seeds'][:2]; c['validation_seeds']=c['validation_seeds'][:2]
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())
    except (subprocess.CalledProcessError,FileNotFoundError):
        commit=None; dirty=None
    versions={x:importlib.metadata.version(x) for x in ['torch','numpy','stable-baselines3','gymnasium','pandas','matplotlib']}
    hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in list(Path('src').glob('*.py'))+list(Path('configs').glob('*'))}
    manifest=dict(settings=c, matrix=MATRIX, requested_steps=steps, seeds=seeds, configs=names,
                  commit=commit,dirty=dirty,source_sha256=hashes,versions=versions,device=args.device,
                  hardware=dict(platform=platform.platform(),processor=platform.processor(),
                                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))
    # Unique per invocation: independent scheduler jobs cannot overwrite one another's manifests.
    tag='_'.join(names)+'_'+'_'.join(map(str,seeds))
    (out/f'manifest_{tag}.json').write_text(json.dumps(manifest,indent=2))
    for name in names:
        for seed in seeds:
            run=out/'raw'/f'{name}_seed{seed}'
            if (run/'final.csv').exists():
                raise FileExistsError(f'{run}: refusing to overwrite completed run')
            run.mkdir(parents=True,exist_ok=True)
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            env=Monitor(SingleRoomHVACEnv(c['physics']),str(run/'monitor.csv'))
            model=PPO('MlpPolicy',env,seed=seed,device=args.device,verbose=0,
                      **c['ppo'],**MATRIX[name])
            model.set_logger(configure(str(run),['csv']))
            callback=ValidationCallback(c,run/'validation.csv',name,seed)
            start=time.monotonic(); model.learn(total_timesteps=steps,callback=callback)
            model.save(run/'model')
            final,trace=evaluate(model,c['test_seeds'],c['physics'],name,seed,model.num_timesteps,True)
            final.to_csv(run/'final.csv',index=False)
            # Prespecified example: first held-out seed and first training seed.
            trace[trace.scenario==c['test_seeds'][0]].to_csv(run/'trajectory.csv',index=False)
            (run/'timing.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,actual_steps=model.num_timesteps)))
            env.close(); print(f'Completed {name} seed={seed}, actual_steps={model.num_timesteps}',flush=True)
    for name,policy in [('thermostat',thermostat),('always_off',always_off)]:
        # Per invocation directories also avoid baseline writes from competing scheduler tasks.
        run=out/'baselines'/tag/name; run.mkdir(parents=True,exist_ok=True)
        df,trace=evaluate(policy,c['test_seeds'],c['physics'],name,-1,0,True)
        df.to_csv(run/'final.csv',index=False)
        trace[trace.scenario==c['test_seeds'][0]].to_csv(run/'trajectory.csv',index=False)

if __name__=='__main__': main()
