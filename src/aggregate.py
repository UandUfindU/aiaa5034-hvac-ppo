import argparse
from pathlib import Path
import pandas as pd

METRICS=['episode_return','energy_kwh','comfort_pct','deviation_c']
def main():
    p=argparse.ArgumentParser(); p.add_argument('--input-dir',default='outputs'); p.add_argument('--output-dir',default='outputs/tables')
    a=p.parse_args(); root=Path(a.input_dir); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    files=list(root.glob('raw/*/final.csv'))+list(root.glob('baselines/*/*/final.csv'))
    if not files: raise RuntimeError('No completed evaluation data')
    df=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
    df=df.drop_duplicates(['config','seed','scenario'])
    df.to_csv(out/'per_scenario.csv',index=False)
    summary=df.groupby(['config','seed'])[METRICS].mean().reset_index()
    summary.to_csv(out/'results_summary.csv',index=False)
    agg=summary.groupby('config')[METRICS].agg(['mean','std'])
    agg.columns=['_'.join(x) for x in agg.columns]
    agg['n_train_seeds']=summary.groupby('config').size()
    agg.to_csv(out/'aggregated.csv')
    val=list(root.glob('raw/*/validation.csv'))
    if val: pd.concat([pd.read_csv(f) for f in val]).to_csv(out/'validation.csv',index=False)
    traces=list(root.glob('raw/*/trajectory.csv'))+list(root.glob('baselines/*/*/trajectory.csv'))
    if traces:
        pd.concat([pd.read_csv(f) for f in traces]).drop_duplicates(['config','seed','scenario','hour']).to_csv(out/'trajectories.csv',index=False)
    print(agg.to_string())
if __name__=='__main__': main()
