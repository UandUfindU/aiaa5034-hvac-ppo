"""Plot only saved tables. No training imports or execution."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input-dir',default='outputs/tables'); p.add_argument('--output-dir',default='outputs/figures')
    a=p.parse_args(); root=Path(a.input_dir); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    def save(fig,name):
        fig.tight_layout(); fig.savefig(out/(name+'.png'),dpi=300); fig.savefig(out/(name+'.pdf')); plt.close(fig)
    s=pd.read_csv(root/'results_summary.csv')
    order=['default','clip_010','clip_030','gae_090','gae_099','epochs_3','epochs_20','thermostat','always_off']
    names=[x for x in order if x in set(s.config)]
    for metric,ylabel,name in [('comfort_pct','Comfort violation rate (%)','fig_comfort_ablation'),('energy_kwh','Energy (kWh / 168-hour episode)','fig_energy_ablation')]:
        g=s.groupby('config')[metric]
        fig,ax=plt.subplots(figsize=(10,4))
        ax.bar(names,[g.mean()[x] for x in names],yerr=[g.std().fillna(0)[x] for x in names],capsize=3,color='steelblue')
        ax.set_ylabel(ylabel); ax.tick_params(axis='x',rotation=35)
        ax.set_title('Final policy: mean +/- SD across training seeds; references have no training variance')
        save(fig,name)
    fig,ax=plt.subplots(figsize=(9,5))
    for name in names:
        z=s[s.config==name]; x=z.energy_kwh; y=z.comfort_pct
        ax.errorbar(x.mean(),y.mean(),xerr=x.std() if len(x)>1 else 0,yerr=y.std() if len(y)>1 else 0,fmt='o',capsize=3)
        ax.annotate(name,(x.mean(),y.mean()),xytext=(5,5),textcoords='offset points',fontsize=8)
    ax.set_xlabel('Energy (kWh / episode)'); ax.set_ylabel('Comfort violation rate (%)'); save(fig,'fig_comfort_energy_tradeoff')
    v=pd.read_csv(root/'validation.csv').groupby(['config','seed','steps']).episode_return.mean().reset_index()
    fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
    for ax,group,title in zip(axes,[['default','clip_010','clip_030'],['default','gae_090','gae_099'],['default','epochs_3','epochs_20']],['Clip range','GAE lambda','Epochs']):
        for name in group:
            z=v[v.config==name]
            if z.empty: continue
            z=z.copy(); z['smooth']=z.groupby('seed').episode_return.transform(lambda q:q.rolling(3,min_periods=1).mean())
            g=z.groupby('steps').smooth.agg(['mean','std','count']); x=g.index.to_numpy(); m=g['mean'].to_numpy(); sd=g['std'].fillna(0).to_numpy()
            ax.plot(x,m,label=f'{name} (n={int(g["count"].max())})'); ax.fill_between(x,m-sd,m+sd,alpha=.15)
        ax.set_title(title); ax.set_xlabel('Environment steps'); ax.legend(fontsize=8)
    axes[0].set_ylabel('Validation return (3-checkpoint trailing mean)'); save(fig,'fig_learning_curves')
    tr=pd.read_csv(root/'trajectories.csv'); fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    for name,seed in [('default',0),('thermostat',-1)]:
        z=tr[(tr.config==name)&(tr.seed==seed)&(tr.scenario==1000)]
        if z.empty: continue
        axes[0].plot(z.hour,z.indoor_temp,label=name); axes[1].plot(z.hour,z.hvac_action,label=name)
        if name=='thermostat': axes[0].plot(z.hour,z.outdoor_temp,ls='--',label='Outdoor')
    axes[0].axhspan(22,26,color='green',alpha=.12,label='Comfort band')
    axes[0].set_ylabel('Temperature (C)'); axes[0].legend(); axes[0].set_title('Prespecified example: held-out scenario 1000, PPO training seed 0')
    axes[1].set_ylabel('HVAC action'); axes[1].set_xlabel('Hour (temperature measured after action)'); save(fig,'fig_temperature_trajectory')
if __name__=='__main__': main()
