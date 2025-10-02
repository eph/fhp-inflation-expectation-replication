import numpy as np
import sys;
from dsge import read_yaml

from fortress import load_estimates

from angeletos import compute_spectral_density

from figures import saved_figure, create_508
import pandas as p

k=4
FHPrep = read_yaml('models/finite_horizon.yaml')
linear_model = FHPrep.compile_model(k=k)

results_noexp =load_estimates(f'../__fortress_FHPrep{"k"+str(k) if k!=1 else ""}noFpi/outpu*',
                              paranames=linear_model.parameter_names)

p0 = results_noexp.mean()[linear_model.parameter_names].copy()
CC, TT, RR, QQ, DD, ZZ, HH = linear_model.system_matrices(p0)


def vd_comp_period(period):

    total_variance = ZZ @ compute_spectral_density(TT, RR * np.sqrt(np.diag(QQ))[np.newaxis,:], 2*np.pi/period) @ ZZ.T
    total_variance = total_variance
    vd = {}
    for i, name in enumerate(linear_model.shock_names):

        RRcopy = np.zeros_like(RR)
        RRcopy[:,i] = RR[:,i]

        variance_due_to_i = ZZ @ compute_spectral_density(TT, RRcopy * np.sqrt(np.diag(QQ))[np.newaxis,:], 2*np.pi/period) @ ZZ.T
        vd[name] = np.diag(variance_due_to_i) / np.diag(total_variance)

    return p.DataFrame(vd,index=linear_model.obs_names).T

period_0 = 1
period_1 = 170
res_vd = p.concat([vd_comp_period(k) 
                   for k in range(period_0, period_1)], 
                  keys=range(period_0, period_1+1))
res_vd = res_vd.apply(np.real)


with saved_figure(f'output/FHP-variance-decomposition-freq-domain-k-{k}.pdf') as (fig, ax):
    to_plot = res_vd.infl.unstack().cumsum(1).iloc[:,:3]
    to_plot.index = np.log(to_plot.index)
    ax.fill_between(to_plot.index[2:], 0, to_plot.iloc[2:,0], alpha=0.3)
    ax.fill_between(to_plot.index[2:], to_plot.iloc[2:,0], to_plot.iloc[2:,1], alpha=0.3)
    ax.fill_between(to_plot.index[2:], to_plot.iloc[2:,1], to_plot.iloc[2:,2], alpha=0.3)

    ax.fill_between(to_plot.index[5:33], 0, to_plot.iloc[5:33,0], alpha=1, color='C0')
    ax.fill_between(to_plot.index[5:33], to_plot.iloc[5:33,0], to_plot.iloc[5:33,1], alpha=1, color='C1')
    ax.fill_between(to_plot.index[5:33], to_plot.iloc[5:33,1], to_plot.iloc[5:33,2], alpha=1, color='C2')


    ax.legend(['__nolegend__']*3 +['Demand', 'Supply', 'Monetary Policy'], ncol=3, bbox_to_anchor=(0.82,-0.11))
    ax.set_xlabel('period (log scale, quarters)');

    ax.set_xlim(np.log(4), np.log(120))
    ax.set_xticks(np.log([4,6,16,32,60,120]))
    ax.set_xticklabels([4,6,16,32,60,120], rotation=0)
    ax.axvline(np.log(6), color='black')
    ax.axvline(np.log(33), color='black')
    ax.set_ylim(0,1)


create_508(to_plot, 'Figure 5')
