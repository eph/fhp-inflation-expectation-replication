import numpy as np
import sys;
import matplotlib.pyplot as plt
from fortress import load_estimates
from angeletos import posterior_predictive_checks
import pandas as p

from dsge import read_yaml

from figures import saved_figure, create_508
from scipy.stats import gaussian_kde

np.random.seed(2666)
k=4
FHP = read_yaml('models/finite_horizon.yaml')
linear_model = FHP.compile_model(k=k, expectations=5)

results =load_estimates(f'../__fortress_FHPrep{"k"+str(k) if k!=1 else ""}noFpi/outpu*',
                        paranames=linear_model.parameter_names)




nsim = 500

nmcmc = results.shape[0]
gap = nmcmc // nsim


xgrid = np.linspace(-0.6, 0.4)
sec508 = {}

para = results.loc[::gap][linear_model.parameter_names].values
ppc = posterior_predictive_checks(linear_model, para)


with saved_figure(f'output/FHP[k={k}]-KW-full.pdf',
                  nrows=2, ncols=2, 
                  sharex=True, sharey=True) as (fig, ax):

    ppc.kw_beta.plot(kind='kde', ax=ax[0,0])
    ax[0,0].set_title(r'$\hat\beta_{KW}^4$: all shocks',size=17)
    mean, std, prob = ppc.kw_beta.mean(), ppc.kw_beta.std(), (ppc.kw_beta<0).mean()
    ax[0,0].text(-0.65, 5, f'mean = {mean:5.2f}\nstd = {std:5.2f}\nprob < 0 = {prob:5.2f}')
    total = ppc.copy()
     
    sec508['all shocks'] = gaussian_kde(ppc.kw_beta)(xgrid)
    shocks = ['σ_ξ', 'σ_y', 'σ_i']
    names = ['demand', 'supply', 'monetary policy']
    for axi, shock, name in zip(ax.reshape(-1)[1:], shocks, names):
        params = results.iloc[::gap][linear_model.parameter_names].copy()
        params[shocks] = 0
        params[shock] = results.iloc[::gap][shock]
     
        ppc = posterior_predictive_checks(linear_model, params.values)
       
        ppc.kw_beta.plot(kind='kde', ax=axi,linewidth=2)
        total.kw_beta.plot(kind='kde', ax=axi,linewidth=2, color='grey', alpha=0.5)
     
        axi.set_title(r'$\hat\beta_{KW}^4$: ' + name + ' only',size=16)
        mean, std, prob = ppc.kw_beta.mean(), ppc.kw_beta.std(), (ppc.kw_beta<0).mean()
        axi.text(-0.65, 5, f'mean = {mean:5.2f}\nstd = {std:5.2f}\nprob < 0 = {prob:5.2f}')
     
        sec508[f'{name} only'] = gaussian_kde(ppc.kw_beta)(xgrid)
     
    ax[0,0].xaxis.set_tick_params(which='both', labelbottom=True)
    ax[0,1].xaxis.set_tick_params(which='both', labelbottom=True)
    ax[0,0].yaxis.set_tick_params(which='both', labelbottom=True)
    ax[0,1].yaxis.set_tick_params(which='both', labelbottom=True)
    fig.set_size_inches(8,6)

sec508 = p.DataFrame(sec508, index=xgrid)
create_508(sec508, 'Figure 7')
