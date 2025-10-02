import numpy as np
import sys;
import matplotlib.pyplot as plt
from fortress import load_estimates
from angeletos import angeletos
import pandas as p

from dsge import read_yaml

from figures import saved_figure, create_508

SI = read_yaml('models/sticky_information.yaml')
SI['__data__']['estimation']['data']['file'] = '.cache/observables/longsample_with_average_inflation_expectations.txt'
linear_model = SI.compile_model()
c = linear_model.yy.columns[-1]
linear_model.yy.loc['1970Q1', [c]] = 3.02
linear_model.yy.loc['1974Q3', [c]] = 6.11
results_noexp = load_estimates(f'.cache/compiled_models/macro/SI/outpu*',
                         paranames=linear_model.parameter_names)

nmcmc = results_noexp.shape[0]
nsim = 200
gap = nmcmc // nsim
para_FHPrep = results_noexp.loc[::gap][linear_model.parameter_names].values
sim_FHPrep, actual = angeletos(linear_model, para_FHPrep, T=168)
 
    
with saved_figure('output/angeletos-finite-sample-SI.pdf',
                  ncols=3, nrows=1, sharex=True) as (fig, ax):
    sim_FHPrep['diff'] = sim_FHPrep.infl - sim_FHPrep.Fπ
    sim_FHPrep.groupby(level=1).mean()[['infl','Fπ','diff']].plot(ax=ax, subplots=True, legend=False, color='C0')
    q05 = sim_FHPrep.groupby(level=1).quantile(0.05)
    q95 = sim_FHPrep.groupby(level=1).quantile(0.95)
    q16 = sim_FHPrep.groupby(level=1).quantile(0.16)
    q84 = sim_FHPrep.groupby(level=1).quantile(0.84)
    [axi.fill_between(q05.index, q05[v], q95[v], color='C0', alpha=0.3) for axi, v in zip(ax, ['infl','Fπ','diff'])]
    [axi.fill_between(q16.index, q16[v], q84[v], color='C0', alpha=0.3) for axi, v in zip(ax, ['infl','Fπ','diff'])]
    [axi.set_title(t) for axi, t in zip(ax, [r'$\pi_t^A$', r'$E_{t-4}[\pi_t^A]$', r'$\pi_t^A -E_{t-4}[\pi_t^A]$'])]
    actual['diff'] = actual.infl - actual.Fπ
    actual[['infl','Fπ','diff']].plot(ax=ax, subplots=True, color='black')
    ax[2].axhline(0, color='black')
    [axi.legend('') for axi in ax]
    fig.set_size_inches(10,3)

sec508 = p.concat([sim_FHPrep.groupby(level=1).mean()[['infl','Fπ','diff']],
                   q05[['infl','Fπ','diff']], q95[['infl','Fπ','diff']], 
                   q16[['infl','Fπ','diff']], q84[['infl','Fπ','diff']], 
                   actual[['infl','Fπ','diff']]], 
                  keys=['mean','q05','q95','q16','q84','actual'], axis=1)
 
create_508(sec508, 'Figure 8')
