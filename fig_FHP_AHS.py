from dsge import read_yaml

import numpy as np
import pandas as p

from fortress import load_estimates
from angeletos import angeletos
from figures import saved_figure, create_508

k=4
FHP = read_yaml('models/finite_horizon.yaml')
FHP['estimation']['data']['file'] = '.cache/observables/longsample_with_average_inflation_expectations.txt'
linear_model = FHP.compile_model(k=k)
c = linear_model.yy.columns[-1]
linear_model.yy.loc['1970Q1', [c]] = 3.02
linear_model.yy.loc['1974Q3', [c]] = 6.11
results_noexp =load_estimates(f'../__fortress_FHPrep{"k"+str(k) if k!=1 else ""}noFpi/outpu*',
                              paranames=linear_model.parameter_names)

np.random.seed(1848)
nmcmc = results_noexp.shape[0]
nsim = 200
gap = nmcmc // nsim
para_FHPrep = results_noexp.loc[::gap][linear_model.parameter_names].values
sim_FHPrep, actual = angeletos(linear_model, para_FHPrep)
 
with saved_figure(f'output/angeletos-finite-sample-k-{k}.pdf',
                  ncols=3, nrows=1, sharex=True, sharey=True) as (fig, ax):
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
 
create_508(sec508, 'Figure 4')

nsim = 1000
gap = nmcmc // nsim
para_FHPrep = results_noexp.loc[::gap][linear_model.parameter_names].values
irfs = p.concat([p.concat(linear_model.impulse_response(para_FHPrep[i],h=60))
                for i in range(nsim)],keys=range(nsim))

irfs['Fπ'] = irfs['π(1)'] + irfs['π(2)'] + irfs['π(3)'] + irfs['π(4)']


def istar(x):
    difference = (x.infl - x.Fπ) 
    pop = np.sign(difference * difference.shift(1))

    pop.iloc[:4] = 0
    crosses = np.where(pop < 0)[0]
    if len(crosses):
        return crosses[0]
    else:
        return np.nan

    
irfs['infl'] = irfs.π.rolling(4, min_periods=0).apply(lambda x: x.fillna(0).sum())
irfs['Fπ'] = irfs.Fπ.shift(4).fillna(0)

prob = irfs.groupby(level=[0,1]).apply(istar).groupby(level=[1]).apply(lambda x: np.mean(x < 40))
mu = irfs.groupby(level=[0,1]).apply(istar).groupby(level=[1]).apply(lambda x: np.mean(x[x<40]))
sig = irfs.groupby(level=[0,1]).apply(istar).groupby(level=[1]).apply(lambda x: np.std(x[x<40]))
res = p.concat([prob, mu, sig], keys=['Prob', 'Mean', 'StD'], axis=1)

AHSistar = sim_FHPrep.groupby(level=0).apply(istar) 
AHSprob = (AHSistar < 40).mean()
AHSmu = AHSistar[AHSistar < 40].mean()
AHSsig = AHSistar[AHSistar < 40].std()

res.loc['AHS',:] = [AHSprob, AHSmu, AHSsig]
print(res.round(2))
