
import pandas as p
import numpy as np
from dsge import read_yaml

from fortress import load_estimates
from angeletos import posterior_predictive_checks

from figures import saved_figure, create_508
from scipy.stats import gaussian_kde

np.random.seed(1848)

cg_beta = np.loadtxt('output/cg_coefficient_h3.txt')[-1]

SI = read_yaml('models/sticky_information.yaml')
linear_model = SI.compile_model()
results = load_estimates(f'.cache/compiled_models/macro/SI/outpu*',
                         paranames=linear_model.parameter_names)

nsim = 500
T = 168
nmcmc = results.shape[0]
gap = nmcmc // nsim


ppc = {}
params = results.iloc[::gap][linear_model.parameter_names].values
ppc['all'] = posterior_predictive_checks(linear_model, params, T=T)

xgrid_cg = np.linspace(-1, 4)
xgrid_kw = np.linspace(-0.6, 0.6)
create_508(p.DataFrame({'cg': gaussian_kde(ppc['all'].cg_beta)(xgrid_cg),
                        'cg_x': xgrid_cg,
                        'kw': gaussian_kde(ppc['all'].kw_beta)(xgrid_kw),
                        'kw_x': xgrid_kw}),
           'Figure 9')



shocks = ['σ_ξ', 'σ_y', 'σ_i']
names = ['demand', 'supply', 'monetary policy']
for shock, name in zip(shocks, names):
    params = results.iloc[::gap][linear_model.parameter_names].copy()
    params[shocks] = 0
    params[shock] = results.iloc[::gap][shock]
    ppc[name] = posterior_predictive_checks(linear_model, params.values, T=T)



with saved_figure(f'output/SI-CG-KW-T{T}.pdf',
                  nrows=1, ncols=2, sharey=False) as (fig, ax):
    ppc['all'].cg_beta.plot(kind='kde', ax=ax[0], linewidth=2)
    ax[0].set_title(r'$\hat\beta_{CG}^4$',size=17)
    ax[0].axvline(1.19, color='black', linewidth=2)
    mean, std = ppc['all'].cg_beta.mean(), ppc['all'].cg_beta.std()
    ax[0].text(-1.1, 0.8, f'mean = {mean:5.2f}\nstd = {std:5.2f}')
     
    ppc['all'].kw_beta.plot(kind='kde', ax=ax[1], linewidth=2)
    ax[1].set_title(r'$\hat\beta_{KW}$',size=17)
    mean, std, prob = ppc['all'].kw_beta.mean(), ppc['all'].kw_beta.std() , (ppc['all'].kw_beta < 0).mean()
    ax[1].text(-0.7, 3.1, f'mean = {mean:5.2f}\nstd = {std:5.2f}\nprob < 0 = {prob:5.2f}')
    fig.set_size_inches(7,3)



# with saved_figure(f'output/SI-CG-full-T{T}.pdf',
#                    nrows=2, ncols=2, 
#                    sharex=True, sharey=True) as (fig, ax):
#     ppc['all'].cg_beta.plot(kind='kde', ax=ax[0,0])
#     ax[0,0].axvline(cg_beta, color='black', linewidth=2)
#     ax[0,0].set_title(r'$\hat\beta_{CG}^4$: all shocks',size=17)
#     mean, std = ppc['all'].cg_beta.mean(), ppc['all'].cg_beta.std()
#     ax[0,0].text(-2.8, 1.0, f'mean = {mean:5.2f}\nstd = {std:5.2f}')
     
#     for axi, name in zip(ax.reshape(-1)[1:], names):
#         ppc[name].cg_beta.plot(kind='kde', ax=axi,linewidth=2)
#         ppc['all'].cg_beta.plot(kind='kde', ax=axi,linewidth=2, color='grey', alpha=0.5)
#         axi.axvline(cg_beta, color='black')
#         axi.set_title(r'$\hat\beta_{CG}^4$: ' + name + ' only',size=16)
#         mean, std = ppc[name].cg_beta.mean(), ppc[name].cg_beta.std()
#         axi.text(-2.5, 1.0, f'mean = {mean:5.2f}\nstd = {std:5.2f}')
        
#         # lam = results.iloc[::gap].λ
#         # (lam/(1-lam)).plot(kind='kde', ax=axi,linestyle='dashed')





