import sys;

from dsge import read_yaml
from fortress import load_estimates
from figures import saved_figure, create_508

import pandas as p 

FHPrep = read_yaml('models/finite_horizon.yaml').fix_parameters(ρ_η=0,σ_η=0)
linmod = FHPrep.compile_model()
keys = list(range(5)) + [10, 15, 30, 40]

results =p.concat([load_estimates(f'.cache/compiled_models/macro/FHP[[]k={k}[]]/outpu*',
                                  paranames=linmod.parameter_names)
                   for k in keys], keys=keys)

with saved_figure('output/FHP_mdds_across_k.pdf') as (fig, ax):
    mu = results.groupby(level=[0,1]).mean().groupby(level=0).mean().logmdd
    sig = results.groupby(level=[0,1]).mean().groupby(level=0).std().logmdd
    ax = mu.plot(marker='o',linestyle=None); 
    ax.fill_between(sig.index, mu - 1.96*sig, mu + 1.96*sig, alpha=0.3, color='C0')
    ax.set_xlabel(r'$k$')

    sec508 = p.DataFrame({'mean': mu, 'q025': mu - 1.96*sig, 'q975': mu + 1.96*sig})
    create_508(sec508, 'Figure 2')
