from dsge import read_yaml

from fortress import load_estimates
from figures import replace
k = 'tvp'
FHPrep = read_yaml('models/finite_horizon.yaml').fix_parameters(ρ_η=0,σ_η=0)

linmod = FHPrep.compile_model()
para = linmod.parameter_names + ['lambar','rho_lam', 'sig_lam'] + [f'x[{d}]' for d in range(168)]
para = ['rA', 'piA', 'yQ', 'kappa', 'sigma', 'phipi',
        'phiy', 'sigxi', 'sigy', 'sigi', 'rhoxi', 'rhoi', 'rhoy', 'gamma', 'gammaf',
        'phipiLR', 'phiyLR'] + ['lambar','rho_lam', 'sig_lam'] + [f'x[{d}]' for d in range(168)]
results_tvp =load_estimates(f'.cache/compiled_models/macro/FHP[[]k=tvp[]]/output-*.json',

                              paranames=para)
results_k4 = load_estimates(f'.cache/compiled_models/macro/FHP[[]k=4[]]/output-01.json',
                              paranames=para)
results_k30 = load_estimates(f'.cache/compiled_models/macro/FHP[[]k=30[]]/output-01.json',
                              paranames=para)

import matplotlib.pyplot as plt

ax = results_tvp.plot(kind='scatter',x='σ_y',y='sig_lam', alpha=0.2)
# results_k4.plot(kind='scatter',x='σ_y',y='sig_lam', ax=ax, color='C1', alpha=0.2)
# results_k30.plot(kind='scatter',x='σ_y',y='sig_lam', ax=ax, color='C2', alpha=0.2, xlabel='σ_y', ylabel='ρ_y')
ax.legend(['tvp','k=4','k=30'])
plt.show()
