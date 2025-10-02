import numpy as np
import pandas as p
from itertools import product

from dsge import read_yaml

from figures import saved_figure, create_508


fhp = read_yaml('models/partial_equilibrium.yaml')

h = 4
ks = np.arange(1, 20)
beta = 0.99
kappa = 0.05
rhos = [0.25, 0.9]
gammas = [0.25, 0.9]


def find_threshold(series):
    pop = np.sign(series*series.shift(1))
    return np.where(pop < 0)[0][0]


res = []
for k, gamma, rho in product(ks, gammas, rhos):
    fhplin = fhp.compile_model(k=k, expectations=h)
    p0 = [beta, kappa, gamma, rho]

    irf = fhplin.impulse_response(p0,h=40)['e_y']
    irf['F'] = (irf.pi.shift(-h) - irf[f'pi({h})'])

    resi = {'k': k, 'rho': rho, 'gamma': gamma,
            'istar': find_threshold(irf.F)}

    res.append(resi)
coefficients = p.DataFrame(res).set_index(['k','rho','gamma']).unstack(['rho','gamma'])


with saved_figure('output/partial_equilibrium_istar.pdf', nrows=2) as (fig, ax):
    ax[0].set_title(rf'Persistent Shock ($\rho = {rhos[1]}$)')
    ax[0].set_ylabel('Threshold\n value $(i^*)$', rotation=0, fontsize=8, labelpad=-25, loc='top')
    coefficients.loc[:,('istar', rhos[1], gammas[0])].plot(ax=ax[0],linestyle='dashed',marker='o',color='black')
    coefficients.loc[:,('istar', rhos[1], gammas[1])].plot(ax=ax[0],linestyle='dashed',marker='^',color='red')
    ax[0].legend([rf'$\gamma_p = {g}$' for g in gammas])
    coefficients.loc[:,('istar', rhos[1], gammas[0])].plot(ax=ax[1],linestyle='dashed',marker='o',color='black' )
    coefficients.loc[:,('istar', rhos[0], gammas[0])].plot(ax=ax[1],linestyle='dashed',marker='^',color='red' )
    ax[1].set_title(rf'Gradual Learning ($\gamma = {gammas[0]}$)')
    ax[1].legend([rf'$\rho = {r}$' for r in rhos[::-1]])
    ax[1].set_ylabel('Threshold\n value $(i^*)$', rotation=0, fontsize=8, labelpad=-25, loc='top')
    [axi.set_xticks(ks) for axi in ax]
    [axi.set_xlabel('Planning Horizon ($k$)') for axi in ax]
    [axi.set_yticks(np.arange(0,32,4)) for axi in ax]
    fig.tight_layout()


    sec508_top = p.concat([coefficients.loc[:,('istar', rhos[1], gammas[0])], 
                           coefficients.loc[:,('istar', rhos[1], gammas[1])]],
                          keys = [f'gamma = {g}' for g in gammas], axis=1)

    sec508_bottom = p.concat([coefficients.loc[:,('istar', rhos[1], gammas[0])],
                              coefficients.loc[:,('istar', rhos[0], gammas[0])]],
                             keys = [f'rho = {r}' for r in rhos[::-1]], axis=1)

    sec508 = p.concat([sec508_top, sec508_bottom], 
                      keys=['Persistent Shock', 'Gradual Learning'], axis=1)

    create_508(sec508, 'Figure 1')


