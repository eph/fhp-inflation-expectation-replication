import numpy as np
import pandas as p

from dsge import read_yaml

from itertools import product
from scipy.linalg import solve_discrete_lyapunov

average = False


fhp_file = 'models/partial_equilibrium.yaml'
fhp = read_yaml(fhp_file)

hs = [1, 4]
ks = [1, 4]
beta = 0.99
kappa = 0.05
rhos = [0.25, 0.9]
gammas = [0.25, 0.9]

def augment_system(linear_model, p0, variable, expectations, average=False):
    _, TT, RR, QQ, *_ = linear_model.system_matrices(p0)
    n_add = expectations*(expectations+1)//2
    v_add = [f'E_[t-{j}][{variable}({i})]' for i in range(1, expectations+1) for j in range(1, i+1)] + [f'{variable}_lag_{i}' for i in range(1, expectations+1)]
    n = TT.shape[0]
    TT_add = np.zeros((n_add+n+expectations, n_add+n+expectations))
    TT_add[:n, :n] = TT
    states_add = linear_model.state_names + v_add
    for i in range(1, expectations+1):
        curr = variable + f'({i})'
        for j in range(1, i+1):
            prev = curr
            curr = f'E_[t-{j}][{variable}({i})]'
            TT_add[states_add.index(curr), states_add.index(prev)] = 1
    RR_add = np.zeros((n_add+n+expectations, RR.shape[1]))
    RR_add[:n, :] = RR
    prev = variable
    for i in range(1, expectations+1):
        curr = variable + f'_lag_{i}'
        TT_add[states_add.index(curr), states_add.index(prev)] = 1
        prev = curr

    return TT_add, RR_add, QQ, states_add

def compute_cg_coefficient(linear_model, p0, variable, h, use_avg=False):
    TT, RR, QQ, states = augment_system(linear_model, p0, variable, h+1, average=use_avg)
    GAMMA0 = solve_discrete_lyapunov(TT, RR @ QQ @ RR.T)
    vector = np.zeros((GAMMA0.shape[0], 2))
    # print(p0, GAMMA0[states.index(f'{variable}'), states.index(f'{variable}')]) 
    vector[states.index(f'E_[t-{h}][{variable}({h})]'), 0] = 1
    vector[states.index(f'E_[t-{h+1}][{variable}({h+1})]'), 0] = -1

    vector[states.index(f'{variable}'), 1] = 1
    vector[states.index(f'E_[t-{h}][{variable}({h})]'), 1] = -1

    cov = vector.T @ GAMMA0 @ vector
    return cov[0, 1]/cov[0, 0]

def compute_kw_coefficient(linear_model, p0, variable, h, use_avg=False):
    TT, RR, QQ, states = augment_system(linear_model, p0, variable, h+1, average=use_avg)
    GAMMA0 = solve_discrete_lyapunov(TT, RR @ QQ @ RR.T)
    vector = np.zeros((GAMMA0.shape[0], 2))

    vector[states.index(f'{variable}'), 0] = 1
    vector[states.index(f'E_[t-{h}][{variable}({h})]'), 0] = -1
    vector[states.index(f'{variable}_lag_{h}'), 1] = 1

    cov = vector.T @ GAMMA0 @ vector
    return cov[0, 1]/cov[1, 1]


res = []
for h, k, gamma, rho in product(hs, ks, gammas, rhos):
    fhp_k = fhp.compile_model(k=k, expectations=h+1)
    p0 = [beta, kappa, gamma, rho]

    resi = {'k': k, 'rho': rho, 'gamma': gamma, 'h': h,
            'cg': compute_cg_coefficient(fhp_k, p0, 'pi', h, average),
            'kw': compute_kw_coefficient(fhp_k, p0, 'pi', h, average)
            }
    res.append(resi)
coefficients = p.DataFrame(res)
coefficients.set_index(['h', 'k', 'rho', 'gamma'], inplace=True)

print('Coibion-Gorodnichenko coefficients:')
print(coefficients.unstack('h').unstack('rho')['cg'].round(2))
print('\n\n')
print('Kohlhas and Walther coefficients:')
print(coefficients.unstack('h').unstack('rho')['kw'].round(2))
