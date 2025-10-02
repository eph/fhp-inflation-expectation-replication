import numpy as np
import pandas as p
import sys;
from dsge import read_yaml
from fortress import load_estimates 
import matplotlib.pyplot as plt

from figures import saved_figure, create_508

fix_parameters=False
average=True

cg_beta_h0 = np.loadtxt('output/cg_coefficient_h0.txt')[-1]
cg_beta_h3 = np.loadtxt('output/cg_coefficient_h3.txt')[-1]

def simulate(model, p0, nsim=200):

    CC, TT, RR, QQ, DD, ZZ, HH = model.system_matrices(p0)
    ysim = np.zeros((nsim * 2, DD.size))
    ssim = np.zeros((nsim * 2, CC.size))
    At = np.zeros((TT.shape[0],))

    for i in range(nsim * 2):
        e = np.random.multivariate_normal(np.zeros((QQ.shape[0])), QQ)
        At = CC + TT.dot(At) + RR.dot(e)

        h = np.random.multivariate_normal(np.zeros((HH.shape[0])), HH)
        At = np.asarray(At).squeeze()
        ysim[i, :] = DD.T + ZZ.dot(At) + np.atleast_2d(h)
        ssim[i, :] = At

    return (p.DataFrame(ysim[nsim:], columns=model.yy.columns), 
            p.DataFrame(ssim[nsim:,:len(model.state_names)], columns=model.state_names))



def cg_comparison(k=1, h1=1, h2=4, nsim=200, T=168, fix_parameters=True, average=True):


    FHPrep = read_yaml('models/finite_horizon.yaml').fix_parameters(ρ_η=0,σ_η=0)
    linear_model = FHPrep.compile_model(k=k,expectations=h1+1)

    if fix_parameters:
        name = 'FHP[[]k=4[]]'
    else:
        name = f'FHP[[]k={k}[]]'
    para = load_estimates(f'.cache/compiled_models/macro/{name}/output*',
                          paranames=linear_model.parameter_names)
    

    
    np.random.seed(2666)
    beta_cg_1 = np.zeros(nsim)

    for i in range(nsim):
        p0 = para[linear_model.parameter_names].iloc[i]
        ysim, ssim = simulate(linear_model, p0, nsim=168)
     
        results = p.DataFrame()
        if average:
            forecast = [f'π({hi})' for hi in range(1,h1+1)]
            lagged = [f'π({hi})' for hi in range(2,h1+2)]
            results['y'] = ssim.π.rolling(h1).mean() - ssim[forecast].mean(1).shift(h1)
            results['x'] = ssim[forecast].mean(1).shift(h1) - ssim[lagged].mean(1).shift(h1+1)
        else:
            results['y'] = ssim.π - ssim[f'π({h1})'].shift(h1)
            results['x'] = ssim[f'π({h1})'].shift(h1) - ssim[f'π({h1+1})'].shift(h1+1)

        results['one'] = 1
     
     
        data = results.dropna().values
        x, y = data[:,1:], data[:,0]
        beta_cg_1[i] = (np.linalg.inv(x.T @ x) @ x.T @ y)[0]


    linear_model = FHPrep.compile_model(k=k,expectations=h2+1)

    np.random.seed(2666)
    beta_cg_2 = np.zeros(nsim)
    for i in range(nsim):
        p0 = para[linear_model.parameter_names].iloc[i]
        ysim, ssim = simulate(linear_model, p0, nsim=168)

        results = p.DataFrame()
        if average:
            forecast = [f'π({hi})' for hi in range(1,h2+1)]
            lagged = [f'π({hi})' for hi in range(2,h2+2)]
            results['y'] = ssim.π.rolling(h2).mean() - ssim[forecast].mean(1).shift(h2)
            results['x'] = ssim[forecast].mean(1).shift(h2) - ssim[lagged].mean(1).shift(h2+1)
        else:
            results['y'] = ssim.π - ssim[f'π({h2})'].shift(h2)
            results['x'] = ssim[f'π({h2})'].shift(h2) - ssim[f'π({h2+1})'].shift(h2+1)
        results['one'] = 1

        data = results.dropna().values
        x, y = data[:,1:], data[:,0]
        beta_cg_2[i] = (np.linalg.inv(x.T @ x) @ x.T @ y)[0]

    results = p.DataFrame({f'h = {h1}':beta_cg_1, f'h = {h2}': beta_cg_2})

    return results

results = cg_comparison(k=1,fix_parameters=fix_parameters, average=average)
results_k4 = cg_comparison(k=4, fix_parameters=fix_parameters, average=average)
#results_k8 = cg_comparison(k=2, fix_parameters=fix_parameters, average=average)
with saved_figure(f'output/k1_vs_k4_fix_para_{fix_parameters}_average_{average}.pdf') as (fig, ax):
    results.plot.scatter(ax=ax,x='h = 1',y='h = 4',alpha=0.5);
    results_k4.plot.scatter(ax=ax, x='h = 1',y='h = 4',alpha=0.5, color='C1');
    #results_k8.plot.scatter(ax=ax, x='h = 1',y='h = 4',alpha=0.3, color='C2');
    ax.legend(['k=1','k=4'])
    ax.scatter(cg_beta_h0, cg_beta_h3, color='black', s=50)

sec508 = p.concat([results, results_k4, results_k8,
                   p.DataFrame({'h = 1':cg_beta_h0, 'h = 4':cg_beta_h3}, index=[0])], 
                   keys=['k=1', 'k=4', 'Data'])
create_508(sec508, 'Figure 3')
