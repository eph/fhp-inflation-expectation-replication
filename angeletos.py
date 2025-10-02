import numpy as np
import pandas as p
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tools.tools import add_constant
from numpy.linalg import matrix_power
import matplotlib.pyplot as plt
from tqdm import tqdm
from pylatexenc.latexencode import unicode_to_latex

def replace(x): 
    if x == 'πA': return r'$\pi^A$'
    elif x == 'yQ': return r'$\gamma^Q$'
    elif x == 'rA': return r'$r^A$'
    else: return (unicode_to_latex(r'$'+x+'$', non_ascii_only=True)
                  .replace('Phi','phi')
                  .replace(r'\ensuremath{\phi}_\ensuremath{\pi}_LR', r'\bar{\phi}_\pi')
                  .replace(r'\ensuremath{\phi}_y_LR', r'\bar{\phi}_{y}'))


def period_to_frequency(period):
    return 2 * np.pi / period

def compute_spectral_density(TT, RR, omega):
    identity_matrix = np.eye(TT.shape[0])
    exp_negative_i_omega = np.exp(-1j * omega)
    return np.linalg.inv(identity_matrix - exp_negative_i_omega * TT) @ RR @ RR.T @ np.linalg.inv(identity_matrix - exp_negative_i_omega * TT).conj().T

def compute_variance_over_frequency_range(TT, RR, omega_0, omega_1, num_points=100):
    omega_range = np.linspace(omega_0, omega_1, num_points)
    total_variance = np.zeros((TT.shape[0], TT.shape[0]))

    for omega in omega_range:
        spectral_density = compute_spectral_density(TT, RR, omega)
        total_variance += np.real(spectral_density)

    return total_variance/num_points

def compute_variance_over_period_range(TT, RR, period_0, period_1, num_points=100):
    omega_0 = period_to_frequency(period_1)  # Smaller frequency corresponds to the larger period
    omega_1 = period_to_frequency(period_0)  # Larger frequency corresponds to the smaller period
    return compute_variance_over_frequency_range(TT, RR, omega_0, omega_1, num_points)

def simulate(model, p0, nsim=500):

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



def angeletos(model, parameters, k=16, nlags=4, T=None):
    if T is None:
        T, ny = model.yy.shape
    else:
        _, ny = model.yy.shape
    results = []
    for i, p0 in enumerate(parameters):
        ysim, ssim = simulate(model, p0, nsim=T)
        x, y = lagmat(ysim, maxlag=nlags, trim='both', original='sep')
        x = add_constant(x, prepend=False)

        phi = np.linalg.inv(x.T @ x) @ x.T @ y
        sigma = (y - x @ phi).T @ (y - x @ phi) / y.shape[0]
        chol_sigma = np.linalg.cholesky(sigma)

        alpha = np.ones((4,)) / 2
        index = 1

        def companion_form(phi, nlags):
            TT = np.zeros((nlags*ny, nlags*ny))
            for i in range(nlags):
                TT[:ny, i*ny:(i+1)*ny] = phi[i*ny:(i+1)*ny, :].T

            for i in range(nlags-1):
                TT[(i+1)*ny:(i+2)*ny, i*ny:(i+1)*ny] = np.eye(ny)

            return TT 


        TT = companion_form(phi, nlags)
        RR = np.zeros((TT.shape[0], ny))
        RR[:ny,:ny] = np.eye(ny)

        def objective_frequency_domain(alpha, add_penalty=True, period_0=6, period_1=32, num_points=100):
            RR = np.zeros((TT.shape[0], ny))
            RR[:ny,:ny] = chol_sigma
            total_variance = compute_variance_over_period_range(TT, RR, period_0=period_0, period_1=period_1, num_points=num_points)
            A0 = (chol_sigma @ alpha[:,np.newaxis])
            RR *= 0 
            RR[:ny,0] = A0.squeeze()
            variance_due_to_shock = compute_variance_over_period_range(TT, RR, period_0=period_0, period_1=period_1, num_points=num_points)
            return -(variance_due_to_shock[index,index] / total_variance[index,index]) + add_penalty*1000*(alpha @ alpha - 1)**2


        from scipy.optimize import fmin

        x0 = fmin(objective_frequency_domain, alpha, disp=False)
        x0 = x0 / np.linalg.norm(x0)
        print(objective_frequency_domain(x0, add_penalty=False), x0.round(2))

        RR *= 0 
        A0 = (chol_sigma @ x0[:,np.newaxis])
        RR[:ny,0] = A0.squeeze()

        s = np.zeros((40, TT.shape[0]))
        s[0] = RR[:,0]
        for i in range(1, 40):
            s[i] = TT @ s[i-1]

        df = p.DataFrame(s[:,:ny],columns=model.obs_names)
        # weird indexing bug
        df.iloc[:,1] = df.iloc[:,1].rolling(4,min_periods=0).apply(lambda x: x.fillna(0).sum()/4)
        df.iloc[:,3] = df.iloc[:,3].shift(4).fillna(0)
        results.append(df)


    x, y = lagmat(model.yy.dropna(), maxlag=nlags, trim='both', original='sep')
    x = add_constant(x, prepend=False)

    phi = np.linalg.inv(x.T @ x) @ x.T @ y
    sigma = (y - x @ phi).T @ (y - x @ phi) / y.shape[0]
    chol_sigma = np.linalg.cholesky(sigma)

    alpha = np.ones((4,)) / 2
    index = 1

    def companion_form(phi, nlags):
        TT = np.zeros((nlags*ny, nlags*ny))
        for i in range(nlags):
            TT[:ny, i*ny:(i+1)*ny] = phi[i*ny:(i+1)*ny, :].T

        for i in range(nlags-1):
            TT[(i+1)*ny:(i+2)*ny, i*ny:(i+1)*ny] = np.eye(ny)

        return TT 


    TT = companion_form(phi, nlags)
    RR = np.zeros((TT.shape[0], ny))
    RR[:ny,:ny] = np.eye(ny)

    def objective_frequency_domain(alpha, add_penalty=True, period_0=6, period_1=32, num_points=100):
        RR = np.zeros((TT.shape[0], ny))
        RR[:ny,:ny] = chol_sigma
        total_variance = compute_variance_over_period_range(TT, RR, period_0=period_0, period_1=period_1, num_points=num_points)
        A0 = (chol_sigma @ alpha[:,np.newaxis])
        RR *= 0 
        RR[:ny,0] = A0.squeeze()
        variance_due_to_shock = compute_variance_over_period_range(TT, RR, period_0=period_0, period_1=period_1, num_points=num_points)
        return -(variance_due_to_shock[index,index] / total_variance[index,index]) + add_penalty*1000*(alpha @ alpha - 1)**2


    from scipy.optimize import fmin

    x0 = fmin(objective_frequency_domain, alpha,disp=False)
    x0 = x0 / np.linalg.norm(x0)
    print(objective_frequency_domain(x0, add_penalty=False), x0.round(2))

    RR *= 0 
    A0 = (chol_sigma @ x0[:,np.newaxis])
    RR[:ny,0] = A0.squeeze()

    s = np.zeros((40, TT.shape[0]))
    s[0] = RR[:,0]
    for i in range(1, 40):
        s[i] = TT @ s[i-1]

    df = p.DataFrame(s[:,:ny],columns=model.obs_names)
    # weird indexing was df.iloc[:,1], df.iloc[:,3]
    df.iloc[:,1]= df.iloc[:,1].rolling(4,min_periods=0).apply(lambda x: x.fillna(0.).sum()/4)
    df.iloc[:,3] = df.iloc[:,3].shift(4).fillna(0.)

    return p.concat(results, keys=range(len(results))), df

def posterior_predictive_checks(model, parameters, h=4, T=168, transform=None):
    cg_betas = np.zeros((parameters.shape[0],2))
    kw_betas = np.zeros((parameters.shape[0],2))

    for i, p0 in enumerate(tqdm(parameters)):
        yy, ss = simulate(model, p0, nsim=T)
        CC, TT, RR, QQ, DD, ZZ, HH = model.system_matrices(p0)
        yy.columns = [str(x) for x in yy.columns]

        if transform != None:
            ss = transform(ss)

        if 'Fπ' not in ss.columns:
            ss['Fπ'] = (ss['π(1)'] + ss['π(2)'] + ss['π(3)'] + ss['π(4)'] ) / 4

        if 'Fπlag' not in ss.columns:
            ss['Fπlag'] = (ss['π(2)'] + ss['π(3)'] + ss['π(4)'] + ss['π(5)'] ) / 4

        ss['err'] = ss.π.rolling(4).mean().shift(-4) - ss.Fπ 
        ss['rev'] = ss.Fπ - ss.Fπlag.shift(1)
        ss['ONE'] = 1
        ss = ss.dropna()
        X, Y = ss[['ONE', 'rev']].values, ss.err.values

        cg_betas[i] = np.linalg.inv(X.T @ X) @ X.T @ Y 


        ss['πlag'] = ss.π.rolling(4).mean()
        ss = ss.dropna()
        X, Y = ss[['ONE', 'πlag']].values, ss.err.values
        kw_betas[i] = np.linalg.inv(X.T @ X) @ X.T @ Y 


    betas = np.c_[cg_betas, kw_betas]
    return p.DataFrame(betas, columns=['cg_alpha', 'cg_beta', 'kw_alpha', 'kw_beta'])

