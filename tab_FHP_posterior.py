from dsge import read_yaml

from fortress import load_estimates
from figures import replace
k = 4
FHPrep = read_yaml('models/finite_horizon.yaml').fix_parameters(ρ_η=0,σ_η=0)

linmod = FHPrep.compile_model()

results_noexp =load_estimates('.cache/compiled_models/macro/FHP[[]k=4[]]/output-*',
                              paranames=linmod.parameter_names)

paras = [('γ',        r'Household learning rate'),                                  
         ('γ_f',       r'Firm learning rate'),                                       
         ('κ',        r'Slope of the Phillips curve'),                              
         ('σ',         r'Coef. of relative risk aversion'),                          
         ('Φ_π',      r'Int. rule response to \(\tilde \pi_t\)'),                   
         ('Φ_y',      r'Int. rule response to \(\tilde y_t\)' ),                    
         ('Φ_π_LR',   r'Int. rule response to \(\overline \pi_t\)'),                
         ('Φ_y_LR',   r'Int. rule response to \(\overline y_t\)'),                  
         ('ρ_ξ',      r'AR coeff. for  demand shock'),                              
         ('ρ_i',      r'AR coeff. for monetary policy shock'),                      
         ('ρ_y',      r'AR coeff. for supply shock')]                               


with open(f'output/selected-posterior-FHP-k-{k}.tex', 'w') as f:
    sys.stdout = f
    print(r"""\begin{tabular}{llcc}\hline\hline
    & Description & Mean & [0, 95] \\
    \hline""")

    for para, text in paras:
        print(rf'{replace(para)} & {text} & {results_noexp[para].mean():5.2f} & [{results_noexp[para].quantile(0.05):5.2f}, {results_noexp[para].quantile(0.95):5.2f}] \\')
    print(r'\hline\end{tabular}')

sys.stdout = sys.__stdout__








