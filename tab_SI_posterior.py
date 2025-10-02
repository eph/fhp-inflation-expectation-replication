import sys;
from dsge import read_yaml

from fortress import load_estimates

from figures import replace



SInoFπ = read_yaml('models/sticky_information.yaml')

linear_model = SInoFπ.compile_model()
results_noexp =load_estimates(f'.cache/compiled_models/macro/SI/outpu*',
                              paranames=linear_model.parameter_names)
               
paras = [
    ('λ',        r'Sticky Information Parameter'),
    ('σ',         r'Coef. of relative risk aversion'),                          
    ('ζ', r'Habit Formation'), 
    ('Φ_π',      r'Int. rule response to \(\tilde \pi_t\)'),                   
    ('Φ_y',      r'Int. rule response to \(\tilde y_t\)' ),                    
    ('ρ_ξ',      r'AR coeff. for  demand shock'),                              
    ('ρ_i',      r'AR coeff. for monetary policy shock'),                      
    ('ρ_y',      r'AR coeff. for supply shock')]                               


with open(f'output/selected-posterior-FHP-si.tex', 'w') as f:
    sys.stdout = f
    print(r"""\begin{tabular}{llcc}\hline\hline
    & Description & Mean & [0, 95] \\
    \hline""")

    for para,text  in paras:#linear_model.parameter_names:
        print(rf'{replace(para)} & {text} & {results_noexp[para].mean():5.2f} & [{results_noexp[para].quantile(0.05):5.2f}, {results_noexp[para].quantile(0.95):5.2f}] \\')

    print(rf'\hline Log MDD &  & {results_noexp.logmdd.mean():5.2f} &  \\')
    print(r'\hline\end{tabular}')

sys.stdout = sys.__stdout__
