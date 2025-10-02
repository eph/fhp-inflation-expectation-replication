from contextlib import contextmanager
import matplotlib.pyplot as plt

import os
import pandas as p 
#plt.style.use("seaborn-white")
plt.matplotlib.rcParams["xtick.top"] = False
plt.matplotlib.rcParams["ytick.right"] = False
plt.matplotlib.rcParams["axes.spines.right"] = False
plt.matplotlib.rcParams["axes.spines.top"] = False
#plt.matplotlib.rcParams["font.sans-serif"] = "Fira Code"
#plt.matplotlib.rcParams["text.usetex"] = True


@contextmanager
def saved_figure(fname, **kwds):
    """
    Saves a figure in `fname`.
    """
    fig, ax = plt.subplots(**kwds)
    try:
        yield (fig, ax)
    finally:
        fig.savefig(fname, bbox_inches='tight')
        plt.close(fig)


from pylatexenc.latexencode import unicode_to_latex

def replace(x): 
    if x == 'πA': return r'$\pi^A$'
    elif x == 'yQ': return r'$\gamma^Q$'
    elif x == 'rA': return r'$r^A$'
    else: return (unicode_to_latex(r'$'+x+'$', non_ascii_only=True)
                  .replace('Phi','phi')
                  .replace(r'\ensuremath{\phi}_\ensuremath{\pi}_LR', r'\bar{\phi}_\pi')
                  .replace(r'\ensuremath{\phi}_y_LR', r'\bar{\phi}_{y}'))


import openpyxl
def create_508(df, sheet_name):
    """creates an excel sheet in the 508 file"""
 
    if os.path.isfile('output/508.xlsx'):
        kwds = {'mode': 'a', 'if_sheet_exists': 'replace'}
    else:
        kwds = {}
    with p.ExcelWriter(path='output/508.xlsx', 
                       engine='openpyxl', **kwds) as writer:
        df.to_excel(writer, sheet_name=sheet_name)
    
    # with p.ExcelWriter('output/508.xlsx', engine='openpyxl') as writer:
    #     book = openpyxl.load_workbook('output/508.xlsx')
    #     writer.book = book
    #     writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    #     df.to_excel(writer, sheet_name=sheet_name)

    return None
