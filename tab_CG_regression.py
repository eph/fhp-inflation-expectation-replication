import numpy as np
import pandas as p

# RT data xls is corrupt (!) 
import openpyxl.reader.excel
openpyxl.reader.excel.ARC_CORE = "fjdslf"


#------------------------------------------------------------ 
# SPF Data
#------------------------------------------------------------
philly_fed ='https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/'
spf_url = philly_fed + 'survey-of-professional-forecasters/data-files/files/individual_pgdp.xlsx?la=en'

spf = p.read_excel(spf_url)
spf['date'] = spf.apply(lambda x: p.Period('%dQ%d' %(x.YEAR, x.QUARTER), freq='Q'), axis=1)


spf = p.DataFrame([{'survey_date': row['date'],
                     'Firm': row['ID'],
                     row['date']-1: row['PGDP1'],
                     row['date']: row['PGDP2'],
                     row['date']+1: row['PGDP3'],
                     row['date']+2: row['PGDP4'],                         
                     row['date']+3: row['PGDP5'],                          
                     row['date']+4: row['PGDP6']}
                    for index, row in spf.iterrows()]).set_index(['survey_date','Firm'])

spf.columns = [p.Period(c, freq='Q') for c in spf.columns]

#------------------------------------------------------------
# Real Time Data
#------------------------------------------------------------
def rename_column(column):
    stem = column[-4:]

    if int(stem[:2]) > 50: 
        root = '19'
    else:
        root = '20'

    return p.Period(root+stem,freq='Q')

actual_url = philly_fed + "real-time-data/data-files/xlsx/pqvqd.xlsx"
actual = p.read_excel(actual_url, na_values=["#N/A"])
actual['DATE'] = actual['DATE'].apply(lambda x: p.Period(str(x).replace(':',''),freq='Q'))
actual.set_index('DATE', inplace=True)
actual = actual.rename(columns=rename_column)
actual = actual['1968':'2020']
    
def construct_CG_variables(spf, actual, h=3):
    # h+1 since we are going to from level to changes
    forecast = ( 
        # translate path of price level forecast into path annualized h-period inflation
        (spf.apply(np.log)*400/(h+1)).diff(h+1,axis='columns') # [h+1] is for level -> changes
        # average across firms 
        .groupby('survey_date').mean() 
        # select the h-step ahead inflation forecast
        .apply(lambda x: x[x.name+h], axis=1) )
    lagged_forecast = ( 
        # translate path of price level forecast into path annualized h-period inflation
        (spf.apply(np.log)*400/(h+1)).diff(h+1,axis='columns') # [h+1] is for level -> changes
        # average across firms 
        .groupby('survey_date').mean() 
        # select the h-step ahead inflation forecast
        .apply(lambda x: x[x.name+h+1], axis=1).shift())

    revision = forecast-lagged_forecast

    # Note that CG's actual for say annual inflation is the result of
    # the sum of 4 quarterly actuals, each taken from different
    # vintages.  We use a single vintage (the one 4 quarters after the
    # first realization of annual inflation).
    realtime_actual = (
        # translate price data in paths for annualized h-period inflation for each vintage
        (400/(h+1)*actual.pct_change(h+1, fill_method=None))
        # the `t` index will refer to forecast origination dates, so we shift the actual backward to align
        .shift(-h)
        # to avoid annoying missing column errors
        .loc['1965':'2020',:]
        # to construct univariate series, we take vintage 4 quarters after forecast period 
        # we include h in this because we've shifted teh data backwards.
        .apply(lambda x: x[x.name+4+h], axis=1))

    lagged_inflation = (400/(h+1)*actual.pct_change(h+1)).shift(1).loc['1965':'2020',:].apply(lambda x: x[x.name], axis=1)
    reg_data = p.concat([forecast, revision, realtime_actual,lagged_inflation], axis=1, keys=['Forecast', 'Revision', 'Actual', 'Lagged Inflation']).dropna(how='any')

    idx=p.period_range('1969','2019',freq='Q');
    reg_data['Error'] = reg_data.Actual - reg_data.Forecast

    return reg_data



def run_h_regression(reg_data, dep_var='Error', regressor='Revision', start='1969Q4', end='2013Q3'):
    consensus_reg = (smf.ols(f'{dep_var} ~ {regressor}', data=reg_data[f'{start}':f'{end}'])
                        .fit()
                        .get_robustcov_results('HAC', maxlags=4))
    return consensus_reg

import statsmodels.formula.api as smf

if __name__ == "__main__":
    cg_data = construct_CG_variables(spf, actual, h=3)
    cg_reg_h3 = run_h_regression(cg_data, dep_var='Error', regressor='Revision', end='2007Q4')

    cg_data = construct_CG_variables(spf, actual, h=0)
    cg_reg_h0 = run_h_regression(cg_data, dep_var='Error', regressor='Revision', end='2007Q4')
    
    from statsmodels.iolib.summary2 import summary_col
    info_dict = {'Number of observations' : lambda x: f"{int(x.nobs):d}", 
                 'Adjusted R-squared' : lambda x: f"{x.rsquared_adj:.2f}"}

    
    dfoutput = summary_col([cg_reg_h0, cg_reg_h3], stars=False,
                           info_dict=info_dict, model_names=['h=0','h=3'],
                           float_format='%.2f')


    dfoutput.tables[0].drop(['R-squared', 'R-squared Adj.'],inplace=True)  # Remove specified rows
    dfoutput.title = 'CG Regression Results'
    with open('output/cg-regression-results.tex', 'w') as f:
        f.write(dfoutput.as_latex(label='tab:cg'))
    np.savetxt('output/cg_coefficient_h0.txt', cg_reg_h0.params)
    np.savetxt('output/cg_coefficient_h3.txt', cg_reg_h3.params)
