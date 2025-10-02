import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import os

# -----------------------
# Create cache directory if it doesn't exist
# -----------------------
if not os.path.exists(".cache/observables"):
    os.makedirs(".cache/observables")

# -----------------------
# Load FRED macro data
# -----------------------
fred_series = ["GDPC1", "GDPDEF", "CNP16OV", "FEDFUNDS"]

data = (
    pdr.DataReader(fred_series, "fred", start="1965-01-01", end="2007-12-31")
    .resample("QE")  # quarterly average
    .mean()
    .to_period("Q")
)

# LNS index and output
data["LNSindex"] = data["CNP16OV"] / data["CNP16OV"]["1992Q3"]
data["output"] = np.log(data["GDPC1"] / data["LNSindex"]) * 100

# Inflation and interest rate
data["inflation"] = np.log(data["GDPDEF"] / data["GDPDEF"].shift(1)) * 400
data["interest_rate"] = data["FEDFUNDS"]

# Output growth
data["output_growth"] = data["output"].diff()

# Placeholder for inflation expectations
data["inflation_expectations"] = np.nan

# Trim to 1966Q1 - 2007Q4
data = data["1966Q1":"2007Q4"]

# Columns to keep
base_cols = ["output_growth", "inflation", "interest_rate", "inflation_expectations"]

# -----------------------
# Save 1: full sample no expectations
# -----------------------
fullsample = data[base_cols]
np.savetxt(".cache/observables/fullsample_with_nan_inflation_expectations.txt", fullsample.values, delimiter='\t', fmt='%.6f')


# -----------------------
# Load SPF GDP deflator level forecasts
# -----------------------
spf_url = (
    "https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/"
    "survey-of-professional-forecasters/data-files/files/individual_{:}gdp.xlsx?la=en"
)

def load_spf_data(series='p'):
    first_letter = series[0]
    spf = pd.read_excel(spf_url.format(series), engine='openpyxl')

    # Keep only rows where YEAR and QUARTER are numbers
    spf = spf[pd.to_numeric(spf['YEAR'], errors='coerce').notna()]
    spf = spf[pd.to_numeric(spf['QUARTER'], errors='coerce').notna()]

    # Convert YEAR and QUARTER to int
    spf['YEAR'] = spf['YEAR'].astype(int)
    spf['QUARTER'] = spf['QUARTER'].astype(int)

    # Create Period objects
    spf['date'] = spf.apply(lambda x: pd.Period(year=x.YEAR, quarter=x.QUARTER, freq='Q'), axis=1)

    # Build wide-format dataframe
    df = pd.DataFrame([
        {
            'survey_date': row['date'],
            'Firm': row['ID'],
            row['date']-1: row[first_letter.upper()+'GDP1'],
            row['date']: row[first_letter.upper()+'GDP2'],
            row['date']+1: row[first_letter.upper()+'GDP3'],
            row['date']+2: row[first_letter.upper()+'GDP4'],
            row['date']+3: row[first_letter.upper()+'GDP5'],
            row['date']+4: row[first_letter.upper()+'GDP6'],
        }
        for _, row in spf.iterrows()
    ]).set_index(['survey_date','Firm'])

    return df

def get_1q_ahead_cg_style(spf_mean, periods):
    """
    Compute 1-quarter ahead inflation expectations, Coibion-Gorodnichenko style.
    This is 400 * ln(forecast of p[t] / forecast of p[t-1]), where forecasts are made at t.
    """
    expectations = []
    for t in periods:
        try:
            level_t = spf_mean.at[t, t]        # nowcast for t made at t
            level_prev = spf_mean.at[t, t - 1] # backcast for t-1 made at t
            infl = np.log(level_t / level_prev) * 400
        except KeyError:
            infl = np.nan
        expectations.append(infl)
    return pd.Series(expectations, index=periods)

def get_4q_ahead_cg_style(spf_mean, periods):
    """
    Compute 1-year ahead inflation expectations, Coibion-Gorodnichenko style.
    This is 100 * ln(forecast of p[t+3] / forecast of p[t-1]), where forecasts are made at t.
    """
    expectations = []
    for t in periods:
        try:
            level_t_plus_3 = spf_mean.at[t, t + 3] # forecast for t+3 made at t
            level_t_minus_1 = spf_mean.at[t, t - 1] # backcast for t-1 made at t
            infl = np.log(level_t_plus_3 / level_t_minus_1) * 100
        except KeyError:
            infl = np.nan
        expectations.append(infl)
    return pd.Series(expectations, index=periods)

spf = load_spf_data("p")

# Average across firms
spf_mean = spf.groupby("survey_date").mean()  # level forecasts
sample_periods = data.index

# -----------------------
# Save 2: 1-quarter ahead expectations
# -----------------------
data_1q = fullsample.copy()
data_1q["inflation_expectations"] = get_1q_ahead_cg_style(spf_mean, sample_periods)
np.savetxt(".cache/observables/fullsample_with_1q_inflation_expectations.txt", data_1q.values, delimiter='\t', fmt='%.6f')


# -----------------------
# Save 3: 4-quarter ahead expectations
# -----------------------
data_4q = fullsample.copy()
data_4q["inflation_expectations"] = get_4q_ahead_cg_style(spf_mean, sample_periods)
np.savetxt(".cache/observables/fullsample_with_4q_inflation_expectations.txt", data_4q.values, delimiter='\t', fmt='%.6f')

print("All three datasets saved successfully to .cache/observables.")
