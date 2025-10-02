
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr

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
fullsample.to_csv("fullsample.txt", sep="\t", float_format="%.6f")

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

def get_ahead_expectations(spf_mean, periods, horizon=1):
    """
    Compute annualized inflation expectations from SPF level forecasts.
    
    spf_mean: DataFrame, index=survey_date (forecast origin), columns=target dates
    periods: PeriodIndex of the desired sample
    horizon: int, quarters ahead (1 or 4)
    """
    expectations = []
    for t in periods:
        origin = t - horizon  # forecast origin
        try:
            level_t = spf_mean.at[origin, t]        # forecast for t made at origin
            level_prev = spf_mean.at[origin, t - 1] # previous level at origin
            infl = np.log(level_t / level_prev) * 400  # annualized quarterly
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
# Compute 1q ahead inflation from levels (annualized)
data_1q["inflation_expectations"] = get_ahead_expectations(spf_mean, sample_periods, horizon=1)
data_1q.to_csv("fullsample_with_1q_inflation_expectations.txt", sep="\t", float_format="%.6f")

# -----------------------
# Save 3: 4-quarter ahead expectations
# -----------------------
data_4q = fullsample.copy()
data_4q["inflation_expectations"] = get_ahead_expectations(spf_mean, sample_periods, horizon=4)
data_4q.to_csv("fullsample_with_4q_inflation_expectations.txt", sep="\t", float_format="%.6f")
np.savetxt('fullsample_with_4q_inflation_expectations.txt',data_4q)
# -----------------------
# Save 4: placeholder column (all NaNs)
# -----------------------
data_nan = fullsample.copy()
data_nan["inflation_expectations"] = np.nan
#data_nan.to_csv("fullsample_with_nan_inflation_expectations.txt", sep="\t", float_format="%.6f")
print("All four datasets saved successfully.")
