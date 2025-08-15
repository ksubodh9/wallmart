# fe.py
import numpy as np
import pandas as pd

FEATURES = [
    'Store', 'Fuel_Price', 'Unemployment', 'Holiday_Flag',
    'Week_sin', 'Week_cos', 'Month_sin', 'Month_cos',
    'Sales_lag_1', 'Sales_lag_2', 'Sales_lag_3', 'Sales_roll_3'
]

def add_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    # ISO week is robust for weekly data
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Month'] = df['Date'].dt.month.astype(int)
    df['Year'] = df['Date'].dt.year.astype(int)
    # cyclical encodings
    df['Week_sin'] = np.sin(2*np.pi*df['Week']/52)
    df['Week_cos'] = np.cos(2*np.pi*df['Week']/52)
    df['Month_sin'] = np.sin(2*np.pi*df['Month']/12)
    df['Month_cos'] = np.cos(2*np.pi*df['Month']/12)
    return df

def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['Store','Date']).copy()
    for lag in [1,2,3]:
        df[f'Sales_lag_{lag}'] = df.groupby('Store')['Weekly_Sales'].shift(lag)
    # rolling mean of the previous 3 weeks (shifted so it uses only past info)
    df['Sales_roll_3'] = (
        df.groupby('Store')['Weekly_Sales']
          .apply(lambda s: s.shift(1).rolling(3).mean())
          .reset_index(level=0, drop=True)
    )
    return df

def build_training_frame(raw: pd.DataFrame) -> pd.DataFrame:
    df = add_time_parts(raw)
    df = add_lags(df)
    # we train on rows where all lags exist
    df = df.dropna(subset=['Sales_lag_1','Sales_lag_2','Sales_lag_3','Sales_roll_3'])
    return df

def make_feature_row(next_date, store, fuel_price, unemployment, holiday_flag,
                     lag1, lag2, lag3) -> pd.DataFrame:
    """Build ONE row with the exact FEATURES for inference."""
    next_date = pd.to_datetime(next_date)
    week = int(next_date.isocalendar().week)
    month = int(next_date.month)
    row = {
        'Store': store,
        'Fuel_Price': float(fuel_price),
        'Unemployment': float(unemployment),
        'Holiday_Flag': int(holiday_flag),
        'Week_sin': np.sin(2*np.pi*week/52),
        'Week_cos': np.cos(2*np.pi*week/52),
        'Month_sin': np.sin(2*np.pi*month/12),
        'Month_cos': np.cos(2*np.pi*month/12),
        'Sales_lag_1': float(lag1),
        'Sales_lag_2': float(lag2),
        'Sales_lag_3': float(lag3),
        'Sales_roll_3': float(np.mean([lag1, lag2, lag3]))
    }
    return pd.DataFrame([row], columns=FEATURES)
