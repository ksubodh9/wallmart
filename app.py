import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from fe import FEATURES, add_time_parts, make_feature_row  # shared code

st.set_page_config(layout="wide", page_title="Walmart Sales Forecast")

@st.cache_resource
def load_model():
    model = joblib.load("models/rf_model.pkl")
    with open("models/features.json","r") as f:
        feats = json.load(f)
    return model, feats

@st.cache_data
def load_data():
    df = pd.read_csv("data/train.csv", parse_dates=['Date'])
    df = df.sort_values(['Store','Date']).reset_index(drop=True)
    # we keep raw history for plotting; no lags needed here
    return add_time_parts(df)

def get_store_hist(df, store):
    s = df[df['Store']==store].sort_values('Date').reset_index(drop=True)
    return s

def recursive_forecast(model, feats, hist_df, store, weeks, fuel_price, unemployment, holiday_list):
    """Forecast next `weeks` recursively using last observed sales as lags."""
    if len(hist_df) < 3:
        raise ValueError("Not enough history (need ≥3 rows) to build lags.")

    last_date = pd.to_datetime(hist_df['Date'].max())

    # initialize lags with last actuals
    lag1 = float(hist_df['Weekly_Sales'].iloc[-1])
    lag2 = float(hist_df['Weekly_Sales'].iloc[-2])
    lag3 = float(hist_df['Weekly_Sales'].iloc[-3])

    out_rows = []
    for i in range(1, weeks+1):
        next_date = last_date + pd.DateOffset(weeks=i)
        hol = holiday_list[i-1] if i-1 < len(holiday_list) else 0

        Xrow = make_feature_row(
            next_date, store, fuel_price, unemployment, hol,
            lag1=lag1, lag2=lag2, lag3=lag3
        )
        # ensure column order matches training
        Xrow = Xrow[feats]

        yhat = float(model.predict(Xrow)[0])
        out_rows.append({"Date": next_date, "Predicted_Sales": yhat})

        # update lags for next step
        lag3, lag2, lag1 = lag2, lag1, yhat

    return pd.DataFrame(out_rows)

# -------- UI --------
model, feats = load_model()
df = load_data()

st.sidebar.header("Inputs")
store = st.sidebar.number_input(
    "Store ID",
    min_value=int(df['Store'].min()),
    max_value=int(df['Store'].max()),
    value=int(df['Store'].min())
)
weeks_ahead = st.sidebar.slider("Weeks to predict", 1, 12, 4)

use_last = st.sidebar.checkbox("Use last observed Fuel & Unemployment", value=True)
if use_last:
    last_row = df[df['Store']==store].iloc[-1]
    fuel_price = float(last_row['Fuel_Price'])
    unemployment = float(last_row['Unemployment'])
else:
    fuel_price = st.sidebar.number_input("Fuel Price", value=3.00, step=0.01, format="%.2f")
    unemployment = st.sidebar.number_input("Unemployment", value=7.0, step=0.1, format="%.1f")

st.sidebar.markdown("Future Holiday flags (0/1), comma-separated. Leave empty for all 0.")
holiday_text = st.sidebar.text_area("e.g. 0,0,1,0", value="")
if holiday_text.strip():
    holiday_list = [int(x.strip()) for x in holiday_text.split(",")][:weeks_ahead]
    if len(holiday_list) < weeks_ahead:
        holiday_list += [0]*(weeks_ahead-len(holiday_list))
else:
    holiday_list = [0]*weeks_ahead

if st.button("Run Forecast"):
    hist = get_store_hist(df, store)
    if hist.shape[0] < 3:
        st.error("Not enough historical data (need at least 3 weeks).")
    else:
        fc = recursive_forecast(model, feats, hist, store, weeks_ahead, fuel_price, unemployment, holiday_list)

        st.subheader(f"Forecast for Store {store} — next {weeks_ahead} weeks")
        st.dataframe(fc.assign(Date=fc['Date'].dt.strftime('%Y-%m-%d')))

        # Plot: last 52 weeks + forecasts
        plt.figure(figsize=(10,4))
        last_hist = hist.tail(52)
        plt.plot(last_hist['Date'], last_hist['Weekly_Sales'], marker='o', label='Historical')
        plt.plot(fc['Date'], fc['Predicted_Sales'], marker='o', linestyle='--', label='Forecast')
        plt.xlabel('Date'); plt.ylabel('Weekly Sales'); plt.legend()
        plt.title(f"Store {store} — Historical vs Forecast")
        st.pyplot(plt.gcf())
