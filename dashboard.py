import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression


# =========================
# CONFIG
# =========================
API_TOKEN = "66b2ee90a79e96.06006024"


# =========================
# DOWNLOAD DATA
# =========================
def get_daily_data(symbol, start_date, end_date):
    url = (
        f"https://eodhd.com/api/eod/{symbol}"
        f"?api_token={API_TOKEN}"
        f"&from={start_date}"
        f"&to={end_date}"
        f"&fmt=json"
    )

    response = requests.get(url)

    if response.status_code == 200:
        df = pd.DataFrame(response.json())

        if df.empty:
            return pd.DataFrame()

        df["datetime"] = pd.to_datetime(df["date"])
        df.set_index("datetime", inplace=True)
        df.drop(columns=["date"], inplace=True)
        return df

    return pd.DataFrame()


# =========================
# CALCULATIONS
# =========================
def calculate_regression_channel(data):
    df = data.copy().reset_index()

    df["day_num"] = np.arange(len(df))
    X = df[["day_num"]]
    y = df["close"]

    model = LinearRegression()
    model.fit(X, y)

    df["trend"] = model.predict(X)

    rolling_std = df["close"].rolling(20).std()
    rolling_std.fillna(df["close"].std(), inplace=True)

    df["upper1"] = df["trend"] + rolling_std
    df["upper2"] = df["trend"] + 2 * rolling_std
    df["lower1"] = df["trend"] - rolling_std
    df["lower2"] = df["trend"] - 2 * rolling_std

    df["zscore"] = (df["close"] - df["trend"]) / rolling_std
    df["distance_pct"] = ((df["close"] - df["trend"]) / df["trend"]) * 100

    return df


# =========================
# SIGNALS
# =========================
def generate_signal(last_row):
    if last_row["zscore"] <= -2:
        return "BUY"
    elif last_row["zscore"] >= 2:
        return "SELL"
    else:
        return "NEUTRAL"


# =========================
# PLOT
# =========================
def plot_chart(df, ticker):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["close"],
        mode="lines", name="Price"
    ))

    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["trend"],
        mode="lines", name="Trend"
    ))

    for band in ["upper1", "upper2", "lower1", "lower2"]:
        fig.add_trace(go.Scatter(
            x=df["datetime"],
            y=df[band],
            mode="lines",
            name=band
        ))

    fig.update_layout(
        title=f"{ticker} Mean Reversion Dashboard",
        template="plotly_dark",
        height=700
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================
# STREAMLIT APP
# =========================
st.set_page_config(layout="wide")
st.title("Mean Reversion Trading Dashboard")

ticker = st.sidebar.text_input("Ticker", "GBPCAD.FOREX")
months = st.sidebar.slider("Months Back", 1, 12, 3)

end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
start_date = (
    pd.Timestamp.today() - pd.DateOffset(months=months)
).strftime("%Y-%m-%d")

data = get_daily_data(ticker, start_date, end_date)

st.write("Ticker:", ticker)
st.write("Rows downloaded:", len(data))
st.write(data.head())

if not data.empty:
    df = calculate_regression_channel(data)
    latest = df.iloc[-1]

    signal = generate_signal(latest)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Last Price", round(latest["close"], 4))
    col2.metric("Trend", round(latest["trend"], 4))
    col3.metric("Z-Score", round(latest["zscore"], 2))
    col4.metric("Distance %", f"{latest['distance_pct']:.2f}%")

    st.subheader(f"Signal: {signal}")

    plot_chart(df, ticker)

else:
    st.error("No data found.")