import streamlit as st
import yfinance as yf
import pandas_ta as ta
import numpy as np
import joblib
from datetime import datetime, timezone, timedelta

st.set_page_config(page_title="GBP/USD Dashboard", layout="wide")

# Load the saved model
model = joblib.load("gbpusd_model.pkl")

# Fetch latest data
data = yf.download("GBPUSD=X", period="18mo", interval="1h")
data.columns = data.columns.get_level_values(0)

# Add indicators
data["RSI"]         = ta.rsi(data["Close"], length=14)
data["EMA9"]        = ta.ema(data["Close"], length=9)
data["EMA21"]       = ta.ema(data["Close"], length=21)

# Calculate MACD manually
exp1 = data["Close"].ewm(span=12, adjust=False).mean()
exp2 = data["Close"].ewm(span=26, adjust=False).mean()
data["MACD"]        = exp1 - exp2
data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

bb                  = ta.bbands(data["Close"])
upper_col           = [c for c in bb.columns if c.startswith("BBU")][0]
lower_col           = [c for c in bb.columns if c.startswith("BBL")][0]
data["BB_Upper"]    = bb[upper_col]
data["BB_Lower"]    = bb[lower_col]

data["ATR"]         = ta.atr(data["High"], data["Low"], data["Close"], length=14)
stoch               = ta.stoch(data["High"], data["Low"], data["Close"])
data["STOCH"]       = stoch["STOCHk_14_3_3"]
data["CCI"]         = ta.cci(data["High"], data["Low"], data["Close"], length=20)

daily               = yf.download("GBPUSD=X", period="18mo", interval="1d")
daily.columns       = daily.columns.get_level_values(0)
daily["Prev_High"]  = daily["High"].shift(1)
daily["Prev_Low"]   = daily["Low"].shift(1)
daily.index         = daily.index.tz_localize("UTC")
data["Prev_High"]   = daily["Prev_High"].reindex(data.index, method="ffill")
data["Prev_Low"]    = daily["Prev_Low"].reindex(data.index, method="ffill")
data["Above_Prev_High"] = (data["Close"] > data["Prev_High"]).astype(int)
data["Below_Prev_Low"]  = (data["Close"] < data["Prev_Low"]).astype(int)

data["Body"]        = abs(data["Close"] - data["Open"])
data["Upper_Wick"]  = data["High"] - data[["Close", "Open"]].max(axis=1)
data["Lower_Wick"]  = data[["Close", "Open"]].min(axis=1) - data["Low"]
data["Is_Bullish"]  = (data["Close"] > data["Open"]).astype(int)
data["Doji"]        = (data["Body"] < data["Body"].rolling(14).mean() * 0.1).astype(int)
data["Hammer"]      = ((data["Lower_Wick"] > 2 * data["Body"]) & (data["Upper_Wick"] < data["Body"])).astype(int)
data["Shooting_Star"] = ((data["Upper_Wick"] > 2 * data["Body"]) & (data["Lower_Wick"] < data["Body"])).astype(int)
data["Bullish_Engulf"] = ((data["Is_Bullish"] == 1) & (data["Close"] > data["Open"].shift(1)) & (data["Open"] < data["Close"].shift(1))).astype(int)
data["Bearish_Engulf"] = ((data["Is_Bullish"] == 0) & (data["Close"] < data["Open"].shift(1)) & (data["Open"] > data["Close"].shift(1))).astype(int)

window = 20
data["Resistance"]      = data["High"].rolling(window).max()
data["Support"]         = data["Low"].rolling(window).min()
data["Dist_Resistance"] = (data["Resistance"] - data["Close"]) / data["Close"]
data["Dist_Support"]    = (data["Close"] - data["Support"]) / data["Close"]
data["Break_Resistance"] = (data["Close"] > data["Resistance"].shift(1)).astype(int)
data["Break_Support"]   = (data["Close"] < data["Support"].shift(1)).astype(int)

data["Sweep_High"]      = ((data["High"] > data["Prev_High"]) & (data["Close"] < data["Prev_High"])).astype(int)
data["Sweep_Low"]       = ((data["Low"] < data["Prev_Low"]) & (data["Close"] > data["Prev_Low"])).astype(int)
data["Strong_Bullish"]  = ((data["Close"] > data["Open"]) & (data["Body"] > data["Body"].rolling(20).mean() * 1.5)).astype(int)
data["Strong_Bearish"]  = ((data["Close"] < data["Open"]) & (data["Body"] > data["Body"].rolling(20).mean() * 1.5)).astype(int)
data["In_Demand_Zone"]  = ((data["Close"] <= data["Low"].shift(1)) & (data["Strong_Bullish"].shift(1) == 1)).astype(int)
data["In_Supply_Zone"]  = ((data["Close"] >= data["High"].shift(1)) & (data["Strong_Bearish"].shift(1) == 1)).astype(int)

data.dropna(inplace=True)

# Get latest signal
features  = ["RSI","EMA9","EMA21","MACD","MACD_Signal",
            "BB_Upper","BB_Lower","ATR","STOCH","CCI",
            "Prev_High","Prev_Low","Above_Prev_High",
            "Below_Prev_Low","Body","Upper_Wick","Lower_Wick",
            "Is_Bullish","Doji","Hammer","Shooting_Star",
            "Bullish_Engulf","Bearish_Engulf","Resistance",
            "Support","Dist_Resistance","Dist_Support",
            "Break_Resistance","Break_Support","Sweep_High",
            "Sweep_Low","Strong_Bullish","Strong_Bearish",
            "In_Demand_Zone","In_Supply_Zone"]

latest    = data[features].iloc[[-1]]
pred      = model.predict(latest)[0]
proba      = model.predict_proba(latest)[0]
label_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
signal    = label_map[pred]
confidence = round(float(max(proba)) * 100, 1)
price     = round(float(data["Close"].iloc[-1]), 5)
rsi_val   = round(float(data["RSI"].iloc[-1]), 2)

kampala = timezone(timedelta(hours=3))
time_now  = datetime.now(kampala).strftime("%Y-%m-%d %H:%M Ugandan Time")

color_map = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}

# Only show signal if confidence is above 60%
if confidence >= 60:
    final_signal = signal
    signal_note  = f"High confidence — act on this!"
else:
    final_signal = "HOLD"
    signal_note  = f"Confidence too low ({confidence}%) — stay out"

# Dashboard
st.title(" 📈 GBP/USD Signal Dashboard")
st.caption(f"Last updated: {time_now}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Price", price)
col2.metric("Signal", f"{color_map[final_signal]} {final_signal}")
col3.metric("Confidence", f"{confidence}%")
col4.metric("RSI", rsi_val)

st.divider()

# Confidence bars
st.subheader("Signal Breakdown")
col1, col2, col3 = st.columns(3)
col1.metric("🔴 SELL", f"{round(float(proba[0])*100, 1)}%")
col2.metric("🟡 HOLD", f"{round(float(proba[1])*100, 1)}%")
col3.metric("🟢 BUY",  f"{round(float(proba[2])*100, 1)}%")

st.divider()

st.subheader("Recent Price Chart")
st.line_chart(data["Close"].tail(100))

st.divider()

st.subheader("Latest Indicator Values")
st.dataframe(latest)

st.info(signal_note)

import time
st.divider()
placeholder = st.empty()

for remaining in range(300, 0, -1):
    minutes = remaining // 60
    seconds = remaining % 60
    placeholder.caption(f"⏱️ Refreshing in {minutes:02d}:{seconds:02d} minutes...")
    time.sleep(1)

st.rerun()
