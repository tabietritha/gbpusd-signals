import yfinance as yf
import pandas_ta as ta  
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

data = yf.download("GBPUSD=X", period="18mo", interval="1h")

data.columns = data.columns.get_level_values(0)

#indicators now 
data["RSI"]  = ta.rsi(data["Close"], length=14)
data["EMA9"] = ta.ema(data["Close"], length=9)
data["EMA21"]= ta.ema(data["Close"], length=21) 

#macd and boilenger bands
exp1 = data["Close"].ewm(span=12, adjust=False).mean()
exp2 = data["Close"].ewm(span=26, adjust=False).mean()
data["MACD"] = exp1 - exp2
data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

bb = ta.bbands(data["Close"])
upper_col = [c for c in bb.columns if c.startswith ("BBU")][0]
lower_col = [c for c in bb.columns if c.startswith ("BBL")][0]
data["BB_Upper"] = bb[upper_col]
data["BB_Lower"] = bb[lower_col]

data["ATR"]   = ta.atr(data["High"], data["Low"], data["Close"], length=14)
stoch         = ta.stoch(data["High"], data["Low"], data["Close"])
data["STOCH"] = stoch["STOCHk_14_3_3"]
data["CCI"]   = ta.cci(data["High"], data["Low"], data["Close"], length=20)

# Previous day high and low
daily = yf.download("GBPUSD=X", period="18mo", interval="1d")
daily.columns = daily.columns.get_level_values(0)
daily["Prev_High"] = daily["High"].shift(1)
daily["Prev_Low"]  = daily["Low"].shift(1)

# Resample to match hourly data
daily.index = daily.index.tz_localize("UTC")
data["Prev_High"] = daily["Prev_High"].reindex(data.index, method="ffill")
data["Prev_Low"]  = daily["Prev_Low"].reindex(data.index, method="ffill")

# Is price above or below previous day levels?
data["Above_Prev_High"] = (data["Close"] > data["Prev_High"]).astype(int)
data["Below_Prev_Low"]  = (data["Close"] < data["Prev_Low"]).astype(int)

# Candle patterns
data["Body"]      = abs(data["Close"] - data["Open"])
data["Upper_Wick"] = data["High"] - data[["Close", "Open"]].max(axis=1)
data["Lower_Wick"] = data[["Close", "Open"]].min(axis=1) - data["Low"]
data["Is_Bullish"] = (data["Close"] > data["Open"]).astype(int)

# Doji — body is very small
data["Doji"] = (data["Body"] < data["Body"].rolling(14).mean() * 0.1).astype(int)

# Hammer — lower wick is 2x the body, small upper wick
data["Hammer"] = (
    (data["Lower_Wick"] > 2 * data["Body"]) &
    (data["Upper_Wick"] < data["Body"])
).astype(int)

# Shooting Star — upper wick is 2x the body, small lower wick
data["Shooting_Star"] = (
    (data["Upper_Wick"] > 2 * data["Body"]) &
    (data["Lower_Wick"] < data["Body"])
).astype(int)

# Bullish Engulfing
data["Bullish_Engulf"] = (
    (data["Is_Bullish"] == 1) &
    (data["Close"] > data["Open"].shift(1)) &
    (data["Open"] < data["Close"].shift(1))
).astype(int)

# Bearish Engulfing
data["Bearish_Engulf"] = (
    (data["Is_Bullish"] == 0) &
    (data["Close"] < data["Open"].shift(1)) &
    (data["Open"] > data["Close"].shift(1))
).astype(int)

# Support and Resistance
window = 20
data["Resistance"] = data["High"].rolling(window).max()
data["Support"]    = data["Low"].rolling(window).min()

# How close is price to support or resistance? (as a %)
data["Dist_Resistance"] = (data["Resistance"] - data["Close"]) / data["Close"]
data["Dist_Support"]    = (data["Close"] - data["Support"]) / data["Close"]

# Is price breaking above resistance or below support?
data["Break_Resistance"] = (data["Close"] > data["Resistance"].shift(1)).astype(int)
data["Break_Support"]    = (data["Close"] < data["Support"].shift(1)).astype(int)

# Liquidity Sweeps
data["Sweep_High"] = (
    (data["High"] > data["Prev_High"]) &
    (data["Close"] < data["Prev_High"])
).astype(int)

data["Sweep_Low"] = (
    (data["Low"] < data["Prev_Low"]) &
    (data["Close"] > data["Prev_Low"])
).astype(int)

# Supply and Demand Zones
# Strong bullish candle = potential demand zone below
data["Strong_Bullish"] = (
    (data["Close"] > data["Open"]) &
    (data["Body"] > data["Body"].rolling(20).mean() * 1.5)
).astype(int)

# Strong bearish candle = potential supply zone above
data["Strong_Bearish"] = (
    (data["Close"] < data["Open"]) &
    (data["Body"] > data["Body"].rolling(20).mean() * 1.5)
).astype(int)

# Price returning to demand zone (previous strong bullish area)
data["In_Demand_Zone"] = (
    (data["Close"] <= data["Low"].shift(1)) &
    (data["Strong_Bullish"].shift(1) == 1)
).astype(int)

# Price returning to supply zone (previous strong bearish area)
data["In_Supply_Zone"] = (
    (data["Close"] >= data["High"].shift(1)) &
    (data["Strong_Bearish"].shift(1) == 1)
).astype(int)

#looks for 3 hours ahead
future = data["Close"].shift(-6)

# Calculate change in percentage
pct_change = (future - data["Close"]) / data["Close"]

# Label each row
data["Label"] = np.where(pct_change > 0.001, 2,
                np.where(pct_change < -0.001, 0, 1))

#dropping empty rows
data.dropna(inplace=True)

#features and target
features = ["RSI","EMA9","EMA21","MACD","MACD_Signal",
            "BB_Upper","BB_Lower","ATR","STOCH","CCI",
            "Prev_High","Prev_Low","Above_Prev_High",
            "Below_Prev_Low","Body","Upper_Wick","Lower_Wick",
            "Is_Bullish","Doji","Hammer","Shooting_Star",
            "Bullish_Engulf","Bearish_Engulf","Resistance",
            "Support","Dist_Resistance","Dist_Support",
            "Break_Resistance","Break_Support","Sweep_High",
            "Sweep_Low","Strong_Bullish","Strong_Bearish",
            "In_Demand_Zone","In_Supply_Zone"]
X = data[features]
y = data["Label"]

#training and testing split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print(f"Training rows: {len(X_train)}")
print(f"Testing rows:  {len(X_test)}")

#train the model
model = XGBClassifier(n_estimators=300, 
                      max_depth=5, 
                      learning_rate=0.05,
                      random_state=42)
model.fit(X_train, y_train)

#testing the model here
y_pred = model.predict(X_test)
print("\nModel Performance:")
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, "gbpusd_model.pkl")
print("Model saved successfully")

#load the model
loaded_model = joblib.load("gbpusd_model.pkl")
print("Model loaded successfully")  

#test it on the latest row of data
latest = X.iloc[[-1]]
prediction = loaded_model.predict(latest)[0]
proba = loaded_model.predict_proba(latest)[0]

label_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
signal = label_map[prediction]
confidence = round(float (max(proba)) * 100, 1)


print(f"Latest Signal: {signal}")    
print(f"Confidence: {confidence}%")
print(f"SELL : {round(proba[0]*100,1)}%")
print(f"HOLD : {round(proba[1]*100,1)}%")
print(f"BUY : {round(proba[2]*100,1)}%")
