import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st
import time

# Define Nifty 50 stocks
nifty_stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']

# Streamlit app setup
st.title("Nifty 50 Machine Learning Trading Strategy")
initial_balance = st.number_input("Initial Capital (₹):", 100000)
risk_per_trade = st.slider("Risk per Trade (%):", 1, 5, 2)

# Download data for Nifty 50 stocks
all_data = {}
for stock in nifty_stocks:
    try:
        all_data[stock] = yf.download(stock, start='2019-01-01', end='2024-01-01')
    except Exception as e:
        st.write(f"Error downloading data for {stock}: {e}")
        continue

# Prepare data for ML
features, labels = [], []
for stock, data in all_data.items():
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = 100 - (100 / (1 + (data['Close'].diff(1).clip(lower=0).rolling(window=14).mean() /
                                     data['Close'].diff(1).clip(upper=0).abs().rolling(window=14).mean())))
    data['RSI'] = 100 - (100 / (1 + (data['Close'].diff(1).clip(lower=0).rolling(window=14).mean().align(
    data['Close'].diff(1).clip(upper=0).abs().rolling(window=14).mean(), axis=0, copy=False)[0])))
    data['ATR'] = data['High'].subtract(data['Low']).rolling(window=14).mean()
    data['Volatility'] = data['Close'].pct_change().rolling(window=14).std()
    data['Target'] = np.where(
    (data['Close'].align(data['SMA20'], axis=0, copy=False)[0] > data['SMA20']) &
    (data['RSI'].align(data['Close'], axis=0, copy=False)[0] < 40), 1,
    np.where(
        (data['Close'].align(data['SMA50'], axis=0, copy=False)[0] < data['SMA50']) &
        (data['RSI'].align(data['Close'], axis=0, copy=False)[0] > 70), -1, 0
    )
)
    data = data.dropna()
    features.extend(data[['Close', 'SMA20', 'SMA50', 'RSI', 'ATR', 'Volatility']].values.tolist())
    labels.extend(data['Target'].values.tolist())

# Hyperparameter tuning
param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, 30], 'min_samples_split': [2, 5, 10]}
model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
grid_search.fit(features, labels)

best_model = grid_search.best_estimator_
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# Display accuracy
st.write("Model Accuracy:", accuracy_score(y_test, y_pred))
st.text(classification_report(y_test, y_pred))

# Backtesting with dynamic ATR-based stop-loss and position sizing
balance = initial_balance
position = 0
atr_multiplier = 1.5

for stock, data in all_data.items():
    data['Signal'] = best_model.predict(
        data[['Close', 'SMA20', 'SMA50', 'RSI', 'ATR', 'Volatility']].dropna().align(
            data[['Close', 'SMA20', 'SMA50', 'RSI', 'ATR', 'Volatility']].dropna(), axis=0, copy=False)[0].values
    )
    for i in range(len(data)):
        if not np.isnan(data['ATR'][i]) and not np.isnan(data['Close'][i]):
            if data['Signal'][i] == 1 and balance > 0:
                risk_amount = balance * (risk_per_trade / 100)
                position_size = risk_amount / (atr_multiplier * data['ATR'][i])
                position = min(balance / data['Close'][i], position_size)
                balance -= position * data['Close'][i]
                stop_loss = data['Close'][i] - (atr_multiplier * data['ATR'][i])
            elif position > 0 and (data['Close'][i] < stop_loss or data['Signal'][i] == -1):
                balance += position * data['Close'][i]
                position = 0
# Final performance evaluation
final_balance = balance if balance > 0 else position * data['Close'].fillna(0).iloc[-1]
profit = final_balance - initial_balance
performance = (profit / initial_balance) * 100
cagr = ((final_balance / initial_balance) ** (1 / 5)) - 1

st.metric("Final Balance", f"₹{final_balance:.2f}")
st.metric("Total Profit", f"₹{profit:.2f} ({performance:.2f}%)")
st.metric("CAGR", f"{cagr:.2%}")

# Visualization
st.write("## Performance Chart")
plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label='Close Price', alpha=0.5)
plt.scatter(data.index[data['Signal'] == 1], data['Close'][data['Signal'] == 1], label='Buy Signal', marker='^', color='green', alpha=1)
plt.scatter(data.index[data['Signal'] == -1], data['Close'][data['Signal'] == -1], label='Sell Signal', marker='v', color='red', alpha=1)
plt.title(f"Machine Learning Backtest on {stock} with ATR Stop-loss & Position Sizing")
plt.legend()
st.pyplot(plt)

# Live price stream for last stock
st.write("## Live Price Stream")
try:
    latest_price = yf.download(stock, period='1d', interval='1m')
    latest_close = latest_price['Close'].iloc[-1]
    st.metric("Latest Price", f"₹{latest_close:.2f}")
except Exception as e:
    st.write(f"Error fetching live price: {e}")
