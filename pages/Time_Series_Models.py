import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

# --- Streamlit setup ---
st.set_page_config(page_title="Time Series Models",
                   page_icon="📈", layout="wide")
st.title("📈 Time Series Forecasting Models")

# --- User input ---
col1, col2 = st.columns(2)
with col1:
    model_type = st.selectbox("Select Forecasting Model", ("ARIMA", "SARIMA", "Prophet","LSTM"))
    stock_symbol = st.text_input("Enter Stock Symbol", "AAPL")
with col2:
    years = st.number_input("Years of historical data", 1, 10, step=1)
    forecast_horizon = st.number_input("Days to Forecast Ahead", 1, 60, value=30)

# --- Load stock data ---
@st.cache_data
def load_data(symbol, years):
    data = yf.download(symbol, period=f"{years}y")
    df = data[["Close"]].reset_index()
    df.columns = ["Date", "Close"]
    return df

# --- ARIMA Forecast ---
def arima_forecast(series, order, steps):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

# --- SARIMA Forecast ---
def sarima_forecast(series, order, seasonal_order, steps):
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

# --- Prophet Forecast ---
def prophet_forecast(data, periods):
    df = data.rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
# --- LSTM Forecast ---
def lstm_forecast(data, forecast_horizon, look_back=30, epochs=10):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data["Close"].values.reshape(-1, 1))
    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i-look_back:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    #             (sample    ,timestamp,feature)   

    model = keras.Sequential([
        keras.layers.LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
        keras.layers.LSTM(50),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, batch_size=16, verbose=0)

    last_seq = scaled[-look_back:]
    preds = []
    for _ in range(forecast_horizon):
        inp = last_seq.reshape((1, look_back, 1))
        pred = model.predict(inp, verbose=0)[0][0]
        preds.append(pred)
        last_seq = np.append(last_seq[1:], [[pred]], axis=0)
    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return preds

# --- Main Forecast Logic ---
try:
    if stock_symbol:
        data = load_data(stock_symbol, years)
        st.subheader(f"📊 {stock_symbol} Historical Closing Price")
        st.line_chart(data.set_index("Date"))

        if model_type == "ARIMA":
            st.subheader("🔮 ARIMA Forecast")
            order = (1, 1, 1)
            forecast = arima_forecast(data["Close"], order, forecast_horizon)

            # Create forecast dates
            last_date = data["Date"].iloc[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon)

            # Combine for plot
            forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": forecast})
            full_df = pd.concat([
                data[["Date", "Close"]].rename(columns={"Close": "Value"}),
                forecast_df.rename(columns={"Forecast": "Value"})
            ])
            st.line_chart(full_df.set_index("Date"))
            st.dataframe(forecast_df, use_container_width=True,hide_index=True)

        elif model_type == "SARIMA":
            st.subheader("🔮 SARIMA Forecast")
            order = (1, 1, 1)
            seasonal_order = (1, 1, 1, 12)
            forecast = sarima_forecast(data["Close"], order, seasonal_order, forecast_horizon)

            last_date = data["Date"].iloc[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon)

            forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": forecast})
            full_df = pd.concat([
                data[["Date", "Close"]].rename(columns={"Close": "Value"}),
                forecast_df.rename(columns={"Forecast": "Value"})
            ])
            st.line_chart(full_df.set_index("Date"))
            st.dataframe(forecast_df, use_container_width=True,hide_index=True)

        elif model_type == "Prophet":
            st.subheader("🔮 Prophet Forecast")
            forecast = prophet_forecast(data, forecast_horizon)
            st.line_chart(forecast.set_index("ds")[["yhat"]])
            forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={
                'ds': 'Date',
                'yhat': 'Forecast',
                'yhat_lower': 'Lower Bound',
                'yhat_upper': 'Upper Bound'
            })
            st.dataframe(forecast_display.tail(forecast_horizon), use_container_width=True,hide_index=True)

        elif model_type == "LSTM":
            st.subheader("🔮 LSTM Forecast")
            preds = lstm_forecast(data, forecast_horizon)
            last_date = data["Date"].iloc[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon)
            forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": preds})
            full_df = pd.concat([
                data[["Date", "Close"]].rename(columns={"Close": "Value"}),
                forecast_df.rename(columns={"Forecast": "Value"})
            ])
            st.line_chart(full_df.set_index("Date"))
            st.dataframe(forecast_df, use_container_width=True)


except Exception as e:
    st.error("⚠️ An error occurred:")
    st.exception(e)
