import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import tensorflow as tf
from tensorflow import keras
import plotly.graph_objs as go

st.set_page_config(page_title="Model Tuning", layout="wide")
st.title("⚙️ Time Series Model Comparison & Tuning")

# --- User Input ---
col1, col2 = st.columns(2)
with col1:
    stock_symbol = st.text_input("Stock Symbol", "AAPL")
    models = st.multiselect("Select Models", ["ARIMA", "SARIMA", "Prophet", "LSTM"], default=["ARIMA", "LSTM"])
with col2:
    years = st.slider("Years of Historical Data", 1, 10, value=3)
    forecast_horizon = st.slider("Forecast Horizon (Days)", 1, 60, value=30)

st.divider()

# --- Model Tuning Controls ---
with st.expander("🔧 Tuning Parameters"):
    st.markdown("### ARIMA")
    arima_p = st.slider("ARIMA p", 0, 10, value=5)
    arima_d = st.slider("ARIMA d", 0, 2, value=1)
    arima_q = st.slider("ARIMA q", 0, 10, value=0)

    st.markdown("### SARIMA")
    sarima_p = st.slider("SARIMA p", 0, 5, value=1)
    sarima_d = st.slider("SARIMA d", 0, 2, value=1)
    sarima_q = st.slider("SARIMA q", 0, 5, value=1)
    sarima_P = st.slider("SARIMA P", 0, 5, value=1)
    sarima_D = st.slider("SARIMA D", 0, 2, value=1)
    sarima_Q = st.slider("SARIMA Q", 0, 5, value=1)
    sarima_m = st.slider("SARIMA seasonal period (m)", 1, 30, value=12)

    st.markdown("### LSTM")
    lstm_look_back = st.slider("LSTM Look-back Window", 5, 90, value=30)
    lstm_epochs = st.slider("LSTM Epochs", 1, 100, value=10)

# --- Load Data ---
@st.cache_data
def load_data(symbol, years):
    df = yf.download(symbol, period=f"{years}y")[["Close"]].reset_index()
    df.columns = ["Date", "Close"]
    return df

# --- Forecast Functions ---
def arima_forecast(series, order, steps):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    return model_fit.forecast(steps=steps)

def sarima_forecast(series, order, seasonal_order, steps):
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    return model_fit.forecast(steps=steps)

def prophet_forecast(df, periods):
    df_prophet = df.rename(columns={"Date": "ds", "Close": "y"})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[["ds", "yhat"]].tail(periods)["yhat"].values

def lstm_forecast(data, forecast_horizon, look_back=30, epochs=10):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data["Close"].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i-look_back:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

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

    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

# --- Main Execution ---
try:
    if stock_symbol and models:
        df = load_data(stock_symbol, years)
        st.subheader(f"📊 Historical Data for {stock_symbol}")
        st.line_chart(df.set_index("Date")["Close"])

        last_date = df["Date"].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon)
        results = {}

        # Model Execution
        for model_name in models:
            if model_name == "ARIMA":
                preds = arima_forecast(df["Close"], (arima_p, arima_d, arima_q), forecast_horizon)
            elif model_name == "SARIMA":
                preds = sarima_forecast(
                    df["Close"],
                    (sarima_p, sarima_d, sarima_q),
                    (sarima_P, sarima_D, sarima_Q, sarima_m),
                    forecast_horizon
                )
            elif model_name == "Prophet":
                preds = prophet_forecast(df, forecast_horizon)
            elif model_name == "LSTM":
                preds = lstm_forecast(df, forecast_horizon, look_back=lstm_look_back, epochs=lstm_epochs)
            results[model_name] = preds

        # 📉 Forecast Plot
        st.subheader("📈 Forecast Comparison")
        fig = go.Figure()
        for model_name, preds in results.items():
            fig.add_trace(go.Scatter(x=future_dates, y=preds, mode='lines', name=model_name))
        fig.update_layout(xaxis_title="Date", yaxis_title="Forecasted Price", height=450)
        st.plotly_chart(fig, use_container_width=True)

        # 📏 Evaluation
        st.subheader("📏 Evaluation (on last N days)")
        if len(df) > forecast_horizon:
            actual = df["Close"].iloc[-forecast_horizon:].values
            eval_table = []
            for model_name, preds in results.items():
                mae = mean_absolute_error(actual, preds)
                rmse = np.sqrt(mean_squared_error(actual, preds))
                eval_table.append({"Model": model_name, "MAE": round(mae, 2), "RMSE": round(rmse, 2)})
            st.dataframe(pd.DataFrame(eval_table).set_index("Model"))
        else:
            st.info("Not enough data for evaluation window.")

except Exception as e:
    st.error("⚠️ An error occurred:")
    st.exception(e)
