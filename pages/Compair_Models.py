import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras


st.set_page_config(page_title="Model Comparison", page_icon="🔎", layout="wide")
st.title("🔎 Time Series Model Comparison & Tuning")

# --- User input ---
col1, col2 = st.columns(2)
with col1:
    stock_symbol = st.text_input("Enter Stock Symbol", "AAPL")
    models = st.multiselect(
        "Select Models", ["ARIMA", "SARIMA", "Prophet", "LSTM"])
with col2:
    years = st.number_input("Years of historical data", 1, 10, step=1)
    forecast_horizon = st.number_input(
        "Days to Forecast Ahead", 1, 60, value=30)


@st.cache_data
def load_data(symbol, years):
    data = yf.download(symbol, period=f"{years}y")
    df = data[["Close"]].reset_index()
    df.columns = ["Date", "Close"]
    return df


def arima_forecast(series, order, steps):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast


def sarima_forecast(series, order, seasonal_order, steps):
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast


def prophet_forecast(data, periods):
    df = data.rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(periods)['yhat'].values


def lstm_forecast(data, forecast_horizon, look_back=30, epochs=10):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

    if len(scaled) <= look_back:
        raise ValueError("Not enough data points for the selected look_back window.")

    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i-look_back:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    model = keras.Sequential([
        keras.layers.LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
        keras.layers.LSTM(50),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, batch_size=16, verbose=0)

    # Recursive prediction
    last_seq = scaled[-look_back:]
    preds = []
    for _ in range(forecast_horizon):
        inp = last_seq.reshape((1, look_back, 1))
        pred = model.predict(inp, verbose=0)[0][0]
        preds.append(pred)
        last_seq = np.append(last_seq[1:], [[pred]], axis=0)

    # Inverse transform to original scale
    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return preds


try:
    if stock_symbol and models:
        data = load_data(stock_symbol, years)
        st.subheader(f"📊 {stock_symbol} Historical Closing Price")
        st.line_chart(data.set_index("Date"))

        last_date = data["Date"].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=forecast_horizon)
        results = {}

        if "ARIMA" in models:
            arima_pred = arima_forecast(
                data["Close"], (1, 1, 1), forecast_horizon)
            results["ARIMA"] = arima_pred

        if "SARIMA" in models:
            sarima_pred = sarima_forecast(
                data["Close"], (1, 1, 1), (1, 1, 1, 12), forecast_horizon)
            results["SARIMA"] = sarima_pred

        if "Prophet" in models:
            prophet_pred = prophet_forecast(data, forecast_horizon)
            results["Prophet"] = prophet_pred
            
        if "LSTM" in models:
           lstm_pred = lstm_forecast(data, forecast_horizon, look_back=30, epochs=10)
           results["LSTM"] = lstm_pred
        # Combine results for plotting
        forecast_df = pd.DataFrame({"Date": future_dates})
        fig = go.Figure()
        st.subheader("🔮 Forecast Comparison")
        for model_name, preds in results.items():
            forecast_df[model_name] = preds
            fig.add_trace(go.Scatter(
                x=forecast_df["Date"],
                y=preds,
                mode='lines',
                name=model_name
            ))
            fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Forecasted Price",
            legend_title="Model",
            width=900,
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)

       
            
        

        # Model evaluation (simple backtest on last forecast_horizon days)
        st.subheader("📏 Model Evaluation (last N days)")
        
        if len(data) > forecast_horizon:
            actual = data["Close"].iloc[-forecast_horizon:].values
            print("actual",actual)
            print("Predicted",preds)
            eval_table = []
            for model_name, preds in results.items():
                mae = mean_absolute_error(actual, preds)
                rmse = np.sqrt(mean_squared_error(actual, preds))
                eval_table.append({"Model": model_name, "MAE": mae, "RMSE": rmse})
            st.dataframe(pd.DataFrame(eval_table),hide_index=True)
        else:
            st.info("Not enough data for evaluation window.")

except Exception as e:
    st.error("⚠️ An error occurred:")
    st.exception(e)