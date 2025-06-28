import streamlit as st

st.set_page_config(page_title="Stock Forecasting Dashboard", page_icon="📊", layout="wide")

st.title("📊 Stock Forecasting & Analysis Dashboard")
st.markdown("""
Welcome to the Stock Forecasting & Analysis Dashboard!

This app provides interactive tools for:
- **Time Series Forecasting** using ARIMA, SARIMA, Prophet, and LSTM models
- **Model Comparison & Tuning** to evaluate and optimize forecasting models
- **CAPM Analysis** for portfolio risk and return estimation

---

**Navigation:**
- Use the sidebar or top menu to access different modules:
    - 📈 **Time Series Models**: Forecast a single stock with your chosen model
    - 🔎 **Model Comparison**: Compare multiple models on the same stock
    - ⚙️ **Model Tuning**: Tune model parameters and see their impact
    - 📉 **CAPM**: Analyze portfolio risk and return

---

**Get Started:**  
Select a page from the sidebar to begin your analysis!
""")