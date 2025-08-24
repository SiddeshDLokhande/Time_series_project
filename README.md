# 📊 Stock Forecasting & Analysis Dashboard



An interactive web application built with Streamlit for comprehensive stock market analysis, including time series forecasting, model comparison, and portfolio analysis using the Capital Asset Pricing Model (CAPM).

## ✨ Features

This dashboard provides a suite of tools for financial analysis:

-   **Time Series Forecasting**: Predict future stock prices using various statistical and machine learning models.
    -   ARIMA (AutoRegressive Integrated Moving Average)
    -   SARIMA (Seasonal ARIMA)
    -   Prophet (by Facebook)
    -   LSTM (Long Short-Term Memory) Neural Networks
-   **Model Comparison**: Evaluate and compare the performance of different forecasting models on a given stock to identify the most accurate one.
-   **Model Tuning**: Interactively tune hyperparameters for the forecasting models and visualize their impact on performance.
-   **CAPM Analysis**: Calculate and visualize the Capital Asset Pricing Model for a stock or a portfolio to understand its risk and expected return relative to the market.

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

-   Python 3.8+
-   pip

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    *(Note: A `requirements.txt` file should be present in the repository. If not, you will need to create one based on the project's imports.)*
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    The main entry point for the app is `HomePage.py`.
    ```bash
    streamlit run HomePage.py
    ```

    Your web browser should open with the application running at `http://localhost:8501`.

## 🛠️ Usage

Once the application is running:

1.  The **Home Page** provides an overview of the dashboard's capabilities.
2.  Use the **sidebar navigation** to select an analysis module:
    -   📈 **Time Series Models**: Choose a stock ticker, a date range, and a forecasting model to generate predictions.
    -   🔎 **Model Comparison**: Select a stock and see a side-by-side comparison of forecasts from multiple models.
    -   ⚙️ **Model Tuning**: Experiment with model parameters to see how they affect forecast accuracy.
    -   📉 **CAPM**: Input stock tickers and a market index to perform a CAPM analysis.

## 💻 Technologies Used

-   **Frontend**: Streamlit
-   **Data Handling**: Pandas
-   **Financial Data**: yfinance
-   **Forecasting Models**:
    -   Statsmodels (for ARIMA, SARIMA)
    -   Prophet
    -   TensorFlow / Keras (for LSTM)
-   **Plotting**: Plotly, Matplotlib

---

Enjoy your analysis!
