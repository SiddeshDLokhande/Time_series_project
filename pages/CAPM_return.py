import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import pandas_datareader as web
import capmFunctions as capmFunctions

st.set_page_config(page_title="CAPM",
                   page_icon="chart_with_upwards_trend", layout="wide")
st.title("Capital Asset Pricing Model")

# Input from user
col1, col2 = st.columns([1, 1])
with col1:
    stock_list = st.multiselect("Choose any 4 Stocks", ("TSLA", "AAPL", "NFLX", "MSFT",
                                "MGM", "AMZN", "NVDA", "GOOGL", "META"), ["TSLA", "NVDA", "META", "MSFT"])
with col2:
    year = st.number_input("Number of years", 1, 30)


# download data

try:
    end = datetime.date.today()
    start = datetime.date(datetime.date.today().year-year,
                        datetime.date.today().month, datetime.date.today().day)

    sp500 = web.DataReader(['sp500'], 'fred', start, end)
    # print(sp500.tail())

    stocks_df = pd.DataFrame()

    for stock in stock_list:
        data = yf.download(stock, period=f'{year}y')
        # print(data.head())
        stocks_df[f'{stock}'] = data['Close']

    # print(stocks_df.head())

    stocks_df.reset_index(inplace=True)
    sp500.reset_index(inplace=True)

    # print(stocks_df.dtypes)
    # stocks_df["Date"]=stocks_df["Date"].astype('datetime64[ns]')
    # print(stocks_df.dtypes)

    stocks_df["Date"] = stocks_df["Date"].apply(lambda x: str(x)[:10])
    stocks_df["Date"] = pd.to_datetime(stocks_df["Date"])
    # print(stocks_df["Date"].dtypes)
    sp500.columns = ["Date", "sp500"]
    stocks_df = pd.merge(stocks_df, sp500, how='inner', on="Date")
    # print(stocks_df)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### Dataframe Head")
        st.dataframe(stocks_df.head(), use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### Dataframe Tail")
        st.dataframe(stocks_df.tail(), use_container_width=True, hide_index=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### Price of all Stocks")
        st.plotly_chart(capmFunctions.plot(stocks_df))
    with col2:
        # print(capmFunctions.normalize(stocks_df))
        st.markdown("### Price of all Stocks after Normalization")
        st.plotly_chart(capmFunctions.plot(capmFunctions.normalize(stocks_df)))


    stocks_daily_return = capmFunctions.daily_return(stocks_df)
    print(stocks_daily_return.head())

    beta = {}
    alpha = {}

    for i in stocks_daily_return.columns:
        if i != 'Date' and i != 'sp500':
            b, a = capmFunctions.calculate_beta(stocks_daily_return, i)

            beta[i] = b
            alpha[i] = a

    print(beta, alpha)

    beta_df = pd.DataFrame(columns=["Stocks", "Beta Value"])

    beta_df["Stocks"] = beta.keys()
    beta_df["Beta Value"] = [str(round(i, 2)) for i in beta.values()]

    with col1:
        st.markdown('### Calculated Beta Value')
        st.dataframe(beta_df, use_container_width=True, hide_index=True)

    # risk free
    rf = 0

    # portfolio return
    rm = stocks_daily_return['sp500'].mean()*252

    return_df = pd.DataFrame()
    return_value = []
    for stock, value in beta.items():
        return_value.append(str(round(rf+(value*(rm-rf)), 2)))


    return_df["Stock"] = stock_list
    return_df["Return Value"] = return_value

    with col2:
        st.markdown("### Calculated Return using CAPM")
        st.dataframe(return_df, use_container_width=True, hide_index=True)
except:
    st.write("Please select valid input")



