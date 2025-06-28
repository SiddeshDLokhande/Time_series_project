import plotly.express as px
import numpy as np


# plot plotly chart


def plot(df):
    fig = px.line()
    for i in df.columns[1:]:
        fig.add_scatter(x=df["Date"], y=df[i], name=i)
    fig.update_layout(width=450, margin=dict(l=20, r=20, t=50, b=20), legend=dict(
        orientation='h', yanchor='bottom', y=1.02, xanchor="right", x=1))
    return fig


# normalization function

def normalize(df1):
    df=df1.copy()
    for i in df.columns[1:]:
        # df[i] = df[i]/df[i][0]
        df[i] = df[i] / df[i].iloc[0]
    return df


# daily return

def daily_return(df):
    df_daily_return=df.copy()
    for i in df.columns[1:]:
        for j in range(1, len(df)):
        #     df_daily_return[i][j] = (df[i][j]-df[i][j-1])*100
        # df_daily_return[i][0] = 0
         df_daily_return[i] = df[i].pct_change()
    df_daily_return.fillna(0,inplace=True)
    return df_daily_return

#beta calculation

# def calculate_beta(stocks_daily_return,stock):
#     rm=stocks_daily_return['sp500'].mean()*252

#     b,a= np.polyfit(stocks_daily_return['sp500'],stocks_daily_return[stock],1)

#     return b,a

def calculate_beta(stocks_daily_return, stock):
    df_clean = stocks_daily_return[['sp500', stock]].dropna()
    b, a = np.polyfit(df_clean['sp500'], df_clean[stock], 1)
    return b, a
