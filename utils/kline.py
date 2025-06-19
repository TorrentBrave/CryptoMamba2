# Import libraries
import random
import os
import numpy as np 
import pandas as pd 
import requests
import pandas_datareader.data as web

# Date
from datetime import date, timedelta, datetime

# EDA
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
import ydata_profiling as yp

# Time Series - EDA and Modelling
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA


import warnings
warnings.filterwarnings("ignore")


# import plotly.graph_objects as go

def Kline(df, title="Cryptocurrency"):
    """绘制加密货币蜡烛图(K线图)

    参数:
        df (pd.DataFrame): 包含 ['Open', 'High', 'Low', 'Close'] 列的数据，索引为时间。
        title (str): 图表标题。

    返回:
        plotly.graph_objects.Figure: 蜡烛图对象。
    """
    fig = go.Figure(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    ))

    fig.update_layout(
        title={'text': f'{title} Candlestick Chart', 'x': 0.5, 'y': 0.85, 'xanchor': 'center', 'yanchor': 'top'},
        xaxis_title='Time',
        yaxis_title='Price in USD',
        xaxis_rangeslider_visible=True,
        yaxis_ticksuffix='$',
        template="plotly_dark",  # 可选美化
        height=600
    )
    
    return fig




if __name__ == "__main__":
    
    cryptocurrency = 'BTC'
    # date_start = datetime(2020, 4, 1)
    date_start = datetime(2015, 2, 9)
    date_end = datetime(2025, 6, 16)
    # date_end = dt.datetime.now()

    df = pd.read_csv('../data/Daily.csv', index_col='Date')  # 将Date列设为索引

    fig = Kline(df=df, title='BTC')  # 调用你的Kline函数生成图表
    fig.show()  # 显示图表（会弹出浏览器或Jupyter内显示）

    # fig.write_html("btc_kline.html")  # 保存为HTML文件
    # print("图表已保存为 btc_kline.html")  # 提示文件路径
