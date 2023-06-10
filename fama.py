import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
import pandas_datareader.data as web
from sklearn.linear_model import LinearRegression


def get_single_factor_loadings(tickers, start, end):
    # Read in price data from Yahoo Finance and Calculate Daily Returns
    price_data = yf.download(tickers, start=start, end=end)['Adj Close']
    stock_returns = price_data.pct_change()*100
    stock_returns = stock_returns.dropna()

    # Obtain Market risk premium from 3 Factor Fama French Data
    ff_data = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench')[0]
    ff_sub = ff_data[stock_returns.index[0]:stock_returns.index[-1]]
    ff_sub = ff_sub.drop(['SMB', 'HML'], axis=1)

    # Regress Stock Excess returns against Market Excess returns for Beta
    x = np.array(ff_sub['Mkt-RF']).reshape(-1, 1)
    y = stock_returns.sub(ff_sub['RF'], axis=0)
    loadings = pd.DataFrame()
    for stock in y.columns:
        reg = LinearRegression()
        reg.fit(x, y[stock])
        loadings[stock] = reg.coef_

    loadings.index = ['Beta']
    return loadings


def get_three_factor_loadings(tickers, start, end):
    # Read in price data from Yahoo Finance and Calculate Daily Returns
    price_data = yf.download(tickers, start=start, end=end)['Adj Close']
    stock_returns = price_data.pct_change()*100
    stock_returns = stock_returns.dropna()

    # Obtain Market risk premium from 3 Factor Fama French Data
    ff_data = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench')[0]
    ff_sub = ff_data[stock_returns.index[0]:stock_returns.index[-1]]

    # Regress Stock Excess returns against Market Excess returns for Beta
    x = ff_sub[['Mkt-RF', 'SMB', 'HML']].to_numpy()
    y = stock_returns.sub(ff_sub['RF'], axis=0)
    loadings = pd.DataFrame()
    for stock in y.columns:
        reg = LinearRegression()
        reg.fit(x, y[stock])
        loadings[stock] = reg.coef_

    loadings.index = ['Mkt-RF', 'SMB', 'HML']
    return loadings


def get_five_factor_loadings(tickers, start, end):
    # Read in price data from Yahoo Finance and Calculate Daily Returns
    price_data = yf.download(tickers, start=start, end=end)['Adj Close']
    stock_returns = price_data.pct_change()*100
    stock_returns = stock_returns.dropna()

    # Obtain Market risk premium from 5 Factor Fama French Data
    ff_data = web.DataReader('F-F_Research_Data_5_Factors_2x3_daily', 'famafrench')[0]
    ff_sub = ff_data[stock_returns.index[0]:stock_returns.index[-1]]

    # Regress Stock Excess returns against Market Excess returns for Beta
    x = ff_sub[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']].to_numpy()
    y = stock_returns.sub(ff_sub['RF'], axis=0)
    loadings = pd.DataFrame()
    for stock in y.columns:
        reg = LinearRegression()
        reg.fit(x, y[stock])
        loadings[stock] = reg.coef_

    loadings.index = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    return loadings


if __name__ == '__main__':
    tickers = ['AAPL', 'MS']
    start = datetime(2020, 1, 1)
    end = datetime(2022, 1, 1)

    #test = get_single_factor_loadings(tickers, start, end)
    #print(test)
    #test = get_three_factor_loadings(tickers, start, end)
    #print(test)
    test = get_five_factor_loadings(tickers, start, end)
    print(test)
