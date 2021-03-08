#!/usr/bin/env python3
"""visualize data"""
from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file


def daily_open(array):
    """return max daily open"""
    return max(array)


def daily_close(array):
    """return last price close"""
    return array[-1]


df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

del df['Weighted_Price']
df = df.rename(columns={"Timestamp": 'Date'})
df['Date'] = pd.to_datetime(df['Date'], unit='s')
df = df.set_index('Date')
df = df.loc['2017-01':]

df['Volume_(BTC)'].fillna(value=0, inplace=True)
df['Volume_(Currency)'].fillna(value=0, inplace=True)
df['Open'].fillna(method='ffill', inplace=True)
df['High'].fillna(method='ffill', inplace=True)
df['Low'].fillna(method='ffill', inplace=True)
df['Close'].fillna(method='ffill', inplace=True)

n_df = pd.DataFrame()
n_df['Open'] = df.Open.resample('D').apply(daily_open)
n_df['High'] = df.High.resample('D').max()
n_df['Low'] = df.Low.resample('D').min()
n_df['Close'] = df.Close.resample('D').apply(daily_close)
n_df['Volume_(BTC)'] = df['Volume_(BTC)'].resample('D').max()
n_df['Volume_(Currency)'] = df['Volume_(Currency)'].resample('min').mean()

n_df.plot()
plt.show()
