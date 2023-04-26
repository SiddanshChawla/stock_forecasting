import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
from calculate_rsi import calculate_rsi
from calculate_sma import calculate_sma


def data(ticker, start_date ,end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return pd.DataFrame(data)

def download_data(ticker, start_date, end_date):
    #download and import data
    df = data(ticker, start_date, end_date)
    df = df.dropna()
    close_prices = df['Close'].values

    #calculate rsi
    rsi_values = calculate_rsi(close_prices)
    df['RSI'] = rsi_values

    #calculate sma
    sma_values = calculate_sma(close_prices)
    df['SMA'] = sma_values

    df = df[['Open', 'High', 'Low', 'RSI', 'SMA', 'Close']]
    #download bse and nifty data
    bse = data('^BSESN', start_date, end_date)
    nifty = data('^NSEI', start_date, end_date)

    #select required data from bse and nifty
    bse['BSE'] = bse['Close']
    bse = bse['BSE']
    nifty['Nifty'] = nifty['Close']
    nifty = nifty['Nifty']

    #merge data and retrieve final data
    merged_df = pd.merge(df, bse, on='Date')
    final_df = pd.merge(merged_df, nifty, on='Date')

    # insert Close column at the end
    col_to_move = 'Close'
    col = df[col_to_move]
    final_df = final_df.drop(col_to_move, axis=1)
    final_df[col_to_move] = col
    return final_df
