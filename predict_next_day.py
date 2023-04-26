from download_data import download_data
from create_dataset import create_dataset
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date
from datetime import timedelta


def predict_next_day(model, sc, sc_close, n_steps, ticker):

    today = date.today()
    new_data = download_data(ticker, '2023-01-01', today)
    new_data = new_data.dropna()
    
    features = new_data.drop('Close', axis=1)
    
    x_ft = sc.transform(features.values)

    X_new, _ = create_dataset(x_ft, n_steps)
    y_predict = model.predict(X_new)
    
    y_predict_unscaled = sc_close.inverse_transform(y_predict)
    predicted_price = np.squeeze(y_predict_unscaled[-1])
    formatted_price = "{:.2f}".format(predicted_price)
    
    return formatted_price
    
