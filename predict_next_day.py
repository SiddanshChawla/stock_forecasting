from download_data import download_data
from create_dataset import create_dataset
import numpy as np
import pandas as pd
import streamlit as st


def predict_next_day(model, sc, sc_close, start_date, end_date, n_steps, ticker):
    new_data = download_data(ticker, start_date, end_date)
    new_data = new_data.dropna()
    
    features = new_data.drop('Close', axis=1)
    
    x_ft = sc.transform(features.values)

    X_new, _ = create_dataset(x_ft, n_steps)
    y_predict = model.predict(X_new)
    
    y_predict_unscaled = sc_close.inverse_transform(y_predict)
    
    return y_predict_unscaled
    
