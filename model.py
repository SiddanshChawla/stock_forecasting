import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.optimizers import Adam
import yfinance as yf
from datetime import date
from datetime import timedelta

from download_data import download_data
from create_dataset import create_dataset

def forecast_prices(dataset, n_steps, lstm_units, epochs, train_test_split, activation):
    today = date.today()
    yesterday = today - timedelta(days = 1)

  #read data
    df = download_data(dataset,'2015-01-01', yesterday)
    df['Close'] = df['Close'].shift(-1)
    df = df.dropna()

  #define features and target var
    features = df.drop('Close', axis=1)
    target = df['Close']
    
    
  #normalize data using standard scaler
    sc = StandardScaler()
    x_ft = sc.fit_transform(features.values)
    x_ft = pd.DataFrame(columns=features.columns, data = x_ft, index = features.index)
    
    sc_close = StandardScaler()
    y_ft = sc_close.fit_transform(target.values.reshape(-1,1))


  #split data in train and test
    train_split = train_test_split
    split_idx = int(np.ceil(len(x_ft)*train_split))
    date_index = x_ft.index

    X_train, X_test = x_ft[:split_idx], x_ft[split_idx:]
    y_train, y_test = y_ft[:split_idx], y_ft[split_idx:]
    X_train_date, X_test_date = date_index[:split_idx], date_index[split_idx:]
    
    
  #create sequences of input data and corresponding output values
    X_train, y_train = create_dataset(X_train.values, n_steps)
    X_test, y_test = create_dataset(X_test.values, n_steps)
    
  #make a model
    model = Sequential()
    model.add(LSTM(units = lstm_units, input_shape=(X_train.shape[1],X_train.shape[2]), activation=activation, return_sequences=True))
    model.add(LSTM(units = lstm_units, activation=activation, return_sequences=True))
    model.add(LSTM(units = lstm_units, activation=activation))
    model.add(Dense(units = 1))

  #compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

  #run the model
    history = model.fit(X_train, y_train, epochs=epochs, verbose = 1, shuffle=False)
  #predict values
    y_predict = model.predict(X_test)
  
  # unscale the predicted and actual values
    y_predict_unscaled = sc_close.inverse_transform(y_predict)
    y_test_unscaled = sc_close.inverse_transform(y_test.reshape(-1, 1)).flatten()

  #test scores
    rmse = mean_squared_error(y_test, y_predict, squared = False)
    mape = mean_absolute_percentage_error(y_test, y_predict)
    st.write("RMSE: " , rmse)
    st.write("MAPE: " , mape)

  #plots showing predicted and actual value
    
    fig, ax = plt.subplots()
    ax.plot(y_test_unscaled, label='actual')
    ax.plot(y_predict_unscaled, label='predicted')
    ax.legend()
    st.pyplot(fig)
    
    return model, sc, sc_close
