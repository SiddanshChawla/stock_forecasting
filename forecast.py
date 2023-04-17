import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import yfinance as yf
import plotly.express as px
import plotly.graph_objs as go

def main():

    # Define company options
    COMPANY_OPTIONS = ('RELIANCE.NS', 'TCS.NS', 'TATAMOTORS.NS', 'ADANIENT.NS')
    
    # Create Streamlit form
    st.header("Select Company and Model Configuration")
    dataset = st.selectbox('Select a company', COMPANY_OPTIONS)
    n_steps = st.slider('Select steps to create dataset', 2, 10, 1)
    lstm_units = st.slider('Select number of units for LSTM', 32, 128, 1)
    epochs = st.slider('Select number of epochs for training', 0, 100, step=1)
    train_test_split = st.slider('Select Train/Test Split', 0.0, 1.0, step=0.1)
    activation = st.selectbox('Select an activation function', ('linear', 'sigmoid', 'relu', 'softmax'))
    optimizer = st.selectbox('Select an optimizer', ('adam', 'sgd', 'adagrad'))

    submitted = st.button('Submit')


    def data(ticker):
        data = yf.download(ticker, start='2015-01-01', end='2023-01-01')
        return pd.DataFrame(data)

    def download_data(ticker):
        #download and import data
        df = data(ticker)
        progress_bar.progress(20)
        df = df.dropna()
        close_prices = df['Close'].values

        #calculate rsi
        rsi_values = calculate_rsi(close_prices)
        df['RSI'] = rsi_values

        #calculate sma
        sma_values = calculate_sma(close_prices)
        df['SMA'] = sma_values

        df = df[['Open', 'High', 'Low', 'RSI', 'SMA', 'Close']]
        progress_bar.progress(45)
        #download bse and nifty data
        bse = data('^BSESN')
        nifty = data('^NSEI')

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
        progress_bar.progress(50)
        return final_df


    def create_dataset(data, n_steps):
        X, y = [], []
        if len(data.shape) == 1:
            # Handle 1D input array
            for i in range(len(data) - n_steps + 1):
                X.append(data[i:i + n_steps])
                y.append(data[i + n_steps - 1])
        else:
            # Handle 2D input array
            for i in range(len(data) - n_steps + 1):
                X.append(data[i:i + n_steps, :-1])
                y.append(data[i + n_steps - 1, -1])
        progress_bar.progress(55)
        return np.array(X), np.array(y)


    def forecast_prices(dataset, n_steps, lstm_units, epochs, train_test_split, activation, optimizer):
      #read data
        df = download_data(dataset)
        df = df.dropna()
        st.write(df.shape)

      #define features and target var
        features = df.drop('Close', axis=1)
        target = df['Close']
        
        
      #normalize data using standard scaler
        sc = StandardScaler()
        x_ft = sc.fit_transform(features.values)
        x_ft = pd.DataFrame(columns=features.columns, data = x_ft, index = features.index)
        st.write(x_ft.shape)
        
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
        progress_bar.progress(70)
        
      #make a model
        model = Sequential()
        model.add(LSTM(units = lstm_units, input_shape=(X_train.shape[1],X_train.shape[2]), activation=activation, return_sequences=True))
        model.add(LSTM(units = lstm_units, activation=activation))
        model.add(Dense(units = 1))

      #compile the model
        model.compile(loss='mean_squared_error', optimizer=optimizer)

      #run the model
        history = model.fit(X_train, y_train, epochs=epochs, verbose = 1, shuffle=False)
        
      #predict values
        y_predict = model.predict(X_test)
      
      # unscale the predicted and actual values
        y_predict_unscaled = sc_close.inverse_transform(y_predict)



    # test scores

      #test scores
        rmse = mean_squared_error(y_test, y_predict, squared = False)
        mape = mean_absolute_percentage_error(y_test, y_predict)
        progress_bar.progress(100)
        st.write("RMSE: " , rmse)
        st.write("MAPE: " , mape)
        
        st.write(y_test.shape)
        st.write(y_predict_unscaled.shape)

      #plots showing predicted and actual value
        
        fig, ax = plt.subplots()
        ax.plot(y_test, label='actual')
        ax.plot(y_predict_unscaled, label='predicted')
        ax.legend()
        st.pyplot(fig)
        
    
    def calculate_rsi(prices, n=14):
        deltas = np.diff(prices)
        seed = deltas[:n+1]
        up = seed[seed >= 0].sum()/n
        down = -seed[seed < 0].sum()/n
        rs = up/down
        rsi = np.zeros_like(prices)
        rsi[:n] = 100. - 100./(1.+rs)
        for i in range(n, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            up = (up*(n-1) + upval)/n
            down = (down*(n-1) + downval)/n
            if down == 0:
                rs = 100
            else:
                rs = up/down
            if np.isinf(rs):
                rs = 100
            rsi[i] = 100. - 100./(1.+rs)
        progress_bar.progress(30)
        return rsi
        
    def calculate_sma(prices, n=10):
        sma = np.zeros_like(prices)
        for i in range(n, len(prices)):
            sma[i] = prices[i-n:i].mean()
        progress_bar.progress(35)
        return sma
        
    if submitted:
        with st.spinner('Running Machine Learning model...'):
            progress_bar = st.progress(0)
            forecast_prices(dataset, n_steps, lstm_units, epochs, train_test_split, activation, optimizer)
            progress_bar.empty()
            st.success('Model run successfully!')
    
if __name__ == '__main__':
    main()
