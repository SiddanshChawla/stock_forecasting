import streamlit as st
from model import forecast_prices
from predict_next_day import predict_next_day
import datetime

def main():

    # Define company options
    COMPANY_OPTIONS = ('RELIANCE.NS', 'TCS.NS', 'TATAMOTORS.NS', 'ADANIENT.NS')
    
    # Create Streamlit form
    st.header("Select Company and Model Configuration")
    dataset = st.selectbox('Select a company', COMPANY_OPTIONS)
    n_steps = st.slider('Select steps to create dataset', 1, 100, 10)
    lstm_units = st.slider('Select number of units for LSTM', 32, 128, 1)
    epochs = st.slider('Select number of epochs for training', 0, 100, step=1)
    train_test_split = st.slider('Select Train/Test Split', 0.0, 1.0, step=0.1)
    activation = st.selectbox('Select an activation function', ('linear', 'sigmoid', 'relu', 'softmax'))
    
    start_date = st.date_input("Enter start date of prediction")
    end_date = st.date_input("Enter end date of prediction")

    submitted = st.button('Submit')
    
 
    if submitted:
        model, sc, sc_close = forecast_prices(dataset, n_steps, lstm_units, epochs, train_test_split, activation)
        st.success('Model run successfully!')
#        new_date = date.today()
        st.write('Predictions for given date range is/are: ')
        answer = predict_next_day(model, sc, sc_close, start_date, end_date, n_steps, dataset)
        
        st.write(answer)
if __name__ == '__main__':
    main()



