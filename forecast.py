import streamlit as st
from model import forecast_prices
from predict_next_day import predict_next_day
import datetime
from news_sent import predict_sentiment

def main():
    n_steps = 2

    # Define company options
    COMPANY_OPTIONS = ('RELIANCE.NS', 'TCS.NS', 'TATAMOTORS.NS', 'ADANIENT.NS')

    st.header("Select Company and Model Configuration : ")
    with st.form(key='form1'):
        col1, col2 = st.columns(2)
        with col1:
            dataset = st.selectbox('Select a company', COMPANY_OPTIONS)
        with col2:
            activation = st.selectbox('Select an activation function', ('relu', 'sigmoid', 'linear', 'softmax'))
        with col1:
            epochs = st.slider('Select number of epochs for training', 0, 100, 50, step=1)
        st.write("")
        with col2:
            n_steps1 = st.slider('Select lookback to create dataset', 1, 100, 2)
        with col1:
            lstm_units = st.selectbox('Select number of units for LSTM', {32, 64, 128})
        with col2:
            train_test_split = st.selectbox('Select Train/Test Split', {0.7, 0.8, 0.9})
        submitted1 = st.form_submit_button('Submit')
        if submitted1:
            my_bar = st.progress(20, text='Model training in progress')
            model, sc, sc_close = forecast_prices(dataset, n_steps, lstm_units, epochs, train_test_split, activation)
            my_bar.progress(60 + 1, text='Model trained successfully!')
            answer = predict_next_day(model, sc, sc_close, n_steps, dataset)
            my_bar.progress(90 + 1, text='Prediction in progress!')
            st.write("Today's price prediction is : ", answer)
            my_bar.progress(100 + 1, text='Prediction complete!')
    
    st.header("News Sentiment analyser: ")
    with st.form(key='form2'):
        query = st.text_input("Enter company name or related info for news sentiment generation")
        submitted2 = st.form_submit_button('Submit')
        if submitted2:
            news_sentiment, results_df = predict_sentiment(query)
            st.write(news_sentiment)
            st.table(results_df)
        
if __name__ == '__main__':
    main()



