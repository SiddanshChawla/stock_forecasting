from tensorflow.keras.models import load_model
import streamlit as st
from newsapi import NewsApiClient
from datetime import date
from datetime import timedelta
from preprocess import preprocess, get_result, tokenize_pad
import numpy as np
import pandas as pd
from news import extract_phrases
import nltk
nltk.download('stopwords')
nltk.download('punkt')

def predict_sentiment(description):
    model = load_model('gru_sentiment_model.h5')

    polarity = []
    headlines1 = []
    sentiments1 = []
    today = date.today()
    yesterday = today - timedelta(days = 1)
    
    keywords = extract_phrases(description)
    important_keywords = keywords[:2]

    newsapi = NewsApiClient(api_key='0eab831ebf9c402ba6f4f2312b355ad6')
    for keyword in important_keywords:
        keyword = keyword
        all_articles = newsapi.get_everything(q=keyword,
                                              from_param=yesterday,
                                              to=today,
                                              language='en')

        number_of_results = all_articles['totalResults']
        i = 0

        while (i<number_of_results):
            headline = all_articles['articles'][i]['title']
            with st.container():
                headline_preprocess = preprocess(headline)
                input = tokenize_pad(headline_preprocess)
                prediction = model.predict(input)
                centered_pred = prediction - 0.5
                polarity.append(centered_pred)
                if prediction > 0.5:
                    predict = "Positive."
                else:
                    predict = "Negative."
                headlines1.append(headline)
                sentiments1.append(predict)
            i = i+1
    results_df = pd.DataFrame({'Headline': headlines1, 'Sentiment': sentiments1})
    mean = np.mean(polarity) + 0.5
    if mean > 0.5:
        answer = "Overall news sentiment polarity is POSITIVE"
    else:
        answer = "Overall news sentiment polarity is NEGATIVE"
    
    return answer, results_df
        

