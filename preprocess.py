import spacy
nlp=spacy.load('en_core_web_sm')
import string
from textblob import TextBlob
import numpy as np
import pandas as pd
import re
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

    
def preprocess(input_text):
    input_text = input_text.lower()

    output_list = [word for word in input_text.split() if word.isalpha()]
    output_list_1 = ' '.join(output_list)

    list_1 = [str(TextBlob(word).correct()) for word in output_list_1.split()]
    output_text= ' '.join(list_1)

    doc = nlp(output_text)
    lemmas=[token.lemma_ for token in doc]
    output_text_1=' '.join(lemmas)

    stopwords=spacy.lang.en.stop_words.STOP_WORDS
    output_list_2=[word for word in output_text_1.split() if word not in stopwords and not(word=='-PRON-') ]
    
    return ' '.join(output_list_2)
    
def tokenize_pad(text):
    # Tokenize the text
    num_words=1000
    max_tokens = 18
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts([text])
    text_tokens = tokenizer.texts_to_sequences([text])

    # Pad the sequence
    text_pad = pad_sequences(text_tokens, maxlen=max_tokens)
    return text_pad

def get_result(statement, model):
    result = model.predict([statement])
    pos = np.where(result[1][0] == np.amax(result[1][0]))
    pos = int(pos[0])
    sentiment_dict = {0:'positive',1:'negative',2:'neutral'}
    print(sentiment_dict[pos])
    
def tokenizer1(text):
    headline_preprocess = preprocess(headline)
    input = tokenize_pad(headline_preprocess)
    return input
