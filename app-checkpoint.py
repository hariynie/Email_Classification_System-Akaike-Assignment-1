from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from gensim import models
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Reshape, Flatten, concatenate, Input, Conv1D,LSTM, GlobalMaxPooling1D, Embedding,Bidirectional

from keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import collections
import re
import string
import matplotlib.pyplot as plt
import pickle
import nltk
from nltk import word_tokenize, WordNetLemmatizer
nltk.download('punkt_tab')
import emoji
def convert_emojis_to_text(text):
    return emoji.demojize(text)
from langdetect import detect
from googletrans import Translator
from flask import Flask, render_template, request
import re
translator = Translator()

def detect_and_translate(text):
    try:
        lang = detect(text)
        if lang == "ta":  # Tamil language code
            translated_text = translator.translate(text, src="ta", dest="en").text
            return translated_text
        else:
            return text
    except Exception as e:
        return text
      
def remove_punct(text):
    text_nopunct = ''
    text_nopunct = re.sub('['+string.punctuation+']', '', text)
    return text_nopunct
def lower_token(tokens): 
    return [w.lower() for w in tokens]
def remove_stop_words(tokens): 
    return [word for word in tokens if word not in stoplist]
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)
MAX_SEQUENCE_LENGTH = 300
EMBEDDING_DIM = 300
label_names = ['Change', 'Incident', 'Problem', 'Request']
from nltk.corpus import stopwords
stoplist = stopwords.words('english')
def predictsentiment(inputtext):
    cleantext = [remove_punct(inputtext)]
    
    tokens = [word_tokenize(sen) for sen in cleantext] 
    lower_tokens = [lower_token(token) for token in tokens] 
    from nltk.corpus import stopwords
    stoplist = stopwords.words('english')
    filtered_words = [remove_stop_words(sen) for sen in lower_tokens]
    result = [' '.join(sen) for sen in filtered_words] 
    test_sequence = tokenizer.texts_to_sequences(result)
    test_data = pad_sequences(test_sequence, maxlen=MAX_SEQUENCE_LENGTH)
    Y_pred = modelsinfo.predict(test_data)
    Y_pred_classes = np.argmax(Y_pred,axis = 1)

    return label_names[Y_pred_classes[0]]
    
    
   
app=Flask(__name__)
import tensorflow as tf
from tensorflow.keras.layers import (
    Embedding, Input, LSTM, Bidirectional, Conv1D, GlobalMaxPooling1D, 
    Dropout, Dense, concatenate
)
from tensorflow.keras.models import Model

def ConvBiLSTMNet(embeddings, max_sequence_length, num_words, embedding_dim, labels_index):
    """
    Combines CNN and BiLSTM layers for text classification.
    
    Args:
        embeddings: Pre-trained embedding weights (e.g., GloVe).
        max_sequence_length: Maximum length of input sequences.
        num_words: Vocabulary size.
        embedding_dim: Dimension of the embedding vectors.
        labels_index: Number of output labels (for classification).

    Returns:
        Compiled Keras model.
    """
    # Embedding layer
    embedding_layer = Embedding(
        input_dim=num_words,
        output_dim=embedding_dim,
        weights=[embeddings],
        input_length=max_sequence_length,
        trainable=False
    )
    
    # Input layer
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    
    # BiLSTM layer with global max pooling
    lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(embedded_sequences)
    lstm_pooled = GlobalMaxPooling1D()(lstm_layer)
    
    # CNN layers with different filter sizes
    convs = []
    filter_sizes = [2, 3, 4, 5, 6]
    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=200, kernel_size=filter_size, activation='relu')(embedded_sequences)
        l_pool = GlobalMaxPooling1D()(l_conv)
        convs.append(l_pool)
    
    # Merge CNN and BiLSTM outputs
    cnn_merged = concatenate(convs, axis=1)
    combined = concatenate([cnn_merged, lstm_pooled], axis=1)
    
    # Fully connected layers with dropout
    x = Dropout(0.1)(combined)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    preds = Dense(labels_index, activation='softmax')(x)
    
    # Build and compile model
    model = Model(sequence_input, preds)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['acc']
    )
    model.summary()
    return model
import pickle
with open('train_embedding_weights.pkl', 'rb') as file:
    train_embedding_weights = pickle.load(file)
with open('train_word_index.pkl', 'rb') as file:
    train_word_index = pickle.load(file)

label_names = ['Change', 'Incident', 'Problem', 'Request']
MAX_SEQUENCE_LENGTH=300
EMBEDDING_DIM=300
modelsinfo = ConvBiLSTMNet(train_embedding_weights, MAX_SEQUENCE_LENGTH, len(train_word_index)+1, EMBEDDING_DIM, 
                len(list(label_names)))
modelsinfo.load_weights('finalmodel.h5')
@app.route('/')
def home():
    return render_template('home.html')
   
@app.route('/predict', methods=['POST'])
def predict():

   

    if request.method=='POST':
        comment=request.form['comment']
        data=comment
        
        
        my_prediction=predictsentiment(data)
            
    return render_template('result.html', prediction=my_prediction)

if __name__== '__main__':
    app.run(debug=True)
