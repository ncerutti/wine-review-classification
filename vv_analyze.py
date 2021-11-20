# -*- coding: utf-8 -*-
"""
@author: Nico
"""

import pandas as pd
import numpy as np
import os
import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import text
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, LSTM, Dropout, Bidirectional, Conv1D, MaxPooling1D
from keras.utils import to_categorical
from langdetect import detect
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import tensorflow

df = pd.read_csv("D:/Nextcloud/applications/20211120_UniPotsdam/scraping_ml/dset.csv")
listid = []
for item, row in df.iterrows():
    if isinstance(row['text'], str):
        if (len(row['text']) > 20):
            lang=detect(row['text'])
            if lang != 'en':
                listid.append(row['id'])
        else:
            listid.append(row['id'])
    else:
        listid.append(row['id'])


df = df[~df.id.isin(listid)]

df["split"] = df.apply(lambda x: "train" if random.randrange(0,100) > 10 else "valid", axis=1)
df["split"].value_counts()

df_train = df[df["split"] == "train"]
df_val = df[df["split"] == "valid"]

tokenizer=Tokenizer(oov_token="'oov'")
tokenizer.fit_on_texts(df_train['text'])

maxlen = 200
train_X = pad_sequences(tokenizer.texts_to_sequences(df_train['text']), maxlen=maxlen)
val_X = pad_sequences(tokenizer.texts_to_sequences(df_val['text']), maxlen=maxlen)

train_Y = df_train["label"]
val_Y = df_val["label"]
train_Y_cat = to_categorical(df_train["label"]-1, num_classes=5)
val_Y_cat = to_categorical(df_val["label"]-1, num_classes=5)

glove_dir="D:/Anaconda3/glove/"
embedding_index = {}
f = open(os.path.join(glove_dir,'glove.6B.100d.txt'),encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:],dtype='float32')
    embedding_index[word] = coefs
f.close()
print('Found %s word vectors ' % len(embedding_index))


max_words = len(tokenizer.word_index) + 1
embedding_dim = 100
embedding_matrix = np.zeros((max_words,embedding_dim))

for word, idx in tokenizer.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[idx]=embedding_vector
        
model=Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen, weights=[embedding_matrix], trainable=False))
model.add(Bidirectional(LSTM(16)))
model.add(Dense(16, activation="relu"))
model.add(Dense(5, activation="softmax"))
model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

model.fit(train_X, train_Y_cat, epochs=20, batch_size=256, validation_data=(val_X, val_Y_cat))

pred = model.predict(val_X)

accuracy_score(val_Y, [np.argmax(p)+1 for p in pred])