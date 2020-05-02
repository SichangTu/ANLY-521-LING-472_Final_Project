# -*- coding: utf-8 -*-
"""Neural_Network.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/113u2P58rt2SwWDGFNJhq_rnoZ0eoxQl2
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
import argparse

# from google.colab import drive
# drive.mount('/content/drive')

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, Activation, Reshape
from tensorflow.keras.layers import LSTM, Bidirectional 
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from keras import metrics


# set random seed for numpy and tensorflow
np.random.seed(93)
tf.random.set_seed(93)

from util import preprocess_text, shuffle_dataset, split_data

def main(datafile):

    # read_file = df = pd.read_table(datafile, sep='\t',header=None,names=['label','msg'])
    # read_file.head()

    # create train, test sets
    with open(datafile, encoding='utf-8') as f:
        texts = f.read().splitlines()

    labels = []
    corpus = []
    for text in texts:
        label, msg = preprocess_text(text)
        labels.append(label)
        corpus.append(msg)

    train, test = split_data(corpus, labels, 0.2)
    y_train = np.asarray(train[1]).astype('int32').reshape((-1,1))
    y_test = np.asarray(test[1]).astype('int32').reshape((-1,1))
    # Converting training and validation data into sequences

    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(corpus)

    print(y_train.shape)

    # culculate max length

    num_tokens = [len(tokens) for tokens in tokenizer.texts_to_sequences(train[0]) + tokenizer.texts_to_sequences(test[0])]
    num_tokens = np.array(num_tokens)

    # max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
    # max_tokens = int(max_tokens)

    # print(max_tokens)

    # np.sum(num_tokens < max_tokens) / len (num_tokens)
    #
    # max_length = np.max(num_tokens)
    # print(max_length)

    # pad the sequence
    max_tokens = 100

    train_seq = sequence.pad_sequences(tokenizer.texts_to_sequences(train[0]),maxlen=max_tokens,padding='post',truncating='post')
    # train_seq = np.expand_dims(train_seq,-1)
    test_seq = sequence.pad_sequences(tokenizer.texts_to_sequences(test[0]),maxlen=max_tokens,padding='post',truncating='post')
    # test_seq = np.expand_dims(test_seq,-1)
    vocab_size = len(tokenizer.word_counts)

    print(train_seq.shape)
    print(test_seq.shape)

    # define parameters in the model
    epochs = 20
    embedding_dim = 128
    unit_dim = 64
    batch_size = 32

    def bilstm_model():
        model = tf.keras.Sequential()
        model.add(Embedding(input_dim=vocab_size+1,
                            output_dim=embedding_dim,
                            input_length=max_tokens,
                            name='layer_embedding'))
        model.add(Bidirectional(LSTM(units=unit_dim, return_sequences=True)))
        model.add(Dropout(0.3))
        model.add(Dense(vocab_size, activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))
        model.add(Reshape((-1,1)))
        model.summary()

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def cnn_model():
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size+1,
                            output_dim=embedding_dim,
                            input_length=max_tokens,
                            name='layer_embedding'))
        model.add(Dropout(0.2))
        model.add(Conv1D(256,
                         3,
                         padding='valid',
                         activation='relu',
                         strides=1))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(256))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.summary()
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def fit_model(model, x, y):
        model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2)

    print("-----Model: TextCNN ----")
    cnn = cnn_model()
    fit_model(cnn, train_seq, y_train)

    score = cnn.evaluate(test_seq, y_test)
    print("\n=========================\n")
    print(f"The loss and accuracy of TextCNN model is: {score}")

    print("-----Model: TextRNN ----")
    bilstm = bilstm_model()
    fit_model(bilstm, train_seq, y_train)

    score = bilstm.evaluate(test_seq, y_test)
    print("\n=========================\n")
    print(f"The loss and accuracy of TextRNN model is: {score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafile", type=str, default="SMSSpamCollection.txt", help="SMSSpamCollection data")

    args = parser.parse_args()

    main(args.datafile)