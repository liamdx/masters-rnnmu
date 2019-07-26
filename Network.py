import tensorflow as tf
import keras
from keras.callbacks import TensorBoard
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Activation, Input, Dropout, Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, CuDNNGRU
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.losses import categorical_crossentropy
from keras.optimizers import adam
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
import datetime
import json
import numpy as np
from util.rprop import * 

def genNormalizedNetworkData(processedData, processedLabels):
    print("Performing train-test split")
    trainX, testX, trainY, testY = train_test_split(
        np.array(processedData),
        np.array(processedLabels),
        test_size=0.25,
        random_state=42,
    )
    print("...Done")
    return trainX, testX, trainY, testY


def genTokenizedNetworkData(processedData, processedLabels, test_split):
    print("Performing train-test split")
    separator = int(len(processedLabels) * (1 - test_split))
    trainX = processedData[:separator]
    testX = processedData[separator:]
    trainY = processedLabels[:separator]
    testY = processedLabels[separator:]

    return trainX, testX, trainY, testY


def evaluate(model, train_x, train_y, test_x, test_y):
    test_loss, test_acc = model.evaluate(test_x, test_y)
    train_loss, train_acc = model.evaluate(train_x, train_y)
    print(
        "Train Loss: {}, Train Accuracy: {}; Test Loss: {}, Test Accuracy: {}".format(
            train_loss, train_acc, test_loss, test_acc
        )
    )

def runTokenNetwork2(trainX, testX, trainY, testY,  _learning_rate,
    _batch_size, _epochs, sequence_length
        ):
    
    model = Sequential()
    model.add(
        LSTM(
            units=128, input_shape=(sequence_length, 10), return_sequences=True, dropout=0.3, recurrent_dropout=0.3
        )
    )
    model.add(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))
    model.add(Dense(128, activation = "relu"))
    model.add(LSTM(32, dropout=0.3, recurrent_dropout=0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(10, activation = "relu"))
    model_optimizer = adam(lr=_learning_rate)
    # model_optimizer = iRprop_()
    model.compile(
        loss="mean_squared_error", optimizer=model_optimizer, metrics=["accuracy" , "mae",]
    )

    model.summary()

    print("Network.py: Beginning model training")
    model.fit(
        trainX,
        trainY,
        validation_data=(testX, testY),
        epochs=_epochs,
        batch_size=_batch_size,
    )

    directory = "debug/models/"
    filename = "token-c2-test-{date:%Y-%m-%d-%H-%M-%S}".format(
        date=datetime.datetime.now()
    )

    model.save(directory + filename + ".h5")

    return model, filename


def runTokenNetwork3(trainX, testX, trainY, testY,  _learning_rate,
    _batch_size, _epochs, sequence_length
        ):
    
    model = Sequential()
    model.add(
        CuDNNGRU(
            units=100, input_shape=(sequence_length, 10), return_sequences=True
        )
    )
    model.add(Dropout(0.3))
    model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="elu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(CuDNNGRU(50))
    model.add(Dense(100, activation="elu"))
    model.add(Dense(10, activation="elu"))
    model_optimizer = adam(lr=_learning_rate)
    model.compile(
        loss="mean_squared_error", optimizer=model_optimizer, metrics=["accuracy" , "mae",]
    )

    model.summary()

    print("Network.py: Beginning model training")
    model.fit(
        trainX,
        trainY,
        validation_data=(testX, testY),
        epochs=_epochs,
        batch_size=_batch_size,
    )

    directory = "debug/models/"
    filename = "token-c3-test-{date:%Y-%m-%d-%H-%M-%S}".format(
        date=datetime.datetime.now()
    )

    model.save(directory + filename + ".h5")

    return model, filename
    
def runTokenNetwork4(trainX, testX, trainY, testY,  _learning_rate,
    _batch_size, _epochs, sequence_length, tokens
        ):
    
    model = Sequential()
    model.add(
        CuDNNGRU(
            units=256, input_shape=(sequence_length, 10), return_sequences=True
        )
    )
    model.add(Dropout(0.3))
    model.add(CuDNNGRU(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(CuDNNGRU(960))
    model.add(Dense(960, activation="elu"))
    model.add(Dropout(0.3))
    model.add(Dense(len(tokens), activation="elu"))
    model.add(Dense(10, activation="elu"))
    model_optimizer = adam(lr=_learning_rate)
    model.compile(
        loss="mean_squared_error", optimizer=model_optimizer, metrics=["accuracy" , "mae",]
    )

    model.summary()

    print("Network.py: Beginning model training")
    model.fit(
        trainX,
        trainY,
        validation_data=(testX, testY),
        epochs=_epochs,
        batch_size=_batch_size,
    )

    directory = "debug/models/"
    filename = "token-c3-test-{date:%Y-%m-%d-%H-%M-%S}".format(
        date=datetime.datetime.now()
    )

    model.save(directory + filename + ".h5")

    return model, filename
    

def runTokenizedNetwork(
    trainX, testX, trainY, testY, sequence_length, _learning_rate,
    _batch_size, _epochs
):
    num_features = 4
    embedding_vecor_length = 128


    model = Sequential()
    model.add(Embedding(512, embedding_vecor_length, input_length=sequence_length))
    model.add(LSTM(200, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Conv1D(filters=32, kernel_size=3, padding="same", activation="elu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation="elu"))
    model.add(Dense(4, activation = "elu"))
    model_optimizer = adam(lr=_learning_rate)
    model.compile(
        loss="mean_squared_error", optimizer=model_optimizer, metrics=["accuracy" , "mae", "categorical_accuracy"]
    )

    model.summary()

    print("Network.py: Beginning model training")
    model.fit(
        trainX,
        trainY,
        validation_data=(testX, testY),
        epochs=_epochs,
        batch_size=_batch_size,
    )

    directory = "debug/models/"
    filename = "token-test-{date:%Y-%m-%d-%H-%M-%S}".format(
        date=datetime.datetime.now()
    )
    
    model.save(directory + filename + ".h5")

    return model, trainX, trainY, testX, testY


def runNormalizedNetwork(
    trainX, testX, trainY, testY, sequence_length, _learning_rate, _batch_size, _epochs
):

    num_features = 4

    model = Sequential()
    model.add(
        LSTM(
            units=256, input_shape=(sequence_length, num_features), return_sequences=True
        )
    )
    model.add(Dropout(0.3))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(LSTM(128))
    model.add(Dropout(0.15))
    model.add(Dense(128,activation="relu"))
    model.add(Dense(4, activation="sigmoid"))

    model_optimizer = adam(lr=_learning_rate)
    model.compile(
        loss="mean_squared_error", optimizer=model_optimizer, metrics=["accuracy", "mae"]
    )

    model.summary()

    print("Network.py: Beginning model training")
    model.fit(
        trainX,
        trainY,
        validation_data=(testX, testY),
        epochs=_epochs,
        batch_size=_batch_size,
    )

    directory = "debug/models/"
    filename = "norm-test-{date:%Y-%m-%d-%H-%M-%S}".format(date=datetime.datetime.now())

    model.save(directory + filename + ".h5")

    return model, trainX, trainY, testX, testY
