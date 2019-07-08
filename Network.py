import tensorflow as tf
import keras
from keras.callbacks import TensorBoard
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Activation, Input, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.losses import categorical_crossentropy
from keras.optimizers import adam
from util.tensorflow_utils import *
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
import datetime
import json


def genNormalizedNetworkData(processedData, processedLabels):
    trainX, testX, trainY, testY = train_test_split(
        np.array(processedData),
        np.array(processedLabels),
        test_size=0.25,
        random_state=42,
    )
    return trainX, testX, trainY, testY


def genTokenizedNetworkData(processedData, processedLabels):
    trainX, testX, trainY, testY = train_test_split(
        np.array(processedData),
        np.array(processedLabels),
        test_size=0.25,
        random_state=42,
    )
    return trainX, testX, trainY, testY


def evaluate(model, train_x, train_y, test_x, test_y):
    test_loss, test_acc = model.evaluate(test_x, test_y)
    train_loss, train_acc = model.evaluate(train_x, train_y)
    print(
        "Train Loss: {}, Train Accuracy: {}; Test Loss: {}, Test Accuracy: {}".format(
            train_loss, train_acc, test_loss, test_acc
        )
    )


def runTokenizedNetwork(
    trainX, testX, trainY, testY, sequence_length, _learning_rate, _batch_size, _epochs
):
    num_features = 4
    embedding_vecor_length = 32
    max_song_length = 5000

    trainX = sequence.pad_sequences(trainX, maxlen=max_song_length)
    testX = sequence.pad_sequences(testX, maxlen=max_song_length)

    model = Sequential()
    model.add(Embedding(512, embedding_vecor_length, input_length=max_song_length, return_sequences=True))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="sigmoid"))
    model.add(LSTM(32))
    model.add(Dense(4, activation="sigmoid"))

    model_optimizer = adam(lr=_learning_rate)
    model.compile(
        loss="categorical_crossentropy", optimizer=model_optimizer, metrics=["accuracy"]
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
            units=32, input_shape=(sequence_length, num_features), return_sequences=True
        )
    )
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="sigmoid"))
    model.add(LSTM(32))
    model.add(Dense(4, activation="sigmoid"))

    model_optimizer = adam(lr=_learning_rate)
    model.compile(
        loss="categorical_crossentropy", optimizer=model_optimizer, metrics=["accuracy"]
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
