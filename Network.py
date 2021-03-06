import tensorflow as tf
import keras
import keras.backend as K
from keras.callbacks import TensorBoard
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.losses import categorical_crossentropy
from keras.optimizers import adam, adadelta
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
import datetime
import json
import numpy as np
from util.rprop import *
from keras.layers import (
    Dense,
    LSTM,
    Flatten,
    Activation,
    Input,
    Dropout,
    Conv1D,
    MaxPooling1D,
    Conv2D,
    MaxPooling2D,
    CuDNNGRU,
    CuDNNLSTM,
)


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


def MethodBC1(trainX, testX, trainY, testY, tokens, params):
    K.clear_session()
    model = Sequential()
    model.add(
        CuDNNLSTM(
            units=512,
            input_shape=(params["sequence_length"], params["num_simultaneous_notes"]),
            return_sequences=True,
        )
    )
    model.add(Dropout(0.3))
    model.add(CuDNNLSTM(256, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation="elu"))
    model.add(CuDNNLSTM(params["sequence_length"], return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Dense(params["sequence_length"], activation="elu"))
    model.add(CuDNNLSTM(len(tokens)))
    model.add(Dense(len(tokens), activation="elu"))
    model.add(Dense(params["num_simultaneous_notes"], activation="elu"))
    model_optimizer = adam(lr=params["learning_rate"])
    model.compile(
        loss="mean_squared_error", optimizer=model_optimizer, metrics=["mae", "acc"]
    )

    model.summary()

    a = datetime.datetime.now()
    timestamp = "%s%s%s%s%s" % (a.second, a.minute, a.hour, a.day, a.month)
    
    tensorboard = buildTensorBoard(params,"MethodB", "C1", timestamp)
    
    print("Network.py: Beginning model training")
    model.fit(
        trainX,
        trainY,
        validation_data=(testX, testY),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        callbacks=[tensorboard]
    )

    directory = "res/models/"
    # properties are: dataset, learning rate, batch size, epochs, sequence length,
    # num tokens, num simultaneous notes, timestep resolution, composition length, percentage of data used

    filename = "MethodB-%s-C1-%f-%d-%d-%d-%d-%d-%d-%d-%f-%s" % (
        params["dataset"],
        params["learning_rate"],
        params["batch_size"],
        params["epochs"],
        params["sequence_length"],
        len(tokens),
        params["num_simultaneous_notes"],
        params["timestep_resolution"],
        params["composition_length"],
        params["data_amount"],
        timestamp,
    )

    model.save(directory + filename + ".h5")

    return model, filename


def MethodBC2(trainX, testX, trainY, testY, tokens, params):
    numTokens = len(tokens)
    K.clear_session()
    model = Sequential()

    testX = testX.reshape(
        len(testX), params["sequence_length"] * params["num_simultaneous_notes"]
    )
    trainX = trainX.reshape(
        len(trainX), params["sequence_length"] * params["num_simultaneous_notes"]
    )

    # num simultaneous notes = 4, sequence length = 75
    model.add(
        Embedding(
            numTokens,
            params["num_simultaneous_notes"],
            input_length=params["sequence_length"] * params["num_simultaneous_notes"],
        )
    )
    model.add(CuDNNLSTM(256, return_sequences=True))
    model.add(Dense(256, activation="elu"))
    model.add(CuDNNLSTM(params["sequence_length"] * 4, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Dense(params["sequence_length"] * 4, activation="elu"))
    model.add(CuDNNLSTM(numTokens))
    model.add(Dropout(0.3))
    model.add(Dense(numTokens, activation="elu"))
    model.add(Dense(params["num_simultaneous_notes"], activation="elu"))
    model_optimizer = adam(lr=params["learning_rate"])
    model.compile(
        loss="mean_squared_error",
        optimizer=model_optimizer,
        metrics=["mae", "acc"],
    )

    model.summary()

    a = datetime.datetime.now()
    timestamp = "%s%s%s%s%s" % (a.second, a.minute, a.hour, a.day, a.month)
    
    tensorboard = buildTensorBoard(params,"MethodB", "C2", timestamp)
    
    print("Network.py: Beginning model training")
    model.fit(
        trainX,
        trainY,
        validation_data=(testX, testY),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        callbacks=[tensorboard],
    )

    directory = "res/models/"
    # properties are: dataset, learning rate, batch size, epochs, sequence length,
    # num tokens, num simultaneous notes, timestep resolution, composition length, percentage of data used

    filename = "MethodB-%s-C2-%f-%d-%d-%d-%d-%d-%d-%d-%f-%s" % (
        params["dataset"],
        params["learning_rate"],
        params["batch_size"],
        params["epochs"],
        params["sequence_length"],
        numTokens,
        params["num_simultaneous_notes"],
        params["timestep_resolution"],
        params["composition_length"],
        params["data_amount"],
        timestamp,
    )

    model.save(directory + filename + ".h5")

    return model, filename


def MethodAC1(trainX, testX, trainY, testY, tokens, params):
    K.clear_session()
    numTokens = len(tokens)
    model = Sequential()
    model.add(
        CuDNNLSTM(
            units=512,
            input_shape=(params["sequence_length"], 4),
            return_sequences=True,
        )
    )
    model.add(Dropout(0.3))
    model.add(Dense(512, activation="elu"))
    model.add(Dense(256, activation="elu"))
    model.add(CuDNNLSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation="elu"))
    model.add(Dense(4, activation="elu"))
    model_optimizer = adam(lr=params["learning_rate"])
    model.compile(
        loss="mean_squared_error", optimizer=model_optimizer, metrics=["mae", "acc"]
    )


    model.summary()

    a = datetime.datetime.now()
    timestamp = "%s%s%s%s%s" % (a.second, a.minute, a.hour, a.day, a.month)
    
    tensorboard = buildTensorBoard(params,"MethodA", "C1", timestamp)
    
    print("Network.py: Beginning model training")
    model.fit(
        trainX,
        trainY,
        validation_data=(testX, testY),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        callbacks=[tensorboard],
    )

    directory = "res/models/"
    # properties are: dataset, learning rate, batch size, epochs, sequence length,
    # num tokens, num simultaneous notes, timestep resolution, composition length, percentage of data used

    filename = "MethodA-%s-C1-%f-%d-%d-%d-%d-%f-%s" % (
        params["dataset"],
        params["learning_rate"],
        params["batch_size"],
        params["epochs"],
        params["sequence_length"],
        2305,
        params["data_amount"],
        timestamp,
    )

    model.save(directory + filename + ".h5")

    return model, filename


def MethodAC2(trainX, testX, trainY, testY, params):
    K.clear_session()
    model = Sequential()
    model.add(
        CuDNNLSTM(
            units=512, input_shape=(params["sequence_length"], 4), return_sequences=True
        )
    )
    model.add(Dropout(0.3))
    model.add(Dense(512, activation="elu"))
    model.add(CuDNNLSTM(256, return_sequences= True))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation="elu"))
    model.add(CuDNNLSTM(params["sequence_length"] , return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Dense(params["sequence_length"], activation="elu"))
    model.add(CuDNNLSTM(4))
    model.add(Dense(4, activation="elu"))
    model_optimizer = adam(lr=params["learning_rate"])
    model.compile(
        loss="mean_squared_error", optimizer=model_optimizer, metrics=["mae", "acc"]
    )

    model.summary()

    a = datetime.datetime.now()
    timestamp = "%s%s%s%s%s" % (a.second, a.minute, a.hour, a.day, a.month)

    tensorboard = buildTensorBoard(params,"MethodA", "C2", timestamp)

    print("Network.py: Beginning model training")
    model.fit(
        trainX,
        trainY,
        validation_data=(testX, testY),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        callbacks=[tensorboard],
    )

    directory = "res/models/"
    # properties are: dataset, learning rate, batch size, epochs, sequence length,
    # num tokens, num simultaneous notes, timestep resolution, composition length, percentage of data used

    filename = "MethodA-%s-C2-%f-%d-%d-%d-%d-%f-%s" % (
        params["dataset"],
        params["learning_rate"],
        params["batch_size"],
        params["epochs"],
        params["sequence_length"],
        2305,
        params["data_amount"],
        timestamp,
    )


    model.save(directory + filename + ".h5")

    return model, filename


def buildTensorBoard(params, method, combination, timestamp):
    boardString = "res/logs/%s-%s-%s-%f-%d-%d-%d-%d-%f-%s" % (
        method,
        combination,
        params["dataset"],
        params["learning_rate"],
        params["batch_size"],
        params["epochs"],
        params["sequence_length"],
        2305,
        params["data_amount"],
        timestamp,
    )
    tensorboard = TensorBoard(log_dir=boardString, histogram_freq=0, write_grads=True)

    return tensorboard