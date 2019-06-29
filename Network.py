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


def genNetworkData(processedData, processedLabels):
    # Should we one-hot encode the labels ?
    # lb = MultiLabelBinarizer()
    # labels_encoded = lb.fit_transform(labels)
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


def runNetwork(x, y, sequence_length, _learning_rate, _batch_size, _epochs):
    trainX, testX, trainY, testY = genNetworkData(x, y)

    num_features = 4

    model = Sequential()
    model.add(
        LSTM(
            units=3, input_shape=(sequence_length, num_features), return_sequences=True
        )
    )
    model.add(Dropout(0.1))
    model.add(LSTM(128))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="sigmoid"))
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

    return model, trainX, trainY, testX, testY

    # Final evaluation of the model
    # evaluate(model, trainX, trainY, testX, testY)

    # model.summary()

    # model.save('imdb-{0}-{1}-{2}-{3}-{4}.cpkt'.format(combination, _learning_rate, _epochs, _batches, _seed)
