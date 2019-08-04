import tensorflow as tf
import keras
from keras.models import load_model
import copy
import numpy as np
from tqdm import tqdm


def loadModel(modelName):
    directory = "res/models/"
    model = load_model(directory + modelName, compile=True)
    return model


def startMethodANetworkRun(model, sequence, sequence_length, notesToProduce):
    preds = []
    sample = copy.deepcopy(sequence)
    for i in tqdm(range(notesToProduce)):
        # sequence shape = (1,notesToProduce,4)
        # pred shape = (1,4)
        pred = model.predict(sample)
        # shift the array down so that the last row = 0,0,0,0
        # then replace the null values with pred
        sample = np.roll(sample, -1, axis=1)
        sample[0, len(sample[0]) - 1] = pred[0]
        preds.append(pred)

    return preds


def startMethodBNetworkRun(model, sequence, timestep_resolution, secondsOfMusic):
    preds = []
    sample = copy.deepcopy(sequence)
    length__ = int(timestep_resolution * secondsOfMusic)
    print("Composing: Genius at work")
    for i in tqdm(range(length__)):
        pred = model.predict(sample)
        preds.append(pred)
        #        print("last row  before roll = ")
        #        print(sample[0][len(sample) - 1])
        sample = np.roll(sample, -1, 1)
        sample[0][len(sample) - 1] = pred
    #        print("last row  after roll = ")
    #        print(sample[0][len(sample) - 1])
    return preds


def startMethodBEmbeddingNetworkRun(
    model, sequence, timestep_resolution, secondsOfMusic
):
    preds = []
    sample = copy.deepcopy(sequence)
    length__ = int(timestep_resolution * secondsOfMusic)
    print("Composing: Genius at work")
    for i in range(length__):
        pred = model.predict(sample)
        preds.append(pred)
        print("sample before roll = ")
        print(sample[0])
        sample = np.roll(sample, -4)
        for j in range(4):
            index = len(sample) - (4 - j)
            sample[0][index] = pred[0][j]
        print("Sample after roll = ")
        print(sample[0])
    return preds
