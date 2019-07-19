import tensorflow as tf
import keras
from keras.models import load_model
import copy
import numpy as np
from tqdm import tqdm


def loadModel(modelName):
    directory = "debug/models/"
    model = load_model(directory + modelName, compile=True)
    return model


def startNormalizedNetworkRun(model, sequence, sequence_length, notesToProduce):
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

def startTokenizedNetworkRun(model, sequence, sequence_length, notesToProduce):
    preds = []
    sample = copy.deepcopy(sequence)
    for i in range(notesToProduce):
        # sequence shape = (1,notesToProduce,4)
        # pred shape = (1,4)
        pred = model.predict(sample)
        print("\nPred =")
        print(pred)
        # shift the array down so that the last row = 0,0,0,0
        # then replace the null values with pred
        print("\nsample before Roll")
        print(sample)
        sample = np.roll(sample, -4, axis=1)
        print("\nsample after roll")
        print(sample)
        for j in range(4):
            # we want to replace the last 4 values in the list
            index = ((sequence_length * 4) - 4) + j
            sample[0][index] = pred[0][j]
        print("\nSample after substitution\n")
        print(sample)
        preds.append(pred)

    return preds