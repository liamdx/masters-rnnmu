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


def startNetworkRun(model, sequence, sequence_length, notesToProduce):
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
