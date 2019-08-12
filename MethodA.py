from DataGen import *
from DataAnalysisHelpers import *
from Network import *
from NetworkRun import *
import numpy as np
import copy
import time
import random
import csv
from tqdm import tqdm

def runFullMethodA():
    params = {}
    params["learning_rate"] = 0.000002
    params["batch_size"] = 64
    params["epochs"] = 90
    params["dataset"] = "classical"
    params["sequence_length"] = 15
    params["data_amount"] = 1.0

    # get the filepaths and load for all .midi files in the dataset
    filepaths = getCleanedFilePaths(params["dataset"])
    # load appropriate amount of dataset in
    # limit the notes to the range c3 - c6 (3 full octaves)
    midi_data, key_distributions = loadMidiData(
        filepaths[: int(len(filepaths) * params["data_amount"]) - 1], 36, 84
    )


    # convert into python arrays andd dicts
    data, vectors, timeScalars = getKerasData(midi_data, key_distributions)
    del midi_data  # no longer need original data, free the memory

    # convert to normalized form for network training
    processedData, processedLabels, tokens, token_cutoffs = processKerasDataMethodA(
        data, timeScalars, params["sequence_length"]
    )
    del data  # no longer need unscaled data


    # train / test split
    trainX, testX, trainY, testY = genTokenizedNetworkData(processedData, processedLabels, 0.25)
    del processedData
    del processedLabels


    # Begin training the neural network based on the above parameters
    model, filename = MethodAC1(trainX, testX, trainY, testY, tokens, params)
    del model


    # Load a pretrained model and generate some music
    # filename = "norm-test-2019-07-28-13-27-26"
    loaded_model = loadModel(filename + ".h5")


    # take some test data
    tempData = copy.deepcopy(testX)


    testComp = [testY[0:100]]

    # inverse tokens for conversion back to midi
    inv_tokens = invertDictionary(tokens)


    # convertMethodADataToMidi(testComp, inv_tokens, token_cutoffs, "DebugMidiTest")

    # produce 20 compositions
    for i in range(100):
        upperbound = len(tempData)
        bounds = []
        for j in range(2):
            bounds.append(random.randint(0, upperbound))
        bounds.sort()
        print(bounds)
        sample = tempData[bounds[0] : bounds[0]+1]
        # Use network to generate some notes
        composition = startMethodANetworkRun(
            loaded_model, sample, params["sequence_length"], 300
        )
        # Output to .midi file
        convertMethodADataToMidi(composition, inv_tokens, token_cutoffs, filename)

