from DataGen import *
from DataAnalysisHelpers import *
from Network import *
from NetworkRun import *
import copy
import time
import random
from keras.preprocessing import sequence
from keras.utils import plot_model
import numpy as np


def runFullMethodBC1():
    # parameter dictionary
    params = {}
    params["learning_rate"] = 0.000004
    params["batch_size"] = 128
    params["epochs"] = 150
    params["timestep_resolution"] = 10
    params["composition_length"] = 60
    params["dataset"] = "classical"
    params["sequence_length"] = 100
    params["num_simultaneous_notes"] = 4
    params["data_amount"] = 1.0
    
    # get the filepaths and load for all .midi files in the dataset
    filepaths = getCleanedFilePaths(params["dataset"])
    # load appropriate amount of dataset in
    # limit the notes to the range c3 - c6 (3 full octaves)
    midi_data, key_distributions = loadMidiData(
        filepaths[: int(len(filepaths) * params["data_amount"]) - 1], 48, 83
    )
    
    
    # convert into python arrays andd dicts
    data, vectors, timeScalars = getKerasData(midi_data, key_distributions)
    del midi_data
    # convert to normalized form for network training
    processedData, processedLabels, tokens = processKerasDataMethodB(
        data,
        params["sequence_length"],
        params["composition_length"] * params["timestep_resolution"],
        params["timestep_resolution"],
        params["num_simultaneous_notes"],
    )
    # no longer need unscaled data
    del data

    finalData, finalLabels = finaliseTokenNetworkData(processedData, processedLabels)
    del processedData
    del processedLabels
    
    # train / test split
    # trainX, testX, trainY, testY = genNetworkData(processedData, processedLabels)
    trainX, testX, trainY, testY = genTokenizedNetworkData(finalData, finalLabels, 0.25)
    del finalData
    del finalLabels
    
    # Begin training the neural network based on the above parameters
    model, filename = MethodBC1(trainX, testX, trainY, testY, tokens, params)
    
    del model
    # filename = "token-c3-test-2019-07-30-03-14-42"
    # Load a pretrained model and generate some music
    loaded_model = loadModel(filename + ".h5")
    loaded_model.summary()
    
    # take some test data
    tempData = copy.deepcopy(testX)
    # plot_model(loaded_model, to_file="debug_model.png")
    # how many compositions should we produce?
    for i in range(30):
        upperbound = len(tempData)
        bounds = []
        for j in range(2):
            bounds.append(random.randint(0, upperbound))
        bounds.sort()
        print(bounds)
        sample = tempData[bounds[0] : bounds[0] + 1]
        # Use network to generate some notes
        composition = startMethodBNetworkRun(
            loaded_model, sample, params["timestep_resolution"], 60
        )
        # Output to .midi file
        convertMethodBDataToMidi(composition, tokens, filename, params["timestep_resolution"])
    
    del loaded_model
    del tempData


runFullMethodBC1()