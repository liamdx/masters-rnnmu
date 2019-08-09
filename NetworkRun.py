import tensorflow as tf
import keras
from keras.models import load_model
import copy
import numpy as np
from tqdm import tqdm
from Network import *
from DataGen import *
from DataAnalysisHelpers import *

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



def startFullMethodBRun(num_compositions):
    params = {}
    params["learning_rate"] = 0.000004
    params["batch_size"] = 128
    params["epochs"] = 150
    params["timestep_resolution"] = 10
    params["composition_length"] = 60
    params["dataset"] = "classical"
    params["sequence_length"] = 100
    params["num_simultaneous_notes"] = 4
    params["data_amount"] = 0.5
    
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

    filename = "final_method_b_model"
    loaded_model = loadModel(filename + ".h5")
    loaded_model.summary()
    
    # take some test data
    tempData = copy.deepcopy(testX)
    # plot_model(loaded_model, to_file="debug_model.png")
    # how many compositions should we produce?
    for i in range(num_compositions):
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



def startFullMethodARun(num_compositions):
    # parameter dictionary
    params = {}
    params["learning_rate"] = 0.000003
    params["batch_size"] = 128
    params["epochs"] = 20
    params["dataset"] = "classical"
    params["sequence_length"] = 10
    params["data_amount"] = 1.0
    # get the filepaths and load for all .midi files in the dataset
    filepaths = getCleanedFilePaths(params["dataset"])
    # load appropriate amount of dataset in
    # limit the notes to the range c3 - c6 (3 full octaves)
    midi_data, key_distributions = loadMidiData(
        filepaths[: int(len(filepaths) * params["data_amount"]) - 1], 48, 83)
    # convert into python arrays andd dicts
    data, vectors, timeScalars = getKerasData(midi_data, key_distributions)
    del midi_data  # no longer need original data, free the memory    
    # convert to normalized form for network training
    processedData, processedLabels, tokens, token_cutoffs = processKerasDataMethodA(
        data, timeScalars, params["sequence_length"]
    )
    del data  # no longer need unscaled data 
    # train / test split
    trainX, testX, trainY, testY = genNormalizedNetworkData(processedData, processedLabels)
    del processedData
    del processedLabels

    filename = "final_method_a_model"
    loaded_model = loadModel(filename + ".h5")
    # take some test data
    tempData = copy.deepcopy(testX)
    
    # inverse tokens for conversion back to midi
    inv_tokens = invertDictionary(tokens)
    # produce 20 compositions
    for i in tqdm(range(num_compositions)):
        upperbound = len(tempData)
        bounds = []
        for j in range(2):
            bounds.append(random.randint(0, upperbound))
        bounds.sort()
        print(bounds)
        sample = tempData[bounds[0] : bounds[0] + 50]
        # Use network to generate some notes
        composition = startMethodANetworkRun(
            loaded_model, sample, params["sequence_length"], 150
        )
        # Output to .midi file
        convertMethodADataToMidi(composition, inv_tokens, token_cutoffs, filename)
    
    del loaded_model
    del tempData
    del inv_tokens
    del tokens
