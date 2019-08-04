from DataGen import *
from DataAnalysisHelpers import *
from Network import *
from NetworkRun import *
import numpy as np
#  from util.tensorflow_utils import *
import copy
import time
import random
import csv
from tqdm import tqdm

# from https://stackoverflow.com/questions/50459119/writing-a-3d-numpy-array-to-a-csv-file?noredirect=1&lq=1
    
#x = np.array(processedData).tolist()
#with open("MethodA_X_DATA.csv", "w") as csvfile:
#    writer = csv.writer(csvfile, delimiter=',')
#    writer.writerows(x)
#
#y = np.array(processedLabels).tolist()
#with open("MethodA_Y_LABELS.csv", "w") as csvfile:
#    writer = csv.writer(csvfile, delimiter=',')
#    writer.writerows(y)
#
#
#with open ("MethodA_X_DATA.csv", "r") as f:
#    reader = csv.reader(f)
#    examples = list(reader)
#
#nwexamples = []
#for row in tqdm(examples):
#    nwrow = []
#    for r in row:
#        nwrow.append(eval(r))
#    nwexamples.append(nwrow)
#nwx = np.array(nwexamples)
#
#del nwx
#del nwexamples
#del examples

def runFullMethodA():
    # parameter dictionary
    params = {}
    params["learning_rate"] = 0.0005
    params["batch_size"] = 64
    params["epochs"] = 100
    params["timestep_resolution"] = 15
    params["composition_length"] = 32
    params["dataset"] = "classical"
    params["sequence_length"] = 5
    params["num_simultaneous_notes"] = 4
    params["data_amount"] = 0.3
    
    # get the filepaths and load for all .midi files in the dataset
    filepaths = getCleanedFilePaths(params["dataset"])
    # load appropriate amount of dataset in
    # limit the notes to the range c3 - c6 (3 full octaves)
    midi_data, key_distributions = loadMidiData(
        filepaths[: int(len(filepaths) * params["data_amount"]) - 1], 48, 83
    )
    
    # convert into python arrays andd dicts
    data, vectors, timeScalars = getKerasData(midi_data, key_distributions)
    del midi_data  # no longer need original data, free the memory
    
    
    # convert to normalized form for network training
    processedData, processedLabels = processKerasDataMethodA(
        data, timeScalars, params["sequence_length"]
    )
    del data  # no longer need unscaled data
    
    
    # train / test split
    trainX, testX, trainY, testY = genNormalizedNetworkData(processedData, processedLabels)
    del processedData
    del processedLabels
    
    # Begin training the neural network based on the above parameters
    model, filename = MethodAC1(trainX, testX, trainY, testY, params)
    
    
    # Load a pretrained model and generate some music
    # filename = "norm-test-2019-07-28-13-27-26"
    loaded_model = loadModel(filename + ".h5")
    
    
    # take some test data
    tempData = copy.deepcopy(testX)
    # Even when we pass in one sequence, the RNN expects the shape to be 3D
    # e.g. shape of input must be at least (1, sequence_length, num_features)
    
    for i in range(20):
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
        convertNormalizedDataToMidi(composition, timeScalars, filename)
        
    
runFullMethodA()
