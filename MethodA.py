from DataGen import *
from DataAnalysisHelpers import *
from Network import *
from NetworkRun import *

#  from util.tensorflow_utils import *
import copy
import time
import random

# parameter dictionary
params = {}
params["learning_rate"] = 0.00003
params["batch_size"] = 128
params["epochs"] = 8
params["timestep_resolution"] = 15
params["composition_length"] = 32
params["dataset"] = "classical"
params["sequence_length"] = 15
params["num_simultaneous_notes"] = 4
params["data_amount"] = 0.2

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
model, filename = MethodAC1(
    trainX, testX, trainY, testY, params
)


# Load a pretrained model and generate some music
# filename = "norm-test-2019-07-28-13-27-26"
loaded_model = loadModel(filename + ".h5")


# take some test data
tempData = copy.deepcopy(testX)
# Even when we pass in one sequence, the RNN expects the shape to be 3D
# e.g. shape of input must be at least (1, sequence_length, num_features)

for i in range(15):
    upperbound = len(tempData)
    bounds = []
    for j in range(2):
        bounds.append(random.randint(0, upperbound))
    bounds.sort()
    print(bounds)
    sample = tempData[bounds[0] : bounds[0] + 50]
    # Use network to generate some notes
    composition = startMethodANetworkRun(loaded_model, sample, params["sequence_length"], 150)
    # Output to .midi file
    convertNormalizedDataToMidi(composition, timeScalars, filename)
