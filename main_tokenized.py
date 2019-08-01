from DataGen import *
from DataAnalysis import *
from Network import *
from NetworkRun import *
import copy
import time
import random
from keras.preprocessing import sequence
from keras.utils import plot_model

# parameter dictionary
params = {}
params["learning_rate"] = 0.000001
params["batch_size"] = 128
params["epochs"] = 4
params["timestep_resolution"] = 15
params["composition_length"] = 32
params["dataset"] = "maestro"
params["sequence_length"] = 150
params["num_simultaneous_notes"] = 4
params["data_amount"] = 0.3

# Network parameters
learning_rate = 0.000001
batch_size = 128
epochs = 4
# data parameters
timestep_resolution = 15
composition_length = 32
dataset = "maestro"
sequence_length = 150
num_simultaneous_notes = 4

# get the filepaths and load for all .midi files in the dataset
filepaths = getCleanedFilePaths(params["dataset"])
# load appropriate amount of dataset in
midi_data, key_distributions = loadMidiData(
    filepaths[: int(len(filepaths) * params["data_amount"]) - 1]
)

# convert into python arrays andd dicts
data, vectors, timeScalars = getKerasData(midi_data, key_distributions)

# convert to normalized form for network training
processedData, processedLabels, tokens = processKerasDataTokenized(
    data,
    params["sequence_length"],
    params["composition_length"] * params["timestep_resolution"],
    params["timestep_resolution"],
    params["num_simultaneous_notes"],
)
# no longer need unscaled data
del data
del midi_data

finalData, finalLabels = finaliseTokenNetworkData(processedData, processedLabels)

del processedData
del processedLabels

# train / test split
# trainX, testX, trainY, testY = genNetworkData(processedData, processedLabels)
trainX, testX, trainY, testY = genTokenizedNetworkData(finalData, finalLabels, 0.25)
del finalData
del finalLabels

# Begin training the neural network based on the above parameters
model, filename = runTokenNetwork4(trainX, testX, trainY, testY, tokens, params)


# filename = "token-c3-test-2019-07-30-03-14-42"
# Load a pretrained model and generate some music
loaded_model = loadModel(filename + ".h5")
loaded_model.summary()

# plot_model(loaded_model, to_file="debug_model.png")

# take some test data
tempData = copy.deepcopy(testX)

# how many compositions should we produce?
for i in range(10):
    upperbound = len(tempData)
    bounds = []
    for j in range(2):
        bounds.append(random.randint(0, upperbound))
    bounds.sort()
    print(bounds)
    sample = tempData[bounds[0] : bounds[0] + 1]
    # Use network to generate some notes
    composition = startTokenizedNetworkRun(
        loaded_model, sample, timestep_resolution, 16
    )
    # Output to .midi file
    convertTokenizedDataToMidi2(composition, tokens, filename, timestep_resolution)
