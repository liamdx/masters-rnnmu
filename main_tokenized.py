from DataGen import *
from DataAnalysis import *
from Network import *
from NetworkRun import *
from util.tensorflow_utils import *
import copy
import time
import random

# debug timing how long dataset takes to load
start_time = time.process_time()

# Network parameters
learning_rate = 0.3
batch_size = 128
epochs = 2
sequence_length = 15
dataset = "classical"


# get the filepaths and load for all .midi files in the dataset
filepaths = getCleanedFilePaths(dataset)
midi_data, key_distributions = loadMidiData(filepaths[0:50])

# convert into python arrays andd dicts
data, vectors, timeScalars = getKerasData(midi_data, key_distributions)
# no longer need original data, free the memory

# convert to normalized form for network training
processedData, processedLabels, tokens = processKerasDataTokenized(
    data, sequence_length
)
# no longer need unscaled data
del data, midi_data
# train / test split
# trainX, testX, trainY, testY = genNetworkData(processedData, processedLabels)

trainX, testX, trainY, testY = genNormalizedNetworkData(processedData, processedLabels)
del processedData
del processedLabels

print(
    "Time taken to load dataset = %d minutes"
    % ((time.process_time() - start_time) / 60.0)
)
# Begin training the neural network based on the above parameters
model = runTokenizedNetwork(
    trainX, testX, trainY, testY, sequence_length, learning_rate, batch_size, epochs
)


# Load a pretrained model and generate some music
loaded_model_name = "token-test-2019-07-08-19-43-03"
loaded_model = loadModel(loaded_model_name + ".h5")


# take some test data
tempData = copy.deepcopy(testX)
pred = loaded_model.predict(tempData[0:10])
