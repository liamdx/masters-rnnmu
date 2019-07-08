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
learning_rate = 0.00005
batch_size = 64
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
loaded_model_name = "test-2019-07-03-15-54-13"
loaded_model = loadModel(loaded_model_name + ".h5")


# take some test data
tempData = copy.deepcopy(testX)
# Even when we pass in one sequence, the RNN expects the shape to be 3D
# e.g. shape of input must be at least (1, sequence_length, num_features)

for i in range(30):
    upperbound = len(tempData)
    bounds = []
    for j in range(2):
        bounds.append(random.randint(0, upperbound))
    bounds.sort()
    print(bounds)
    sample = tempData[bounds[0] : bounds[0] + 50]
    # Use network to generate some notes
    composition = startNetworkRun(loaded_model, sample, sequence_length, 300)
    # Output to .midi file
    convertDataToMidi(composition, timeScalars, loaded_model_name)
