from DataGen import *
from DataAnalysis import *
from Network import *
from NetworkRun import *
import copy
import time
import random
from keras.preprocessing import sequence
# debug timing how long dataset takes to load
start_time = time.process_time()

# Network parameters
learning_rate = 0.00001
batch_size = 64
epochs = 10

# data parameters
timestep_resolution = 30
composition_length = 32
dataset = "classical"
sequence_length = 240


# get the filepaths and load for all .midi files in the dataset
filepaths = getCleanedFilePaths(dataset)
midi_data, key_distributions = loadMidiData(filepaths[:100])

# convert into python arrays andd dicts
data, vectors, timeScalars = getKerasData(midi_data, key_distributions)

# no longer need original data, free the memory

# convert to normalized form for network training
processedData, processedLabels, tokens = processKerasDataTokenized(
    data, sequence_length, composition_length * timestep_resolution , timestep_resolution
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

print(
    "Time taken to load dataset = %d minutes"
    % ((time.process_time() - start_time) / 60.0)
)


# Begin training the neural network based on the above parameters
model, filename = runTokenNetwork4(
    trainX, testX, trainY, testY, learning_rate, batch_size, epochs, sequence_length, tokens
)


# filename = "token-c3-test-2019-07-24-22-25-05"
# Load a pretrained model and generate some music
loaded_model = loadModel(filename + ".h5")
loaded_model.summary()

# null token
tokens[(0,0)] = 0

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
    composition = startTokenizedNetworkRun(loaded_model, sample, timestep_resolution, 32)
    # Output to .midi file
    convertTokenizedDataToMidi2(composition, tokens, filename, timestep_resolution)    

