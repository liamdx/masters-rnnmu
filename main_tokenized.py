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
learning_rate = 0.01
batch_size = 64
epochs = 2
sequence_length = 240
dataset = "classical"


# get the filepaths and load for all .midi files in the dataset
filepaths = getCleanedFilePaths(dataset)
midi_data, key_distributions = loadMidiData(filepaths[0:10])

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

trainX, testX, trainY, testY = genTokenizedNetworkData(processedData, processedLabels, 0.25)
del processedData
del processedLabels

print(
    "Time taken to load dataset = %d minutes"
    % ((time.process_time() - start_time) / 60.0)
)
# Begin training the neural network based on the above parameters
model, filename = runTokenNetwork2(
    trainX, testX, trainY, testY, learning_rate, batch_size, epochs, sequence_length
)


# Load a pretrained model and generate some music
loaded_model = loadModel(filename + ".h5")
loaded_model.summary()

tokens[(0,0)] = 0

# take some test data
tempData = copy.deepcopy(testX)

prediction = loaded_model.predict(tempData[3840:7860])

# how many compositions should we produce?
for i in range(1):
    upperbound = len(tempData)
    bounds = []
    for j in range(2):
        bounds.append(random.randint(0, upperbound))
    bounds.sort()
    print(bounds)
    sample = tempData[bounds[0] : bounds[0] + 1]
    # Use network to generate some notes
    composition = startTokenizedNetworkRun(loaded_model, sample, sequence_length, 1)
    # Output to .midi file
    convertTokenizedDataToMidi2(composition, tokens, filename, 60)    

