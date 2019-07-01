from DataGen import *
from DataAnalysis import *
from Network import *
from NetworkRun import *
from util.tensorflow_utils import *
import copy
import time

# debug timing how long dataset takes to load
start_time = time.process_time()

# Network parameters
learning_rate = 0.0003
batch_size = 64
epochs = 3
sequence_length = 9

dataset = "maestro"

# get the filepaths and load for all .midi files in the dataset
filepaths = getCleanedFilePaths(dataset)
midi_data, key_distributions = loadMidiData(filepaths)

# convert into python arrays andd dicts
data, vectors, timeScalars = getKerasData(midi_data, key_distributions)
del midi_data # no longer need original data, free the memory
# convert to normalized form for network training
processedData, processedLabels = processKerasData(data, timeScalars, sequence_length)
del data # no longer need unscaled data

# train / test split
trainX, testX, trainY, testY = genNetworkData(processedData, processedLabels)
del processedData
del processedLabels

print("Time taken to load dataset = %f minutes" % ((time.process_time() - start_time) / 60.0))

# Begin training the neural network based on the above parameters
model = runNetwork(
    trainX, testX, trainY, testY,
    sequence_length, learning_rate, batch_size, epochs
)



# Load a pretrained model and generate some music
loaded_model_name = "test-2019-07-02-00-01-28"
loaded_model = loadModel(loaded_model_name + ".h5")

# take some test data
tempData = copy.deepcopy(testX[30:40])
# Even when we pass in one sequence, the RNN expects the shape to be 3D
# e.g. shape of input must be at least (1, sequence_length, num_features)
sample = tempData[0:1]
# Use network to generate some notes
composition = startNetworkRun(loaded_model, sample, 9, 500)
# Output to .midi file
convertDataToMidi(composition, timeScalars, loaded_model_name)
