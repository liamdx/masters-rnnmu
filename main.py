from DataGen import *
from DataAnalysis import *
from Network import *
from NetworkRun import *
from util.tensorflow_utils import *
import copy

learning_rate = 0.0003
batch_size = 64
epochs = 3
sequence_length = 9

# gen midi
filepaths = getCleanedFilePaths("maestro")
midi_data, key_distributions = loadMidiData(filepaths[0:10])

# genImagesFromMidiData(midi_data)
# read it back in
# image_filepaths = getImageFilepaths("debug")

data, vectors, timeScalars = getKerasData(midi_data, key_distributions)
del midi_data
processedData, processedLabels = processKerasData(data, timeScalars, sequence_length)
del data

# Load data in to spyder
# RE RUNS RUN Me ONLY
model, trainX, trainY, testX, testY = runNetwork(
    processedData, processedLabels, sequence_length, learning_rate, batch_size, epochs
)


tempData = copy.deepcopy(trainX[8:10])
preds = []
sample = tempData[0:1]

for i in range(200):
    # sample shape = (1,15,4) 
    print("Sample Shape = ")
    print(sample.shape)
    print(sample)
    # pred shape = (1,4)
    pred = model.predict(sample)
    print("Pred Shape = ")
    print(pred.shape)
    print(pred[0])
    # shift the array down so that the last row = 0,0,0,0
    # then replace the null values with pred
    sample = np.roll(sample, -1, axis=1)
    sample[0,len(sample[0]) - 1,] = pred[0]
    print("sample after substitution\n")
    print(sample)
    preds.append(pred)

convertDataToMidi(preds, timeScalars)

