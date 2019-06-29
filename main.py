from DataGen import *
from DataAnalysis import *
from Network import *
from util.tensorflow_utils import *
from scipy.ndimage.interpolation import shift

learning_rate = 0.001
batch_size = 64
epochs = 3
sequence_length = 15

# gen midi
filepaths = getCleanedFilePaths("maestro")
midi_data, key_distributions = loadMidiData(filepaths[0:20])

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

tempData = trainX[4:10]
preds = []

for i in range(2):
    # sample shape = (1,15,4)
    sample = tempData[0:1]
    print("Sample Shape = ")
    print(sample.shape)
    # pred shape = (1,4)
    pred = model.predict(sample)
    # shift the array down so that the last row = 0,0,0,0
    # then replace the null values with pred
    sample[0, len(sample) - 1] = pred[0]
    tempData[0] = sample[0]
    preds.append(pred)

convertDataToMidi(preds, timeScalars)

