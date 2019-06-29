# from DataGenerator import *
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
# tempData = tempData.tolist()

# final notes which will be sent to a midi file
preds = []

for i in range(2):
    sample = tempData[0:1]
    pred = model.predict(sample)
    # sample = shift(sample, 1, cval=np.NaN)
    sample[0, len(sample) - 1] = pred[0]
    tempData[0] = sample[0]
    preds.append(pred)
#    notes.append(pred)
#    sample.pop(0)
#    sample.append(pred)
convertDataToMidi(preds, timeScalars)
# genNetworkData(imageFilepaths, labels)
# processedMidi = getMidiDataFromImages(imageFilepaths)
# dumpMidi(processedMidi)
