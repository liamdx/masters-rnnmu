# from DataGenerator import *
from DataGen import *
from DataAnalysis import *
from Network import *
from util.tensorflow_utils import *
from scipy.ndimage.interpolation import shift

learning_rate = 0.005 
batch_size = 64
epochs = 3
sequence_length = 25

# gen midi
filepaths = getCleanedFilePaths("maestro")
midi_data, key_distributions = loadMidiData(filepaths[0:200])

# genImagesFromMidiData(midi_data)
# read it back in
# image_filepaths = getImageFilepaths("debug")

data, vectors, timeScalars = getKerasData(midi_data, key_distributions)
processedData, processedLabels = processKerasData(data, timeScalars, sequence_length)

del(data, midi_data)

# Load data in to spyder


# RE RUNS RUN Me ONLY
model = runNetwork(processedData, processedLabels, sequence_length, learning_rate, batch_size, epochs)

tempData = processedData[1:5]
tempData = tempData.tolist()

# final notes which will be sent to a midi file
preds = []
#notes = []
for i in range(500):
    pred = model.predict(tempData[0:1])
    sample = tempData[0]
    # sample = shift(sample, 1, cval=np.NaN)
    sample[len(sample) - 1] = pred
    tempData[0] = sample
    preds.append(pred)
#    notes.append(pred)
#    sample.pop(0)
#    sample.append(pred)
convertDataToMidi(preds, timeScalars)  
# genNetworkData(imageFilepaths, labels)
# processedMidi = getMidiDataFromImages(imageFilepaths)
# dumpMidi(processedMidi)