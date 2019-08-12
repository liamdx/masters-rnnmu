import random
import pretty_midi
from DataGen import *

num_songs = 50
filepaths = getCleanedFilePaths("classical")


def genRandomData(num_songs_to_compose):
    # load in midi data for occurence counting
    midi_data, key_distributions = loadMidiData(
        filepaths, 36, 84
    )
    # convert into python arrays andd dicts
    data, vectors, timeScalars = getKerasData(midi_data, key_distributions)
    del midi_data  # no longer need original data, free the memory

    pitches_occurences, offsets_occurences, durations_occurences, velocities_occurences = countTokenOccurences(
            data
    )
    
    # using real values caused analysis software to crash
    minDuration = min(list(durations_occurences.keys()))
    maxDuration = 8
    # cap offset and duration to 8 seconds 
    minOffset = min(list(offsets_occurences.keys()))
    maxOffset = 8

    minPitch = min(list(pitches_occurences.keys()))
    maxPitch = max(list(pitches_occurences.keys()))

    minVelocity = min(list(velocities_occurences.keys()))
    maxVelocity = max(list(velocities_occurences.keys()))


    for i in range(num_songs_to_compose):
        num_notes = random.randint(1, 1000)
        print("Num notes = %d" % num_notes)
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(0)
        counter = 0.0
        for j in range(num_notes):
            pitch = random.randint(minPitch, maxPitch)
            velocity = random.randint(minVelocity, maxVelocity)
            duration = remap(random.random(), 0.0, 1.0, minDuration, maxDuration) 
            offset = remap(random.random(), 0.0, 1.0, minOffset, maxOffset) 
            counter += offset
            # print("Note: Pitch = %d, Velocity = %d, Duration = %f, Offset = %f" % (pitch, velocity, duration, offset))
            n = pretty_midi.Note(velocity, pitch, counter, counter + duration )
            inst.notes.append(n)
        pm.instruments.append(inst)
        pm.write(getMidiRunName("Random"))

genRandomData(num_songs)