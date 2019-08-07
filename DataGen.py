import pretty_midi
import os
from os import listdir
from os.path import isfile, join
import random
import string
import copy
import numpy as np
import scipy.misc as smp
from matplotlib import pyplot as plt
from DataAnalysisHelpers import *
from util.rnnmu_utils import similar
from tqdm import tqdm
import collections
import operator
import mido
from collections import OrderedDict
import bisect

def getCleanedFilePaths(dataset):
    subfolders = [f.path for f in os.scandir(dataset) if f.is_dir()]
    allfiles = []
    print("Collecting filepath")
    for s in subfolders:
        currentFileArray = [f.path for f in os.scandir(s)]
        for f in currentFileArray:
            allfiles.append(f)

    cleanfiles = []
    counter = 0
    lastfile = ""

    # not returning "cleaned" files, do we need this check?
    print("Removing Duplicates")

    for f in tqdm(allfiles):
        currentSimilarityRatio = similar(lastfile, f)

        if "_format0" not in f:
            cleanfiles.append(f)

        lastfile = f
        counter += 1

    return cleanfiles


# pretty midi object
def limitMidiOctaves(pm, min, max):
    for note in pm.instruments[0].notes:
        while note.pitch < min or note.pitch > max:
            if note.pitch > max:
                # subtract an octave
                note.pitch = note.pitch - 12
            elif note.pitch < min:
                # add an octave
                note.pitch = note.pitch + 12


def loadMidiData(cleanfiles, minOctave, maxOctave):
    print("Loading Midifiles")
    midifiles = {}
    key_distributions = {}
    counter = 0
    for c in tqdm(cleanfiles):
        try:
            mid = pretty_midi.PrettyMIDI(c)
            limitMidiOctaves(mid, minOctave, maxOctave)
            midifiles[c] = mid
            total_velocity = sum(sum(mid.get_chroma()))
            key_weight_distribution = [
                sum(semitone) / total_velocity for semitone in mid.get_chroma()
            ]
            # key_distributions[c[c.rfind(os.path.sep) + 1:c.rfind(".")] + "_"] = key_weight_distribution
            key_distributions[c] = key_weight_distribution

        except OSError:
            print("Midifile: " + c + " could not be loaded)")
        except ValueError:
            print("Midifile: " + c + " could not be loaded)")
        except IndexError:
            print("Midifile: " + c + " could not be loaded)")
        except KeyError:
            print("Midifile: " + c + " could not be loaded)")
        except mido.KeySignatureError:
            print("Midifile: " + c + " could not be loaded)")
        except EOFError:
            print("Midifile: " + c + " could not be loaded)")

        counter += 1

    return midifiles, key_distributions


def getKerasData(mididict, key_distributions):
    binaryVecs = []
    data = []

    earliestNoteTime, latestNoteTime = 0, 0
    shortestNoteDuration, longestNoteDuration = 0, 0
    lowestNotePitch, highestNotePitch = 0, 1

    for k, v in tqdm(mididict.items()):
        raw_notes = v.instruments[0].notes
        notes = []
        # not being used right now
        binaryKeyVector = getBinaryKeyVector(key_distributions[k])
        lastNote = raw_notes[0]
        for note in raw_notes:
            noteTime = note.start - lastNote.start
            noteDuration = note.end - note.start
            noteVelocity = note.velocity
            notePitch = note.pitch
            # do some stuff
            # 1st Start Time 2nd Duration of Note, 3rd Note Velocity, 4th Note Pitch
            notes.append((noteTime, noteDuration, noteVelocity, notePitch))
            # for normalisation in neural network

            lastNote = note

            if noteTime >= latestNoteTime:
                latestNoteTime = noteTime
            elif noteTime <= earliestNoteTime:
                earliestNoteTime = noteTime
            # duration
            if noteDuration >= longestNoteDuration:
                longestNoteDuration = noteDuration
            elif noteDuration <= shortestNoteDuration:
                shortestNoteDuration = noteDuration
            # pitch
            if notePitch >= highestNotePitch:
                highestNotePitch = notePitch
            elif notePitch <= lowestNotePitch:
                lowestNotePitch = notePitch

        data.append(notes)
        binaryVecs.append(key_distributions[k])

    return (
        data,
        binaryVecs,
        (
            earliestNoteTime,
            latestNoteTime,
            shortestNoteDuration,
            longestNoteDuration,
            lowestNotePitch,
            highestNotePitch,
        ),
    )


def processKerasDataMethodA(data, timeScalars, sequenceSize):
    newNotes = []
    newLabels = []
    print("Converting to LSTM Format")
    counter = 0

    pitches_occurences, offsets_occurences, durations_occurences, velocities_occurences = countTokenOccurences(
        data
    )


    tokens, token_cutoffs = getTokensA(pitches_occurences, durations_occurences, offsets_occurences, 500)
    inv_tokens = invertDictionary(tokens)
    pitch_tokens = invertDictionary(slice_odict(inv_tokens,1, token_cutoffs["last_pitch"] + 1))
    velocity_tokens = invertDictionary(slice_odict(inv_tokens,token_cutoffs["last_pitch"] + 1, token_cutoffs["last_velocity"] +1))
    duration_tokens = invertDictionary(slice_odict(inv_tokens,token_cutoffs["last_velocity"] + 1, token_cutoffs["last_duration"] + 1))
    offset_tokens = invertDictionary(slice_odict(inv_tokens,token_cutoffs["last_duration"]+1, token_cutoffs["last_offset"] + 1))


    for i in tqdm(range(len(data))):
    #     # only take first 1000 notes of song (for now)
    #     # for j in range(len(data[i])):
    #     # for j in each note in each song
        for j in range(len(data[i])):
            if j < sequenceSize:
               j = sequenceSize
            note = data[i][j]

            note_time = note[0]
            note_duration = note[1]
            note_velocity = note[2]
            note_pitch = note[3]

            # data[min(data.keys(), key=lambda k: abs(k-num))]) 
            token_offset = getNoteTokenA(note_time, offset_tokens)
            # token_offset = bisect.bisect_left(list(offset_tokens.keys()), note_time)
            token_duration = getNoteTokenA(note_duration, duration_tokens)
            #token_duration = bisect.bisect_left(list(duration_tokens.keys()), note_duration)
            token_velocity = getNoteTokenA(note_velocity, velocity_tokens)
            # token_velocity = bisect.bisect_left(list(velocity_tokens.keys()), note_velocity)
            token_pitch = getNoteTokenA(note_pitch, pitch_tokens)
            # token_pitch = bisect.bisect_left(list(pitch_tokens.keys()), note_pitch)


            newLabels.append((token_pitch, token_velocity, token_duration, token_offset))

            priorNotes = []
            for k in range(sequenceSize):
                pnote = data[i][j - (k + 1)]
                pnote_time = pnote[0]
                pnote_duration = pnote[1]
                pnote_velocity = pnote[2]
                pnote_pitch = pnote[3]

                ptoken_offset = getNoteTokenA(pnote_time, offset_tokens)
                ptoken_duration = getNoteTokenA(pnote_duration, duration_tokens)
                ptoken_velocity = getNoteTokenA(pnote_velocity, velocity_tokens)
                ptoken_pitch = token_pitch = getNoteTokenA(pnote_pitch, pitch_tokens)
                priorNotes.append((ptoken_pitch, ptoken_velocity , ptoken_duration, ptoken_offset))
            
            newNotes.append(priorNotes)

    return (np.array(newNotes), np.array(newLabels), tokens, token_cutoffs)



def invertDictionary(old_dict):
    inverted_dict = OrderedDict([[v,k] for k,v in old_dict.items()])
    return(inverted_dict)


def processKerasDataMethodB(
    data, sequence_size, num_timesteps, timestep_resolution, num_simultaneous_notes
):
    processedData = []
    pitches_occurences, offsets_occurences, durations_occurences, velocities_occurences = countTokenOccurences(
        data
    )
    tokens, velocities = getTokensB(pitches_occurences, velocities_occurences, 500)

    print("\nConverting data to final network representation\n")

    y = []

    for song in tqdm(data):
        lastStartTime = 0
        song_data = np.zeros((num_timesteps, num_simultaneous_notes), dtype=int)

        for note in song:
            time = note[0]
            duration = note[1]
            velocity = note[2]
            pitch = note[3]
            lastStartTime += time

            note_token = getNoteTokenB(pitch, velocity, tokens, velocities)
            note_duration = int(timestep_resolution * duration)
            note_start = int((lastStartTime) * timestep_resolution)

            if note_start + note_duration >= num_timesteps:
                break
            else:
                # place the note
                for i in range(note_duration):
                    for j in range(num_simultaneous_notes):
                        value = song_data[note_start + i][j]
                        if value == 0:
                            song_data[note_start + i][j] = note_token
                            break
        y.append(song_data)

    x = []
    new_y = []
    for song in tqdm(y):
        song_train = []
        song_label = []
        for i in range(len(song)):
            if i < sequence_size:
                i = sequence_size
            song_label.append(song[i])
            lastNotes = []
            for j in range(sequence_size):
                lastNotes.append(song[i - (j + 1)])
            song_train.append(lastNotes)
        x.append(song_train)
        new_y.append(song_label)
    
    del y

    print("...Done\n")
    # debug to not break compatibility
    return x, new_y, tokens


def getNoteTokenB(notePitch, noteVelocity, tokens, velocities):
    inv_tokens = dict((v, k) for k, v in tokens.items())
    if (notePitch, noteVelocity) in tokens.values():
        return inv_tokens[(notePitch, noteVelocity)]
    else:
        # find the veocity with the minimum distance from current velocityy
        nearestVelocity = min(velocities, key=lambda x: abs(x - noteVelocity))
        return inv_tokens[(notePitch, nearestVelocity)]


def getNoteTokenA(value, dictionary):
    if value in list(dictionary.keys()):
        return(dictionary[value])
    else:
        lowestDifference = 3000
        for k, v in dictionary.items():
            diff = abs(k - value)
            if diff < lowestDifference:
                lowestDifference = k
        return(dictionary[lowestDifference])
        



def finaliseTokenNetworkData(x, y):
    print("Finalising network data")
    X = np.concatenate(x)
    Y = np.vstack(y)
    return (X, Y)


def distance(a, b):
    if a > b:
        return a - b
    elif a < b:
        return b - a
    else:
        return 0


def slice_odict(odict, start=None, end=None):
    return OrderedDict([
        (k,v) for (k,v) in odict.items() 
        if k in (list(odict.keys())[start:end])
    ])

def getTokensA(pitches_occurences, durations_occurences, offsets_occurences, max_tokens):
    tokens = OrderedDict()
    tokens[0] = 0
    counter = 1
    
    for pitch, occurences in pitches_occurences.items():
        if pitch not in tokens.keys():
            tokens[pitch] = counter
            counter += 1
    
    last_pitch = counter - 1

    tokens[32] = counter
    counter += 1
    tokens[45] = counter
    counter += 1 
    tokens[90] = counter
    counter += 1
    tokens[120] = counter
    counter += 1
    
    last_velocity = counter - 1
    
    remaining_tokens = int(max_tokens - (counter + 1)) 
    current_counter = counter
    # durations
    durations = list(durations_occurences.keys())
    i = 0

    canEmplace = True
    while canEmplace:
        currentDuration = abs(durations[i])
        i += 1
        if currentDuration <= 1.0:
            if currentDuration not in tokens:
                tokens[currentDuration] = counter
                counter += 1
            if len(tokens) >= current_counter + (remaining_tokens / 2):
                canEmplace = False
        
    last_duration = counter - 1
    current_counter = counter
    canEmplace = True
    # offsets
    i = 0
    offsets = list(offsets_occurences.keys())
    while canEmplace:
        currentOffset = abs(offsets[i])
        i += 1
        if currentOffset <= 2.0:
            if currentOffset not in tokens:
                tokens[currentOffset] = counter
                counter += 1
            if len(tokens) >= current_counter + (remaining_tokens / 2):
                canEmplace = False

        

    last_offset = counter - 1 

    token_cutoffs = {}
    token_cutoffs["last_pitch"] = last_pitch
    token_cutoffs["last_velocity"] = last_velocity
    token_cutoffs["last_duration"] = last_duration
    token_cutoffs["last_offset"] = last_offset

    return tokens, token_cutoffs


def getTokensB(pitch_occurences, velocity_occurences, max_tokens):
    tokens = {}
    tokens[0] = (0, 0)
    maxVelocitiesPerPitch = 4
    counter = 1
    velocities = []
    for pitch in pitch_occurences:
        for j in range(maxVelocitiesPerPitch):
            if counter >= max_tokens:
                break
            else:
                tokens[counter] = (int(pitch), int(128 / (j + 1)))
                counter += 1

    for i in range(maxVelocitiesPerPitch):
        velocities.append(int(128 / (j + 1)))

    return (tokens, velocities)


def countTokenOccurences(data):
    note_pitches = collections.OrderedDict()
    note_offsets = collections.OrderedDict()
    note_durations = collections.OrderedDict()
    note_velocities = collections.OrderedDict()

    print("Counting Occurences for Tokenization")
    for song in data:

        for note in song:
            pitch = note[3]
            velocity = note[2]
            duration = note[1]
            offset = note[0]

            if pitch not in note_pitches:
                note_pitches[pitch] = 1
            else:
                note_pitches[pitch] += 1

            if velocity not in note_velocities:
                note_velocities[velocity] = 1
            else:
                note_velocities[velocity] += 1

            if duration not in note_durations:
                note_durations[duration] = 1
            else:
                note_durations[duration] += 1

            if offset not in note_offsets:
                note_offsets[offset] = 1
            else:
                note_offsets[offset] += 1

    return note_pitches, note_offsets, note_durations, note_velocities


def getBinaryKeyVector(key_distribution):
    binaryVector = []

    maxWeightIndex = key_distribution.index(max(key_distribution))
    tmp = key_distribution
    tmp.remove(max(key_distribution))
    secondWeight = max(tmp)
    secondWeightIndex = key_distribution.index(secondWeight)

    for i in range(12):
        if i == maxWeightIndex or i == secondWeightIndex:
            binaryVector.append(1)
        else:
            binaryVector.append(0)

    return binaryVector


def remap(OldValue, OldMin, OldMax, NewMin, NewMax):
    OldRange = OldMax - OldMin
    if OldRange == 0:
        NewValue = NewMin
    else:
        NewRange = NewMax - NewMin
        NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    return NewValue


def convertMethodADataToMidi(notes, tokens, token_cutoffs, model_name):
    print("Converting raw notes to MIDI")
    mid = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    # inst.program = 0
    lastNoteStart = 0
    for note in tqdm(notes):
        # remap output of neural network / normalise notes
        pitch_token = clamp(int(note[0][0]), 0, len(tokens))
        velocity_token  = clamp(int(note[0][1]), 0, len(tokens))
        duration_token = clamp(int(note[0][2]), 0 , len(tokens))
        offset_token = clamp(int(note[0][3]),0, len(tokens))
        
        pitch_token = int(clamp(pitch_token, 1, token_cutoffs["last_pitch"]))
        velocity_token = int(clamp(offset_token, token_cutoffs["last_pitch"] + 1, token_cutoffs["last_velocity"]))
        velocity_token = int(clamp(offset_token, token_cutoffs["last_velocity"] + 1, token_cutoffs["last_duration"]))
        velocity_token = int(clamp(offset_token, token_cutoffs["last_duration"] + 1, token_cutoffs["last_offset"]))
        
        pitch = np.uint8(tokens[pitch_token])
        offset = tokens[offset_token]
        duration = tokens[duration_token]
        velocity = np.uint8(tokens[velocity_token])

        start = lastNoteStart + offset
        pmnote = pretty_midi.Note(velocity, pitch, start, start + duration)
        inst.notes.append(pmnote)
        lastNoteStart = start

    mid.instruments.append(inst)
    cleanMidiA(mid)
    mid.write(getMidiRunName(model_name))

def convertMethodBDataToMidi(data, tokens, model_name, timestep_resolution):
    print("Converting raw notes to MIDI")
    mid = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    timestepCounter = 0
    # what notes are currently being played
    current_notes = {}
     # unnormalize the data at each step
    for d in tqdm(data):
        for timestep in d:
            # for all notes in this timestep
            rounded_timestep = [round(x) for x in timestep]
            for msg in rounded_timestep:
                # if we encounter a new note start counting its duration
                if msg not in current_notes:
                    current_notes[msg] = []
                    current_notes[msg].append(timestepCounter)
                    current_notes[msg].append(1)
                else:
                    current_notes[msg][1] = current_notes[msg][1] + 1
            # if any of the currently playing notes are not
            # being played at this timestep, remove it from current notes7
            notesToRemove = []
            for note, timing in current_notes.items():
                if note not in rounded_timestep:
                    # print("Logic is sound, this is being called :) ")
                    currentToken = abs(int(note))
                    if note >= 1:
                        pitch, velocity = tokens[currentToken]
                        if pitch > 127:
                            pitch = np.uint8(127)
                        if velocity > 127:
                            velocity = np.uint8(127)
                        start = timing[0] / timestep_resolution
                        duration = timing[1] / timestep_resolution
                        # print("Note start time = %d, note duration = %d" % (start, duration))
                        new_note = pretty_midi.Note(
                            velocity, pitch, start, start + duration
                        )
                        inst.notes.append(new_note)
                        notesToRemove.append(note)

            for note in notesToRemove:
                current_notes.pop(note)

            timestepCounter += 1

    if bool(current_notes) == True:
        for note, timing in current_notes.items():
            currentToken = abs(int(note))
            if currentToken != 0:
                pitch, velocity = tokens[currentToken]
                if pitch > 127:
                    pitch = np.uint8(127)
                if velocity > 127:
                    velocity = np.uint8(127)
                start = timing[0] / timestep_resolution
                duration = timing[1] / timestep_resolution
                # print("Note start time = %d, note duration = %d" % (start, duration))
                new_note = pretty_midi.Note(velocity, pitch, start, start + duration)
                inst.notes.append(new_note)

    del current_notes
    print("total number of notes = %d" % len(inst.notes))
    mid.instruments.append(inst)
    cleanMidiB(mid)
    mid.write(getMidiRunName(model_name))


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i : i + n]

def cleanMidiB(midi):
    print("Cleaning Midi..")
    pitch_randomiser = random.randint(-11, 11)
    for instrument in midi.instruments:
        for note in instrument.notes:
            note.pitch += pitch_randomiser
            note.start = note.start * 2
            note.end = note.end * 2

def cleanMidiA(midi):
    print("Cleaning Midi..")
    pitch_randomiser = random.randint(-11, 11)
    for instrument in midi.instruments:
        for note in instrument.notes:
            note.pitch += pitch_randomiser
        

def getMidiRunName(model_name):
    directory = "res/midi/" + model_name

    if not os.path.exists(directory):
        os.makedirs(directory)

    files = [f.path for f in os.scandir(directory)]
    existingFiles = len(files)
    if existingFiles == 0:
        ret = directory + "/1.mid"
        return ret
    else:
        ret = directory + "/" + str(existingFiles + 1) + ".mid"
        return ret
