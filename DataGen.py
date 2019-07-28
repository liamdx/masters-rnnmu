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
from DataAnalysis import *
from util.rnnmu_utils import similar
from tqdm import tqdm
import collections
import operator


def getCleanedFilePaths(dataset):
    subfolders = [f.path for f in os.scandir(dataset) if f.is_dir()]
    allfiles = []
    print("Collecting filepath")
    for s in subfolders:
        currentFileArray = [f.path for f in os.scandir(s)]
        for f in currentFileArray:
            allfiles.append(f)

    cleanfiles = []
    numFiles = len(allfiles)
    counter = 0
    lastfile = ""

    # not returning "cleaned" files, do we need this check?
    print("Removing Duplicates")

    for f in tqdm(allfiles):

        currentSimilarityRatio = 0
        currentSimilarityRatio = similar(lastfile, f)

        if currentSimilarityRatio < 0.9:
            cleanfiles.append(f)

        lastfile = f
        counter += 1

    return allfiles


def loadMidiData(cleanfiles):
    print("Loading Midifiles")
    midifiles = {}
    key_distributions = {}
    counter = 0
    for c in tqdm(cleanfiles):
        try:
            mid = pretty_midi.PrettyMIDI(c)
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


def processKerasDataNormalized(data, timeScalars, sequenceSize):
    newNotes = []
    newLabels = []
    print("Converting to LSTM Format")
    counter = 0
    for i in tqdm(range(len(data))):
        # only take first 1000 notes of song (for now)
        # for j in range(len(data[i])):
        # for j in each note in each song
        for j in range(len(data[i])):
            if j < sequenceSize:
                j = sequenceSize
            note = data[i][j]
            # remap note time from min note time in set, max note time in set to 0, 1
            newNoteTime = float(remap(note[0], timeScalars[0], timeScalars[1], 0, 1))
            # remap note time from min note duration in set, max note duration in set to 0, 1
            newNoteDuration = float(
                remap(note[1], timeScalars[2], timeScalars[3], 0, 1)
            )
            # 0, 127 -> 0, 1
            newNoteVelocity = float(remap(note[2], 0, 127, 0, 1))
            # 0, 127 -> 0, 1
            newNotePitch = float(remap(note[3], timeScalars[4], timeScalars[5], 0, 1))
            newLabels.append(
                (newNoteTime, newNoteDuration, newNoteVelocity, newNotePitch)
            )
            priorNotes = []
            for k in range(sequenceSize):
                # get previous notes of length sequenceSize
                prevNote = data[i][j - (k + 1)]
                pNoteTime = float(
                    remap(prevNote[0], timeScalars[0], timeScalars[1], 0, 1)
                )
                pNoteDuration = float(
                    remap(prevNote[1], timeScalars[2], timeScalars[3], 0, 1)
                )
                pNoteVelocity = float(remap(prevNote[2], 0, 127, 0, 1))
                pNotePitch = float(remap(prevNote[3], 0, 127, 0, 1))
                priorNotes.append((pNoteTime, pNoteDuration, pNoteVelocity, pNotePitch))
            newNotes.append(priorNotes)
        counter += 1
    return (np.array(newNotes), np.array(newLabels))


def processKerasDataTokenized(data, sequence_size, num_timesteps, timestep_resolution, num_simultaneous_notes):
    processedData = [] 
    pitches_occurences, offsets_occurences, durations_occurences, velocities_occurences = countTokenOccurences(
        data
    )
    tokens, velocities = getTokens2(pitches_occurences, velocities_occurences, 500)

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

            note_token = getNoteToken(pitch, velocity, tokens, velocities)
            note_duration = int(timestep_resolution * duration)
            note_start = int((lastStartTime) * timestep_resolution)

            if (note_start + note_duration >= num_timesteps):
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
    for song in tqdm(y):
        song_train = []
        for i in range(len(song)):
            if(i < sequence_size):
                i = sequence_size
            
            lastNotes = []
            for j in range(sequence_size):
                lastNotes.append(song[i - j])
            song_train.append(lastNotes)
        x.append(song_train)
    
    print("...Done\n")
    # debug to not break compatibility
    return x, y , tokens


def getNoteToken(notePitch, noteVelocity, tokens, velocities):
    inv_tokens = dict((v,k) for k,v in tokens.items())
    if (notePitch, noteVelocity) in tokens.values():
        return (inv_tokens[(notePitch, noteVelocity)])
    else:
        # find the veocity with the minimum distance from current velocityy
        nearestVelocity = min(velocities, key=lambda x:abs(x-noteVelocity))
        return(inv_tokens[(notePitch, nearestVelocity)])


def finaliseTokenNetworkData(x,y):
    print("Finalising network data")
    X = np.concatenate(x)
    Y = np.vstack(y)
    return(X, Y)
def distance(a, b):
    if a > b:
        return a - b
    elif a < b:
        return b - a
    else:
        return 0


def getTokens(pitch_occurences, velocity_occurences):
    tokens = {}
    counter = 1
    for pitch in pitch_occurences:
        for velocity in velocity_occurences:
            currentCombination = (pitch, velocity)
            tokens[currentCombination] = counter
            counter += 1
    return(tokens)

def getTokens2(pitch_occurences, velocity_occurences, max_tokens):
    tokens = {}
    tokens[0] = (0,0)
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

    return(tokens, velocities)

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


def convertNormalizedDataToMidi(notes, timeScalars, model_name):
    print("Converting raw notes to MIDI")
    mid = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    # inst.program = 0
    lastNoteStart = 0
    for n in tqdm(notes):
        # remap output of neural network / normalise notes
        pitch = np.uint8(remap(n[0][3], 0, 1, timeScalars[4], timeScalars[5]))
        velocity = np.uint8(remap(n[0][2], 0, 1, 0, 127))
        offset = remap(n[0][0], 0, 1, timeScalars[0], timeScalars[1])
        duration = remap(n[0][1], 0, 1, timeScalars[2], timeScalars[3])
        start = lastNoteStart + offset
        note = pretty_midi.Note(velocity, pitch, start, start + duration)
        inst.notes.append(note)
        lastNoteStart = start

    mid.instruments.append(inst)
    mid.write(getMidiRunName(model_name))

def convertTokenizedDataToMidi(data, tokens, model_name, timestep_resolution):
    print("Converting raw notes to MIDI")
    mid = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    lastNoteStart, timestepCounter = (0, 0)
    # what notes are currently being played
    current_notes = {}
    timestep_resolution * data
    for timestep in data:
        for event in timestep:
            # if we encounter a new note 
            print("Event Type = ")
            print(type(event))
            if event not in current_notes:
                current_notes[event] = (timestepCounter, 1)
            else:
                current_notes[event][1] += 1
            # if any of the currently playing notes are not
            # being played at this timestep, remove it from current notes
            for note, timing in current_notes.items():
                if note not in timestep:
                    pitch, velocity = tokens[note]
                    start = timing[0] / timestep_resolution
                    duration = timing / timestep_resolution
                    new_note = pretty_midi.Note(velocity, pitch, start, start + duration)
                    inst.notes.append(new_note)
                    current_notes.pop(note)
        timestepCounter += 1
    
    mid.instruments.append(inst)
    mid.write(getMidiRunName(model_name))

def convertTokenizedDataToMidi2(data, tokens, model_name, timestep_resolution):
    print("Converting raw notes to MIDI")
    mid = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inv_tokens = dict((v,k) for k,v in tokens.items())
    normalizer = 1
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
                   currentToken = abs(int(note * normalizer))
                   if note >= 1:
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
                        notesToRemove.append(note)                   
        
            for note in notesToRemove:
                current_notes.pop(note)

            timestepCounter += 1
        
    
    if bool(current_notes) == True:
        for note, timing in current_notes.items():
            currentToken = abs(int(note * normalizer))
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
    mid.write(getMidiRunName(model_name))

    
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def getMidiRunName(model_name):
    directory = "debug/midi/" + model_name

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
