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


def processKerasDataTokenized(data, sequence_size):
    # 128 resolution for notes,velocity, 128 separate durations and offsets remembered
    # need to convert existing data to tokens.
    # also need to recreate the prior notes for tokens too.
    max_tokens = 512
    # tokens_remaining = copy.copy(max_tokens)
    pitches_occurences, offsets_occurences, durations_occurences, velocities_occurences = countTokenOccurences(
        data
    )
    tokens = getTokens(offsets_occurences, durations_occurences)
    offsetTokens = tokens[256:384]
    durationTokens = tokens[384:512]
    new_data = []
    labels = []
    print("\nTokenizing Data\n")
    for song in tqdm(data):
        for j in range(len(song)):
            if j < sequence_size:
                j = sequence_size
            note = song[j]
            n = []
            # pitch is just pitch, same with velocity
            pitch = note[3]
            velocity = note[2]
            raw_offset = note[0]
            raw_duration = note[1]
            n.append(pitch)
            n.append(128 + velocity)

            closest_offset = min(offsetTokens, key=lambda x: distance(x[1], raw_offset))
            n.append(tokens[int(closest_offset[0])][0])
            closest_duration = min(
                durationTokens, key=lambda x: distance(x[1], raw_duration)
            )
            n.append(tokens[int(closest_duration[0])][0])

            labels.append(n)

            x = []
            for k in range(sequence_size):
                # get previous notes of length sequenceSize
                prevNote = song[j - (k + 1)]
                pitch = prevNote[3]
                velocity = prevNote[2]
                raw_offset = prevNote[0]
                raw_duration = prevNote[1]
                x.append(pitch)
                x.append(128 + velocity)

                closest_offset = min(
                    offsetTokens, key=lambda x: distance(x[1], raw_offset)
                )
                x.append(tokens[int(closest_offset[0])][0])
                closest_duration = min(
                    durationTokens, key=lambda x: distance(x[1], raw_duration)
                )
                x.append(tokens[int(closest_duration[0])][0])

            new_data.append(x)
            # print("Raw offset = %f - Closest Offset Found = %f" % (raw_offset, closest_offset[0]))
    return new_data, labels, tokens
    # get the closest time in the tokens;


def distance(a, b):
    if a > b:
        return a - b
    elif a < b:
        return b - a
    else:
        return 0


def getTokens(offset_occurences, duration_occurences):
    # offset tokenization
    c_offset_occurences = copy.deepcopy(offset_occurences)
    offsets = []
    for i in range(128):
        currentMaxKey = max(c_offset_occurences.items(), key=operator.itemgetter(1))[0]
        offsets.append(currentMaxKey)
        c_offset_occurences.pop(currentMaxKey)

    # duration tokenization
    c_duration_occurences = copy.deepcopy(duration_occurences)
    durations = []
    for i in range(128):
        currentMaxKey = max(c_duration_occurences.items(), key=operator.itemgetter(1))[
            0
        ]
        durations.append(currentMaxKey)
        c_duration_occurences.pop(currentMaxKey)

    del c_duration_occurences
    del c_offset_occurences

    tokens = []
    for i in range(128):
        tokens.append((i, i))
    for i in range(128):
        tokens.append((128 + i, i))
    for i in range(128):
        tokens.append((256 + i, offsets[i]))
    for i in range(128):
        tokens.append((384 + i, durations[i]))

    return tokens


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

def convertTokenizedDataToMidi(notes, tokens, model_name):
    print("Converting raw notes to MIDI")
    mid = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    lastNoteStart = 0
    raw_notes = []
    for token in tqdm(notes):
        for msg in token[0]:
            raw_notes.append(tokens[int(msg)][1])
        
    new_notes = list(chunks(raw_notes, 4))
    
    for note in new_notes:
        pitch = note[0]
        velocity = note[1]
        duration = note[3]
        start = lastNoteStart + note[2] # offset
        n = pretty_midi.Note(velocity, pitch, start, start + duration)
        inst.notes.append(n)
    
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
