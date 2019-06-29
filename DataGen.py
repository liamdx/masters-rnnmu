import pretty_midi
import os
from os import listdir
from os.path import isfile, join
import random
import string
from difflib import SequenceMatcher
import numpy as np
import scipy.misc as smp
from matplotlib import pyplot as plt
from DataAnalysis import *


def remap(OldValue, OldMin, OldMax, NewMin, NewMax):
    OldRange = OldMax - OldMin
    if OldRange == 0:
        NewValue = NewMin
    else:
        NewRange = NewMax - NewMin
        NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    return NewValue


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


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

    for f in allfiles:
        if counter % 128 == 0:
            percentage = (counter / numFiles) * 100
            print("Processing Files : %f percent" % percentage)

        currentSimilarityRatio = 0
        currentSimilarityRatio = similar(lastfile, f)

        if currentSimilarityRatio < 0.8:
            cleanfiles.append(f)

        lastfile = f
        counter += 1

    return allfiles


def loadMidiData(cleanfiles):
    print("Loading Midifiles")
    midifiles = {}
    key_distributions = {}
    counter = 0
    for c in cleanfiles:
        if counter % 16 == 0:
            percentage = (counter / len(cleanfiles)) * 100
            print("Processing Files : %f percent" % percentage)
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

    for k, v in mididict.items():
        raw_notes = v.instruments[0].notes
        notes = []
        # not being used right now
        binaryKeyVector = getBinaryKeyVector(key_distributions[k])
        for note in raw_notes:
            noteTime = note.start
            noteDuration = note.end - note.start
            noteVelocity = note.velocity
            notePitch = note.pitch
            # do some stuff
            # 1st Start Time 2nd Duration of Note, 3rd Note Velocity, 4th Note Pitch
            notes.append((noteTime, noteDuration, noteVelocity, notePitch))
            # for normalisation in neural network
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


def processKerasData(data, timeScalars, sequenceSize):
    newNotes = []
    newLabels = []
    print("Converting to LSTM Format")
    counter = 0
    for i in range(len(data)):
        # for i in all songs
        if counter % 16 == 0:
            percentage = (counter / len(data)) * 100
            print("Processing Files for Keras : %f percent" % percentage)
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


def processKerasDataLastNote(data, timeScalars):
    processedData = []
    print("Printing Notes :) ")
    for i in range(len(data)):
        newNotes = []
        for note in data[i]:
            newNoteTime = float(remap(note[0], timeScalars[0], timeScalars[1], 0, 1))
            newNoteDuration = float(
                remap(note[1], timeScalars[2], timeScalars[3], 0, 1)
            )
            newNoteVelocity = float(remap(note[2], 0, 127, 0, 1))
            newNotePitch = float(remap(note[3], timeScalars[4], timeScalars[5], 0, 1))
            lastNote = newNotes[len(newNotes) - 1]
            newNotes.append(
                (
                    newNoteTime,
                    newNoteDuration,
                    newNoteVelocity,
                    newNotePitch,
                    lastNote[0],
                    lastNote[1],
                    lastNote[2],
                    lastNote[3],
                )
            )
        processedData.append(newNotes)
    return processedData


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


def convertDataToMidi(notes, timeScalars):
    mid = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    # inst.program = 0

    for n in notes:
        # remap output of neural network / normalise notes
        pitch = np.uint8(remap(n[0][3], 0, 1, timeScalars[4], timeScalars[5]))
        velocity = np.uint8(remap(n[0][2], 0, 1, 0, 127))
        start = remap(n[0][0], 0, 1, timeScalars[0], timeScalars[1])
        duration = remap(n[0][1], 0, 1, timeScalars[2], timeScalars[3])
        note = pretty_midi.Note(velocity, pitch, start, start + duration)
        inst.notes.append(note)

    mid.instruments.append(inst)
    mid.write("debug/debug.mid")
