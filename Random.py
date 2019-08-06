import random
import pretty_midi
from DataGen import *

num_songs = 5

def genRandomData(num_songs_to_compose):
    for i in range(num_songs_to_compose):
        num_notes = random.randint(1, 1000)
        print("Num notes = %d" % num_notes)
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(0)
        counter = 0.0
        for j in range(num_notes):
            pitch = random.randint(0, 127)
            velocity = random.randint(0, 127)
            duration = random.random() * 3
            offset = random.random() * 3
            counter += offset
            # print("Note: Pitch = %d, Velocity = %d, Duration = %f, Offset = %f" % (pitch, velocity, duration, offset))
            n = pretty_midi.Note(velocity, pitch, counter, counter + duration )
            inst.notes.append(n)
        pm.instruments.append(inst)
        pm.write(getMidiRunName("Random"))

genRandomData(num_songs)