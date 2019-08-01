import bimpy
import pygame
import time
import music21

# init UI context
ctx = bimpy.Context()
ctx.init(1200, 600, "Hello")

print("loading midifile")
# load in a midi file
path = "./classical/bach/bach_846.mid"
mf = music21.midi.MidiFile()
mf.open(str(path))
mf.read()
mf.close()
print("converting to stream")
# convert it to a stream
s = music21.midi.translate.midiFileToStream(mf)
# use stream player to play it
print("play back")
sp = music21.midi.realtime.StreamPlayer(s)
sp.play()

songs = []
songPlaying = False


while not ctx.should_close():
    with ctx:
        if bimpy.button("Button", bimpy.Vec2(90, 40)):
            print("Ya wee shite")
