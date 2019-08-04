import pretty_midi
import librosa.display
import xmltodict


def getInstrumentType(instrument):
    programId = instrument.program
    type = ""
    if programId <= 7:
        type = "Piano"
    elif programId > 7 and programId <= 15:
        type = "Percussion"
    elif programId > 15 and programId <= 23:
        type = "Organ"
    elif programId > 23 and programId <= 31:
        type = "Guitar"
    elif programId > 31 and programId <= 39:
        type = "Bass"
    elif programId > 39 and programId <= 47:
        type = "Strings"
    elif programId > 47 and programId <= 55:
        type = "Ensemble"
    elif programId > 55 and programId <= 63:
        type = "Brass"
    elif programId > 63 and programId <= 71:
        type = "Reed"
    elif programId > 71 and programId <= 79:
        type = "Pipe"
    elif programId > 79 and programId <= 87:
        type = "Synth Lead"
    elif programId > 87 and programId <= 95:
        type = "Synth Pad"
    elif programId > 95 and programId <= 103:
        type = "Synth Effects"
    elif programId > 103 and programId <= 111:
        type = "World Instruments"
    elif programId > 111 and programId <= 119:
        type = "Percussive"
    else:
        type = "na"
    return type


def plot_piano_roll(pm, start_pitch=0, end_pitch=127, fs=120):
    # Use librosa's specshow function for displaying the piano roll
    return librosa.display.specshow(
        pm.get_piano_roll(fs)[start_pitch:end_pitch],
        hop_length=1,
        sr=fs,
        x_axis="time",
        y_axis="cqt_note",
        fmin=pretty_midi.note_number_to_hz(start_pitch),
    )


def parseAnalysisData(filepath):
    with open(filepath) as fd:
        doc = xmltodict.parse(fd.read())
        return doc
    return FileNotFoundError
