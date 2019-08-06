from DataAnalysisHelpers import *
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
# used for plot formatting
rcParams.update({"figure.autolayout": True})

def getFeatures(data):
    # make the data workable and not xml
    feature_list = []
    for song in dataset_data:
        features = song["feature"]
        for od in features:
            fname = od["name"]
            if fname not in feature_list:
                feature_list.append(fname)
    return(feature_list)
    

def formatData(data):
    # make the data workable and not xml
    new_data = {}
    for song in data:
        name = song["data_set_id"].rsplit("\\", 1)[1]
        name = name.rsplit(".", 1)[0]
        new_data[name] = {}
        features = song["feature"]

        for od in features:
            fname = od["name"]
            v = od["v"]
            if type(v) == str:
                val = float(v)
                new_data[name][fname] = val
            elif type(v) == list:
                l = []
                for val in v:
                    l.append(float(val))
                new_data[name][fname] = l
    return(new_data)


def aggregateData(formatted_data, labels):
    aggregated_data = [0] * int(len(labels) + 1)
    counter = 0  # for meaning
    for k, v in formatted_data.items():
        counter += 1
        for l, index in labels.items():
           aggregated_data[index - 1] += v[l]

    aggregated_data = [x / len(formatted_data) for x in aggregated_data]
    return(aggregated_data)


# features we are going to analyse
labels = {
    "Amount of Arpeggiation": 1,  # fast melodic runs within confines of a key
    "Average Note Duration": 2,  # Length of note
    "Average Number of Simultaneous Pitches": 3,  # how many notes are active at once
    "Variability of Note Durations": 4,  # how much does the note length vary
    "Rhythmic Value Variability": 5,  # similar but more broad (e.g. overall rhythmic feel of the piece)
    "Metrical Diversity": 6,  # again similar
    "Chromatic Motion": 7,  # motion of melody is chromatic (one after the other e.g. c1 -> c#1 -> d1 ->d#1)
    "Contrary Motion": 8,  # motion of melody(s) run counter to one another (indicator of complexity)
    "Stepwise Motion": 9,  # motion of melody moves in steps of they key (e.g. c3->e3->g3 perfect c major)
    "Similar Motion": 10,  # motion of melody moves in same way, with different pitches (again indicator of complexityy)
    "Parallel Motion": 11,
    "Variation of Dynamics": 12,  # how much dynamics variation is there
    "Melodic Embellishments": 13,  # small melodic passages that slightly change or enhance a larger music phrase (complexity)
    "Repeated Notes": 14,  # number of notes that are repeated one after the other (indicator of low complexity)
}


training_analysis_filepath = "res/analysis_data/dataset_analysis.xml"
generated_analysis_filepath = "res/analysis_data/dataset_analysis.xml"

dataset_raw = parseAnalysisData(training_analysis_filepath)
dataset_data = deepcopy(dataset_raw["feature_vector_file"]["data_set"])
generated_raw = parseAnalysisData(generated_analysis_filepath)
generated_data = deepcopy(generated_raw["feature_vector_file"]["data_set"])
del dataset_raw
del generated_raw

features = getFeatures(dataset_data)
dataset_final_data = formatData(dataset_data)
generated_final_data = formatData(generated_data)


mean_dataset_data = aggregateData(dataset_final_data, labels)
mean_generated_data = aggregateData(generated_final_data, labels)

y_pos = np.arange(len(mean_dataset_data))

fig = plt.figure()
plt.bar(y_pos, mean_dataset_data, align="center", alpha=0.5, width=0.3)
plt.xticks(y_pos, list(labels.keys()))
plt.ylabel("Mean Score")
plt.title("Music Descriptors Dataset")
plt.autoscale()
fig.autofmt_xdate()
