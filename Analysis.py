from DataAnalysisHelpers import *
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

filepath = "dataset_analysis.xml"
dataset_raw = parseAnalysisData(filepath)
dataset_data = deepcopy(dataset_raw["feature_vector_file"]["data_set"])
del dataset_raw

processed_data = {}
feature_list = []
# make the data workable and not xml  
for song in dataset_data:
    name = song["data_set_id"].rsplit("\\",1)[1]
    name = name.rsplit(".", 1)[0]
    processed_data[name] = {}
    features = song["feature"]

    for od in features:
        lastKey = ""
        fname = od["name"]
        v = od["v"]
        if(type(v) == str):
            val = float(v)
            processed_data[name][fname] = val
        elif(type(v) == list):
            l = []
            for val in v:
                l.append(float(val))
            processed_data[name][fname] = l
        if fname not in feature_list:
            feature_list.append(fname)
del dataset_data


# features we are going to analyse
labels = {
    "Amount of Arpeggiation" : 1,
    "Average Note Duration" : 2,
    "Average Number of Simultaneous Pitches" : 3,
    "Variability of Note Durations" : 4,
    "Rhythmic Value Variability" : 5,
    "Metrical Diversity" : 6,
    "Variability in Rhythmic Value Run Lengths" : 7,
    "Chromatic Motion" : 8,
    "Contrary Motion" : 9,
    "Stepwise Motion" : 10,
    "Similar Motion" : 11,
    "Parallel Motion" : 12,
    "Variation of Dynamics" : 13,
    "Melodic Embellishments" : 14,
    "Repeated Notes" : 15,
}

final_data = [0] * int(len(labels) + 1)

counter = 0 # for meaning
for k, v in processed_data.items():
    counter += 1
    for l, index in labels.items():
        final_data[index-1] += v[l]
        
final_data = [x / len(processed_data) for x in final_data]

y_pos = np.arange(len(final_data))

fig = plt.figure()
plt.bar(y_pos, final_data, align='center', alpha=0.5, width = 0.3)
plt.xticks(y_pos, list(labels.keys()))
plt.ylabel('Mean Score')
plt.title('Music Descriptors Dataset')
plt.autoscale()
fig.autofmt_xdate()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        