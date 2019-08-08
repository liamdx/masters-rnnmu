from DataAnalysisHelpers import *
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import statistics

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



def plotFeatureComparison(organised_data, feature):
    print("Hello")
    # import statistics
    # numbers = [G[key] for key in G]
    # mean_ = statistics.mean(numbers)
    dat = organised_data[feature]
    plot_dat = {}
    for dk, dv in dat.items():
        plot_dat[dk] = {}
        numbers = [dv[key] for key in dv]
        d_std = statistics.stdev(numbers)
        d_mean = statistics.mean(numbers)
        plot_dat[dk]["std"] = d_std
        plot_dat[dk]["mean"] = d_mean

    fig, ax = plt.subplots()
    index = np.arange(1)
    bar_width = 0.15
    opacity = 0.8


    plt.title("%s comparison" % feature)
    rects1 = plt.bar(index, plot_dat["dataset"]["mean"], bar_width,
    alpha=opacity,
    color='r',
    label='Dataset',
    align="center",
    yerr=plot_dat["dataset"]["std"])

    rects2 = plt.bar(index + bar_width, plot_dat["generated"]["mean"], bar_width,
    alpha=opacity,
    color='g',
    label='Generated',
    align="center",
    yerr=plot_dat["generated"]["std"])

    rects3 = plt.bar(index + (bar_width * 2), plot_dat["random"]["mean"], bar_width,
    alpha=opacity,
    color='b',
    label='Random',
    align="center",
    yerr=plot_dat["random"]["std"])

    plt.legend()
    plt.show()

    return plot_dat
    
    

# features we are going to analyse
labels = {
    "Amount of Arpeggiation": 1,  # fast melodic runs within confines of a key
    "Average Note Duration": 2,  # Length of note
    "Average Number of Simultaneous Pitches": 3,  # how many notes are active at once
    "Variability of Note Durations": 4,  # how much does the note length vary
    # "Rhythmic Value Variability": 5,  # similar but more broad (e.g. overall rhythmic feel of the piece)
    "Metrical Diversity": 5,  # again similar
    "Chromatic Motion": 6,  # motion of melody is chromatic (one after the other e.g. c1 -> c#1 -> d1 ->d#1)
    "Contrary Motion": 7,  # motion of melody(s) run counter to one another (indicator of complexity)
    "Stepwise Motion": 8,  # motion of melody moves in steps of they key (e.g. c3->e3->g3 perfect c major)
    "Similar Motion": 9,  # motion of melody moves in same way, with different pitches (again indicator of complexityy)
    "Parallel Motion": 10,
    "Variation of Dynamics": 11,  # how much dynamics variation is there
    "Melodic Embellishments": 12,  # small melodic passages that slightly change or enhance a larger music phrase (complexity)
    "Repeated Notes": 13,  # number of notes that are repeated one after the other (indicator of low complexity)
}


training_analysis_filepath = "res/analysis_data/dataset_analysis.xml"
generated_analysis_filepath = "res/analysis_data/generated_analysis.xml" # generated_analysis.xml
random_analysis_filepath = "res/analysis_data/random_analysis.xml" # random_analysis.xml

dataset_raw = parseAnalysisData(training_analysis_filepath)
dataset_data = deepcopy(dataset_raw["feature_vector_file"]["data_set"])
generated_raw = parseAnalysisData(generated_analysis_filepath)
generated_data = deepcopy(generated_raw["feature_vector_file"]["data_set"])
random_raw = parseAnalysisData(random_analysis_filepath)
random_data = deepcopy(random_raw["feature_vector_file"]["data_set"])

del dataset_raw
del generated_raw
del random_raw

# list of all the descriptors we extract from the midi files
# needed for plotting
features = getFeatures(dataset_data)

dataset_final_data = formatData(dataset_data)
generated_final_data = formatData(generated_data)
random_final_data = formatData(random_data)

del dataset_data
del generated_data
del random_data

mean_dataset_data = aggregateData(dataset_final_data, labels)
mean_generated_data = aggregateData(generated_final_data, labels)
mean_random_data = aggregateData(random_final_data, labels)


all_data = {}
all_data["dataset"] = dataset_final_data
all_data["generated"] = generated_final_data
all_data["random"] = random_final_data


# should have mean, standard deviation
organised_data = {}
for k, v in labels.items():
    organised_data[k] = {}
    for dk, dv in all_data.items():
        organised_data[k][dk] = {}
        for sk, sv in dv.items():
            organised_data[k][dk][sk] =sv[k]
        
del all_data
del dataset_final_data
del random_final_data
del generated_final_data


for k, v in labels.items():
    x = plotFeatureComparison(organised_data, k)





# # Overall comparison of aggregated results
# # Data, Generated & Random
# fig, ax  = plt.subplots()
# index = np.arange(len(mean_dataset_data))
# bar_width = 0.35
# opacity = 0.8

# rects1 = plt.bar(index, mean_dataset_data, bar_width,
# alpha=opacity,
# color='b',
# label='Dataset',
# align="center")

# rects2 = plt.bar(index + bar_width, mean_generated_data, bar_width,
# alpha=opacity,
# color='g',
# label='Generated',
# align="center")

# rects2 = plt.bar(index + (bar_width * 2), mean_generated_data, bar_width,
# alpha=opacity,
# color='r',
# label='Generated',align="center")


# plt.xlabel('Descriptor')
# plt.ylabel('Scores')
# plt.title('MIR Descriptors')
# plt.xticks(index + bar_width, labels ,rotation=30)


# fig.tight_layout()
# # plt.tight_layout()
# plt.legend()
# plt.show()

