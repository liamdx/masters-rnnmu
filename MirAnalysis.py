from DataAnalysisHelpers import *
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import matplotlib.ticker as ticker
import numpy as np
from matplotlib import rcParams
import statistics
from DataGen import *
import pandas as pd


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

def plotOverallDifferenceComparison(organised_data):
    means = {}
    stds = {} 

    for feature, values in organised_data.items():
        dat = organised_data[feature]
        plot_dat = {}
        for dk, dv in dat.items():
            plot_dat[dk] = {}
            numbers = [dv[key] for key in dv]
            d_std = statistics.stdev(numbers)
            d_mean = statistics.mean(numbers)
            d_max = max(numbers)
            d_min = min(numbers)
            plot_dat[dk]["std"] = remap(d_std, d_min, d_max, 0, 1)
            plot_dat[dk]["mean"] = remap(d_mean, d_min, d_max, 0, 1)
        
        means[feature] = []
        stds[feature] = []
        # Human vs Method A
        means[feature].append(abs(plot_dat["dataset"]["mean"] - plot_dat["methodA"]["mean"]))
        stds[feature].append(abs(plot_dat["dataset"]["std"] - plot_dat["methodA"]["std"]))
        # Human vs Method B
        means[feature].append(abs(plot_dat["dataset"]["mean"] - plot_dat["methodB"]["mean"]))
        stds[feature].append(abs(plot_dat["dataset"]["std"] - plot_dat["methodB"]["std"]))
        # Human vs Random
        means[feature].append(abs(plot_dat["dataset"]["mean"] - plot_dat["random"]["mean"]))
        stds[feature].append(abs(plot_dat["dataset"]["std"] - plot_dat["random"]["std"]))


    fig, ax = plt.subplots()
    df = pd.DataFrame(means).T
    df.plot(kind='bar', ax=ax)
    ax.legend(["Method A", "Method B", "Random"]);
    ax.set_xlabel("MIR Descriptor")
    ax.set_ylabel("Mean % Difference from Human Compositions")
    
    plt.savefig("res/plots/overall_mean_difference.png")

    fig, ax = plt.subplots()
    df = pd.DataFrame(stds).T
    df.plot(kind='bar', ax=ax)
    ax.legend(["Method A", "Method B", "Random"]);
    ax.set_xlabel("MIR Descriptor")
    ax.set_ylabel("% Deviation from Human Compositions")

    plt.savefig("res/plots/overall-deviation-difference.png")


def plotFeatureAgainstHumanComparison(organised_data, feature):
    dat = organised_data[feature]
    plot_dat = {}
    means = []
    stds = [] 
    
    for dk, dv in dat.items():
        plot_dat[dk] = {}
        numbers = [dv[key] for key in dv]
        d_std = statistics.stdev(numbers)
        d_mean = statistics.mean(numbers)
        plot_dat[dk]["std"] = d_std
        plot_dat[dk]["mean"] = d_mean
        
    # Human vs Method A
    means.append(abs(plot_dat["dataset"]["mean"] - plot_dat["methodA"]["mean"]))
    stds.append(abs(plot_dat["dataset"]["std"] - plot_dat["methodA"]["std"]))
    # Human vs Method B
    means.append(abs(plot_dat["dataset"]["mean"] - plot_dat["methodB"]["mean"]))
    stds.append(abs(plot_dat["dataset"]["std"] - plot_dat["methodB"]["std"]))
    # Human vs Method A
    means.append(abs(plot_dat["dataset"]["mean"] - plot_dat["random"]["mean"]))
    stds.append(abs(plot_dat["dataset"]["std"] - plot_dat["random"]["std"]))
    
    l = ["Method A", "Method B", "Random"]
    fig, ax = plt.subplots()
    index = np.arange(len(l))
    bar_width = 0.5
    opacity = 0.8
    
    ax.yaxis.grid(True)
        
    plt.title("Mean %s absolute difference from human composition" % feature)
    plt.xticks(index, l)
    plt.ylabel("Descriptor Value")
    plt.xlabel("\nComposition Type")
    
    rects = plt.bar(index, means, bar_width,
    alpha=opacity,
    color='#FC6600',
    label='Dataset',
    align="center",
    yerr=stds,
    capsize=6,
    ecolor="black")

    plt.savefig("res/plots/%s-absolute-difference" % feature)
    
    

def plotFeatureComparison(organised_data, feature):

    # import statistics
    # numbers = [G[key] for key in G]
    # mean_ = statistics.mean(numbers)
    dat = organised_data[feature]
    plot_dat = {}
    means = []
    stds = [] 
    
    for dk, dv in dat.items():
        plot_dat[dk] = {}
        numbers = [dv[key] for key in dv]
        d_std = statistics.stdev(numbers)
        d_mean = statistics.mean(numbers)
        plot_dat[dk]["std"] = d_std
        plot_dat[dk]["mean"] = d_mean
        means.append(d_mean)
        stds.append(d_std)

    fig, ax = plt.subplots()
    index = np.arange(len(dat))
    bar_width = 0.5
    opacity = 0.8

    ax.yaxis.grid(True)
    
    plt.title("Mean %s comparison" % feature)
    plt.xticks(np.arange(len(dat)), dat.keys())
    plt.ylabel("Descriptor Value")
    plt.xlabel("\nComposition Type")
    
    rects = plt.bar(index, means, bar_width,
    alpha=opacity,
    color='#1f77b4',
    label='Dataset',
    align="center",
    yerr=stds,
    capsize=6,
    ecolor="black")
    
    plt.savefig("res/plots/%s-overall-comparison" % feature)
    
    plt.show()
    return plot_dat
    
    

# features we are going to analyse
labels = {
    "Amount of Arpeggiation": 1,  # fast melodic runs within confines of a key
    "Average Note Duration": 2,  # Length of note
    "Average Number of Simultaneous Pitches": 3,  # how many notes are active at once
    "Variability of Note Durations": 4,  # how much does the note length vary
    "Metrical Diversity": 5,  # Broadfer than noteduration, rhythmic feel of a section
    "Chromatic Motion": 6,  # motion of melody is chromatic (one after the other e.g. c1 -> c#1 -> d1 ->d#1)
    "Contrary Motion": 7,  # motion of melody(s) run counter to one another (indicator of complexity) 
    "Stepwise Motion": 8,  # motion of melody moves in steps of they key (e.g. c3->e3->g3 perfect c major)
    "Similar Motion": 9,  # motion of melody moves in same way, with different pitches (again indicator of complexityy)
    "Variation of Dynamics": 10,  # how much dynamics variation is there
    "Melodic Embellishments": 11,  # small melodic passages that slightly change or enhance a larger music phrase (complexity)
    "Repeated Notes": 12,  # number of notes that are repeated one after the other (indicator of low complexity)
    "Melodic Pitch Variety" : 13
}


training_analysis_filepath = "res/analysis_data/human.xml"
methodA_analysis_filepath = "res/analysis_data/method_a.xml" 
methodB_analysis_filepath = "res/analysis_data/method_b.xml" 
random_analysis_filepath = "res/analysis_data/random.xml" # random_analysis.xml

dataset_raw = parseAnalysisData(training_analysis_filepath)
dataset_data = deepcopy(dataset_raw["feature_vector_file"]["data_set"])
methodA_raw = parseAnalysisData(methodA_analysis_filepath)
methodA_data = deepcopy(methodA_raw["feature_vector_file"]["data_set"])
methodB_raw = parseAnalysisData(methodB_analysis_filepath)
methodB_data = deepcopy(methodB_raw["feature_vector_file"]["data_set"])
random_raw = parseAnalysisData(random_analysis_filepath)
random_data = deepcopy(random_raw["feature_vector_file"]["data_set"])

del dataset_raw
del methodA_raw
del methodB_raw
del random_raw

# list of all the descriptors we extract from the midi files
# needed for plotting
features = getFeatures(dataset_data)

dataset_final_data = formatData(dataset_data)
methodA_final_data = formatData(methodA_data)
methodB_final_data = formatData(methodB_data)
random_final_data = formatData(random_data)

del dataset_data
del methodA_data
del methodB_data
del random_data

mean_dataset_data = aggregateData(dataset_final_data, labels)
mean_methodA_data = aggregateData(methodA_final_data, labels)
mean_methodB_data = aggregateData(methodB_final_data, labels)
mean_random_data = aggregateData(random_final_data, labels)


all_data = {}
all_data["dataset"] = dataset_final_data
all_data["methodA"] = methodA_final_data
all_data["methodB"] = methodB_final_data
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
del methodA_final_data
del methodB_final_data

# perform the plots

plotOverallDifferenceComparison(organised_data)

for k, v in labels.items():
   x = plotFeatureComparison(organised_data, k)
   y = plotFeatureAgainstHumanComparison(organised_data, k)



