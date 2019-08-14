from DataAnalysisHelpers import *
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import statistics
from DataGen import *


def plotAccuracyAll(guesses, answers): 
    # Get accuracy of participants per composition
    normalizer = len(instruments)
    candidate_accuracy = []

    for i in range(len(answers)):
        current_answer = answers[i]
        current_key = "question-%d" % (i + 1)
        m = 0
        for guess in guesses[current_key]:
            if guess == current_answer:
                continue
            else:
                m += 1
        percentage = 1 - (m / normalizer)
        candidate_accuracy.append(percentage)
        
        
    fig, ax = plt.subplots()
    index = np.arange(len(candidate_accuracy))
    plt.title("Participant accuracy by composition")
    labels = [1,2,3,4,5,6,7,8,9]
    plt.xticks(index, labels)
    ax.set_ylabel("Participant Accuracy")
    ax.set_xlabel("Composition No.")
    plt.bar(index, candidate_accuracy)
    plt.savefig("res/plots/accuracy-all.png", bbox_inches="tight")
    plt.show()
    return(candidate_accuracy)


def plotAccuracyByType(candidate_accuracy, types):
    humans = []
    ma = []
    mb = []
        
    for i in range(len(candidate_accuracy)):
        comp_type = types[i]
        if comp_type == 0:
            humans.append(candidate_accuracy[i])
        elif comp_type == 1:
            ma.append(candidate_accuracy[i])
        elif comp_type == 2:
            mb.append(candidate_accuracy[i])

    average_human_score = (sum(humans) / 3)
    average_ma_score = (sum(ma) / 3)
    average_mb_score = (sum(mb) / 3)

    plot_dat = [average_human_score, average_ma_score, average_mb_score]

    fig, ax = plt.subplots()
    index = np.arange(len(plot_dat))
    plt.title("Participant accuracy by composition type")
    labels = ["Human", "Method A", "Method B"]
    plt.xticks(index, labels)
    ax.set_ylabel("Participant Accuracy")
    ax.set_xlabel("Composition Type")
    plt.ylim(0.6, 0.85)
    plt.bar(index, plot_dat, color=['red', 'green', 'blue'])
    plt.savefig("res/plots/accuracy-by-type.png", bbox_inches="tight")
    plt.show()
    

def plotLikertRatingAll(feature_set, feature_name):
    averages = []
    stds = []
    its = feature_set.values()
    for v in its:
        averages.append(sum(v) / len(v))
        stds.append(statistics.stdev(v))
        
    fig, ax = plt.subplots()
    index = np.arange(len(averages))
    plt.title("%s rating by composition" % feature_name)
    labels = [1,2,3,4,5,6,7,8,9]
    plt.xticks(index, labels)
    plt.yticks(np.arange(5), ('Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'))
    ax.set_ylabel("Participant Accuracy")
    ax.set_xlabel("Composition No.")
    plt.bar(index, averages, yerr=stds, capsize=5)
    plt.savefig("res/plots/%s-all.png" % feature_name, bbox_inches="tight")
    plt.show()
    return(averages, stds)


def plotLikertByType(averages, stds, types, feature_name):
    humans_m = []
    ma_m = []
    mb_m = []

    humans_std = []
    ma_std = []
    mb_std = []
        
    for i in range(len(averages)):
        comp_type = types[i]
        if comp_type == 0:
            humans_m.append(averages[i])
            humans_std.append(stds[i])
        elif comp_type == 1:
            ma_m.append(averages[i])
            ma_std.append(stds[i])
        elif comp_type == 2:
            mb_m.append(averages[i])
            mb_std.append(stds[i])

    average_human_score = (sum(humans_m) / 3)
    average_ma_score = (sum(ma_m) / 3)
    average_mb_score = (sum(mb_m) / 3)

    average_human_deviation = (sum(humans_std) / 3)
    average_ma_deviation = (sum(ma_std) / 3)
    average_mb_deviation = (sum(mb_std) / 3)
    

    plot_dat = [average_human_score, average_ma_score, average_mb_score]
    plot_std = [average_human_deviation, average_ma_deviation, average_mb_deviation]

    fig, ax = plt.subplots()
    index = np.arange(len(plot_dat))
    plt.title("Perceived %s by composition type" % feature_name)
    labels = ["Human", "Method A", "Method B"]
    plt.xticks(index, labels)
    plt.yticks(np.arange(5), ('Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'))
    ax.set_ylabel("Participant Accuracy")
    ax.set_xlabel("Composition Type")
    # plt.ylim(0.5, 0.9)
    plt.bar(index, plot_dat, color=['red', 'green', 'blue'], yerr=plot_std, capsize=5)
    plt.savefig("res/plots/%s-by-type.png" % feature_name, bbox_inches="tight")
    plt.show()


sub_analysis_path = "res/analysis_data/subjective_analysis.xml"
sub_raw = parseAnalysisData(sub_analysis_path)
sub_data = sub_raw["Test"]["Participant"]


# 0 = Human, 1 = Methoa A, 2 = Method B
types = [1,0,2,0,1,2,2,0,1]
# correct answers (e.g. human or computer)
answers = [0, 1, 0, 1, 0, 0, 0, 1, 0]

# get the data
guesses = {}
enjoyments = {}
replays = {}
complexities = {}

skills = []
instruments = []
listen_times = []


for participant in sub_data:
    participant_no = eval(participant["participantId"])
    print("Participant %d" % participant_no)
    part1 = participant["test-part1"]
    part2 = participant["test-part2"]
    
    for q, v in part2.items():
        if q not in enjoyments:
            enjoyments[q] = []
            enjoyments[q].append(eval(v["enjoy"]))
        else:
            enjoyments[q].append(eval(v["enjoy"]))
        
        if q not in replays:
            replays[q] = []
            replays[q].append(eval(v["replay"]))
        else:
            replays[q].append(eval(v["replay"]))
            
        if q not in complexities:
            complexities[q] = []
            complexities[q].append(eval(v["complex"]))
        else:
            complexities[q].append(eval(v["complex"]))
        
        if q not in guesses:
            guesses[q] = []
            guesses[q].append(eval(v["human-or-computer"]))
        else:
            guesses[q].append(eval(v["human-or-computer"]))
    
    skills.append(eval(part1["qualification"]))
    instruments.append(eval(part1["num_instruments"]))
    listen_times.append(eval(part1["hours_of_music"]))
    
    

candidate_accuracy = plotAccuracyAll(guesses, answers)
plotAccuracyByType(candidate_accuracy, types)


complexity_averages, complexity_stds = plotLikertRatingAll(complexities, "Complexity")
plotLikertByType(complexity_averages, complexity_stds, types, "Complexity")
enjoyment_averages, enjoyment_stds = plotLikertRatingAll(enjoyments, "Enjoyment")
plotLikertByType(enjoyment_averages, enjoyment_stds, types, "Enjoyment")
replayability_averages, replayability_stds = plotLikertRatingAll(replays, "Replayability")
plotLikertByType(replayability_averages, replayability_stds, types, "Replayability")

# plot participants musical skill
fig, ax = plt.subplots()
index = np.arange(10)
labels = [1,2,3,4,5,6,7,8,9,10]
y_labels = ['No Musical Education', 
            'School', 
            'Privately Tutored', 
            'Student of Music', 
            'Professional']

plt.xticks(index, labels)
plt.yticks(np.arange(1, 6, step=1), y_labels)
plt.title("Skills by participant")
plt.bar(index, skills)
plt.savefig("res/plots/participant-skill-all.png", bbox_inches="tight")
plt.show()


mean_skill = statistics.mean(skills)
std_skill = statistics.stdev(skills)
min_skill = min(skills)
max_skill = max(skills)
med_skill = statistics.median(skills)
skill_comparison = [mean_skill, std_skill, min_skill, max_skill, med_skill]

fig, ax = plt.subplots()
index = np.arange(5)
labels = ['Mean', 'StdDev', 'Min', 'Max', 'Median']

plt.xticks(index, labels)
y_labels = ['No Musical Education', 
            'School', 
            'Privately Tutored', 
            'Student of Music', 
            'Professional']
plt.yticks(np.arange(1, 6, step=1), y_labels)
plt.title("Skills Stats")
plt.bar(index, skill_comparison)
plt.savefig("res/plots/participant-skill-comparison.png", bbox_inches="tight")
plt.show()