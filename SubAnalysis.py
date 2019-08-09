from DataAnalysisHelpers import *
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import statistics

sub_analysis_path = "res/analysis_data/subjective_analysis.xml"

sub_raw = parseAnalysisData(sub_analysis_path)
sub_data = sub_raw["Test"]["Participant"]


enjoyments = {}
replays = {}
complexities = {}
guesses = {}

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
            enjoyments[q].append(v["enjoy"])
        else:
            enjoyments[q].append(v["enjoy"])
        
        if q not in replays:
            replays[q] = []
            replays[q].append(v["replay"])
        else:
            replays[q].append(v["replay"])
            
        if q not in complexities:
            complexities[q] = []
            complexities[q].append(v["human-or-computer"])
        else:
            complexities[q].append(v["human-or-computer"])
        
        if q not in guesses:
            guesses[q] = []
            guesses[q].append(v["human-or-computer"])
        else:
            guesses[q].append(v["human-or-computer"])
    
    skills.append(part1["qualification"])
    instruments.append(part1["num_instruments"])
    listen_times.append(part1["hours_of_music"])