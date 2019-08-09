from DataAnalysisHelpers import *
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import statistics

sub_analysis_path = "res/analysis_data/subjective_analysis.xml"

sub_raw = parseAnalysisData(sub_analysis_path)
sub_data = sub_raw["Test"]["Participant"]


enjoyments = []
replays = []
complexities = []

skills = []
instruments = []
listen_times = []

for participant in sub_data:
    participant_no = eval(participant["participantId"])
    print("Participant %d" % participant_no)
    