import os
from difflib import SequenceMatcher

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()