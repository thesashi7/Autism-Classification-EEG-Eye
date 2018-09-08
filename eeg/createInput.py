# need to create a single csv file from the data folder
import csv
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from numpy import array
from config.config import *
import glob
from pyentrp import entropy as ent
from fextractor import *
from utils.analysis import *

def createInputFile():
    # read file
    # find eye tracking file name for each subject
    # get the diagnosis
    # read eye tracking file csv
    # Now write the featureSet eye trakcing data along with the diagnosis at the end into a new csv file
    feature_type = "fft"
    featureSet = getFeatureSet(type=feature_type, file_type="train")
    writeCSVFile(featureSet, main_path + "train-"+feature_type+"-eeg.csv")
    print("done1")
    #print len(featureSet)
    writeFeatures(main_path + "train-"+feature_type+"-eeg.csv", "train-"+feature_type+"-eeg")
    print("done2")
    featureSet = getFeatureSet(type=feature_type, file_type="test")
    writeCSVFile(featureSet, main_path + "test-"+feature_type+"-eeg.csv")
    writeFeatures(main_path + "test-"+feature_type+"-eeg.csv", "test-"+feature_type+"-eeg")


def createSamples(num_sam = 5, feature_type = "sh_entropy"):
    #divide this data into num_sam
    #return
    splitted = splitSample(main_path + "train-"+feature_type+"-eeg.csv",num_splits=num_sam)
    print len(splitted)
    for sam in splitted:
        print sam[1].shape
    return splitted


#createInputFile()
#createSamples()
# data = np.genfromtxt(main_path+"eeg.csv", delimiter=',',dtype=None)
# plotFeatures(data)
# st_y = standerizeLabel(main_path+"train-eeg-y.csv")
# write_csv(st_y, main_path + "train-eeg-y.csv")
# st_y = standerizeLabel(main_path+"test-eeg-y.csv")
#write_csv(st_y, main_path + "test-eeg-y.csv")
