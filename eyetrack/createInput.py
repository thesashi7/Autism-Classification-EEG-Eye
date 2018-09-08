import csv

import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from sklearn.model_selection import train_test_split
from utils.analysis import splitSample

from config.config import *


def plotFeature(feat):
    plt.figure(1)
    plt.plot(feat)
    plt.show()

def write_csv(data, file_name):
    with open(file_name, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
        # print(type(features[0]))
        if (isinstance(data, np.ndarray)):
            for fv in data:
                print(fv)
                writer.writerow(fv)
        else:  # only one feature vector
            writer.writerow(data)

def writeFeatureFile(featureFile, file_type="train"):
    data = np.genfromtxt(featureFile, delimiter=',')
    np.random.shuffle(data)
    print data.shape
    data = np.delete(data, (0), axis=0)
    data = np.delete(data, (0), axis=1)
    where_are_NaNs = isnan(data)
    data[where_are_NaNs] = 0
    print data.shape
    #print data
    target = np.empty([data.shape[0], 1])
    i = 0
    while i < data.shape[0]:
        if (data[i][data.shape[1] - 1] == 1.0):
            target[i] = 1
        else:
            target[i] = 0
        i += 1
    print target
    data = np.delete(data, np.s_[data.shape[1] - 1: data.shape[1]], axis=1)

    target = target.reshape((target.shape[0], 1))
    write_csv(data, main_path + file_type+"-eye-x.csv")
    write_csv(target, main_path + file_type+"-eye-y.csv")

def createInputFile():
    writeFeatureFile(base_path+"train_eyetrack.csv")
    writeFeatureFile(base_path+"test_eyetrack.csv", file_type="test")

def createSamples(num_sam = 5):
    #divide this data into num_sam
    #return
    splitted = splitSample(base_path + "train_eyetrack.csv",num_splits=num_sam)
    print len(splitted)
    for sam in splitted:
        print sam[1].shape
        where_are_NaNs = isnan(sam[0])
        sam[0][where_are_NaNs] = 0
    return splitted

createInputFile()
