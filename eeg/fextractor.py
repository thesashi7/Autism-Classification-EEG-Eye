# need to create a single csv file from the data folder
import csv
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from numpy import array
from pyentrp import entropy as ent
from utils.reader import  *
from utils.writer import *
from config.config import *

def plotDataFrame(ar_x, ar_y):
   i = 0
   plt.figure(1)
   while(i < len(ar_x)):
      plt.subplot(211+i)
      plt.plot(ar_x[i],ar_y[i])
      i+=1
   plt.show()

def plotFeatures(feature):
   i = 0
   plt.figure(1)
   while(i < 1):
      plt.plot(feature[2])
      i+=1
   plt.show()

def plotFeature(feat):
    plt.figure(1)
    plt.plot(feat)
    plt.show()

def do_fft(all_channel_data):
    """
    Do fft in each channel for all channels.
    Input: Channel data with dimension N x M. N denotes number of channel and M denotes number of EEG data from each channel.
    Output: FFT result with dimension N x M. N denotes number of channel and M denotes number of FFT data from each channel.
    """
    print "i"
    data_fft = map(lambda x: np.fft.fft(x), all_channel_data)
    print "ii"
    return data_fft


def getFrequency(all_channel_data):
    """
    Get frequency from computed fft for all channels. 
    Input: Channel data with dimension N x M. N denotes number of channel and M denotes number of EEG data from each channel.
    Output: Frequency band from each channel: Delta, Theta, Alpha, Beta, and Gamma.
    """
    # Length data channel
    L = len(all_channel_data[0])

    # Sampling frequency
    Fs = 128

    # Get fft data
    data_fft = do_fft(all_channel_data)

    # Compute frequency
    frequency = map(lambda x: abs(x / L), data_fft)
    frequency = map(lambda x: x[: L / 2 + 1] * 2, frequency)

    # List frequency
    delta = map(lambda x: x[L * 1 / Fs - 1: L * 4 / Fs], frequency)
    theta = map(lambda x: x[L * 4 / Fs - 1: L * 8 / Fs], frequency)
    alpha = map(lambda x: x[L * 5 / Fs - 1: L * 13 / Fs], frequency)
    beta = map(lambda x: x[L * 13 / Fs - 1: L * 30 / Fs], frequency)
    gamma = map(lambda x: x[L * 30 / Fs - 1: L * 50 / Fs], frequency)

    return delta, theta, alpha, beta, gamma


def getFrequencyFeature(all_channel_data):
    """
    Get feature from each frequency.
    Input: Channel data with dimension N x M. N denotes number of channel and M denotes number of EEG data from each channel.
    Output: Feature (standard deviasion and mean) from all frequency bands and channels with dimesion 1 x M (number of feature).
    """
    # Get frequency data
    (delta, theta, alpha, beta, gamma) = getFrequency(all_channel_data)

    # Compute feature std
    delta_std = np.std(delta, axis=1)
    theta_std = np.std(theta, axis=1)
    alpha_std = np.std(alpha, axis=1)
    beta_std = np.std(beta, axis=1)
    gamma_std = np.std(gamma, axis=1)

    # Compute feature mean
    delta_m = np.mean(delta, axis=1)
    theta_m = np.mean(theta, axis=1)
    alpha_m = np.mean(alpha, axis=1)
    beta_m = np.mean(beta, axis=1)
    gamma_m = np.mean(gamma, axis=1)

    # Concate feature
    feature = np.array(
        [delta_std, delta_m, theta_std, theta_m, alpha_std, alpha_m, beta_std, beta_m, gamma_std, gamma_m])
    feature = feature.T
    #feature = feature.ravel()
    return feature


# compute fft for each channel
def computeFFT(signal):
    fft_feat = [0] * signal.shape[1]
    # compute fft for each channel
    i = 0
    while (i < signal.shape[1]):
        fft_feat[i] = np.fft.rfftn(signal[:, i])
        i += 1
    return fft_feat


def computeShannonEntropy(signal):
    she_feat = [0] * signal.shape[1]
    # compute fft for each channel
    i = 0
    while (i < signal.shape[1]):
        print i
        she_feat[i] = ent.multiscale_entropy(signal[:, i],2,0.1*np.std(signal[:, i]))
        i += 1
    return she_feat

def flatten_channel(chn):
    flatten = []
    for r in chn:
        for v in r:
            flatten.append(v)
    return np.array(flatten)

def extractFFT(file_name):
    time, signals = readEEG(file_name)
    fft = getFFTWithWindow(signals)
    print len(fft)
    stats_fft = []
    for channel in fft:
        print len(channel)
        print len(channel[0])
        # flatten the channel
        # [[1,2],[3,4]] => [1,2,3,4]
        channel = np.array(channel)
        channel = flatten_channel(channel)

        print channel.shape
        print channel[0].shape
        stats_fft.append(np.std(channel))
        stats_fft.append(np.mean(channel))
    return np.array(stats_fft)

def extractFeatureFrequency(file_name):
    time, signals = readEEG(file_name)
    return getFrequencyFeature(signals.transpose())


def extractFeatureShannonEntropy(file_name):
    time, signals = readEEG(file_name)
    sh_entropy = getShannonEntropyWithWindow(signals)
    stats_entropy = []
    for channel in sh_entropy:
        stats_entropy.append(np.std(channel))
        stats_entropy.append(np.mean(channel))
    print len(stats_entropy)
    return np.array(stats_entropy)

def extractFeatureMultiScaleEntropy(file_name):
    time, signals = readEEG(file_name)
    sh_entropy = getMultiScaleEntropyWithWindow(signals)
    stats_entropy = []
    for channel in sh_entropy:
        print channel
        stats_entropy.append(np.std(channel, axis=1))
        stats_entropy.append(np.mean(channel, axis=1))
    return np.array(stats_entropy)

def extractFeatureAverage(file_name):
    # read eeg file
    # compute fftn.
    # convert to array
    # take average
    # return the final averaged value
    time, signals = readEEG(file_name)
    #plotDataFrame(time, signals)
    #signals = computeFFT(signals)
    #feature = array(signals)
    return np.average(signals, axis=1)


def extractFeatureStd(file_name):
    # read eeg file
    # compute fft
    # convert to array
    # take average
    # return the final averaged value
    time, signals = readEEG(file_name)
    #signals = computeFFT(signals)
    #feature = array(signals)
    return np.std(signals, axis=1)

def extractFeatureMean(file_name):
    # read eeg file
    # compute fft
    # convert to array
    # take average
    # return the final averaged value

    time, signals = readEEG(file_name)
    #signals = computeFFT(signals)
    #feature = array(signals)
    return np.mean(signals, axis=1)

def getFeatureSet(type="freq", file_type="train"):
    # read diagnostic file
    # extract features of each subject one at a time
    features = []
    dg_info = diagnosticFile(file_type)
    for sub in dg_info:
        # col 0 is the subject id
        # col 3 is the diagnosis
        sub =  sub.astype(int)
        print getSubjectFileName(sub[0])
        if(type == "std"):
            cur_feat = extractFeatureStd(getSubjectFileName(sub[0])).real
        elif(type == "ave"):
            cur_feat = extractFeatureAverage(getSubjectFileName(sub[0])).real
        elif(type == "mean"):
            cur_feat = extractFeatureMean(getSubjectFileName(sub[0])).real
        elif(type == "freq"):
            cur_feat = extractFeatureFrequency(getSubjectFileName(sub[0])).real
        elif(type == "sh_entropy"):
            cur_feat = extractFeatureShannonEntropy(getSubjectFileName(sub[0]))
        elif(type == "mul_entropy"):
            cur_feat = extractFeatureMultiScaleEntropy(getSubjectFileName(sub[0]))
        elif(type == "fft"):
            cur_feat = extractFFT(getSubjectFileName(sub[0])).real

        print cur_feat.shape
        while (len(cur_feat) < 129):
            cur_feat = np.insert(cur_feat, cur_feat.shape[0], 0, axis=0)
        if (type == "sh_entropy" or type == "fft"):
            print ("adding extra")
            while (len(cur_feat) < 258):
                cur_feat = np.insert(cur_feat, cur_feat.shape[0], 0, axis=0)
                print "added"
            print len(cur_feat)

        else:
            while(len(cur_feat) > 129):
                cur_feat = np.delete(cur_feat, cur_feat.shape[0]-1, axis=0)
        cur_feat = np.insert(cur_feat, 0, int(sub[0]), axis = 0)
        cur_feat = np.insert(cur_feat, cur_feat.shape[0], sub[3], axis=0)
        print cur_feat.shape
        features.append(cur_feat)
        print cur_feat
        print "sub ex"
    print len(features)
    for f in features:
        print len(f)

    return features

def getFFTWithWindow(signal, type="freq", window_size=40, step_size = 20):
    #testing for entropy

    # Shannon entropy
        #ent = 0.0
        #for freq in freq_list:
        #    ent += freq * np.log2(freq)
        #ent = -ent

    entropy = []
    i = 0
    while(i < signal.shape[1]):
        j=0
        she_feat = []
        while (j < signal.shape[0]):
            #print i
            if (j+window_size < signal.shape[0]):
                sh_ent =  np.fft.rfftn(signal[j:j+window_size+1, i])
                #print sh_ent
                she_feat.append( sh_ent )
            else:
                sh_ent =  np.fft.rfftn(signal[j:j+window_size+1, i])
                #print sh_ent
                she_feat.append( sh_ent )

            j += step_size
        entropy.append(she_feat)
        i+=1
    return entropy

def getShannonEntropyWithWindow(signal, type="freq", window_size=40, step_size = 20):
    #testing for entropy

    # Shannon entropy
        #ent = 0.0
        #for freq in freq_list:
        #    ent += freq * np.log2(freq)
        #ent = -ent

    entropy = []
    i = 0
    while(i < signal.shape[1]):
        j=0
        she_feat = []
        while (j < signal.shape[0]):
            #print i
            if (j+window_size < signal.shape[0]):
                sh_ent =  ent.shannon_entropy(signal[j:j+window_size+1, i])
                #print sh_ent
                she_feat.append( sh_ent )
            else:
                sh_ent =  ent.shannon_entropy(signal[j:j+window_size+1, i])
                #print sh_ent
                she_feat.append( sh_ent )

            j += step_size
        entropy.append(she_feat)
        i+=1
    return entropy

def getMultiScaleEntropyWithWindow(signal, type="freq", window_size=40, step_size = 20):
    entropy = []
    step_size = window_size
    i = 0
    while(i < signal.shape[1]):
        j=0
        she_feat = []
        while (j < signal.shape[0]):
            print i
            if (j+window_size < signal.shape[0]):
                sh_ent = ent.multiscale_entropy(signal[j:j+window_size+1, i],len(signal[j:j+window_size+1, i]), 0.2 * np.std(signal[j:j+window_size+1, i]))
                #print sh_ent
                she_feat.append( sh_ent )
            else:
                sh_ent = ent.multiscale_entropy(signal[j:j+window_size+1, i],len(signal[j:j+window_size+1, i]), 0.2 * np.std(signal[j:j+window_size+1, i]))
                #print sh_ent
                she_feat.append( sh_ent )

            j += step_size
        entropy.append(she_feat)
        i+=1
    return entropy
#test

#time, signals = readEEG("/home/sashi/Documents/masters_research/jamie-eeg/003 Joint Attention.xlsx")
#plotFeature(ent)


#extract and write shannon entropy features
#write into a csv
#feat = getFeatureSet("ave")
#writeCSVFile(feat, main_path+"ave-eeg.csv")
#print("wrote")
#creat train and test file from the final csv with all the features and dataset
#writeTrainAndTestFile( featureFile = main_path+"ave-eeg.csv",type_name= "ave-eeg")
#print("wrote train test")
