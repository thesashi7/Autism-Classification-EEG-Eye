# read one data set
# transform into FFT
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import matplotlib.pyplot as plt
import os
import numpy as np
from numpy import array
from config import *
from sklearn.decomposition import PCA

"""
	@file: excel files path and name
	@returns the time row and the signals. First value is the array with timestamps and the second value is a 2D array with signals
"""
def read_eeg(file_name, sheet_name="sheet"):
   df = pd.read_excel(file_name)
   time_x = df.index.values
   signals = df.values
   #print df.columns
   return time_x, signals

def plot_dataFrame(ar_x, ar_y):
   i = 0
   plt.figure(1)
   while(i < len(ar_x)):
      plt.subplot(211+i)
      plt.plot(ar_x[i],ar_y[i])
      i+=1
   plt.show()

def plot_features(feature):
   i = 0
   plt.figure(1)
   while(i < len(feature)):
      plt.subplot(211+i)
      plt.plot(feature[i])
      i+=1
   plt.show()

#compute fft for each channel
def compute_fft(signal):
   fft_feat = [0] * signal.shape[1]
   #compute fft for each channel
   i = 0
   while(i < signal.shape[1]):
      fft_feat[i] = np.fft.rfftn(signal[:, i])
      i+=1
   return fft_feat

def extract_feature_average(file_name):
#read eeg file
#compute fft
#convert to array
#take average
#return the final averaged value
   time, signals = read_eeg(file_name)
   print signals.shape
   signals = compute_fft(signals)
   feature = array(signals)
   return np.average(feature, axis=1)


def do_fft(all_channel_data):
    """
    Do fft in each channel for all channels.
    Input: Channel data with dimension N x M. N denotes number of channel and M denotes number of EEG data from each channel.
    Output: FFT result with dimension N x M. N denotes number of channel and M denotes number of FFT data from each channel.
    """
    print "do_fft"
    data_fft = []
    i = 0
    for ch in all_channel_data:
        x = np.fft.fft(ch)
        data_fft.append(x)
        print i
        i+=1
    #data_fft = map(lambda x: np.fft.fft(x), all_channel_data)
    return array(data_fft)


def getFrequency(all_channel_data):
    """
    Get frequency from computed fft for all channels. 
    Input: Channel data with dimension N x M. N denotes number of channel and M denotes number of EEG data from each channel.
    Output: Frequency band from each channel: Delta, Theta, Alpha, Beta, and Gamma.
    """
    # Length data channel
    print "getFrequency"
    L = len(all_channel_data[0])

    # Sampling frequency
    Fs = 128

    # Get fft data
    data_fft = do_fft(all_channel_data)

    print "computing frequency"
    # Compute frequency
    frequency = map(lambda x: abs(x / L), data_fft)
    frequency = map(lambda x: x[: L / 2 + 1] * 2, frequency)

    print "list frequency"
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
    print feature.shape
    feature = feature.T
    #feature = feature.ravel()
    print feature.shape
    return feature

def extractFeatureFrequency(file_name):
    time, signals = read_eeg(file_name)
    print "eeg read"
    return getFrequencyFeature(signals.transpose())

def plotFeature(feat):
    plt.figure(1)
    plt.plot(feat)
    plt.show()

"""
print("Starting")
time = [0] * 2
signals = [0] * 2
time[0], signals[0] = read_eeg("/home/sashi/Documents/masters_research/jamie-eeg/test.xlsx", sheet_name='Sheet1')
time[1], signals[1] = read_eeg("/home/sashi/Documents/masters_research/jamie-eeg/test_asd.xlsx", sheet_name='Sheet1')
#print signals[0][:, 1]
#exit(0)
#print signals[0].shape
#print signals[1].shape
signals[0] = compute_fft(signals[0])
signals[1] = compute_fft(signals[1])
print len(signals[0][0])
print len(signals[1][0])
#print np.zeros(time[0].shape[0])
#exit(0)
#plot_dataFrame(time[0],signals[0])
#plot_dataFrame(time[1],signals[1])
# try visualizing signals
#print signals[1]
#print signals[0]
# average each channel
#signals[0] = np.average(signals[0], axis =0)
#signals[1] = np.average(signals[1], axis =0)
#print signals[0].shape
feature1 = array(signals[0])
feature2 = array(signals[1])
print feature1.shape
feature1 = np.average(feature1, axis=1)
feature2 = np.average(feature2, axis=1)
print len(signals[0])
print feature1
features = []
features.append(feature1)
features.append(feature2)
print len(features)
plot_features(features)
"""
#feature = extract_feature_average("/home/sashi/Documents/masters_research/jamie-eeg/003 Joint Attention.xlsx")
#time, signals = read_eeg("/home/sashi/Documents/masters_research/jamie-eeg/003 Joint Attention.xlsx")
#plt.plot(signals)
#plt.show()
#print feature.shape
#plt.plot(feature)
#plt.show()
#now need to create a csv of features
#
time, signals = read_eeg("/home/sashi/Documents/masters_research/jamie-eeg/003 Joint Attention.xlsx")

pca = PCA(n_components=signals.shape[1])
pca.fit(signals)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()


NCOMPONENTS = 35


exit(1)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
from scipy import signal
import matplotlib.pyplot as plt

#time, signals = read_eeg("/home/sashi/Documents/masters_research/jamie-eeg/076 joint attention.xlsx")
freq = extractFeatureFrequency("/home/sashi/Documents/masters_research/jamie-eeg/003 Joint Attention.xlsx")
print freq.shape
plotFeature(freq)

exit(1)
"""
print signals.shape
pca = PCA(n_components=signals.shape[1])
pca.fit(signals)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()

NCOMPONENTS = 100
print signals.shape
pca = PCA(n_components=NCOMPONENTS)
signals = pca.fit_transform(signals)
pca_std = np.std(signals)
print signals.shape
f, t, Sxx = signal.spectrogram(signals[:, 1])
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
"""
