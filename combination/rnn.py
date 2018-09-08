# LSTM for international airline passengers problem with regression framing
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adamax, Adam, SGD
from keras.utils import to_categorical
import keras.initializers
import numpy as np
import numpy
from keras.models import model_from_json
from config import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from keras.callbacks import EarlyStopping
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# Load data from CSV file. Edit this to point to the features file
#data, target = FeatureLoader("data/features.csv").loadLibrosaCSV()
# Load data from CSV file. Edit this to point to the features file
train_data = np.genfromtxt(main_path+"train-comb-x.csv",delimiter=',')
train_target =  np.genfromtxt(main_path+'train-comb-y.csv',delimiter=',')
test_data = np.genfromtxt(main_path+"test-comb-x.csv",delimiter=',')
test_target =  np.genfromtxt(main_path+'test-comb-y.csv',delimiter=',')
print train_target.shape
#print train_target

train_target =  np_utils.to_categorical(train_target, 2)
test_target =  np_utils.to_categorical(test_target, 2)

#sm = SMOTE(random_state=12, ratio = .4)
#train_data, train_target = sm.fit_sample(train_data, [y.argmax() for y in train_target])
#print train_data.shape
#print train_target.shape

#train_target = np_utils.to_categorical(train_target, 2)

pca = PCA(n_components=train_data.shape[1])
pca.fit(train_data)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()


NCOMPONENTS = 25

pca = PCA(n_components=NCOMPONENTS)
X_pca_train = pca.fit_transform(train_data)
X_pca_test = pca.transform(test_data)
pca_std = np.std(X_pca_train)

print(train_data.shape)
print(X_pca_train.shape)
look_back = 1
trainX, trainY = train_data, train_target
testX, testY = test_data, test_target
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(3, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
scaler = MinMaxScaler(feature_range=(0, 1))
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
