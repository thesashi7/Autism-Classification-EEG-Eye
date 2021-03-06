import numpy as np
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from config.config import *
from sklearn.decomposition import PCA
from utils.analysis import *

feature_type = "fft"
# Load data from CSV file. Edit this to point to the features file
train_data = np.genfromtxt(main_path+"train-comb-"+feature_type+"-x.csv",delimiter=',')
train_target =  np.genfromtxt(main_path+"train-comb-"+feature_type+"-y.csv",delimiter=',')
test_data = np.genfromtxt(main_path+"test-comb-"+feature_type+"-x.csv",delimiter=',')
test_target =  np.genfromtxt(main_path+"test-comb-"+feature_type+"-y.csv",delimiter=',')
print train_data.shape
print train_target.shape
#train_target = np.ravel(train_target)
pca = PCA()
pca.fit(train_data)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()


NCOMPONENTS = 24

pca = PCA(n_components=15)
#train_data = pca.fit_transform(train_data)
#test_data = pca.transform(test_data)
pca_std = np.std(train_data)
print train_data.shape
print train_target.shape
print test_target.shape
train_target = train_target.reshape(train_target.shape[0])
test_target = test_target.reshape(test_target.shape[0])
print train_target.shape
logistic = linear_model.LogisticRegression()
logistic.fit(train_data, train_target)
predict = logistic.predict(test_data)
print("ACC: "+str(logistic.score(test_data,test_target)))
confusionMatrix(actual=test_target.astype(int), predicted=predict.astype(int), shape=1)