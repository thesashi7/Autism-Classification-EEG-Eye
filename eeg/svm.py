from sklearn import svm
import numpy as np
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from config.config import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from utils.analysis import *

# Load data from CSV file. Edit this to point to the features file
feat_type = "fft"
train_data = np.genfromtxt(main_path+"train-"+feat_type+"-eeg-x.csv",delimiter=',')
train_target =  np.genfromtxt(main_path+"train-"+feat_type+"-eeg-y.csv",delimiter=',')
test_data = np.genfromtxt(main_path+"test-"+feat_type+"-eeg-x.csv",delimiter=',')
test_target =  np.genfromtxt(main_path+"test-"+feat_type+"-eeg-y.csv",delimiter=',')

pca = PCA(n_components=129)
pca.fit(train_data)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()


NCOMPONENTS = 29

pca = PCA(n_components=NCOMPONENTS)
train_data = pca.fit_transform(train_data)
test_data = pca.transform(test_data)
pca_std = np.std(train_data)
print train_data.shape
svmModel = svm.SVC()
svmModel.fit(train_data, train_target)
predict = svmModel.predict(test_data)

print("Loss: "+str(log_loss(test_target,predict)))
print("Accuracy: "+str(accuracy_score(test_target, predict)))
confusionMatrix(actual=test_target.astype(int), predicted=predict.astype(int), shape=1)