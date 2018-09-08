import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from utils.analysis import *
from config.config import *

# Load data from CSV file. Edit this to point to the features file
train_data = np.genfromtxt(main_path+"train-eye-x.csv",delimiter=',')
train_target =  np.genfromtxt(main_path+'train-eye-y.csv',delimiter=',')
test_data = np.genfromtxt(main_path+"test-eye-x.csv",delimiter=',')
test_target =  np.genfromtxt(main_path+'test-eye-y.csv',delimiter=',')


pca = PCA(n_components=train_data.shape[1])
pca.fit(train_data)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()


NCOMPONENTS = train_data.shape[1]

pca = PCA(n_components=20)
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