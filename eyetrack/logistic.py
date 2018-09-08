import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.decomposition import PCA
from utils.analysis import *
from config.config import *
from sklearn.metrics import accuracy_score
from config.config import *
from utils.analysis import *
from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from createInput import createSamples
from utils.writer import *
# Load data from CSV file. Edit this to point to the features file
train_data = np.genfromtxt(main_path+"train-eye-x.csv",delimiter=',')
train_target =  np.genfromtxt(main_path+'train-eye-y.csv',delimiter=',')
test_data = np.genfromtxt(main_path+"test-eye-x.csv",delimiter=',')
test_target =  np.genfromtxt(main_path+'test-eye-y.csv',delimiter=',')
print train_data.shape
print train_target.shape
#train_target = np.ravel(train_target)
pca = PCA(n_components=train_data.shape[1])
pca.fit(train_data)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()


NCOMPONENTS = 15

pca = PCA(n_components=NCOMPONENTS)
train_data = pca.fit_transform(train_data)
test_data = pca.transform(test_data)
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


kFold = KFold(n_splits=2,shuffle=True, random_state=True)
i=0

## 5x2 Cross validation T-Test
# First divide the data into 5 sets
"""
test_n = 2
while test_n > 0:
    samples = createSamples(num_sam=5)
    i = 0
    print "--- K-Fold Cross Validation ---\n"
    for sample in samples:
        for train, test in kFold.split(sample[0], sample[1]):
            i+=1
            logistic = linear_model.LogisticRegression()
            logistic.fit(sample[0][train], sample[1][train])
            y_pred = logistic.predict(sample[0][test])
            print ("Accuracy: "+ str(accuracy_score(sample[1][test], y_pred, normalize=True)))
            print("Loss: " + str(mean_squared_error(sample[1][test], y_pred)))
            print "\n"
        print "---"
    test_n-=1

"""
i=0
sum_ac = 0.0
n=0

## 10 fold Cross validation T-Test
# First divide the data into 5 sets
kFold = KFold(n_splits=10,shuffle=True, random_state=True)
print "--- K-Fold Cross Validation ---\n"
test_n = 200
ac =[]
while test_n > 0:
    i = 0
    print "-------------------------------------------------\n"
    for train, test in kFold.split(train_data, train_target):
        i+=1
        logistic = linear_model.LogisticRegression()
        logistic.fit(train_data[train], train_target[train])
        y_pred = logistic.predict(train_data[test])
        cur_ac = accuracy_score(train_target[test], y_pred, normalize=True)
        print ("Accuracy: "+ str(cur_ac))
        print("Loss: " + str(mean_squared_error(train_target[test], y_pred)))
        print "\n"
        print "---"
        n+=1
        sum_ac += cur_ac
        ac.append(cur_ac)

    test_n-=1
ac = np.array(ac)
print ("Average Acc: "+ str(ac.mean(axis=0)))
print ("STD: " + str(ac.std(axis=0)))

writeExcel(ac,"results-eye-log")