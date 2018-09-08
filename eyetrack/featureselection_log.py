from mlxtend.feature_selection.sequential_feature_selector import *
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from config import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from config.config import *
from utils.analysis import *
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from utils.writer import *
from sklearn import linear_model

# Load data from CSV file. Edit this to point to the features file
train_data = np.genfromtxt(main_path+"train-eye-x.csv",delimiter=',')
train_target =  np.genfromtxt(main_path+'train-eye-y.csv',delimiter=',')
test_data = np.genfromtxt(main_path+"test-eye-x.csv",delimiter=',')
test_target =  np.genfromtxt(main_path+'test-eye-y.csv',delimiter=',')

logistic = linear_model.LogisticRegression()


print train_data.shape


# Sequential Forward Selection
sfs = SequentialFeatureSelector(logistic,
          k_features=10,
          forward=True,
          floating=False,
          scoring='accuracy',
          cv=4,
          n_jobs=-1)
sfs = sfs.fit(train_data, train_target)

def selectFeatures(feature_ids, data):
    selected_feature = []
    for sf in sfs.k_feature_idx_:
        cf = data[:, sf]
        selected_feature.append(cf)
    selected_feature = np.array(selected_feature)
    selected_feature = selected_feature.transpose()
    return selected_feature

print('\nSequential Forward Selection (k=3):')
print(sfs.k_feature_idx_)
print('CV Score:')
print(sfs.k_score_)
#train_data = selectFeatures(sfs.k_feature_idx_, train_data)
#test_data = selectFeatures(sfs.k_feature_idx_, test_data)
print test_target.shape
logistic = linear_model.LogisticRegression()
logistic.fit(train_data, train_target)
y_pred = logistic.predict(test_data)
# Print results
print accuracy_score(test_target,y_pred, normalize = True)
confusionMatrix(actual=test_target.astype(int), predicted=y_pred.astype(int), shape=1)

## 10 fold Cross validation T-Test
# First divide the data into 5 sets
kFold = KFold(n_splits=10,shuffle=True, random_state=True)
print "--- K-Fold Cross Validation ---\n"
test_n = 200
sum_ac = 0.0
n=0
ac= list()
while test_n > 0:
    i = 0
    print "-------------------------------------------------\n"
    for train, test in kFold.split(train_data, train_target):
        i+=1
        logistic.fit(train_data[train], train_target[train])
        y_pred = logistic.predict(train_data[test])
        print ("Accuracy: "+ str(accuracy_score(train_target[test], y_pred, normalize=True)))
        print("Loss: " + str(mean_squared_error(train_target[test], y_pred)))
        print "\n"
        print "---"
        n+=1
        ac.append(accuracy_score(train_target[test], y_pred, normalize=True))

    test_n-=1

ac = np.array(ac)
print ("Average Acc: "+ str(ac.mean(axis=0)))
print ("STD: " + str(ac.std(axis=0)))
writeExcel(ac,"results-eye-without-logistic")