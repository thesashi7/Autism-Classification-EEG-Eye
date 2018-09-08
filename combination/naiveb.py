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
from createInput import createSamples
# Load data from CSV file. Edit this to point to the features file
feature_type = "sh_entropy"
# Load data from CSV file. Edit this to point to the features file
train_data = np.genfromtxt(main_path+"train-comb-"+feature_type+"-x.csv",delimiter=',')
train_target =  np.genfromtxt(main_path+"train-comb-"+feature_type+"-y.csv",delimiter=',')
test_data = np.genfromtxt(main_path+"test-comb-"+feature_type+"-x.csv",delimiter=',')
test_target =  np.genfromtxt(main_path+"test-comb-"+feature_type+"-y.csv",delimiter=',')

gnb = GaussianNB()

pca = PCA()
pca.fit(train_data)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
#plt.show()


NCOMPONENTS = 50

print train_data.shape
pca = PCA(n_components=NCOMPONENTS)
print train_data.shape

train_data = pca.fit_transform(train_data)
test_data = pca.transform(test_data)
#pca_std = np.std(train_data)
print train_data.shape

# Train classifier
gnb.fit(
    train_data,
    train_target
)
y_pred = gnb.predict(test_data)

# Print results
print accuracy_score(test_target,y_pred, normalize = True)
confusionMatrix(actual=test_target.astype(int), predicted=y_pred.astype(int), shape=1)

## 5x2 Cross validation T-Test
# First divide the data into 5 sets
"""
kFold = KFold(n_splits=2,shuffle=True, random_state=True)

test_n = 2
while test_n > 0:
    samples = createSamples(num_sam=5, feature_type="std")
    i = 0
    print "--- K-Fold Cross Validation ---\n"
    for sample in samples:
        for train, test in kFold.split(sample[0], sample[1]):
            i+=1
            gnb.fit(sample[0][train], sample[1][train])
            y_pred = gnb.predict(sample[0][test])
            print ("Accuracy: "+ str(accuracy_score(sample[1][test], y_pred, normalize=True)))
            print("Loss: " + str(mean_squared_error(sample[1][test], y_pred)))
            print "\n"
        print "---"
    test_n-=1"""

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
        gnb.fit(train_data[train], train_target[train])
        y_pred = gnb.predict(train_data[test])
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
writeExcel(ac,"results-comb-pca-"+feature_type)