from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adamax, Adam, SGD
from keras.utils import to_categorical
import keras.initializers
import numpy as np
import numpy
from keras.models import model_from_json
from config.config import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from keras.callbacks import EarlyStopping
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE
import pandas as pd
from utils.analysis import *
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from createInput import createSamples
from sklearn.metrics import accuracy_score

class DenseNeuralNetwork:

    def __init__(self):
        self.model = None
        self.drp_rate = 0.1


    def learn(self,train_data,train_target, drp_rate=0.1, epochs=150, batch_size=5,
              num_class=2):

        # Trying to get consistent results but this is just the first step
        #   It looks like keras doesn't currenlty allow seeds to be initialized
        #   So every time there is new seed and random new weights so the results can be
        #       different on each run
        numpy.random.seed(0)

        self. model = Sequential()

        num_features = train_data.shape[1]
        # numpy.randomfrom keras.utils import to_categorical.seed(0)
        # Adding first dense layer with acitivation relu and dropout
        self.model.add(Dense(num_features+1, input_dim=num_features))
        self.model.add(Activation('sigmoid'))

        self.model.add(Dense(num_features))
        self.model.add(Activation('sigmoid'))

        self.model.add(Dense(num_features))
        self.model.add(Activation('sigmoid'))

        self.model.add(Dense(num_features))
        self.model.add(Activation('sigmoid'))
        self.model.add(Dropout(drp_rate))

        self.model.add(Dense(num_features))
        self.model.add(Activation('sigmoid'))

        self.model.add(Dense(num_features))
        self.model.add(Activation('sigmoid'))



        # Adding final output layer with softmax
        self.model.add(Dense(units=num_class))
        self.model.add(Activation('softmax'))
        print np.unique(train_target)
        print train_target[0]
        y_classes = [np.argmax(train_target, axis=None, out=None) for y in train_target]
        y_ints = [y.argmax() for y in train_target]
        #print y_classes.shape
        cw = class_weight.compute_class_weight('balanced', np.unique(y_ints),
                                              y_ints)

        print cw
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=955, \
                                  verbose=1, mode='auto')
        callbacks_list = [earlystop]
        callbacks_list = []
        self.model.compile(loss='categorical_crossentropy', optimizer=Adamax())
        self.model.fit(train_data, train_target, batch_size=batch_size, verbose=0,validation_split=0.2, epochs=epochs,  callbacks=callbacks_list)
                      # class_weight = {0:.7, 1:.3})


    def learn_FFT(self,train_data,train_target, drp_rate=0.1, epochs=150, batch_size=5,
              num_class=2):

        # Trying to get consistent results but this is just the first step
        #   It looks like keras doesn't currenlty allow seeds to be initialized
        #   So every time there is new seed and random new weights so the results can be
        #       different on each run
        numpy.random.seed(0)

        self. model = Sequential()

        num_features = train_data.shape[1]
        # numpy.randomfrom keras.utils import to_categorical.seed(0)
        # Adding first dense layer with acitivation relu and dropout
        self.model.add(Dense(num_features+1, input_dim=num_features))
        self.model.add(Activation('sigmoid'))

        self.model.add(Dense(num_features))
        self.model.add(Activation('sigmoid'))

        self.model.add(Dense(num_features))
        self.model.add(Activation('sigmoid'))

        self.model.add(Dense(num_features))
        self.model.add(Activation('sigmoid'))
        self.model.add(Dropout(drp_rate))

        self.model.add(Dense(num_features))
        self.model.add(Activation('sigmoid'))

        self.model.add(Dense(num_features))
        self.model.add(Activation('sigmoid'))



        # Adding final output layer with softmax
        self.model.add(Dense(units=num_class))
        self.model.add(Activation('softmax'))
        print np.unique(train_target)
        print train_target[0]
        y_classes = [np.argmax(train_target, axis=None, out=None) for y in train_target]
        y_ints = [y.argmax() for y in train_target]
        #print y_classes.shape
        cw = class_weight.compute_class_weight('balanced', np.unique(y_ints),
                                              y_ints)

        print cw
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=955, \
                                  verbose=1, mode='auto')
        callbacks_list = [earlystop]
        callbacks_list = []
        self.model.compile(loss='categorical_crossentropy', optimizer=Adamax())
        self.model.fit(train_data, train_target, batch_size=batch_size, verbose=0,validation_split=0.2, epochs=epochs,  callbacks=callbacks_list)
                      # class_weight = {0:.7, 1:.3})



    def test(self,test_data,test_target):
        score = self.model.evaluate(test_data, test_target, batch_size=5)
        print('\n')
        print (score)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        #print recall_score(test_target, self.model.predict(test_data))

    def predict(self, test_data):
        prediction = self.model.predict(test_data)
        return prediction

    def save(self,name=""):
        f_name = "model/model-dnn"
        name = f_name+"-"+name
        model_json = self.model.to_json()
        with open(name+".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(name+".h5")
        print("Saved model to disk")


    def load(self,name=""):
        f_name = "model/model-dnn"
        name = f_name +"-"+name
        # load json and create model
        json_file = open(name+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(name+".h5")
        print("Loaded model from disk")

        # evaluate loaded model on test data
        loaded_model.compile(loss='categorical_crossentropy', optimizer=Adamax(), metrics=['accuracy'])
        self.model = loaded_model

    def predict(self,test_data):
        prediction = self.model.predict(test_data)
        return prediction

def target_vector(target):
    temp_target = numpy.zeros(shape=(train_target.shape[0], 1))
    i = 0
    for t in target:
        temp_target[i][0] = t
        i += 1
    return temp_target

def oneHotDecode(label):
    decoded = []
    for v in label:
        if v[1] == 1:
            decoded.append(1)
        else:
            decoded.append(0)
    return np.array(decoded)

# Load data from CSV file. Edit this to point to the features file
#data, target = FeatureLoader("data/features.csv").loadLibrosaCSV()
# Load data from CSV file. Edit this to point to the features file

feat_type = "sh_entropy"
train_data = np.genfromtxt(main_path+"train-"+feat_type+"-eeg-x.csv",delimiter=',')
train_target =  np.genfromtxt(main_path+"train-"+feat_type+"-eeg-y.csv",delimiter=',')
test_data = np.genfromtxt(main_path+"test-"+feat_type+"-eeg-x.csv",delimiter=',')
test_target =  np.genfromtxt(main_path+"test-"+feat_type+"-eeg-y.csv",delimiter=',')
print train_target.shape
#print train_target

train_target =  np_utils.to_categorical(train_target, 2)
orig_test_target = test_target
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


NCOMPONENTS = 30

pca = PCA(n_components=NCOMPONENTS)
train_data= pca.fit_transform(train_data)
test_data = pca.transform(test_data)
pca_std = np.std(train_data)

print(train_data.shape)


#train model
dnn = DenseNeuralNetwork()
dnn.learn_FFT(train_data,train_target, epochs = 250)
#dnn.test(test_data,test_target)
print test_target.shape

test_pred = dnn.predict(test_data)
test_pred = np.where(test_pred > 0.5, 1, 0)
print ("Accuracy: " + str(accuracy_score(test_target, test_pred, normalize=True)))
print("Loss: " + str(mean_squared_error(test_target, test_pred)))
confusionMatrix(actual=test_target, predicted=test_pred)

"""

i=0

## 5x2 Cross validation T-Test
# First divide the data into 5 sets
kFold = KFold(n_splits=2,shuffle=True, random_state=True)
dnn = DenseNeuralNetwork()
print "5x2 CV T-Test"
test_n = 2
while test_n > 0:
    samples = createSamples(num_sam=5, feature_type="std")
    i = 0
    print "--- K-Fold Cross Validation ---\n"
    for sample in samples:
        for train, test in kFold.split(sample[0], sample[1]):
            i+=1
            train_target = np_utils.to_categorical(sample[1][train], 2)
            test_target = np_utils.to_categorical(sample[1][test], 2)
            dnn.learn(sample[0][train], train_target)
            test_pred = dnn.predict(sample[0][test])
            test_pred= np.where(test_pred > 0.5, 1, 0)

            print ("Accuracy: "+ str(accuracy_score(test_target, test_pred, normalize=True)))
            print("Loss: " + str(mean_squared_error(test_target, test_pred)))
            print "\n"
        print "---"
    test_n-=1"""