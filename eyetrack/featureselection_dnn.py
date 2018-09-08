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
from utils.analysis import *

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
        self.model.fit(train_data, train_target, verbose=0, batch_size=batch_size, validation_split=0.2, epochs=epochs,  callbacks=callbacks_list, class_weight=cw)



    def learn_std(self,train_data,train_target, drp_rate=0.1, epochs=150, batch_size=5,
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
        self.model.add(Activation("sigmoid"))


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
        self.model.compile(loss='categorical_crossentropy', optimizer=Adamax(), metrics=['accuracy'])
        self.model.fit(train_data, train_target, batch_size=batch_size, validation_split=0.2, epochs=epochs,  callbacks=callbacks_list, class_weight=cw)

    def learn_ave(self,train_data,train_target, drp_rate=0.1, epochs=150, batch_size=5,
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
        self.model.compile(loss='categorical_crossentropy', optimizer=Adamax(), metrics=['accuracy'])
        self.model.fit(train_data, train_target, batch_size=batch_size, validation_split=0.2, epochs=epochs,  callbacks=callbacks_list, class_weight=cw)

    def learn_mean(self,train_data,train_target, drp_rate=0.1, epochs=150, batch_size=5,
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
        self.model.fit(train_data, train_target, batch_size=batch_size, verbose=0,validation_split=0.2, epochs=epochs,  callbacks=callbacks_list, class_weight=cw)


    def test(self,test_data,test_target):
        score = self.model.evaluate(test_data, test_target, batch_size=5)
        print('\n')
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        #print recall_score(test_target, self.model.predict(test_data))


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



# Load data from CSV file. Edit this to point to the features file
train_data = np.genfromtxt(main_path+"train-eye-x.csv",delimiter=',')
train_target =  np.genfromtxt(main_path+'train-eye-y.csv',delimiter=',')
test_data = np.genfromtxt(main_path+"test-eye-x.csv",delimiter=',')
test_target =  np.genfromtxt(main_path+'test-eye-y.csv',delimiter=',')

gnb = GaussianNB()


print train_data.shape


# Sequential Forward Selection
sfs = SequentialFeatureSelector(gnb,
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
train_data = selectFeatures(sfs.k_feature_idx_, train_data)
test_data = selectFeatures(sfs.k_feature_idx_, test_data)
print test_target.shape
train_target =  np_utils.to_categorical(train_target, 2)
test_target =  np_utils.to_categorical(test_target, 2)
#train model
dnn = DenseNeuralNetwork()
dnn.learn_ave(train_data,train_target, epochs = 500)
dnn.test(test_data,test_target)
test_pred = dnn.predict(test_data)
confusionMatrix(actual=test_target, predicted=test_pred)

## 10 fold Cross validation T-Test
# First divide the data into 5 sets
kFold = KFold(n_splits=10,shuffle=True, random_state=True)
print "--- K-Fold Cross Validation ---\n"
test_n = 10
sum_ac = 0.0
n=0
ac= list()
while test_n > 0:
    i = 0
    print "-------------------------------------------------\n"
    for train, test in kFold.split(train_data, train_target):
        i+=1
        dnn = DenseNeuralNetwork()
        dnn.learn(train_data[train], train_target[train], 2)
        y_pred = dnn.predict(train_data[test])
        y_pred = np.where(y_pred > 0.5, 1, 0)
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
writeExcel(ac,"results-eye-dnn")