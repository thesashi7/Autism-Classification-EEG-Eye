# from keras.models import Sequential
from keras.utils import np_utils
# from keras.layers.core import Dense, Activation, Dropout

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Dense, Activation, convolutional, pooling, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adamax, Adam, SGD
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from config.config import *
from utils.analysis import *


class ConvolutionNeuralNetwork:

    def __init__(self):
        self.model = None


    def learn(self,train_data, train_target, cn_drp_rate=0.2, d_drp_rate=0.5, epochs=150, batch_size=40,
              num_class=2):


        train_target = np_utils.to_categorical(train_target, num_class)
        train_data = train_data.reshape(train_data.shape[0], 1, train_data.shape[1], 1)

        self.model = Sequential()


        #Adding first convolution layer with dropout and maxpooling with activation relu
        self.model.add(
            convolutional.Conv2D(filters=32, kernel_size=(3,3),strides=(1, 1),padding='same',activation='relu'
                                 ,input_shape=(1,193,1)))
        #model.add(Dropout(cn_drp_rate))
        self.model.add(pooling.MaxPooling2D(pool_size=(2, 2),padding='same'))

        #Adding second convolution layer with activation relu
        self.model.add(convolutional.Conv2D(48,(5,5),padding='same'))
        self.model.add(Activation('relu'))


        #Adding third convolution layer with activation relu and dropout
        self.model.add(convolutional.Conv2D(48,(5,5),padding='same'))
        self.model.add(Activation('relu'))
        #model.add(Dropout(cn_drp_rate))


        #Adding fourth convolution layer with activation relu and dropout
        self.model.add(convolutional.Conv2D(64,(5,5),padding='same'))
        self.model.add(Activation('relu'))
        #model.add(Dropout(cn_drp_rate))


        #Adding fifth convolution layer with activation relu and dropout
        self.model.add(convolutional.Conv2D(64,(5,5),padding='same'))
        self.model.add(Activation('relu'))
        #model.add(Dropout(cn_drp_rate))


        #model.add(pooling.MaxPooling2D(pool_size=(2,2),padding='same'))
        #model.add(convolutional.Conv2D(48,(5,5),padding='same'))
        #model.add(Activation('relu'))
        #model.add(Dropout(cn_drp_rate))
        #model.add(pooling.MaxPooling2D(pool_size=(2,2),padding='same'))

        # Adding flatten layer and fully connected layer or deep neural network
        self.model.add(Flatten())
        #Adding first dense layer with acitivation relu and dropout
        self.model.add(Dense(150))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(d_drp_rate))
        #Adding second dense layer with acitivation relu and dropout
        self.model.add(Dense(150))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(d_drp_rate))

        self.model.add(Dense(150))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(d_drp_rate))



        #Adding final output layer with softmax
        self.model.add(Dense(units=num_class))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer=Adamax(), metrics=['accuracy'])
        self.model.fit(train_data, train_target, batch_size = batch_size, epochs=epochs)

    def test(self, test_data, test_target):
        score = self.model.evaluate(test_data, test_target, batch_size=10)
        print('\n')
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def save(self, name=""):
        f_name = "model/model-cnn"
        name = f_name + "-" + name
        model_json = self.model.to_json()
        with open(name + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(name + ".h5")
        print("Saved model to disk")

    def load(self,name=""):
        # load json and create model
        f_name = "model/model-cnn"
        name = f_name + "-" + name
        # load json and create model
        json_file = open(name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(name + ".h5")
        print("Loaded model from disk")

        # evaluate loaded model on test data
        loaded_model.compile(loss='categorical_crossentropy', optimizer=Adamax(), metrics=['accuracy'])
        self.model = loaded_model

    def predict(self,test_data):
        prediction = self.model.predict(test_data)
        return prediction

# Load data from CSV file. Edit this to point to the features file
train_data = np.genfromtxt(main_path+"train-comb-x.csv",delimiter=',')
train_target =  np.genfromtxt(main_path+'train-comb-y.csv',delimiter=',')
test_data = np.genfromtxt(main_path+"test-comb-x.csv",delimiter=',')
test_target =  np.genfromtxt(main_path+'test-comb-y.csv',delimiter=',')
print train_target.shape
#print train_target

train_target =  np_utils.to_categorical(train_target, 2)
test_target =  np_utils.to_categorical(test_target, 2)
train_data = train_data.reshape(train_data.shape[0],1,train_data.shape[1],1)
test_data = test_data.reshape(test_data.shape[0],1,test_data.shape[1],1)
dnn = ConvolutionNeuralNetwork()
dnn.learn(train_data,train_target, epochs = 150)
dnn.test(test_data,test_target)