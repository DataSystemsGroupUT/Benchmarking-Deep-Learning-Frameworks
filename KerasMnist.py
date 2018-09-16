# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Import for handling arrays
import numpy as np
#Import for Plotting
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

#Import Dataset, neural network
from keras.datasets import mnist
from keras.models import Sequential,load_model
from keras.layers import Dense, Activation
from keras.utils import np_utils

#Load the dataset
#Split the dataset into train set and test set
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# =============================================================================
# 
# #plot 9 examples from the dataset
# fig = plt.figure()
# for i in range(9):
#   plt.subplot(3,3,i+1)
#   plt.tight_layout()
#   plt.imshow(X_train[i], cmap='gray', interpolation='none')
#   plt.title("Digit: {}".format(y_train[i]))
#   plt.xticks([])
#   plt.yticks([])
# fig
# 
# =============================================================================


# The shape before reshaping and normalization
# =============================================================================
# print("X_train shape", X_train.shape)
# print("y_train shape", y_train.shape)
# print("X_test shape", X_test.shape)
# print("y_test shape", y_test.shape)
# 
# =============================================================================
# building the input vector from the 28x28 pixels
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalizing the data to help with the training
X_train /= 255
X_test /= 255

# print the final input shape ready for training
print("Train matrix shape", X_train.shape)
print("Test matrix shape", X_test.shape)

#print(np.unique(y_train, return_counts=True))


# one-hot encoding using keras' numpy-related utilities
#The result is a vector with a length equal to the number of categories.
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)

# building a linear stack of layers with the sequential model
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
#Activation function to introduce nonlinearities
model.add(Activation('relu'))                            

#shape is inferred automatically from first layer
model.add(Dense(512))
model.add(Activation('relu'))

#For the multi-class classification 
#we add another densely-connected (or fully-connected) layer 
#for the 10 different output classes

model.add(Dense(10))
#Activation function for multiclass targets
model.add(Activation('softmax'))

# compiling the  model
#metrics function is cose to the loss function
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')




#the bigger the batch, the more stable our stochastic gradient descent updates will be.
# training the model and saving metrics in SModel
SModel = model.fit(X_train, Y_train,
          batch_size=128, epochs=20,
          verbose=2,
          validation_data=(X_test, Y_test))

# saving the model
save_dir = "/home/nesma/results/"
model_name = 'keras_mnist.h5'
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# plotting the metrics
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(SModel.SModel['acc'])
plt.plot(SModel.SModel['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2,1,2)
plt.plot(SModel.SModel['loss'])
plt.plot(SModel.SModel['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

plt.tight_layout()

fig


#Evaluate the Model's Performance
mnist_model = load_model(model_path)
loss_and_metrics = mnist_model.evaluate(X_test, Y_test, verbose=2)

print("Test Loss", loss_and_metrics[0])
print("Test Accuracy", loss_and_metrics[1])



# load the model and create predictions on the test set
mnist_model = load_model(model_path)
predicted_classes = mnist_model.predict_classes(X_test)

# see which we predicted correctly and which not
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
print()
print(len(correct_indices)," classified correctly")
print(len(incorrect_indices)," classified incorrectly")

# adapt figure size to accomodate 18 subplots
plt.rcParams['figure.figsize'] = (7,14)

figure_evaluation = plt.figure()

# plot 9 correct predictions
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(6,3,i+1)
    plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title(
      "Predicted: {}, Truth: {}".format(predicted_classes[correct],
                                        y_test[correct]))
    plt.xticks([])
    plt.yticks([])

# plot 9 incorrect predictions
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(6,3,i+10)
    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title(
      "Predicted {}, Truth: {}".format(predicted_classes[incorrect], 
                                       y_test[incorrect]))
    plt.xticks([])
    plt.yticks([])

figure_evaluation