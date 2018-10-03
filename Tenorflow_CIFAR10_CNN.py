import tensorflow as tf
import pandas as pd
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
import os
import platform
from subprocess import check_output

def initParameters(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay):
    global Dataset    
    global pbatchSize
    global pnumClasses
    global pEpochs
    global pLearningRate
    global pMomentum
    global pWeightDecay
    Dataset = dataset
    pbatchSize = batchSize
    pnumClasses = numClasses
    pEpochs = epochs
    pLearningRate = learningRate
    pMomentum = momentum
    pWeightDecay = weightDecay
    
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte



def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.


def loadDataCifar10():
    X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
    return X_train, y_train, X_val, y_val, X_test, y_test

####################################################################
    
# define net
class CifarNet():
    def __init__(self):
       
        # To ReLu (?x16x16x32) -> MaxPool (?x16x16x32) -> affine (8192)
        self.Wconv1 = tf.get_variable("Wconv1", shape=[3, 3, 3, 32])
        self.bconv1 = tf.get_variable("bconv1", shape=[32])
        # (32-5)/1 + 1 = 28
        # 28x28x64 = 50176
        self.Wconv2 = tf.get_variable("Wconv2", shape=[3, 3, 32, 32])
        self.bconv2 = tf.get_variable("bconv2", shape=[32])
        
        self.Wconv3 = tf.get_variable("Wconv3", shape=[3, 3, 32, 64])
        self.bconv3 = tf.get_variable("bconv3", shape=[64])
        
        self.Wconv4 = tf.get_variable("Wconv4", shape=[3, 3, 64, 64])
        self.bconv4 = tf.get_variable("bconv4", shape=[64])
        
        # affine layer with 1024
        self.W1 = tf.get_variable("W1", shape=[4096, 1024])
        self.b1 = tf.get_variable("b1", shape=[1024])
        # affine layer with 10
        self.W2 = tf.get_variable("W2", shape=[1024, 10])
        self.b2 = tf.get_variable("b2", shape=[10])        
        
    def forward(self, X, y, is_training):
        
        conv1 = tf.nn.conv2d(X, self.Wconv1, strides=[1, 1, 1, 1], padding='SAME') + self.bconv1
        relu1 = tf.nn.relu(conv1)
        print(X.shape)
        print(relu1.shape)
        # Conv
        conv2 = tf.nn.conv2d(relu1, self.Wconv2, strides=[1, 1, 1, 1], padding='SAME') + self.bconv2
        relu2 = tf.nn.relu(conv2)
        print(conv2.shape)
        
        maxpool = tf.layers.max_pooling2d(relu2, pool_size=(2,2),strides=2)
        drop1 = tf.layers.dropout(inputs=maxpool, training=is_training)
        
        print(drop1.shape)
        conv3 = tf.nn.conv2d(drop1, self.Wconv3, strides=[1, 1, 1, 1], padding='SAME') + self.bconv3
        relu3 = tf.nn.relu(conv3)
        
        conv4 = tf.nn.conv2d(relu3, self.Wconv4, strides=[1, 1, 1, 1], padding='SAME') + self.bconv4
        relu4 = tf.nn.relu(conv4)
        
        maxpool2 = tf.layers.max_pooling2d(relu4, pool_size=(2,2),strides=2)
        drop2 = tf.layers.dropout(inputs=maxpool2, training=is_training)
        
        print(drop2.shape)
        maxpool_flat = tf.reshape(drop2,[-1,4096])
#        # Spatial Batch Normalization Layer (trainable parameters, with scale and centering)
#        bn1 = tf.layers.batch_normalization(inputs=maxpool_flat, center=True, scale=True, training=is_training)
        # Affine layer with 1024 output units
        affine1 = tf.matmul(maxpool_flat, self.W1) + self.b1
        
     
        # ReLU Activation Layer
        relu2 = tf.nn.relu(affine1)
        
        # dropout
        drop1 = tf.layers.dropout(inputs=relu2, training=is_training)
        
        # Affine layer from 1024 input units to 10 outputs
        affine2 = tf.matmul(drop1, self.W2) + self.b2
        
   
        
        self.predict = tf.layers.batch_normalization(inputs=affine2, center=True, scale=True, training=is_training)
        
        return self.predict
    
    def run(self, session, loss_val, Xd, yd,epochs=1, batch_size=64, print_every=100,training=None, plot_losses=False, isSoftMax=False):
        # have tensorflow compute accuracy
        if isSoftMax:
            correct_prediction = tf.nn.softmax(self.predict)
        else:
            correct_prediction = tf.equal(tf.argmax(self.predict,1), y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # shuffle indicies
        train_indicies = np.arange(Xd.shape[0])
        np.random.shuffle(train_indicies)

        training_now = training is not None

        # setting up variables we want to compute (and optimizing)
        # if we have a training function, add that to things we compute
        variables = [mean_loss, correct_prediction, accuracy]
        if training_now:
            variables[-1] = training

        # counter 
        iter_cnt = 0
        for e in range(epochs):
            # keep track of losses and accuracy
            correct = 0
            losses = []
            # make sure we iterate over the dataset once
            for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
                # generate indicies for the batch
                start_idx = (i*batch_size)%Xd.shape[0]
                idx = train_indicies[start_idx:start_idx+batch_size]

                # create a feed dictionary for this batch
                feed_dict = {X: Xd[idx,:],
                             y: yd[idx],
                             is_training: training_now }
                # get batch size
                actual_batch_size = yd[idx].shape[0]

                # have tensorflow compute loss and correct predictions
                # and (if given) perform a training step
                loss, corr, _ = session.run(variables,feed_dict=feed_dict)

                # aggregate performance stats
                losses.append(loss*actual_batch_size)
                correct += np.sum(corr)

                # print every now and then
                if training_now and (iter_cnt % print_every) == 0:
                    print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                          .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
                iter_cnt += 1
            total_correct = correct/Xd.shape[0]
            total_loss = np.sum(losses)/Xd.shape[0]
            print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
                  .format(total_loss,total_correct,e+1))
            if plot_losses:
                plt.plot(losses)
                plt.grid(True)
                plt.title('Epoch {} Loss'.format(e+1))
                plt.xlabel('minibatch number')
                plt.ylabel('minibatch loss')
                plt.show()
        return total_loss, total_correct





def RunCIFAR10(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay):
    
    initParameters(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    
    X_train,y_train,X_val,y_val ,X_test,y_test = loadDataCifar10()
    
    tf.reset_default_graph()
    global X
    global y
    global mean_loss
    global is_training
    X = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int64, [None])
    is_training = tf.placeholder(tf.bool)
    
    net = CifarNet()
    net.forward(X,y,is_training)
    
    
    # Annealing the learning rate
    global_step = tf.Variable(0, trainable=False)

    # Feel free to play with this cell
    mean_loss = None
    optimizer = None
    
    # define our loss
    cross_entr_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y,10), logits=net.predict)
    mean_loss = tf.reduce_mean(cross_entr_loss)
    
    # define our optimizer
    optimizer = tf.train.MomentumOptimizer(learningRate,momentum=momentum)
    
    
    # batch normalization in tensorflow requires this extra dependency
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_step = optimizer.minimize(mean_loss, global_step=global_step)
        
        
    # train with 10 epochs
    sess = tf.Session()
    
    try:
        with tf.device("/cpu:0") as dev:
            sess.run(tf.global_variables_initializer())
            print('Training')
            net.run(sess, mean_loss, X_train, y_train, epochs, batchSize, batchSize, train_step, False)
            print('Validation')
            net.run(sess, mean_loss, X_val, y_val, 1,batchSize )
    except tf.errors.InvalidArgumentError:
        print("no gpu found, please use Google Cloud if you want GPU acceleration")
    
    
    
    # view net model result on train  and validation set
    print('Training')
    net.run(sess, mean_loss, X_train, y_train, 1, batchSize)
    print('Validation')
    net.run(sess, mean_loss, X_val, y_val, 1, batchSize)
    
    
    # check result on test
    print('Test')
    net.run(sess, mean_loss, X_test, y_test, 1, batchSize)
    
    # create a feed dictionary for this batch
    feed_dict = {X: X_test,y: y_test,is_training: False}
    
    # predict
    predict = sess.run(tf.nn.softmax(net.predict), feed_dict=feed_dict)
    predict_df = pd.DataFrame(predict, columns=classes)

    

def runModel(dataset,batchSize=128,numClasses=10,epochs=12,learningRate=0.01,momentum=0.5,weightDecay=1e-6):
    if dataset is "mnist":
        pass#RunMNIST(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    elif dataset is "cifar10":
        RunCIFAR10(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    elif dataset is "cifar100":
        pass#RunCIFAR100(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    elif dataset is "SVHN":
        #fname = 'SVHN/%s_32x32.mat'
        pass#RunSVHN(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay,fname) 
    else:
        print("Choose cifar10 or mnist")

def main():
    
    #runModel("mnist",epochs=1)
    runModel("cifar10",epochs=3)
#    runModel("SVHN",epochs=15)
    #runModel("cifar100",epochs=3)
    
    
  
if __name__ == '__main__':
    main()