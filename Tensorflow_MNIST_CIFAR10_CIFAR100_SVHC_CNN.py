import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
import os
import platform
import gzip
import LoggerYN as YN
import scipy.io as sio
import utilsYN as uYN
import warnings
warnings.filterwarnings("ignore", message="Reloaded modules: <module_name>")

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

def NormalizeData(x_train,x_test):
        x_train /= 255
        x_test /= 255
        return x_train, x_test
    


# CIFAR 10 ###############################################################
def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

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




def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cifar-10-batches-py'
    X_train, y_train, X_test, y_test = loadDataCIFAR10_temp(cifar10_dir)

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

def get_CIFAR100_data():
        # Check if cifar data exists
    if not os.path.exists("./cifar-100-batches-py"):
        print("CIFAR-10 dataset can not be found. Please download the dataset from 'https://www.cs.toronto.edu/~kriz/cifar.html'.")
        return

    # Load the dataset
    print("Loading data...")
    data = loadDataCIFAR100_temp()
    X_train = data['X_train']
    y_train = data['Y_train']
    X_test = data['X_test']
    y_test = data['Y_test']
    num_training = 95000
    num_validation=5000
    num_test=210000
    
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def loadDataCIFAR10_temp(ROOT):
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


def loadDataCIFAR100_temp():
    xs = []
    ys = []
    for j in range(5):
      d = unpickle('cifar-100-batches-py/train')
      x = d['data']
      y = d['fine_labels']
      xs.append(x)
      ys.append(y)
    d = unpickle('cifar-100-batches-py/test')
    xs.append(d['data'])
    ys.append(d['fine_labels'])
    x = np.concatenate(xs)/np.float32(255)

    y = np.concatenate(ys)
    x = x.reshape((x.shape[0], 32, 32, 3))
    pixel_mean = np.mean(x[0:50000],axis=0)
    x -= pixel_mean

    # create mirrored images
    X_train = x[0:50000,:,:,:]
    Y_train = y[0:50000]
    X_train_flip = X_train[:,:,:,::-1]
    Y_train_flip = Y_train
    X_train = np.concatenate((X_train,X_train_flip),axis=0)
    Y_train = np.concatenate((Y_train,Y_train_flip),axis=0)
    X_test = x[50000:,:,:,:]
    Y_test = y[50000:]

    return dict(X_train=X_train,Y_train=Y_train.astype('int32'),X_test = X_test,Y_test = Y_test.astype('int32'),)   



def loadDataCIFAR10():
    X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
    return X_train, y_train, X_val, y_val, X_test, y_test

def loadDataCIFAR100():
    X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR100_data()
    return X_train, y_train, X_val, y_val, X_test, y_test


def loadDataMNIST():
    from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)


    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        return data / np.float32(255)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

#    print(X_train.shape)
    global imgRows
    global imgCols
    global imgRGB_Dimensions
    global inputShape
    
    imgRGB_Dimensions = X_train.shape[1]
    imgRows = X_train.shape[2]
    imgCols = X_train.shape[3]
    
    
    inputShape = (imgRows, imgCols, imgRGB_Dimensions)
    
    num_training=59500
    num_validation=500
    num_test=1000
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]
    X_train = X_train.transpose(0,2,3,1).astype("float")
    X_test = X_test.transpose(0,2,3,1).astype("float")
    X_val = X_val.transpose(0,2,3,1).astype("float")
    return X_train, y_train, X_test, y_test,X_val,y_val


def loadDataSVHN(fname,extra=False):

    def load_mat(fname):
        data = sio.loadmat(fname)
        X = data['X'].transpose(3, 0, 1, 2)
        y = data['y'] % 10  # map label "10" --> "0"
        return X, y

    data = uYN.Dataset()
    data.classes = np.arange(10)


    X, y = load_mat(fname % 'train')
    data.train_images = X
    data.train_labels = y.reshape(-1)

    X, y = load_mat(fname % 'test')
    data.test_images = X
    data.test_labels = y.reshape(-1)

    if extra:
        X, y = load_mat(fname % 'extra')
        data.extra_images = X
        data.extra_labels = y.reshape(-1)
    
    (x_train, y_train), (x_test, y_test)  = (data.train_images,data.train_labels),(data.test_images,data.test_labels)
    
    global imgRows
    global imgCols
    global imgRGB_Dimensions
    global inputShape
    
    imgRows = x_train.shape[1]
    imgCols = x_train.shape[2]

    try:
        imgRGB_Dimensions = x_train.shape[3]
    except Exception:
        imgRGB_Dimensions = 1 #For Gray Scale Images

    
    x_train = x_train.reshape(x_train.shape[0], imgRows, imgCols, imgRGB_Dimensions)
    x_test = x_test.reshape(x_test.shape[0], imgRows, imgCols, imgRGB_Dimensions)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train, x_test = NormalizeData(x_train, x_test)
    inputShape = (imgRows, imgCols, imgRGB_Dimensions)
    
    
    x_train = x_train.reshape(len(y_train), imgRows, imgCols, imgRGB_Dimensions)
    x_test = x_test.reshape(len(y_test), imgRows, imgCols, imgRGB_Dimensions)
    
    num_training= x_train.shape[0]- 3257
    num_validation=3257
    mask = range(num_training, num_training + num_validation)
    x_val = x_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    x_train = x_train[mask]
    y_train = y_train[mask]
    
    
    return x_train,y_train,x_val,y_val,x_test,y_test    
    


### Models
    
class ModelMNIST():
    def __init__(self):
       
        # To ReLu (?x16x16x32) -> MaxPool (?x16x16x32) -> affine (8192)
        self.Wconv1 = tf.get_variable("Wconv1", shape=[3, 3, 1, 32])
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
        self.W1 = tf.get_variable("W1", shape=[3136, 1024])
        self.b1 = tf.get_variable("b1", shape=[1024])
        # affine layer with 10
        self.W2 = tf.get_variable("W2", shape=[1024, 10])
        self.b2 = tf.get_variable("b2", shape=[10])        
        
    def forward(self, X, y, is_training):
        conv1 = tf.nn.conv2d(X, self.Wconv1, strides=[1, 1, 1, 1], padding='SAME') + self.bconv1
        relu1 = tf.nn.relu(conv1)
       
        # Conv
        conv2 = tf.nn.conv2d(relu1, self.Wconv2, strides=[1, 1, 1, 1], padding='SAME') + self.bconv2
        relu2 = tf.nn.relu(conv2)
        
        maxpool = tf.layers.max_pooling2d(relu2, pool_size=(2,2),strides=2)
        drop1 = tf.layers.dropout(inputs=maxpool, training=is_training)
        
        conv3 = tf.nn.conv2d(drop1, self.Wconv3, strides=[1, 1, 1, 1], padding='SAME') + self.bconv3
        relu3 = tf.nn.relu(conv3)
        
        conv4 = tf.nn.conv2d(relu3, self.Wconv4, strides=[1, 1, 1, 1], padding='SAME') + self.bconv4
        relu4 = tf.nn.relu(conv4)
        
        maxpool2 = tf.layers.max_pooling2d(relu4, pool_size=(2,2),strides=2)
        drop2 = tf.layers.dropout(inputs=maxpool2, training=is_training)
        
        maxpool_flat = tf.reshape(drop2,[-1,3136])

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

    
    
class ModelSVHN():
    def __init__(self):
       
        # To ReLu (?x16x16x32) -> MaxPool (?x16x16x32) -> affine (8192)
        self.Wconv1 = tf.get_variable("Wconv1", shape=[5, 5, 3, 48])
        self.bconv1 = tf.get_variable("bconv1", shape=[48])
    
    
        self.Wconv2 = tf.get_variable("Wconv2", shape=[5, 5, 48, 64])
        self.bconv2 = tf.get_variable("bconv2", shape=[64])
        
        self.Wconv3 = tf.get_variable("Wconv3", shape=[5, 5, 64, 128])
        self.bconv3 = tf.get_variable("bconv3", shape=[128])
        
        self.Wconv4 = tf.get_variable("Wconv4", shape=[5, 5, 128, 160])
        self.bconv4 = tf.get_variable("bconv4", shape=[160])
     
        self.Wconv5 = tf.get_variable("Wconv5", shape=[5, 5, 160, 192])
        self.bconv5 = tf.get_variable("bconv5", shape=[192])
    
        self.Wconv6 = tf.get_variable("Wconv6", shape=[5, 5, 192, 192])
        self.bconv6 = tf.get_variable("bconv6", shape=[192])

        self.Wconv7 = tf.get_variable("Wconv7", shape=[5, 5, 192, 192])
        self.bconv7 = tf.get_variable("bconv7", shape=[192])

        self.Wconv8 = tf.get_variable("Wconv8", shape=[5, 5, 192, 192])
        self.bconv8 = tf.get_variable("bconv8", shape=[192])        
        
        
        
        # affine layer with 1024
        self.W1 = tf.get_variable("W1", shape=[192, 1024])
        self.b1 = tf.get_variable("b1", shape=[1024])
        # affine layer with 10
        self.W2 = tf.get_variable("W2", shape=[1024, 10])
        self.b2 = tf.get_variable("b2", shape=[10])        
        
    def forward(self, X, y, is_training):
        
        conv1 = tf.nn.conv2d(X, self.Wconv1, strides=[1, 1, 1, 1], padding='SAME') + self.bconv1
        relu1 = tf.nn.relu(conv1)
        maxpool = tf.layers.max_pooling2d(relu1, pool_size=(2,2),strides=2,padding='SAME')
        drop1 = tf.layers.dropout(inputs=maxpool,rate=0.2, training=is_training)
        
        conv2 = tf.nn.conv2d(drop1, self.Wconv2, strides=[1, 1, 1, 1], padding='SAME') + self.bconv2
        relu2 = tf.nn.relu(conv2)
        maxpool2 = tf.layers.max_pooling2d(relu2, pool_size=(2,2),strides=2,padding='SAME')
        drop2 = tf.layers.dropout(inputs=maxpool2,rate=0.2, training=is_training)
        
        conv3 = tf.nn.conv2d(drop2, self.Wconv3, strides=[1, 1, 1, 1], padding='SAME') + self.bconv3
        relu3 = tf.nn.relu(conv3)
        maxpool3 = tf.layers.max_pooling2d(relu3, pool_size=(2,2),strides=2,padding='SAME')
        drop3 = tf.layers.dropout(inputs=maxpool3,rate=0.2, training=is_training)
        
        conv4 = tf.nn.conv2d(drop3, self.Wconv4, strides=[1, 1, 1, 1], padding='SAME') + self.bconv4
        relu4 = tf.nn.relu(conv4)
        maxpool4 = tf.layers.max_pooling2d(relu4, pool_size=(2,2),strides=2,padding='SAME')
        drop4 = tf.layers.dropout(inputs=maxpool4,rate=0.2, training=is_training)
        
        conv5 = tf.nn.conv2d(drop4, self.Wconv5, strides=[1, 1, 1, 1], padding='SAME') + self.bconv5
        relu5 = tf.nn.relu(conv5)
        maxpool5 = tf.layers.max_pooling2d(relu5, pool_size=(2,2),strides=2,padding='SAME')
        drop5 = tf.layers.dropout(inputs=maxpool5,rate=0.2, training=is_training)
        
        conv6 = tf.nn.conv2d(drop5, self.Wconv6, strides=[1, 1, 1, 1], padding='SAME') + self.bconv6
        relu6 = tf.nn.relu(conv6)
        maxpool6 = tf.layers.max_pooling2d(relu6, pool_size=(2,2),strides=2,padding='SAME')
        drop6 = tf.layers.dropout(inputs=maxpool6,rate=0.2, training=is_training)
        
        
        conv7 = tf.nn.conv2d(drop6, self.Wconv7, strides=[1, 1, 1, 1], padding='SAME') + self.bconv7
        relu7 = tf.nn.relu(conv7)
        maxpool7 = tf.layers.max_pooling2d(relu7, pool_size=(2,2),strides=2,padding='SAME')
        drop7 = tf.layers.dropout(inputs=maxpool7,rate=0.2, training=is_training)
        
        conv8 = tf.nn.conv2d(drop7, self.Wconv8, strides=[1, 1, 1, 1], padding='SAME') + self.bconv8
        relu8 = tf.nn.relu(conv8)
        maxpool8 = tf.layers.max_pooling2d(relu8, pool_size=(2,2),strides=2,padding='SAME')
        drop8 = tf.layers.dropout(inputs=maxpool8,rate=0.2, training=is_training)
        
        
        maxpool_flat = tf.reshape(drop8,[-1,192])
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



class ModelCIFAR10():
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
     
        # Conv
        conv2 = tf.nn.conv2d(relu1, self.Wconv2, strides=[1, 1, 1, 1], padding='SAME') + self.bconv2
        relu2 = tf.nn.relu(conv2)
        
        maxpool = tf.layers.max_pooling2d(relu2, pool_size=(2,2),strides=2)
        drop1 = tf.layers.dropout(inputs=maxpool,rate=0.25, training=is_training)
        
        conv3 = tf.nn.conv2d(drop1, self.Wconv3, strides=[1, 1, 1, 1], padding='SAME') + self.bconv3
        relu3 = tf.nn.relu(conv3)
        
        conv4 = tf.nn.conv2d(relu3, self.Wconv4, strides=[1, 1, 1, 1], padding='SAME') + self.bconv4
        relu4 = tf.nn.relu(conv4)
        
        maxpool2 = tf.layers.max_pooling2d(relu4, pool_size=(2,2),strides=2)
        drop2 = tf.layers.dropout(inputs=maxpool2,rate=0.25, training=is_training)
        
        maxpool_flat = tf.reshape(drop2,[-1,4096])
        affine1 = tf.matmul(maxpool_flat, self.W1) + self.b1
        
     
        # ReLU Activation Layer
        relu2 = tf.nn.relu(affine1)
        
        # dropout
        drop1 = tf.layers.dropout(inputs=relu2,rate=0.5, training=is_training)
        
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

class ModelCIFAR100():
    def __init__(self):
       
        # To ReLu (?x16x16x32) -> MaxPool (?x16x16x32) -> affine (8192)
        self.Wconv1 = tf.get_variable("Wconv1", shape=[3, 3, 3, 128])
        self.bconv1 = tf.get_variable("bconv1", shape=[128])
        # (32-5)/1 + 1 = 28
        # 28x28x64 = 50176
        self.Wconv2 = tf.get_variable("Wconv2", shape=[3, 3, 128, 128])
        self.bconv2 = tf.get_variable("bconv2", shape=[128])
        
        self.Wconv3 = tf.get_variable("Wconv3", shape=[3, 3, 128, 256])
        self.bconv3 = tf.get_variable("bconv3", shape=[256])
        
        self.Wconv4 = tf.get_variable("Wconv4", shape=[3, 3, 256, 256])
        self.bconv4 = tf.get_variable("bconv4", shape=[256])
        
        
        self.Wconv5 = tf.get_variable("Wconv5", shape=[3, 3, 256, 512])
        self.bconv5 = tf.get_variable("bconv5", shape=[512])
        
        
        
        self.Wconv6 = tf.get_variable("Wconv6", shape=[3, 3, 512, 512])
        self.bconv6 = tf.get_variable("bconv6", shape=[512])
        
        
        
        # affine layer with 1024
        self.W1 = tf.get_variable("W1", shape=[8192, 1024])
        self.b1 = tf.get_variable("b1", shape=[1024])
        # affine layer with 10
        self.W2 = tf.get_variable("W2", shape=[1024, 100])
        self.b2 = tf.get_variable("b2", shape=[100])        
        
    def forward(self, X, y, is_training):

        conv1 = tf.nn.conv2d(X, self.Wconv1, strides=[1, 1, 1, 1], padding='SAME') + self.bconv1
        relu1 = tf.nn.relu(conv1)
        
        # Conv
        conv2 = tf.nn.conv2d(relu1, self.Wconv2, strides=[1, 1, 1, 1], padding='SAME') + self.bconv2
        relu2 = tf.nn.relu(conv2)
        
        
        maxpool = tf.layers.max_pooling2d(relu2, pool_size=(2,2),strides=2)
        drop1 = tf.layers.dropout(inputs=maxpool, training=is_training)
        
        conv3 = tf.nn.conv2d(drop1, self.Wconv3, strides=[1, 1, 1, 1], padding='SAME') + self.bconv3
        relu3 = tf.nn.relu(conv3)
        
        conv4 = tf.nn.conv2d(relu3, self.Wconv4, strides=[1, 1, 1, 1], padding='SAME') + self.bconv4
        relu4 = tf.nn.relu(conv4)
        
        maxpool2 = tf.layers.max_pooling2d(relu4, pool_size=(2,2),strides=2)
        drop2 = tf.layers.dropout(inputs=maxpool2, training=is_training)
        
        
        
        conv5 = tf.nn.conv2d(drop2, self.Wconv5, strides=[1, 1, 1, 1], padding='SAME') + self.bconv5
        relu5 = tf.nn.relu(conv5)
        
        conv6 = tf.nn.conv2d(relu5, self.Wconv6, strides=[1, 1, 1, 1], padding='SAME') + self.bconv6
        relu6 = tf.nn.relu(conv6)
        
        maxpool3 = tf.layers.max_pooling2d(relu6, pool_size=(2,2),strides=2)
        drop3 = tf.layers.dropout(inputs=maxpool3, training=is_training)
        
        
        
        maxpool_flat = tf.reshape(drop3,[-1,8192])

        affine1 = tf.matmul(maxpool_flat, self.W1) + self.b1
        
     
        # ReLU Activation Layer
        relu7 = tf.nn.relu(affine1)
        
        # dropout
        drop1 = tf.layers.dropout(inputs=relu7, training=is_training)
        
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

def RunMNIST(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay):
    
    initParameters(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    
    X_train, y_train, X_test, y_test, X_val, y_val = loadDataMNIST()
    tf.reset_default_graph()
    global X
    global y
    global mean_loss
    global is_training
    X = tf.placeholder(tf.float32, [None, 28, 28, 1])
    y = tf.placeholder(tf.int64, [None])
    is_training = tf.placeholder(tf.bool)
    
    net = ModelMNIST()
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


####################################################################






def RunCIFAR10(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay):
    
    initParameters(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    
    X_train,y_train,X_val,y_val ,X_test,y_test = loadDataCIFAR10()
    
    tf.reset_default_graph()
    global X
    global y
    global mean_loss
    global is_training
    X = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int64, [None])
    is_training = tf.placeholder(tf.bool)
    
    net = ModelCIFAR10()
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



def RunCIFAR100(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay):
    
    initParameters(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    
    X_train,y_train,X_val,y_val ,X_test,y_test =  loadDataCIFAR100()
    
    tf.reset_default_graph()
    global X
    global y
    global mean_loss
    global is_training
    X = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int64, [None])
    is_training = tf.placeholder(tf.bool)
    
    net = ModelCIFAR100()
    net.forward(X,y,is_training)
    
    
    # Annealing the learning rate
    global_step = tf.Variable(0, trainable=False)

    # Feel free to play with this cell
    mean_loss = None
    optimizer = None
    
    # define our loss
    cross_entr_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y,100), logits=net.predict)
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
    



def RunSVHN(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay,fname):
    
    initParameters(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    
    X_train,y_train,X_val,y_val ,X_test,y_test = loadDataSVHN(fname)
    
    tf.reset_default_graph()
    global X
    global y
    global mean_loss
    global is_training
    X = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int64, [None])
    is_training = tf.placeholder(tf.bool)
    
    net = ModelSVHN()
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


def runModel(dataset,batchSize=128,numClasses=10,epochs=12,learningRate=0.01,momentum=0.5,weightDecay=1e-6):
    if dataset is "mnist":
        RunMNIST(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    elif dataset is "cifar10":
        RunCIFAR10(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    elif dataset is "cifar100":
        RunCIFAR100(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    elif dataset is "SVHN":
        fname = 'SVHN/%s_32x32.mat'
        RunSVHN(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay,fname) 
    else:
        print("Choose cifar10 or mnist")

def main():
    
#    runModel("mnist",epochs=1)
#    runModel("cifar10",epochs=3)
#    runModel("SVHN",epochs=1)
    runModel("cifar100",epochs=3)
    
    
  
if __name__ == '__main__':
    main()
