# Benchmarking-Deep-Learning-Frameworks
## The frameworks that are included in this benchmark are:
  1- [Keras](https://keras.io/)
<br /> 
  2-[Chainer](https://docs.chainer.org/en/stable/glance.html) <br />
  3-[Tensorflow](https://www.tensorflow.org/) <br />
  4-[Pytorch](https://pytorch.org/) <br />
  5-[Theano](http://deeplearning.net/software/theano/) <br />
  6-[Mxnet](https://mxnet.apache.org/) <br />

## The Experiments comparing those frameworks over 4 datsets:
  1- [MNIST](http://yann.lecun.com/exdb/mnist/) <br />
  2- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) <br />
  3- [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) <br />
  4- [SVHN](http://ufldl.stanford.edu/housenumbers/) <br />

There are two experiments one of them uses CPU and the other uses GPU.

## This repository is divided into 3 folders:
  1- CPU Experiment <br />
     Markup :   * It contain the CPU source code <br />
      The Generated graphs <br />
       The logs of the experiment <br />
       <br/>
  2- GPU Experiment <br />
       It contain the GPU experiment <br />
       The Generated graphs <br />
       The logs of the experiment <br />
              <br/>
  3- Installation Guide <br />
       It contains the required packages to be included for each environment.<br />

  
 ## Experiment Logging:
  there exist 3 files for logging the resources during the experiment: CPU log, GPU Log, memory Log.<br />
  
## How to run? 
 1- Install the environment for each framework using the installation  guide <br />
 2- Clone the project <br />
 3- For running the experiment over MNIST datset for Keras framework for example, you will find in CPU folder the source code, There is a file for each framework. <br />
 4- There is a method in the main function that is called runModel, this methods holds the name of the dataset and the number of       epochs needed for this run.
