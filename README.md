# Benchmarking-Deep-Learning-Frameworks
## The frameworks that are included in this benchmark are:
  1- [Keras](https://keras.io/)
<br /> 
  2-Chainer <br />
  3-Tensorflow <br />
  4-Pytorch <br />
  5-Theano <br />
  6-Mxnet <br />

## The Experiments comparing those frameworks over 4 datsets:
  1- MNIST <br />
  2- CIFAR10 <br />
  3- CIFAR100 <br />
  4- SVHN <br />

There are two experiments one of them uses CPU and the other uses GPU.

## This repository is divided into 3 folders:
  1- CPU Experiment <br />
      It contain the CPU source code <br />
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
