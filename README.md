# MNIST Handwriting Recognition with TensorFlow
This project implements 3-layer linear neural network to classify the
[MNIST dataset](http://yann.lecun.com/exdb/mnist/) using the TensorFlow.
I aimed for more than 98% test set accuracy without overfitting.

## Requirements
- Python 3.6
- TensorFlow
- matplotlib
- numpy

## Results

If you read the MNIST successfully using `tensorflow.examples.tutorials.mnist` the output looks similar to below:
```
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz

Number of training examples: 55000
Number of test examples: 10000
```

While repeating the mini-batch loop, the cost is displayed as below:
```
Cost after epoch 0: 0.476846
Cost after epoch 10: 0.033396
Cost after epoch 20: 0.011072
Cost after epoch 30: 0.005313
Cost after epoch 40: 0.004301
Cost after epoch 50: 0.002868
Cost after epoch 60: 0.003237
Cost after epoch 70: 0.002007
Cost after epoch 80: 0.001794
Cost after epoch 90: 0.002296
```

When the gradient descent with the Adam Optimizer is finished, the cost is displayed as a graph with train and test accuracy.
```
Train Accuracy: 0.999982
Test Accuracy: 0.9842
```
