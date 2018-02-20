# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def load_mnist(mnist_path='MNIST_data/'):
    """
    Load mnist data from tensorflow.

    :param mnist_path: Path where the mnist data will be stored
    :return:
    mnist -- Tensorflow handwriting data
        train - Flattened and normalized images of shape (55000, 784), width and height of a image is 28
        test - Flattened and normalized images of shape (10000, 784), width and height of a image is 28
        (train, test).label - One hot encoded labels of shape (n, 10), n is number of data
    """
    mnist = input_data.read_data_sets(mnist_path, one_hot=True)

    print('Number of training examples: ' + str(mnist.train.num_examples))
    print('Number of test examples: ' + str(mnist.test.num_examples))

    return mnist


def initialize_parameters(num_px, num_class):
    """
    Initialize parameters to build NN.
        w1: (32, num_px)
        b1: (32, 1)
        w2: (16, 32)
        b2: (16, 1)
        w3: (num_class, 16)
        b2: (num_class, 1)

    :param num_px: Number of pixels of a image, 784
    :param num_class: Number of class, 10
    :return:
    parameters -- A python dictionary of tensors
    """
    w1 = tf.get_variable('W1', [32, num_px], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable('b1', [32, 1], initializer=tf.zeros_initializer())
    w2 = tf.get_variable('W2', [16, 32], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable('b2', [16, 1], initializer=tf.zeros_initializer())
    w3 = tf.get_variable('W3', [num_class, 16], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable('b3', [num_class, 1], initializer=tf.zeros_initializer())

    parameters = {'W1': w1,
                  'b1': b1,
                  'W2': w2,
                  'b2': b2,
                  'W3': w3,
                  'b3': b3}

    return parameters


def main():
    mnist = load_mnist()

    # Get transpose of data, because I want one column to represent one datum
    train_x = mnist.train.images.T
    train_y = mnist.train.labels.T
    test_x = mnist.test.images.T
    test_y = mnist.test.labels.T

    num_px = train_x.shape[0]  # number of pixels (26 * 28 = 784)
    num_class = train_y.shape[0]  # number of class should be 10

    # Create placeholders
    x = tf.placeholder(tf.float32, [num_px, None])
    y = tf.placeholder(tf.float32, [num_class, None])

    # Initialize parameters
    parameters = initialize_parameters(num_px, num_class)

    print(str(parameters))


main()
