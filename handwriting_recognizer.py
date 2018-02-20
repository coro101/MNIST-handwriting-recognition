# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data


def load_mnist(mnist_path='MNIST_data/'):
    """
    Load mnist data from tensorflow.

    :param mnist_path: Path where the mnist data will be stored
    :return:
    mnist -- Tensorflow handwriting data
        train - Flattened and normalized images of shape (55000, 784), width and height of a image is 28
        test - Flattened and normalized images of shape (10000, 784), width and height of a image is 28
        label - One hot encoded labels of shape (55000, 10)
    """
    mnist = input_data.read_data_sets(mnist_path, one_hot=True)

    print('Number of training examples: ' + str(mnist.train.num_examples))
    print('Number of test examples: ' + str(mnist.test.num_examples))

    return mnist


def main():
    mnist = load_mnist()


main()
