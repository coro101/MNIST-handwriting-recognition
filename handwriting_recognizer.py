# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np


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
        w1: (num_l1, num_px)
        b1: (num_l1, 1)
        w2: (num_l2, num_l1)
        b2: (num_l2, 1)
        w3: (num_class, num_l2)
        b2: (num_class, 1)

    :param num_px: Number of pixels of a image, 784
    :param num_class: Number of class, 10
    :return:
    parameters -- A python dictionary of tensors
    """
    num_l1 = 512
    num_l2 = 512

    w1 = tf.get_variable('W1', [num_l1, num_px], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable('b1', [num_l1, 1], initializer=tf.zeros_initializer())
    w2 = tf.get_variable('W2', [num_l2, num_l1], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable('b2', [num_l2, 1], initializer=tf.zeros_initializer())
    w3 = tf.get_variable('W3', [num_class, num_l2], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable('b3', [num_class, 1], initializer=tf.zeros_initializer())

    parameters = {'W1': w1,
                  'b1': b1,
                  'W2': w2,
                  'b2': b2,
                  'W3': w3,
                  'b3': b3}

    return parameters


def forward_propagation(x, parameters, keep_prob):
    """
    Implement of forward propagation of the following model.
        LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFT MAX

    :param x: Placeholder of input data set
    :param parameters: Python dictionary containing W, b
    :param keep_prob: Probability of keeping a neuron active during drop-out
    :return:
    z3 -- The output value before soft max activation function
    """
    w1 = parameters['W1']
    b1 = parameters['b1']
    w2 = parameters['W2']
    b2 = parameters['b2']
    w3 = parameters['W3']
    b3 = parameters['b3']

    z1 = tf.add(tf.matmul(w1, x), b1)
    a1 = tf.nn.dropout(tf.nn.relu(z1), keep_prob)
    z2 = tf.add(tf.matmul(w2, a1), b2)
    a2 = tf.nn.dropout(tf.nn.relu(z2), keep_prob)
    z3 = tf.add(tf.matmul(w3, a2), b3)

    return z3


def compute_cost(z3, y, parameters):
    """
    Compute the cost.

    :param z3: Output of forward propagation
    :param y: Labels
    :param parameters: Python dictionary containing W, b
    :return:
    cost -- A Tensor of the cross entropy cost function
    """
    # Calc L2 loss
    lambd = 0.
    w1 = parameters['W1']
    w2 = parameters['W2']
    w3 = parameters['W3']

    regularize = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3)

    logits = tf.transpose(z3)
    labels = tf.transpose(y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels) +
                          lambd * regularize)
    return cost


def model(mnist, learning_rate=0.0002, num_epochs=100, mini_batch_size=128, print_cost=True):
    """
    Implement of 3-layer tensorflow model.

    :param mnist: Tensorflow handwriting data
    :param learning_rate: Learning rate of gradient descent
    :param num_epochs: Number of iteration epoch loop
    :param mini_batch_size: Size of mini batch
    :param print_cost: If true, print the cost
    :return:
    parameters -- Parameters learnt by this model
    """
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
    keep_prob = tf.placeholder(tf.float32)

    # Initialize parameters
    parameters = initialize_parameters(num_px, num_class)

    # Forward propagation
    z3 = forward_propagation(x, parameters, keep_prob)

    # Cost function
    cost = compute_cost(z3, y, parameters)

    # Back propagation
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initialize tensorflow variables
    init = tf.global_variables_initializer()

    costs = []
    with tf.Session() as session:
        session.run(init)

        for epoch in range(num_epochs):
            epoch_cost = 0.
            num_mini_batches = mnist.train.num_examples // mini_batch_size

            for _ in range(num_mini_batches):
                mini_batch_x, mini_batch_y = mnist.train.next_batch(mini_batch_size)
                _, mini_batch_cost = session.run([optimizer, cost], feed_dict={x: mini_batch_x.T,
                                                                               y: mini_batch_y.T,
                                                                               keep_prob: 0.8})
                epoch_cost += mini_batch_cost / num_mini_batches

            if print_cost and epoch % 10 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost and epoch % 5 == 0:
                costs.append(epoch_cost)

        # Plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        parameters = session.run(parameters)

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(z3), tf.argmax(y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({x: train_x, y: train_y, keep_prob: 1}))
        print("Test Accuracy:", accuracy.eval({x: test_x, y: test_y, keep_prob: 1}))

        return parameters


def main():
    mnist = load_mnist()
    model(mnist)


main()
