#!/usr/bin/env python

# Set the seed to make changes deterministic
import random
random.seed(1)

import numpy as np
np.random.seed(1)

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

import network
net = network.Network([784, 30, 10])

net.SGD(training_data, 30, 10, 0.001, test_data=test_data)

