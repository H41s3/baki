import numpy as np

# ONE DATA POINT
# row 4 of the Tier 2 dataset: hours[3] and passed[3]
x = 4.0
y = 1.0

# PARAMETERS
# hand-picked starting values, NOT learned. Hidden units must start with
# different weights, otherwise both compute the same thing forever.
w = np.array([0.5, -0.3])
b = np.array([0.1, 0.2])
v = np.array([0.4, -0.6])
c = 0.05  # output bias
learning_rate = 0.1


def sigmoid(z):
    """Squash any real number into (0, 1)."""
    return 1 / (1 + np.exp(-z))


def forward(x, w, b, v, c):
    """Run one input through the network, keeping every intermediate value."""
    z1 = w * x + b  # raw score of each hidden unit
    h = sigmoid(z1)
    z2 = np.sum(v * h) + c  # raw score of the output unit
    y_hat = sigmoid(z2)  # predicted probability of passing
    return z1, h, z2, y_hat
