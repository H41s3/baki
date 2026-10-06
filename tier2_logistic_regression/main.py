import numpy as np

"""
DATASET:

overlap at hours 4 and 5 is deliberate: if the classes separated perfectly,

the weights would grow forever chasing a "perfect" boundary that never arrives
"""

hours = np.array([1, 2, 3, 4, 5, 6, 7, 8])
passed = np.array([0, 0, 0, 1, 0, 1, 1, 1])


def sigmoid(z):
    """
    Squash any real number

    Args:
        z: raw score (weight * hours + bias), scalar or array

        Returns:
        value(s) between 0 and 1, same shape as z
    """

    return 1 / (1 + np.exp(-z))


print(sigmoid(0))
print(sigmoid(2))
print(sigmoid(-2))
