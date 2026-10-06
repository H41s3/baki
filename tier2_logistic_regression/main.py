import numpy as np

# DATASET
# overlap at hours 4 and 5 is deliberate: if the classes separated perfectly,
# the weights would grow forever chasing a "perfect" boundary that never arrives
hours = np.array([1, 2, 3, 4, 5, 6, 7, 8])
passed = np.array([0, 0, 0, 1, 0, 1, 1, 1])

# MODEL
weight = 0.0
bias = 0.0


def sigmoid(z):
    """Squash any real number into (0, 1) so it reads as a probability."""
    return 1 / (1 + np.exp(-z))


def predict(hours, weight, bias):
    """Probability of passing for each hours value."""
    return sigmoid(weight * hours + bias)


def binary_cross_entropy(predictions, actual):
    """Log loss: lower means the probabilities match the 0/1 labels better."""
    # ln(0) is -infinity, so keep predictions off the exact edges
    eps = 1e-12
    predictions = np.clip(predictions, eps, 1 - eps)
    return -np.mean(
        actual * np.log(predictions) + (1 - actual) * np.log(1 - predictions)
    )


def compute_gradients(hours, predictions, actual):
    """Gradients of the loss w.r.t. weight and bias."""
    errors = predictions - actual
    return np.mean(errors * hours), np.mean(errors)


predictions = predict(hours, weight, bias)
print(predictions)
print(f"Loss: {binary_cross_entropy(predictions, passed):.4f}")
gw, gb = compute_gradients(hours, predictions, passed)
print(f"Gradient weight: {gw}, Gradient bias: {gb}")
