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


# TRAINING LOOP
learning_rate = 0.1
epochs = 20000

for epoch in range(epochs):
    predictions = predict(hours, weight, bias)
    loss = binary_cross_entropy(predictions, passed)
    gw, gb = compute_gradients(hours, predictions, passed)

    weight = weight - learning_rate * gw
    bias = bias - learning_rate * gb

    # printed loss is from BEFORE this epoch's update, same lag as Tier 1
    if epoch < 3 or epoch % 100 == 0:
        print(
            f"Epoch {epoch}: loss = {loss:.4f}, "
            f"weight = {weight:.4f}, bias = {bias:.4f}"
        )

final_loss = binary_cross_entropy(predict(hours, weight, bias), passed)
print(f"Final: weight = {weight:.4f}, bias = {bias:.4f}, loss = {final_loss:.4f}")

# the model's 50/50 point: where w*x + b = 0, so sigmoid outputs 0.5
print(f"Decision boundary: {-bias / weight:.2f} hours")
