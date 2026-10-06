import numpy as np
import matplotlib.pyplot as plt

# DATASET
hours = np.array([1, 2, 3, 4, 5])
scores = np.array([2, 4, 5, 4, 5])

# MODEL
weight = 0.0
bias = 0.0


def predict(hours, weight, bias):
    """
    Compute predicted scores for given hours using current weight/bias.

    Args:
        hours: array of study hours (input feature)
        weight: current model weight (slope)
        bias: current model bias (intercept)

    Returns:
        array of predicted scores, same shape as hours
    """
    return weight * hours + bias


def mse_loss(predictions, actual):
    """
    Mean Squared Error: how wrong the model's predictions are, on average.

    Squaring errors punishes big misses harder than small ones, and keeps
    the function smooth (differentiable) so gradient descent can use it.

    Args:
        predictions: model's predicted scores
        actual: true scores from the dataset

    Returns:
        single float — lower means the model is doing better
    """
    errors = predictions - actual
    return np.mean(errors**2)


def compute_gradients(hours, predictions, actual):
    """
    Compute how much to nudge weight and bias to reduce the loss.

    Derived from the derivative of MSE loss with respect to weight and
    bias — tells gradient descent which direction is "downhill."

    Args:
        hours: input feature values
        predictions: model's current predicted scores
        actual: true scores from the dataset

    Returns:
        (gradient_weight, gradient_bias) tuple
    """
    errors = predictions - actual
    gradient_weight = np.mean(2 * hours * errors)
    gradient_bias = np.mean(2 * errors)
    return gradient_weight, gradient_bias


predictions = predict(hours, weight, bias)
loss = mse_loss(predictions, scores)
print(f"Loss: {loss:.4f}")

gw, gb = compute_gradients(hours, predictions, scores)
print(f"Gradient weight: {gw}, Gradient bias: {gb}")

# TRAINING LOOP
learning_rate = 0.05
epochs = 1000

for epoch in range(epochs):
    predictions = predict(hours, weight, bias)
    (
        gw,
        gb,
    ) = compute_gradients(hours, predictions, scores)

    weight = weight - learning_rate * gw
    bias = bias - learning_rate * gb

    if epoch % 100 == 0:
        loss = mse_loss(predictions, scores)
        print(
            f"Epoch {epoch}: Loss = {loss:.4f}, weight = {weight:.4f}, bias = {bias:.4f}"
        )

final_loss = mse_loss(predict(hours, weight, bias), scores)
print(f"Final: weight= {weight:.4f}, bias = {bias:.4f}, loss = {final_loss:.4f} ")

# predict on the same hours so the line is drawn through the x-values the data lives at
line_predictions = predict(hours, weight, bias)

plt.scatter(hours, scores, label="actual data")
plt.plot(hours, line_predictions, label="fitted line")
plt.xlabel("Hours studied")
plt.ylabel("Exam score")
plt.legend()
plt.show()
