import numpy as np

# ONE DATA POINT
# row 4 of the Tier 2 dataset: hours[3] and passed[3]
x = 4.0
y = 1.0

# PARAMETERS
# hand-picked starting values, NOT learned. Hidden units must start with
# different weights, otherwise both compute the same thing forever.
w = np.array([0.5, -0.3])  # input -> hidden: one weight per hidden unit
b = np.array([0.1, 0.2])  # one bias per hidden unit
v = np.array([0.4, -0.6])  # hidden -> output: one weight per hidden unit
c = 0.05  # output bias
learning_rate = 0.1


def sigmoid(z):
    """Squash any real number into (0, 1)."""
    return 1 / (1 + np.exp(-z))


def forward(x, w, b, v, c):
    """Run one input through the network, keeping every intermediate value."""
    z1 = w * x + b  # raw score of each hidden unit
    h = sigmoid(z1)  # hidden activations
    z2 = np.sum(v * h) + c  # raw score of the output unit
    y_hat = sigmoid(z2)  # predicted probability of passing
    return z1, h, z2, y_hat


def loss_fn(y_hat, y):
    """Binary cross-entropy for a single data point."""
    return -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


# FORWARD PASS
z1, h, z2, y_hat = forward(x, w, b, v, c)
loss = loss_fn(y_hat, y)
print(f"z1 = {z1}")
print(f"h = {h}")
print(f"z2 = {z2:.4f}")
print(f"y_hat = {y_hat:.4f}")
print(f"loss = {loss:.4f}")

# BACKWARD PASS
# each line is one link of the chain rule, working from the loss backwards
dz2 = y_hat - y  # sigmoid + cross-entropy cancel, same as logistic regression
dv = dz2 * h  # how much each hidden->output weight contributed
dc = dz2
dh = dz2 * v  # blame passed back to each hidden unit
dz1 = dh * h * (1 - h)  # pass blame back through each hidden sigmoid
dw = dz1 * x
db = dz1
print(f"dz2 = {dz2:.4f}")
print(f"dv = {dv}")
print(f"dc = {dc:.4f}")
print(f"dh = {dh}")
print(f"dz1 = {dz1}")
print(f"dw = {dw}")
print(f"db = {db}")

# UPDATE
w = w - learning_rate * dw
b = b - learning_rate * db
v = v - learning_rate * dv
c = c - learning_rate * dc
print(f"new w = {w}")
print(f"new b = {b}")
print(f"new v = {v}")
print(f"new c = {c:.4f}")

# did one step help?
_, _, _, y_hat_new = forward(x, w, b, v, c)
print(f"new y_hat = {y_hat_new:.4f}")
print(f"new loss = {loss_fn(y_hat_new, y):.4f}")
