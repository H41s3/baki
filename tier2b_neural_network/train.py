import numpy as np

# DATASET
# same 8 rows as the logistic regression, so the two models are comparable
hours = np.array([1, 2, 3, 4, 5, 6, 7, 8])
passed = np.array([0, 0, 0, 1, 0, 1, 1, 1])
n = len(hours)

# PARAMETERS
# same hand-picked starting values as the one-row trace. The two hidden
# units start different on purpose, otherwise they would stay identical.
w = np.array([0.5, -0.3])  # input -> hidden, one weight per hidden unit
b = np.array([0.1, 0.2])  # one bias per hidden unit
v = np.array([0.4, -0.6])  # hidden -> output, one weight per hidden unit
c = 0.05  # output bias
learning_rate = 0.5
epochs = 20000


def sigmoid(z):
    """Squash any real number into (0, 1)."""
    return 1 / (1 + np.exp(-z))


def forward(hours, w, b, v, c):
    """
    Run every row through the network at once.

    Returns:
        h: hidden activations, shape (8, 2): one row per data point,
           one column per hidden unit
        y_hat: predicted probability of passing, shape (8,)
    """
    # hours[:, None] turns shape (8,) into (8, 1) so it can pair with w's
    # two entries and produce all 8 x 2 hidden scores in one move
    z1 = hours[:, None] * w + b
    h = sigmoid(z1)
    # axis=1 adds across the two hidden units, giving one score per row
    z2 = np.sum(h * v, axis=1) + c
    y_hat = sigmoid(z2)
    return h, y_hat


def binary_cross_entropy(predictions, actual):
    """Log loss: lower means the probabilities match the 0/1 labels better."""
    # ln(0) is -infinity, so keep predictions off the exact edges
    eps = 1e-12
    predictions = np.clip(predictions, eps, 1 - eps)
    return -np.mean(
        actual * np.log(predictions) + (1 - actual) * np.log(1 - predictions)
    )


def backward(hours, h, y_hat, actual, v):
    """Gradients of the loss for all 7 parameters, averaged over the rows."""
    dz2 = y_hat - actual  # shape (8,): one error per row
    dv = np.mean(dz2[:, None] * h, axis=0)
    dc = np.mean(dz2)
    dh = dz2[:, None] * v  # shape (8, 2): blame handed to each hidden unit
    dz1 = dh * h * (1 - h)
    # axis=0 averages down the 8 rows, leaving one number per hidden unit
    dw = np.mean(dz1 * hours[:, None], axis=0)
    db = np.mean(dz1, axis=0)
    return dw, db, dv, dc


# TRAINING LOOP
for epoch in range(epochs):
    h, y_hat = forward(hours, w, b, v, c)
    loss = binary_cross_entropy(y_hat, passed)
    # every gradient is computed BEFORE any update, so they all come from the
    # same snapshot. Updating v first would change dh mid-calculation.
    dw, db, dv, dc = backward(hours, h, y_hat, passed, v)

    w = w - learning_rate * dw
    b = b - learning_rate * db
    v = v - learning_rate * dv
    c = c - learning_rate * dc

    # printed loss is from BEFORE this epoch's update
    if epoch < 3 or epoch % 2000 == 0:
        print(f"Epoch {epoch}: loss = {loss:.4f}")

# RESULTS
_, y_hat = forward(hours, w, b, v, c)
print(f"Final loss = {binary_cross_entropy(y_hat, passed):.4f}")
print(f"w = {w}")
print(f"b = {b}")
print(f"v = {v}")
print(f"c = {c:.4f}")
for hr, label, p in zip(hours, passed, y_hat):
    print(f"hours={hr}  label={label}  probability={p:.4f}")
correct = np.sum((y_hat >= 0.5) == passed)
print(f"Correct at the 0.5 threshold: {correct}/{n}")
print("Logistic regression final loss was 0.3129")
