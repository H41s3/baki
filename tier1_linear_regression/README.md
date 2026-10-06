# Tier 1: Linear regression from scratch

## What it builds

A model that predicts exam score from hours studied, using `y = wx + b`. NumPy only, no ML frameworks.

## The method

- MSE loss to measure how wrong the line is
- Gradients derived by hand, then coded
- A 1000-epoch gradient descent loop that nudges `w` and `b` downhill

## Results

| learning rate | outcome                    | final w | final b | loss    |
|---------------|----------------------------|---------|---------|---------|
| 0.01          | converges slowly           | 0.6177  | 2.1361  | 0.4807  |
| 0.05          | converges, matches algebra | 0.6000  | 2.2000  | 0.4800  |
| 0.1           | diverges (explodes)        | n/a     | n/a     | >10^200 |

## What I learned

- The learning rate has a stability boundary: past it, the loss doesn't just stall, it explodes.
- A shallow valley makes small steps crawl, which is why lr=0.01 is still off after 1000 epochs.
- An exact closed-form answer is a great test for an iterative one.

## Run

From the repo root: `python3 tier1_linear_regression/main.py`
