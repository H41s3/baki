# baki

Machine learning built from scratch in NumPy before any framework is allowed near it.

## Why the name

Baki Hanma keeps picking fights he isn't ready for, loses, and comes back stronger. Every tier here is a fight I'm underprepared for on purpose. I expect to lose a few rounds. That's the curriculum.

## Roadmap

| Tier | Project                              | Status      | Concept                         |
|------|--------------------------------------|-------------|---------------------------------|
| 1    | Linear regression (NumPy only)       | Done        | loss, gradient descent          |
| 2    | Logistic regression + 2-layer NN     | In progress | sigmoid, cross-entropy, backprop |
| 3    | PyTorch CNN image classifier         | Planned     | autograd, tensors               |
| 4    | Attention / transformers             | Planned     | sequence modeling               |
| 5    | RAG system over my own data          | Planned     | embeddings, retrieval           |
| 6    | Model serving (FastAPI)              | Planned     | inference, deployment           |

## Highlights

- Gradient descent was hand-derived and hand-traced before it was coded.
- The result was verified against the exact closed-form solution (w=0.6, b=2.2) and matched to 4 decimals.
- I deliberately broke it: lr=0.1 made the loss explode past 10^200.

## Run it

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then, from the repo root:

```bash
python3 tier1_linear_regression/main.py
```

Heads up: the script opens a matplotlib window at the end, so it won't exit until you close it.
