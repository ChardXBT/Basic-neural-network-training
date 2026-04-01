# Basic Neural Network Training Feedforward Neural Network (FNN)

## Description

This project implements a minimal emergent-communication experiment in which two neural network agents — a **Sender** and a **Receiver** — learn to develop a shared signalling protocol through repeated interaction.

At each training episode, the Sender receives a one-hot encoded target object and produces a discrete signal via the Gumbel-Softmax estimator. The Receiver is then given that signal alongside a randomised lineup of candidate objects and must select the correct target. Both agents are trained end-to-end with a cross-entropy loss using a shared Adam optimiser, so the signal vocabulary emerges solely from the pressure to communicate accurately.

A lightweight HTTP server runs alongside the training loop and streams live metrics to a browser-based dashboard, enabling real-time inspection of accuracy, loss, the learned language map, and a receiver heatmap.

---
## Configuration

All hyper-parameters and display options are defined as module-level constants at the top of `train.py`.

| Constant | Default | Description |
|----------|---------|-------------|
| `N_OBJECTS` | `6` | Number of distinct objects in the world |
| `N_SYMBOLS` | `4` | Size of the signal vocabulary |
| `HIDDEN` | `64` | Hidden layer width for both Sender and Receiver |
| `LINEUP_SIZE` | `4` | Number of candidates shown to the Receiver per episode |
| `EPISODES` | `20,000` | Total number of training episodes |
| `LR` | `3e-3` | Adam learning rate |
| `LOG_EVERY` | `100` | Episode interval between dashboard state updates |
| `PORT` | `7331` | HTTP server port for the dashboard |

---
### `model.py`

Defines two `torch.nn.Module` subclasses:

- **`Sender(n_objects, n_symbols, hidden)`** — a two-layer MLP that maps a one-hot object vector to a hard discrete signal using `gumbel_softmax` (`tau=1.0`, `hard=True`).
- **`Receiver(n_objects, n_symbols, hidden)`** — a two-layer MLP that scores each candidate object by concatenating the received signal with the candidate's one-hot encoding, then returning a scalar logit per candidate.

### `train.py`

Contains the full training loop and the embedded HTTP server. Key responsibilities:

- Randomly samples a target object and a distractor lineup each episode.
- Computes the cross-entropy loss between the Receiver's logits and the correct candidate position.
- Updates the shared state dictionary (protected by a `threading.Lock`) at every `LOG_EVERY` episodes for consumption by the dashboard endpoint.
- Exposes three HTTP routes: `/` (dashboard HTML), `/state` (JSON snapshot), `/stream` (SSE stream).
