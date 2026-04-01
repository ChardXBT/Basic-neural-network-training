# Basic Neural Network Training

## Description

This project implements a minimal emergent-communication experiment in which two neural network agents — a **Sender** and a **Receiver** — learn to develop a shared signalling protocol through repeated interaction.

At each training episode, the Sender receives a one-hot encoded target object and produces a discrete signal via the Gumbel-Softmax estimator. The Receiver is then given that signal alongside a randomised lineup of candidate objects and must select the correct target. Both agents are trained end-to-end with a cross-entropy loss using a shared Adam optimiser, so the signal vocabulary emerges solely from the pressure to communicate accurately.

A lightweight HTTP server runs alongside the training loop and streams live metrics to a browser-based dashboard, enabling real-time inspection of accuracy, loss, the learned language map, and a receiver heatmap.

---

## Requirements

The following packages are required. No `requirements.txt` is included in the repository; install dependencies manually using the commands in the [Installation](#installation) section.

| Package | Purpose |
|---------|---------|
| Python >= 3.8 | Runtime |
| PyTorch >= 1.13 | Neural network construction and training |
| NumPy >= 1.21 | Numerical utilities and random sampling |

The dashboard is rendered entirely in the browser and relies on the following CDN-hosted library (no local installation required):

| Library | Version |
|---------|---------|
| Chart.js | 4.4.1 |

---

## Installation

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate.bat     # Windows
```

2. Install the required Python packages:

```bash
pip install torch numpy
```

---

## Usage

Run the training script from the repository root:

```bash
python train.py
```

On startup, the script will:

1. Launch a local HTTP server on port `7331`.
2. Print the dashboard URL to the terminal.
3. Attempt to open the dashboard automatically in the default browser.
4. Begin the training loop.

Once training is complete (after 20,000 episodes), the server remains running so that the final results can be explored in the dashboard. Press `Ctrl+C` to shut down.

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

## Dashboard

The live dashboard is served at `http://localhost:7331` and provides the following views:

- **Accuracy chart** — rolling recent accuracy and overall accuracy plotted over training episodes.
- **Language map** — the discrete symbol each object is consistently mapped to by the Sender.
- **Receiver heatmap** — a normalised grid showing how strongly the Receiver associates each signal with each object.
- **Last round** — the target object, emitted signal, candidate lineup, and whether the Receiver's choice was correct.
- **Entropy** — the percentage of maximum Shannon entropy in the Sender's output distribution, indicating how deterministic the learned protocol is.

State is pushed from the training thread to the browser via a Server-Sent Events (`/stream`) endpoint and polled at 300 ms intervals.

---

## Project Structure

```
.
├── model.py         # Sender and Receiver network definitions
├── train.py         # Training loop, HTTP server, and dashboard state management
└── dashboard.html   # Browser-based live training dashboard
```

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
