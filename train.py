import torch
import numpy as np
import json
import time
import threading
import webbrowser
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from model import Sender, Receiver

# ── config ────────────────────────────────────────────────────────────────────
N_OBJECTS   = 6      # total number of distinct objects in the world
N_SYMBOLS   = 4      # vocabulary size (number of symbols the Sender can emit)
HIDDEN      = 64     # hidden-layer width for both Sender and Receiver MLPs
LINEUP_SIZE = 4      # number of candidates shown to the Receiver each episode
EPISODES    = 20000  # total number of training episodes
LR          = 3e-3   # Adam learning rate
LOG_EVERY   = 100    # how often (in episodes) to update the dashboard state
PORT        = 7331   # local port for the live-dashboard HTTP server

# Human-readable labels and colours used by the dashboard
OBJECT_NAMES  = ["OBJ-0","OBJ-1","OBJ-2","OBJ-3","OBJ-4","OBJ-5"]
SYMBOL_NAMES  = ["SIG-A","SIG-B","SIG-C","SIG-D"]
OBJECT_COLORS = ["#ff6b6b","#4ecdc4","#ffe66d","#a8e6cf","#c9b1ff","#ff9f43"]
SYMBOL_COLORS = ["#7F77DD","#1D9E75","#EF9F27","#D85A30"]

# ── shared state (written by train thread, read by HTTP server) ───────────────
# This dictionary is updated by the training thread and served as JSON to the
# dashboard via the /state and /stream endpoints.
state = {
    "ep": 0, "done": False,
    "chart": [],                                      # accuracy history for the line chart
    "lang_map": [0] * N_OBJECTS,                      # symbol each object currently maps to
    "heatmap": [[0.0]*N_OBJECTS for _ in range(N_SYMBOLS)],  # receiver scores (symbol × object)
    "recent_acc": 0.0, "overall_acc": 0.0, "loss": 0.0,
    "entropy_pct": 100.0,                             # sender entropy as % of maximum entropy
    "last_round": {"target": 0, "signal": 0, "lineup": [], "choice": 0, "correct": False},
    "config": {
        "n_objects": N_OBJECTS, "n_symbols": N_SYMBOLS,
        "episodes": EPISODES, "lineup_size": LINEUP_SIZE,
        "obj_names": OBJECT_NAMES, "sym_names": SYMBOL_NAMES,
        "obj_colors": OBJECT_COLORS, "sym_colors": SYMBOL_COLORS,
        "chance": round(100/LINEUP_SIZE, 1)           # random-chance baseline accuracy (%)
    }
}
# Lock that must be held whenever reading or writing `state` from either thread
state_lock = threading.Lock()

# ── HTTP server — serves dashboard + SSE stream ───────────────────────────────
# Resolve the dashboard HTML file relative to this script so the server can
# be launched from any working directory.
DASHBOARD_PATH = os.path.join(os.path.dirname(__file__), "dashboard.html")

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass  # silence request logs

    def do_GET(self):
        if self.path == "/":
            # Serve the static dashboard HTML page
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            with open(DASHBOARD_PATH, "rb") as f:
                self.wfile.write(f.read())

        elif self.path == "/state":
            # Return the current training state as a one-shot JSON snapshot
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with state_lock:
                self.wfile.write(json.dumps(state).encode())

        elif self.path == "/stream":
            # Push live state updates to the browser via Server-Sent Events (SSE).
            # The browser opens this endpoint and receives a new JSON event every
            # 300 ms until training is complete.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    with state_lock:
                        data = json.dumps(state)
                    # SSE format: each message must be prefixed with "data: " and end with "\n\n"
                    self.wfile.write(f"data: {data}\n\n".encode())
                    self.wfile.flush()
                    if state["done"]: break  # stop streaming once training finishes
                    time.sleep(0.3)
            except (BrokenPipeError, ConnectionResetError):
                # Client disconnected — exit gracefully without raising an error
                pass
        else:
            self.send_response(404)
            self.end_headers()

# ── helpers ───────────────────────────────────────────────────────────────────
def get_language_map(sender):
    """Return the symbol index that the Sender assigns to each object.

    Runs the Sender's raw network (before Gumbel-softmax) in deterministic mode
    and takes the argmax, giving the most likely symbol for each object.

    Returns:
        A list of length N_OBJECTS where entry i is the symbol index for object i.
    """
    mapping = []
    with torch.no_grad():  # disable gradient tracking for inference
        for i in range(N_OBJECTS):
            obj = torch.zeros(N_OBJECTS); obj[i] = 1.0  # one-hot encode object i
            mapping.append(sender.net(obj).argmax().item())
    return mapping

def get_entropy(sender):
    """Compute the average Shannon entropy of the Sender's symbol distribution.

    High entropy means the Sender spreads probability across many symbols;
    low entropy means it consistently uses a small set of symbols (a more
    structured language).  The result is averaged over all objects.

    Returns:
        Mean entropy (nats) averaged across all N_OBJECTS objects.
    """
    total = 0.0
    with torch.no_grad():
        for i in range(N_OBJECTS):
            obj = torch.zeros(N_OBJECTS); obj[i] = 1.0  # one-hot encode object i
            probs = torch.softmax(sender.net(obj), dim=0).numpy()  # convert logits to probabilities
            total += -np.sum(probs * np.log(probs + 1e-9))         # H = -Σ p log p (1e-9 avoids log(0))
    return total / N_OBJECTS  # average entropy across all objects

def get_heatmap(receiver):
    """Build a normalised heatmap of receiver compatibility scores (symbol × object).

    For every (symbol, object) pair, feeds the concatenated one-hot vectors
    through the Receiver's raw network and records the scalar score.  Each row
    (symbol) is then min-max normalised to [0, 1] so that the dashboard can
    display relative preferences clearly.

    Returns:
        A list of N_SYMBOLS rows, each containing N_OBJECTS floats in [0, 1].
    """
    heatmap = []
    with torch.no_grad():
        for s in range(N_SYMBOLS):
            s_vec = torch.zeros(N_SYMBOLS); s_vec[s] = 1.0  # one-hot encode symbol s
            row = []
            for o in range(N_OBJECTS):
                obj_vec = torch.zeros(N_OBJECTS); obj_vec[o] = 1.0  # one-hot encode object o
                # Score how compatible this symbol is with this object
                row.append(round(receiver.net(torch.cat([s_vec, obj_vec])).item(), 3))
            heatmap.append(row)
    # Min-max normalise each row so values are in [0, 1]
    for row in heatmap:
        mn, mx = min(row), max(row)
        span = mx - mn if mx != mn else 1  # avoid division by zero when all scores are equal
        for j in range(len(row)):
            row[j] = round((row[j] - mn) / span, 3)
    return heatmap

# ── training ──────────────────────────────────────────────────────────────────
def train():
    """Run the emergent-communication referential game.

    Each episode:
      1. A random target object is chosen and one-hot encoded.
      2. A random lineup of LINEUP_SIZE − 1 distractor objects is assembled.
      3. The Sender observes the target and emits a discrete symbol.
      4. The Receiver scores all candidates given the symbol and picks the highest.
      5. Cross-entropy loss between the Receiver's scores and the correct position
         is back-propagated through both agents jointly via Adam.

    Every LOG_EVERY episodes the shared `state` dict is updated so the
    dashboard can display live training metrics.
    """
    sender   = Sender(N_OBJECTS, N_SYMBOLS, HIDDEN)
    receiver = Receiver(N_OBJECTS, N_SYMBOLS, HIDDEN)
    # Jointly optimise both agents with a single Adam optimiser
    opt = torch.optim.Adam(
        list(sender.parameters()) + list(receiver.parameters()), lr=LR
    )
    history = []  # per-episode correctness flags (1 = correct, 0 = wrong)

    for ep in range(EPISODES):
        # ── sample a referential game episode ──────────────────────────────
        target_idx = np.random.randint(N_OBJECTS)           # choose the target object
        target = torch.zeros(N_OBJECTS); target[target_idx] = 1.0  # one-hot encode it

        # Choose LINEUP_SIZE-1 distractors (objects different from the target)
        others = np.random.choice(
            [i for i in range(N_OBJECTS) if i != target_idx],
            size=LINEUP_SIZE - 1, replace=False
        )
        lineup_idxs = [target_idx] + others.tolist()
        np.random.shuffle(lineup_idxs)  # randomise target position in lineup

        # Build one-hot tensors for every candidate in the lineup
        candidates = [torch.zeros(N_OBJECTS) for _ in lineup_idxs]
        for i, idx in enumerate(lineup_idxs): candidates[i][idx] = 1.0

        # ── forward pass ───────────────────────────────────────────────────
        sig_out = sender.forward(target)           # Sender emits a discrete symbol
        logits  = receiver.forward(sig_out, candidates)  # Receiver scores all candidates

        # ── compute loss ───────────────────────────────────────────────────
        correct_pos = lineup_idxs.index(target_idx)       # ground-truth position in lineup
        label = torch.tensor(correct_pos)
        # Cross-entropy penalises the Receiver for placing low probability on the target
        loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), label.unsqueeze(0))

        # ── backward pass + parameter update ──────────────────────────────
        opt.zero_grad(); loss.backward(); opt.step()

        # ── record episode outcome ─────────────────────────────────────────
        sig_idx  = sig_out.argmax().item()    # which symbol the Sender chose
        choice   = logits.argmax().item()     # which candidate the Receiver selected
        correct  = choice == correct_pos      # did the Receiver pick the right object?
        history.append(1 if correct else 0)

        # ── periodic dashboard update ──────────────────────────────────────
        if ep % LOG_EVERY == 0:
            window   = history[-500:] if len(history) >= 500 else history  # recent window
            recent   = round(np.mean(window) * 100, 1)       # accuracy over recent 500 episodes
            overall  = round(np.mean(history) * 100, 1)      # accuracy over all episodes so far
            ent_pct  = round(get_entropy(sender) / np.log(N_SYMBOLS) * 100, 1)  # entropy as % of max
            lang_map = get_language_map(sender)   # deterministic symbol → object mapping
            heatmap  = get_heatmap(receiver)      # normalised receiver score matrix

            with state_lock:
                state["ep"]          = ep
                state["recent_acc"]  = recent
                state["overall_acc"] = overall
                state["loss"]        = round(loss.item(), 4)
                state["entropy_pct"] = ent_pct
                state["lang_map"]    = lang_map
                state["heatmap"]     = heatmap
                state["last_round"]  = {
                    "target": target_idx, "signal": sig_idx,
                    "lineup": lineup_idxs, "choice": lineup_idxs[choice],
                    "correct": correct
                }
                state["chart"].append({"ep": ep, "acc": recent})  # append point to accuracy chart

    # ── training finished — write final state ──────────────────────────────
    with state_lock:
        state["ep"]   = EPISODES
        state["done"] = True                             # signals SSE stream to stop
        state["lang_map"] = get_language_map(sender)    # final language map
        state["heatmap"]  = get_heatmap(receiver)       # final receiver heatmap

    print("\n  training complete — dashboard will show final results.")

# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Start the HTTP server on a background daemon thread so it doesn't block training
    server = HTTPServer(("localhost", PORT), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    url = f"http://localhost:{PORT}"
    print(f"\n  dashboard → {url}")
    print(f"  open it in your browser, then training will begin...\n")
    time.sleep(1)
    webbrowser.open(url)  # automatically open the dashboard in the default browser
    time.sleep(1)

    train()  # run the training loop (blocks until all episodes are complete)

    print(f"  keep the browser open to explore results.")
    print(f"  press Ctrl+C to quit.\n")
    # Keep the main thread alive so the HTTP server keeps running after training
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("  bye!")