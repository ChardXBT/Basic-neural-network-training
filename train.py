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
N_OBJECTS   = 6
N_SYMBOLS   = 4
HIDDEN      = 64
LINEUP_SIZE = 4
EPISODES    = 20000
LR          = 3e-3
LOG_EVERY   = 100
PORT        = 7331

OBJECT_NAMES  = ["OBJ-0","OBJ-1","OBJ-2","OBJ-3","OBJ-4","OBJ-5"]
SYMBOL_NAMES  = ["SIG-A","SIG-B","SIG-C","SIG-D"]
OBJECT_COLORS = ["#ff6b6b","#4ecdc4","#ffe66d","#a8e6cf","#c9b1ff","#ff9f43"]
SYMBOL_COLORS = ["#7F77DD","#1D9E75","#EF9F27","#D85A30"]

# ── shared state (written by train thread, read by HTTP server) ───────────────
state = {
    "ep": 0, "done": False,
    "chart": [],
    "lang_map": [0] * N_OBJECTS,
    "heatmap": [[0.0]*N_OBJECTS for _ in range(N_SYMBOLS)],
    "recent_acc": 0.0, "overall_acc": 0.0, "loss": 0.0,
    "entropy_pct": 100.0,
    "last_round": {"target": 0, "signal": 0, "lineup": [], "choice": 0, "correct": False},
    "config": {
        "n_objects": N_OBJECTS, "n_symbols": N_SYMBOLS,
        "episodes": EPISODES, "lineup_size": LINEUP_SIZE,
        "obj_names": OBJECT_NAMES, "sym_names": SYMBOL_NAMES,
        "obj_colors": OBJECT_COLORS, "sym_colors": SYMBOL_COLORS,
        "chance": round(100/LINEUP_SIZE, 1)
    }
}
state_lock = threading.Lock()

# ── HTTP server — serves dashboard + SSE stream ───────────────────────────────
DASHBOARD_PATH = os.path.join(os.path.dirname(__file__), "dashboard.html")

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass  # silence request logs

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            with open(DASHBOARD_PATH, "rb") as f:
                self.wfile.write(f.read())

        elif self.path == "/state":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with state_lock:
                self.wfile.write(json.dumps(state).encode())

        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    with state_lock:
                        data = json.dumps(state)
                    self.wfile.write(f"data: {data}\n\n".encode())
                    self.wfile.flush()
                    if state["done"]: break
                    time.sleep(0.3)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404)
            self.end_headers()

# ── helpers ───────────────────────────────────────────────────────────────────
def get_language_map(sender):
    mapping = []
    with torch.no_grad():
        for i in range(N_OBJECTS):
            obj = torch.zeros(N_OBJECTS); obj[i] = 1.0
            mapping.append(sender.net(obj).argmax().item())
    return mapping

def get_entropy(sender):
    total = 0.0
    with torch.no_grad():
        for i in range(N_OBJECTS):
            obj = torch.zeros(N_OBJECTS); obj[i] = 1.0
            probs = torch.softmax(sender.net(obj), dim=0).numpy()
            total += -np.sum(probs * np.log(probs + 1e-9))
    return total / N_OBJECTS

def get_heatmap(receiver):
    heatmap = []
    with torch.no_grad():
        for s in range(N_SYMBOLS):
            s_vec = torch.zeros(N_SYMBOLS); s_vec[s] = 1.0
            row = []
            for o in range(N_OBJECTS):
                obj_vec = torch.zeros(N_OBJECTS); obj_vec[o] = 1.0
                row.append(round(receiver.net(torch.cat([s_vec, obj_vec])).item(), 3))
            heatmap.append(row)
    for row in heatmap:
        mn, mx = min(row), max(row)
        span = mx - mn if mx != mn else 1
        for j in range(len(row)):
            row[j] = round((row[j] - mn) / span, 3)
    return heatmap

# ── training ──────────────────────────────────────────────────────────────────
def train():
    sender   = Sender(N_OBJECTS, N_SYMBOLS, HIDDEN)
    receiver = Receiver(N_OBJECTS, N_SYMBOLS, HIDDEN)
    opt = torch.optim.Adam(
        list(sender.parameters()) + list(receiver.parameters()), lr=LR
    )
    history = []

    for ep in range(EPISODES):
        target_idx = np.random.randint(N_OBJECTS)
        target = torch.zeros(N_OBJECTS); target[target_idx] = 1.0

        others = np.random.choice(
            [i for i in range(N_OBJECTS) if i != target_idx],
            size=LINEUP_SIZE - 1, replace=False
        )
        lineup_idxs = [target_idx] + others.tolist()
        np.random.shuffle(lineup_idxs)

        candidates = [torch.zeros(N_OBJECTS) for _ in lineup_idxs]
        for i, idx in enumerate(lineup_idxs): candidates[i][idx] = 1.0

        sig_out = sender.forward(target)
        logits  = receiver.forward(sig_out, candidates)

        correct_pos = lineup_idxs.index(target_idx)
        label = torch.tensor(correct_pos)
        loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), label.unsqueeze(0))

        opt.zero_grad(); loss.backward(); opt.step()

        sig_idx  = sig_out.argmax().item()
        choice   = logits.argmax().item()
        correct  = choice == correct_pos
        history.append(1 if correct else 0)

        if ep % LOG_EVERY == 0:
            window   = history[-500:] if len(history) >= 500 else history
            recent   = round(np.mean(window) * 100, 1)
            overall  = round(np.mean(history) * 100, 1)
            ent_pct  = round(get_entropy(sender) / np.log(N_SYMBOLS) * 100, 1)
            lang_map = get_language_map(sender)
            heatmap  = get_heatmap(receiver)

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
                state["chart"].append({"ep": ep, "acc": recent})

    with state_lock:
        state["ep"]   = EPISODES
        state["done"] = True
        state["lang_map"] = get_language_map(sender)
        state["heatmap"]  = get_heatmap(receiver)

    print("\n  training complete — dashboard will show final results.")

# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    server = HTTPServer(("localhost", PORT), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    url = f"http://localhost:{PORT}"
    print(f"\n  dashboard → {url}")
    print(f"  open it in your browser, then training will begin...\n")
    time.sleep(1)
    webbrowser.open(url)
    time.sleep(1)

    train()

    print(f"  keep the browser open to explore results.")
    print(f"  press Ctrl+C to quit.\n")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("  bye!")