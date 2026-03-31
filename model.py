import torch
import torch.nn as nn


class Sender(nn.Module):
    def __init__(self, n_objects, n_symbols, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_objects, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_symbols)
        )

    def forward(self, obj_onehot):
        raw = self.net(obj_onehot)
        return nn.functional.gumbel_softmax(raw, tau=1.0, hard=True)


class Receiver(nn.Module):
    def __init__(self, n_objects, n_symbols, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_symbols + n_objects, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, sig, cands):
        out = []
        for c in cands:
            x = torch.cat([sig, c])
            out.append(self.net(x))
        return torch.stack(out).squeeze()