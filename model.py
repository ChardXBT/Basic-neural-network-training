import torch
import torch.nn as nn


class Sender(nn.Module):
    """Neural network agent that observes a target object and emits a discrete symbol (signal).

    Architecture: Linear → ReLU → Linear, followed by a hard Gumbel-softmax
    to produce a one-hot symbol vector that is differentiable during training.
    """

    def __init__(self, n_objects, n_symbols, hidden):
        """
        Args:
            n_objects:  Number of distinct objects in the world (input size).
            n_symbols:  Size of the communication vocabulary (output size).
            hidden:     Number of neurons in the hidden layer.
        """
        super().__init__()
        # Two-layer MLP: maps a one-hot object vector to raw symbol logits
        self.net = nn.Sequential(
            nn.Linear(n_objects, hidden),  # project object → hidden representation
            nn.ReLU(),                     # non-linear activation
            nn.Linear(hidden, n_symbols)   # project hidden → one logit per symbol
        )

    def forward(self, obj_onehot):
        """Produce a discrete symbol for the given one-hot object vector.

        Args:
            obj_onehot: One-hot tensor of shape (n_objects,) identifying the target.

        Returns:
            A one-hot tensor of shape (n_symbols,) representing the chosen symbol.
            Gumbel-softmax with hard=True keeps the output discrete (argmax)
            while still passing gradients through the soft relaxation.
        """
        raw = self.net(obj_onehot)  # compute raw logits for each symbol
        # Apply hard Gumbel-softmax to sample a discrete one-hot symbol
        # tau=1.0 controls the temperature (lower = more peaked distribution)
        return nn.functional.gumbel_softmax(raw, tau=1.0, hard=True)


class Receiver(nn.Module):
    """Neural network agent that receives a symbol from the Sender and picks
    the target object out of a lineup of candidates.

    For each candidate object, it concatenates the signal with the candidate's
    one-hot vector and passes it through a shared MLP to produce a compatibility
    score. The candidate with the highest score is chosen as the target.
    """

    def __init__(self, n_objects, n_symbols, hidden):
        """
        Args:
            n_objects:  Number of distinct objects in the world.
            n_symbols:  Size of the communication vocabulary (signal input size).
            hidden:     Number of neurons in the hidden layer.
        """
        super().__init__()
        # Two-layer MLP: maps (signal ‖ candidate) pair → scalar compatibility score
        self.net = nn.Sequential(
            nn.Linear(n_symbols + n_objects, hidden),  # input: signal concat candidate
            nn.ReLU(),                                  # non-linear activation
            nn.Linear(hidden, 1)                        # output: single compatibility score
        )

    def forward(self, sig, cands):
        """Score every candidate object given the received signal.

        Args:
            sig:   One-hot signal tensor of shape (n_symbols,) from the Sender.
            cands: List of one-hot candidate tensors, each of shape (n_objects,).

        Returns:
            A 1-D tensor of length len(cands) containing one score per candidate.
            Higher scores indicate the Receiver believes that candidate is the target.
        """
        out = []
        for c in cands:
            x = torch.cat([sig, c])  # concatenate signal and candidate into one vector
            out.append(self.net(x))  # score this (signal, candidate) pair
        # Stack individual scores into a single tensor and remove the trailing dimension
        return torch.stack(out).squeeze()