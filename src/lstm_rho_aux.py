"""Paper2019 single LSTM with one raw scalar rho auxiliary head."""
import torch
from torch import nn
from src.model import Paper2019LSTMClassifier


class Paper2019LSTMRhoAux(Paper2019LSTMClassifier):
    def __init__(self):
        super().__init__()
        # Preserve the baseline encoder/fc initialization AND next shuffle RNG.
        state = torch.get_rng_state()
        self.rho_head = nn.Linear(30, 1)
        torch.set_rng_state(state)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x.unsqueeze(-1))
        h = h_n[-1]
        return self.fc(h), self.rho_head(h).squeeze(-1), h
