import torch
from torch import nn


class LSTMClassifier(nn.Module):
    """Many-to-one LSTM classifier using final hidden state only."""

    def __init__(self, input_size: int = 1, hidden_size: int = 30, num_layers: int = 1, num_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        return self.fc(last_hidden)
