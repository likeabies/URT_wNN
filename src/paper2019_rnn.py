"""Vanilla RNN extension; existing ANN and LSTM implementations are untouched."""
import torch.nn as nn


class Paper2019RNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=30, num_layers=1,
                          nonlinearity='tanh', batch_first=True, dropout=0)
        self.fc = nn.Linear(30, 2)

    def forward(self, x):
        _, hidden = self.rnn(x.unsqueeze(-1))
        return self.fc(hidden[-1])
