"""Independent binary heads sharing the fixed-cycle LSTM encoder.

Only the direct Stage1 -> Stage2 connection is removed. Both losses still
backpropagate through the same encoder; Stage2 never consumes Stage1 output.
"""
import torch
from torch import nn


class IndependentHeadIntegrationOrderLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=30, num_layers=1,
                            batch_first=True)
        self.stage1 = nn.Linear(30, 1)
        self.stage2 = nn.Linear(30, 1)

    def forward(self, level, difference):
        _, (h_level, _) = self.lstm(level.unsqueeze(-1))
        logit1 = self.stage1(h_level[-1]).squeeze(-1)
        _, (h_diff, _) = self.lstm(difference.unsqueeze(-1))
        logit2 = self.stage2(h_diff[-1]).squeeze(-1)
        return torch.stack((logit1, logit2), dim=1)
