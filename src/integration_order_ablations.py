"""Requested integration-order controls; original model/DGP remain unchanged."""
import torch
from torch import nn
from src.integration_order_lstm import IntegrationOrderLSTM

ARCHITECTURES = ('detach_p1', 'independent_encoders', 'reduced_l2')


class DetachP1LSTM(IntegrationOrderLSTM):
    def forward(self, level, difference):
        _, (h1, _) = self.lstm(level.unsqueeze(-1))
        logit1 = self.stage1(h1[-1]).squeeze(-1)
        _, (h2, _) = self.lstm(difference.unsqueeze(-1))
        p1 = torch.sigmoid(logit1).detach()
        logit2 = self.stage2(torch.cat((h2[-1], p1.unsqueeze(1)), dim=1)).squeeze(-1)
        return torch.stack((logit1, logit2), dim=1)


class IndependentEncodersLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(1, 30, 1, batch_first=True)
        self.stage1 = nn.Linear(30, 1)
        self.lstm2 = nn.LSTM(1, 30, 1, batch_first=True)
        self.stage2 = nn.Linear(30, 1)

    def forward(self, level, difference):
        _, (h1, _) = self.lstm1(level.unsqueeze(-1))
        _, (h2, _) = self.lstm2(difference.unsqueeze(-1))
        return torch.stack((self.stage1(h1[-1]).squeeze(-1),
                            self.stage2(h2[-1]).squeeze(-1)), dim=1)


def make_model(architecture):
    return {'detach_p1': DetachP1LSTM, 'independent_encoders': IndependentEncodersLSTM,
            'reduced_l2': IntegrationOrderLSTM}[architecture]()


def total_loss(parts, stage2_weight):
    return parts[0] + stage2_weight * parts[1]
