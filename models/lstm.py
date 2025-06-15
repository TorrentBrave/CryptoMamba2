import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(
        self,
        num_features=5,
        hidden_size=64,
        num_layers=1,
        bidirectional=False,
    ):
        super().__init__()
        # self.norm = norm_layer((num_features, window_size))
        self.lstm = nn.LSTM(input_size=num_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)
        d = 2 if bidirectional else 1
        self.linear = nn.Linear(d * hidden_size, 1)
        # self.linear2 = nn.Linear(window_size, 1)

    
    def forward(self, x):
        # x = self.norm(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.linear(x[:, -1, :])
        # x = x.permute(0, 2, 1)
        # x = self.linear2(x)
        return nn.functional.tanh(x)
