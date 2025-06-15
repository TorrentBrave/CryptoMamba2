from torch import nn

class GRUModel(nn.Module):
    def __init__(self, num_features=5, hidden_size=50, num_layers=2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(num_features, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        output, _ = self.gru(x)
        output = self.fc(output[:, -1, :])
        return output