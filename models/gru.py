import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

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
    
if __name__ == "__main__":

    model = GRUModel(
        num_features=8,    # 输入特征数
        hidden_size=64,    # 隐藏层维度
        num_layers=2,      # GRU层数
    )
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")