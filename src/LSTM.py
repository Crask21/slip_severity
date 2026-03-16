import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):

    def __init__(self, input_dim=51, hidden_dim=128, layers=2, output_dim=7):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        # x shape: (B, T, 51)

        out, _ = self.lstm(x)

        out = out[:, -1]  # last timestep

        pose = self.fc(out)

        if pose.shape[-1] == 7:

            trans = pose[:, :3]
            quat = pose[:, 3:]

            quat = F.normalize(quat, dim=-1)

            return torch.cat([trans, quat], dim=-1)
        else:
            return pose