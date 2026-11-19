import torch
import torch.nn as nn

class LstmModel(nn.Module):
    def __init__(self, seq_features, lstm_hidden, ctx_hidden, context_dim, dropout_rate):
        super().__init__()
        self.lstm = nn.LSTM(seq_features, lstm_hidden, batch_first=True)
        self.context_net = nn.Sequential(
            nn.Linear(context_dim, ctx_hidden),
            nn.BatchNorm1d(ctx_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden + ctx_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )

    def forward(self, x_seq, x_ctx):
        _, (h_n, _) = self.lstm(x_seq)
        fused = torch.cat([h_n[-1], self.context_net(x_ctx)], dim=1)
        return self.classifier(fused)