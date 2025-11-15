import os


import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class CNNEncoder(nn.Module):
    def __init__(self, in_channels=3, out_dim=128,dropout_rate=0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # -> (B*T, 64, 1, 1)
        )
        self.fc = nn.Linear(64, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):  # x: (B, T, C, N, N)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.cnn(x).view(B * T, -1)  # (B*T, 64)
        x = self.fc(x)                   # (B*T, out_dim)
        x = x.view(B, T, -1)             # (B, T, out_dim)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.xavier_uniform_(self.pos_embedding)

    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1)]


class TimeSeriesTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=6, num_classes=2, dropout=0.1, max_len=10001):
        super().__init__()
        self.pos_encoder = PositionalEncoding(max_len=max_len, d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

        self.attn_weights = None
    def forward(self, x):  # x: (B, T, d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)          # (B, T, d_model)
        x = self.norm(x.mean(dim=1))     # mean pooling + norm
        return self.classifier(x)


class CNNTransformerModel(nn.Module):
    def __init__(self, in_channels=3, cnn_out_dim=128, num_classes=2, max_len=10001):
        super().__init__()
        self.encoder = CNNEncoder(in_channels=in_channels, out_dim=cnn_out_dim)
        self.temporal = TimeSeriesTransformer(d_model=cnn_out_dim, num_classes=num_classes, max_len=max_len)

    def forward(self, x):  # x: (B, T, C, N, N)
        x = self.encoder(x)       # (B, T, cnn_out_dim)
        out = self.temporal(x)    # (B, num_classes)
        return out

    def get_attention(self):
        return self.temporal.attn_weights
