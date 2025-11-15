import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.xavier_uniform_(self.pos_embedding)

    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1)]  # (B, T, D)


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=192, d_model=64, nhead=4, num_layers=6, num_classes=2, dropout=0.1, max_len=10001):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)  #
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

    def forward(self, x):  # x: (B, T, F)
        x = self.input_proj(x)  # (B, T, d_model)
        x = self.pos_encoder(x)  # (B, T, d_model)
        x = x.transpose(0, 1)  # (T, B, d_model)
        x = self.transformer(x)  # (T, B, d_model)
        x = x.transpose(0, 1)  # (B, T, d_model)
        x = self.norm(x.mean(dim=1))  # (B, d_model)
        return self.classifier(x)  # (B, num_classes)


