"""Compact CNN + BiLSTM acoustic model trained with CTC loss (our own implementation)."""
import torch, torch.nn as nn

CHARS = ["<blank>", " ", "'"] + list("abcdefghijklmnopqrstuvwxyz")   # id 0 = CTC blank
CHAR2ID = {c: i for i, c in enumerate(CHARS)}

def encode(text):
    return [CHAR2ID[c] for c in text.lower() if c in CHAR2ID and c != "<blank>"]

class CNNBiLSTMCTC(nn.Module):
    def __init__(self, n_mels=80, hidden=256, layers=3, n_classes=len(CHARS)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.proj = nn.Linear(32 * (n_mels // 4), hidden)
        self.rnn = nn.LSTM(hidden, hidden, layers, batch_first=True, bidirectional=True, dropout=0.2)
        self.out = nn.Linear(hidden * 2, n_classes)

    @staticmethod
    def out_len(L):                       # two stride-2 convs => T/4
        L = (L - 1) // 2 + 1
        return (L - 1) // 2 + 1

    def forward(self, x):                 # x: (B, T, n_mels)
        x = self.conv(x.unsqueeze(1))     # (B, C, T/4, F/4)
        b, c, t, f = x.shape
        x = self.proj(x.permute(0, 2, 1, 3).reshape(b, t, c * f))
        x, _ = self.rnn(x)
        return self.out(x).log_softmax(-1)   # (B, T/4, classes) log-probabilities for CTC
