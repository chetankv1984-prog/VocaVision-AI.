"""Train the CNN-BiLSTM acoustic model with CTC loss.
Usage:  python -m model.train --manifest data/manifest.csv --epochs 20
Manifest CSV columns: path,text   (see data/prepare_librispeech.py)"""
import argparse, os, librosa, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from audio.features import log_mel, SR, N_MELS
from model.acoustic_model import CNNBiLSTMCTC, encode, CHARS
from model.ctc_decoder import greedy_ctc_decode

class SpeechDS(Dataset):
    def __init__(self, csv, max_sec=16):
        self.df = pd.read_csv(csv); self.max_sec = max_sec
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        y, _ = librosa.load(r["path"], sr=SR, duration=self.max_sec)
        return torch.from_numpy(log_mel(y)), torch.tensor(encode(str(r["text"])))

def collate(batch):
    feats, tg = zip(*batch)
    fl = torch.tensor([f.shape[0] for f in feats]); tl = torch.tensor([len(t) for t in tg])
    x = nn.utils.rnn.pad_sequence(feats, batch_first=True)
    return x, fl, torch.cat(tg), tl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dl = DataLoader(SpeechDS(a.manifest), batch_size=a.batch, shuffle=True, collate_fn=collate)
    model = CNNBiLSTMCTC(N_MELS).to(dev)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)                       # <-- CTC loss
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    os.makedirs("checkpoints", exist_ok=True)
    for ep in range(1, a.epochs + 1):
        tot = 0
        for x, fl, tg, tl in dl:
            lp = model(x.to(dev))                                       # (B, T', C)
            out_lens = torch.tensor([CNNBiLSTMCTC.out_len(int(l)) for l in fl])
            loss = ctc(lp.transpose(0, 1), tg, out_lens, tl)            # CTC expects (T, B, C)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); tot += loss.item()
        torch.save(model.state_dict(), "checkpoints/custom_ctc.pt")
        with torch.no_grad():                                           # show a sample decode
            model.eval(); p = model(x[:1].to(dev))[0].exp().cpu().numpy(); model.train()
            hyp = greedy_ctc_decode(p, dict(enumerate(CHARS)), 0, 0.04)["text"]
        print(f"epoch {ep:02d}  CTC loss {tot/len(dl):.3f}  sample: '{hyp}'")

if __name__ == "__main__":
    main()
