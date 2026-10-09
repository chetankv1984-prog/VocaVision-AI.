# VocaVision AI — Speech-to-Text with an Acoustic Model + CTC Loss

## 1. Setup (VS Code terminal)
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```
For mp3/m4a files also install **ffmpeg** (wav/flac work without it).

## 2. Run the dashboard
```bash
streamlit run app.py
```
First run downloads the pretrained CTC model (~360 MB, needs internet once). Then upload/record audio → **Transcribe**.

## 3. (Recommended for marks) Train our own CTC model
```bash
python data/prepare_librispeech.py 1500     # downloads LibriSpeech dev-clean, builds data/manifest.csv
python -m model.train --epochs 20           # CNN-BiLSTM trained with nn.CTCLoss -> checkpoints/custom_ctc.pt
streamlit run app.py                        # pick "Custom CNN-BiLSTM-CTC" in the dropdown
```
A small model trained on little data/CPU will be much less accurate than the pretrained one — say so in the viva;
the point is that the CTC training pipeline is real. Use a GPU (Google Colab) and more data for better accuracy.

## 4. Structure
```
app.py                     dashboard (6 pages)
audio/features.py          loading, log-Mel features
model/acoustic_model.py    CNN + BiLSTM network (outputs log-probs)
model/train.py             CTC training loop (nn.CTCLoss)
model/ctc_decoder.py       our greedy CTC decoder (collapse repeats, remove blanks, word timing, confidence)
model/inference.py         pretrained wav2vec2-CTC + custom model back-ends
analytics/                 speech stats, keywords, PDF export
```

## 5. Viva talking points
* Pipeline: audio → log-Mel → acoustic model → per-frame probabilities → CTC decode → text.
* CTC: blank symbol + sum over all alignments, so no frame-level labels needed. Loss = -log P(Y|X).
* Decoding: argmax per frame → merge repeats → remove blanks.
* Pretrained wav2vec2 weights are used for accuracy; decoding/analytics/visualisation and the custom model + training are ours.
* Limitations: English only, greedy decoding (no language model), accuracy drops with noise.
