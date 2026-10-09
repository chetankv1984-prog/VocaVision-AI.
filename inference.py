"""Two CTC acoustic-model back-ends that both return per-frame probabilities (T, classes)."""
import os, numpy as np, torch
from audio.features import log_mel, SR, N_MELS
from model.acoustic_model import CNNBiLSTMCTC, CHARS

CKPT = "checkpoints/custom_ctc.pt"
MMS = "MMS-1B multilingual CTC (English / Hindi / Kannada / Urdu)"
LARGE = "wav2vec2-large (English only, most accurate)"
PRETRAINED = "wav2vec2-base (English only, faster)"
LANGS = {"English": "eng", "Hindi": "hin", "Kannada": "kan", "Urdu": "urd-script_arabic"}
CUSTOM = "Custom CNN-BiLSTM-CTC (trained by us)"

def available_backends():
    return [LARGE, MMS, PRETRAINED] + ([CUSTOM] if os.path.exists(CKPT) else [])

class MMSCTC:
    """facebook/mms-1b-all: wav2vec2 acoustic model trained with CTC, one small adapter + vocabulary per language."""
    frame_sec = 0.02
    def __init__(self):
        from transformers import Wav2Vec2ForCTC, AutoProcessor
        name = "facebook/mms-1b-all"
        self.proc = AutoProcessor.from_pretrained(name, target_lang="eng")
        self.model = Wav2Vec2ForCTC.from_pretrained(name, target_lang="eng", ignore_mismatched_sizes=True).eval()
        self.lang = "eng"; self._refresh()
        self.n_params = sum(p.numel() for p in self.model.parameters())
    def _refresh(self):
        self.blank_id = self.proc.tokenizer.pad_token_id
        self.id2tok = {i: t for t, i in self.proc.tokenizer.get_vocab().items()}
    def set_lang(self, code):
        if code != self.lang:
            self.proc.tokenizer.set_target_lang(code); self.model.load_adapter(code)
            self.lang = code; self._refresh()
    @torch.no_grad()
    def probs(self, y):
        out, step = [], 30 * SR
        for s in range(0, len(y), step):
            inp = self.proc(y[s:s + step], sampling_rate=SR, return_tensors="pt").input_values
            out.append(self.model(inp).logits[0].softmax(-1).numpy())
        return np.concatenate(out)

class Wav2Vec2CTC:
    frame_sec = 0.02
    def __init__(self, name="facebook/wav2vec2-base-960h"):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        self.proc = Wav2Vec2Processor.from_pretrained(name)
        self.model = Wav2Vec2ForCTC.from_pretrained(name).eval()
        self.blank_id = self.proc.tokenizer.pad_token_id            # CTC blank = <pad>
        self.id2tok = {i: t for t, i in self.proc.tokenizer.get_vocab().items()}
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def set_lang(self, code): pass
    @torch.no_grad()
    def probs(self, y):
        out, step = [], 30 * SR                                      # 30 s chunks
        for s in range(0, len(y), step):
            inp = self.proc(y[s:s + step], sampling_rate=SR, return_tensors="pt").input_values
            out.append(self.model(inp).logits[0].softmax(-1).numpy())
        return np.concatenate(out)

class CustomCTC:
    frame_sec = 0.04
    def __init__(self):
        self.model = CNNBiLSTMCTC()
        self.model.load_state_dict(torch.load(CKPT, map_location="cpu")); self.model.eval()
        self.blank_id, self.id2tok = 0, {i: c for i, c in enumerate(CHARS)}
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def set_lang(self, code): pass
    @torch.no_grad()
    def probs(self, y):
        x = torch.from_numpy(log_mel(y, SR, N_MELS)).unsqueeze(0)
        return self.model(x)[0].exp().numpy()

class WhisperEN:
    """Optional English text refiner (Whisper small.en). Used only to polish the final transcript."""
    def __init__(self, name="openai/whisper-small.en"):
        from transformers import pipeline
        self.pipe = pipeline("automatic-speech-recognition", model=name, chunk_length_s=30)
    def transcribe(self, y):
        return self.pipe({"raw": y, "sampling_rate": SR})["text"].strip()

def get_recognizer(name):
    if name == MMS: return MMSCTC()
    if name == LARGE: return Wav2Vec2CTC("facebook/wav2vec2-large-960h-lv60-self")
    if name == PRETRAINED: return Wav2Vec2CTC()
    return CustomCTC()
