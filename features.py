"""Audio loading + feature extraction (waveform -> log-Mel spectrogram)."""
import os, tempfile
import numpy as np, librosa, soundfile as sf

SR = 16000
N_MELS = 80

def load_audio(uploaded_file):
    """Accepts a Streamlit UploadedFile. Returns mono float32 @16 kHz + original info."""
    name = getattr(uploaded_file, "name", "audio.wav")
    suffix = os.path.splitext(name)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name
    try:
        orig_sr, channels = SR, 1
        try:
            i = sf.info(path); orig_sr, channels = i.samplerate, i.channels
        except Exception:
            pass
        y, _ = librosa.load(path, sr=SR, mono=True)   # m4a/mp3 may need ffmpeg installed
    finally:
        os.remove(path)
    return y.astype(np.float32), {"orig_sr": orig_sr, "channels": channels}

def log_mel(y, sr=SR, n_mels=N_MELS):
    """Returns (T, n_mels) normalised log-Mel features (10 ms hop, 25 ms window)."""
    m = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=400, hop_length=160, n_mels=n_mels)
    lm = np.log(m + 1e-6).T
    return ((lm - lm.mean()) / (lm.std() + 1e-5)).astype(np.float32)

def clean_audio(y, sr=SR, denoise=False):
    """Light clean-up for live recordings: high-pass filter, optional noise reduction, loudness normalisation."""
    from scipy.signal import butter, sosfilt
    if len(y) < sr // 4:
        return y
    y = sosfilt(butter(4, 80, "highpass", fs=sr, output="sos"), y)     # remove rumble / mic hum
    if denoise:
        try:
            import noisereduce as nr
            y = nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=0.6)
        except ImportError:
            pass
    return (y / (np.max(np.abs(y)) + 1e-8) * 0.95).astype(np.float32)   # peak-normalise
