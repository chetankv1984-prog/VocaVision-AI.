import time, os
from datetime import datetime
import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go, plotly.express as px
import librosa

from audio.features import load_audio, log_mel, clean_audio, SR
from model.inference import get_recognizer, available_backends, LANGS, MMS, LARGE
from model.ctc_decoder import greedy_ctc_decode
from analytics.stats import speech_stats, keywords
from analytics.export import to_pdf

st.set_page_config(page_title="VocaVision AI", page_icon="🎙️", layout="wide")
HIST = "data/history.csv"

st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 15% 0%,#1b2350 0%,#0b1020 45%,#080b16 100%);}
h1,h2,h3{letter-spacing:.3px}
.hero{font-size:2.6rem;font-weight:800;background:linear-gradient(90deg,#22d3ee,#8b5cf6,#ec4899);
 -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0}
.sub{color:#9fb0e0;margin-top:-6px;margin-bottom:18px}
.card{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1);border-radius:16px;
 padding:16px 18px;backdrop-filter:blur(8px);text-align:center}
.card .v{font-size:1.9rem;font-weight:800;color:#22d3ee}.card .l{color:#9fb0e0;font-size:.85rem}
.tag{display:inline-block;padding:4px 12px;margin:3px;border-radius:999px;
 background:linear-gradient(90deg,#6d28d9,#0891b2);color:white;font-size:.85rem}
.pipe{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:10px 0}
.pipe div{padding:10px 14px;border-radius:12px;background:rgba(139,92,246,.18);border:1px solid #8b5cf6}
.pipe span{color:#22d3ee;font-weight:800}
.ctc{font-family:monospace;font-size:1.05rem;background:#0a0f22;padding:12px;border-radius:10px;
 border:1px solid #243065;word-break:break-all;letter-spacing:2px}
</style>""", unsafe_allow_html=True)

def card(col, label, value):
    col.markdown(f'<div class="card"><div class="v">{value}</div><div class="l">{label}</div></div>', unsafe_allow_html=True)

def style(fig, h=320):
    fig.update_layout(template="plotly_dark", height=h, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=40, b=10))
    return fig

@st.cache_resource(show_spinner="Loading acoustic model (first run downloads ~360 MB)...")
def load_model(name): return get_recognizer(name)

@st.cache_resource(show_spinner="Loading Whisper English refiner (first run downloads ~480 MB)...")
def load_whisper():
    from model.inference import WhisperEN
    return WhisperEN()

def fmt(sec): return f"{int(sec//60):02d}:{int(sec%60):02d}"

COLS = ["Date", "Audio", "Model", "Duration(s)", "Words", "WPM", "Confidence(%)", "Transcript", "Language"]

def load_hist():
    return pd.read_csv(HIST).reindex(columns=COLS) if os.path.exists(HIST) else pd.DataFrame(
        columns=COLS)

# ---------------- sidebar ----------------
st.sidebar.markdown("## 🎙️ VocaVision AI")
page = st.sidebar.radio("Navigate", ["🏠 Home", "🎙️ Transcription Studio", "🧠 AI Model",
                                     "📊 Speech Analytics", "🔬 Explainable AI", "📚 History"])
st.sidebar.success("● System ready")
st.sidebar.caption("CTC-based acoustic model • Speech-to-Text")
R = st.session_state.get("last")

st.markdown('<p class="hero">VocaVision AI</p><p class="sub">Intelligent Speech-to-Text & Voice Analytics Platform — '
            'Acoustic Model + Connectionist Temporal Classification (CTC)</p>', unsafe_allow_html=True)

# ---------------- HOME ----------------
if page == "🏠 Home":
    st.markdown('<div class="pipe"><div>🎧 Audio</div><span>→</span><div>🔊 Log-Mel Features</div><span>→</span>'
                '<div>🧠 Acoustic Model</div><span>→</span><div>🔗 CTC Loss / Probabilities</div><span>→</span>'
                '<div>🔎 CTC Decoder</div><span>→</span><div>📝 Transcript</div></div>', unsafe_allow_html=True)
    h = load_hist(); c = st.columns(4)
    card(c[0], "Transcriptions", len(h)); card(c[1], "Words recognised", int(h["Words"].sum()) if len(h) else 0)
    card(c[2], "Avg confidence", f"{h['Confidence(%)'].mean():.1f}%" if len(h) else "–")
    card(c[3], "Audio processed", fmt(h["Duration(s)"].sum()) if len(h) else "00:00")
    st.markdown("### What this project does")
    st.write("Speech is converted to log-Mel features, passed to an acoustic model that outputs per-frame character "
             "probabilities, and trained with **CTC loss**, which removes the need for frame-by-frame alignment between "
             "audio and text. A CTC decoder (argmax → collapse repeats → remove blanks) produces the transcript. "
             "Go to **Transcription Studio** to try it.")

# ---------------- STUDIO ----------------
elif page == "🎙️ Transcription Studio":
    backends = available_backends()
    model_name = st.selectbox("Acoustic model", backends)
    cl, cn = st.columns([1, 1])
    lang_name = cl.selectbox("🌐 Spoken language", list(LANGS), help="Multilingual model only. wav2vec2-base is English only.")
    denoise = cn.checkbox("🔇 Noise reduction (only if your room is noisy)", value=False)
    if model_name != MMS and lang_name != "English":  # English-only models
        st.warning("This model only supports English. Choose the MMS multilingual model for Hindi / Kannada / Urdu.")
    refine = st.checkbox("✨ Accuracy boost for English (final text refined with Whisper)", value=True,
                         help="The CTC model still runs for all visualisations; Whisper only polishes the English transcript.")
    tab_rec, tab_up = st.tabs(["🎤 Record live", "📁 Upload file"])
    with tab_rec:
        rec = st.audio_input("Click the mic, speak clearly, then click stop", sample_rate=16000)
        st.caption("Tip: use a headset mic, stay ~15 cm from the mic, quiet room, speak in full sentences.")
    with tab_up:
        up = st.file_uploader("Upload audio (wav / mp3 / flac / m4a)", type=["wav", "mp3", "flac", "m4a", "ogg"])
    src = rec or up
    if src is not None:
        st.audio(src)
        if st.button("🚀 Transcribe", type="primary"):
            with st.spinner("Processing speech..."):
                y, info = load_audio(src)
                y = clean_audio(y, denoise=denoise)
                y = np.pad(y, (SR // 3, SR // 3))   # silence padding so first/last sounds aren't clipped
                dur = len(y) / SR
                m = load_model(model_name); m.set_lang(LANGS[lang_name]); t0 = time.time()
                probs = m.probs(y)
                res = greedy_ctc_decode(probs, m.id2tok, m.blank_id, m.frame_sec)
                ptime = time.time() - t0
                stt = speech_stats(res, dur, y)
                if refine and lang_name == "English":
                    wtxt = load_whisper().transcribe(y)
                    if wtxt:
                        res["ctc_text"], res["text"] = res["text"], wtxt
                        stt["words"], stt["chars"] = len(wtxt.split()), len(wtxt)
                        stt["wpm"] = stt["words"] / (dur / 60) if dur else 0
            name = getattr(src, "name", "recording.wav")
            st.session_state["last"] = dict(y=y, info=info, probs=probs, res=res, stats=stt, name=name,
                                            dur=dur, ptime=ptime, model=model_name, lang=lang_name, id2tok=m.id2tok,
                                            blank=m.blank_id, fs=m.frame_sec, size=len(src.getvalue()))
            row = pd.DataFrame([[datetime.now().strftime("%Y-%m-%d %H:%M"), name, model_name, round(dur, 1),
                                 stt["words"], round(stt["wpm"]), round(stt["confidence"], 1), res["text"], lang_name]],
                               columns=load_hist().columns)
            pd.concat([load_hist(), row]).to_csv(HIST, index=False)
            R = st.session_state["last"]
    if R:
        st.markdown("### 📊 Results"); s = R["stats"]; c = st.columns(6)
        card(c[0], "Words", s["words"]); card(c[1], "Characters", s["chars"]); card(c[2], "Duration", fmt(R["dur"]))
        card(c[3], "Speech rate (WPM)", round(s["wpm"])); card(c[4], "Confidence", f"{s['confidence']:.1f}%")
        card(c[5], "Silence", f"{s['silence_pct']:.1f}%")
        st.caption(f"File: {R['name']} • {R['info']['orig_sr']} Hz • {R['info']['channels']} ch • "
                   f"{R['size']/1024:.0f} KB • processed in {R['ptime']:.2f}s")
        st.markdown("🔈 **Audio sent to the model (after cleaning)**"); st.audio(R["y"], sample_rate=SR)
        y = R["y"]; step = max(1, len(y) // 4000); t = np.arange(0, len(y), step) / SR
        fig = go.Figure(go.Scatter(x=t, y=y[::step], line=dict(color="#22d3ee", width=1), fill="tozeroy"))
        st.plotly_chart(style(fig.update_layout(title="🎧 Waveform", xaxis_title="Time (s)"), 240), use_container_width=True)
        fc = R["res"]["frame_conf"]; tt = np.arange(len(fc)) * R["fs"]
        fig = go.Figure(go.Scatter(x=tt, y=fc * 100, line=dict(color="#8b5cf6"), fill="tozeroy"))
        st.plotly_chart(style(fig.update_layout(title="📈 Model confidence over time (%)", yaxis_range=[0, 100]), 260), use_container_width=True)
        st.markdown("### 📝 Transcript")
        txt = st.text_area("Edit if needed", R["res"]["text"], height=150)
        d = st.columns(3)
        d[0].download_button("📄 Export TXT", txt, "transcript.txt")
        if R.get("lang") == "English":
            d[1].download_button("📑 Export PDF", to_pdf("VocaVision AI Transcript", txt), "transcript.pdf")
        else:
            d[1].caption("PDF export is English-only (font limits). Use TXT for other languages.")
        rtl = 'dir="rtl"' if R.get("lang") == "Urdu" else ""
        st.markdown(f'<div class="ctc" {rtl} style="font-size:1.4rem;letter-spacing:0">{txt}</div>', unsafe_allow_html=True)
        kw = keywords(txt)
        if kw: st.markdown("**🔍 Keywords** " + "".join(f'<span class="tag">{k}</span>' for k in kw), unsafe_allow_html=True)
    else:
        st.info("Upload or record audio, then press **Transcribe**.")

# ---------------- AI MODEL ----------------
elif page == "🧠 AI Model":
    st.markdown('<div class="pipe"><div>Audio</div><span>→</span><div>Log-Mel (80)</div><span>→</span>'
                '<div>2× Conv2D</div><span>→</span><div>3× BiLSTM</div><span>→</span><div>Softmax</div><span>→</span>'
                '<div>CTC Loss</div><span>→</span><div>Greedy CTC Decode</div></div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        st.markdown("### Custom model (our implementation)")
        st.code("""class CNNBiLSTMCTC(nn.Module):
    conv  : 2 x Conv2d(stride 2) + BatchNorm + ReLU   (time / 4)
    proj  : Linear -> 256
    rnn   : 3-layer BiLSTM, hidden 256
    out   : Linear -> 29 classes (blank, space, ', a-z)
    return log_softmax(out)        # fed to nn.CTCLoss""", language="python")
        st.write("Train it with `python -m model.train` (see README). Once `checkpoints/custom_ctc.pt` exists it "
                 "appears in the model dropdown in Transcription Studio.")
    with b:
        st.markdown("### Pretrained model (high accuracy)")
        st.write("`facebook/wav2vec2-base-960h` is a transformer acoustic model **fine-tuned with CTC loss** on 960 h of "
                 "LibriSpeech. Only the network weights are pretrained; the **CTC decoding, confidence analysis and "
                 "alignment are our own code** (`model/ctc_decoder.py`).")
    st.markdown("### Why CTC?")
    st.write("Audio has many more frames than output characters, and we don't know which frame belongs to which letter. "
             "CTC adds a special **blank** symbol and sums the probability of **all alignments** that collapse to the target "
             "text, so no manual alignment is needed.")
    st.latex(r"P(Y|X)=\sum_{A\in\mathcal{B}^{-1}(Y)}\prod_{t=1}^{T}p_t(a_t|X),\qquad \mathcal{L}_{CTC}=-\log P(Y|X)")
    st.code("loss = nn.CTCLoss(blank=0, zero_infinity=True)(log_probs.transpose(0,1), targets, input_lens, target_lens)", language="python")

# ---------------- ANALYTICS ----------------
elif page == "📊 Speech Analytics":
    h = load_hist()
    if h.empty: st.info("No data yet — transcribe something first.")
    else:
        h["Session"] = range(1, len(h) + 1); c = st.columns(2)
        c[0].plotly_chart(style(px.line(h, x="Session", y="Words", markers=True, title="Words per session")), use_container_width=True)
        c[1].plotly_chart(style(px.bar(h, x="Session", y="Confidence(%)", title="Confidence per session", color="Confidence(%)")), use_container_width=True)
        c[0].plotly_chart(style(px.bar(h, x="Session", y="Duration(s)", title="Speech duration (s)")), use_container_width=True)
        c[1].plotly_chart(style(px.line(h, x="Session", y="WPM", markers=True, title="Speaking rate (WPM)")), use_container_width=True)
        if R:
            s = R["stats"]; st.markdown("### Latest recording — speaking analysis"); c = st.columns(4)
            card(c[0], "Speech rate (WPM)", round(s["wpm"])); card(c[1], "Pauses > 0.3 s", s["long_pauses"])
            card(c[2], "Avg word duration", f"{s['avg_word_sec']:.2f}s")
            q = "🟢 Good" if s["confidence"] > 85 else "🟡 Moderate" if s["confidence"] > 65 else "🔴 Poor"
            card(c[3], "Audio quality (by confidence)", q)

# ---------------- EXPLAINABLE AI ----------------
elif page == "🔬 Explainable AI":
    if not R: st.info("Transcribe an audio clip first.")
    else:
        stages = ["1 · Audio", "2 · Features", "3 · Acoustic model output", "4 · CTC frame sequence", "5 · CTC decoding", "6 · Transcript"]
        stg = st.select_slider("How did the AI understand the audio? Click through each stage", stages)
        y, res, probs = R["y"], R["res"], R["probs"]
        if stg == stages[0]:
            st.write(f"Raw waveform: {len(y)} samples at {SR} Hz ({R['dur']:.2f}s).")
            step = max(1, len(y)//4000)
            st.plotly_chart(style(go.Figure(go.Scatter(y=y[::step], line=dict(color="#22d3ee")))), use_container_width=True)
        elif stg == stages[1]:
            st.write("The waveform is converted to a **Mel spectrogram** (80 bands, 25 ms window, 10 ms hop), log-scaled.")
            lm = log_mel(y).T
            st.plotly_chart(style(px.imshow(lm, origin="lower", aspect="auto", color_continuous_scale="Turbo",
                                            labels=dict(x="Frame (10 ms)", y="Mel band")), 340), use_container_width=True)
        elif stg == stages[2]:
            st.write("The acoustic model outputs a **probability distribution over characters for every time frame**.")
            tok = [R["id2tok"][i] for i in range(probs.shape[1])]
            top = np.argsort(-probs.sum(0))[:14]
            w0 = st.slider("Window start (frame)", 0, max(0, len(probs) - 100), 0)
            sub = probs[w0:w0 + 100][:, top].T
            lab = ["blank" if top_i == R["blank"] else tok[top_i] for top_i in top]
            st.plotly_chart(style(px.imshow(sub, x=list(range(w0, w0 + sub.shape[1])), y=lab, aspect="auto",
                                            color_continuous_scale="Viridis", labels=dict(x="Frame", y="Token", color="P")), 380), use_container_width=True)
        elif stg == stages[3]:
            st.write("Best-path: take the **most probable symbol in each frame** (`_` = blank, `·` = space).")
            w0 = st.slider("Window start (frame)", 0, max(0, len(res["raw"]) - 120), 0, key="w3")
            st.markdown(f'<div class="ctc">{"".join(res["raw"][w0:w0+120])}</div>', unsafe_allow_html=True)
        elif stg == stages[4]:
            st.write("**Step 1** merge repeated symbols → **Step 2** delete blanks.")
            raw = "".join(res["raw"][:120]); merged = "".join(ch for i, ch in enumerate(raw) if i == 0 or ch != raw[i-1])
            final = merged.replace("_", "")
            st.markdown("Raw frames"); st.markdown(f'<div class="ctc">{raw}</div>', unsafe_allow_html=True)
            st.markdown("After collapsing repeats"); st.markdown(f'<div class="ctc">{merged}</div>', unsafe_allow_html=True)
            st.markdown("After removing blanks"); st.markdown(f'<div class="ctc">{final}</div>', unsafe_allow_html=True)
            st.caption("Words below are built with their start/end time and confidence:")
            st.dataframe(pd.DataFrame(res["words"]).round(2).head(30), use_container_width=True)
        else:
            if res.get("ctc_text"):
                st.markdown("**Raw CTC decoder output**"); st.code(res["ctc_text"], language=None)
                st.markdown("**Final text (refined with Whisper)**")
            st.success(res["text"]); st.caption(f"CTC acoustic-model confidence {100*res['confidence']:.1f}%")

# ---------------- HISTORY ----------------
else:
    h = load_hist()
    st.dataframe(h.iloc[::-1], use_container_width=True)
    if not h.empty:
        st.download_button("⬇️ Download history CSV", h.to_csv(index=False), "history.csv")
        if st.button("🗑️ Clear history"): os.remove(HIST); st.rerun()
