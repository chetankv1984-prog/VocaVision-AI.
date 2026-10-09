import re, numpy as np, librosa
from collections import Counter
STOP = set("a an the and or but if of to in on at by for with from as is are was were be been it its this that these those you he she we they me my your our their so not no do does did have has had will would can could just than then there here what which who".split())

def speech_stats(res, duration, y, sr=16000):
    w = res["words"]; n = len(w)
    iv = librosa.effects.split(y, top_db=30)
    speech = sum(e - s for s, e in iv) / sr
    pauses = [w[i + 1]["start"] - w[i]["end"] for i in range(n - 1)]
    return {"words": n, "chars": len(res["text"]), "duration": duration,
            "wpm": n / (duration / 60) if duration else 0,
            "silence_pct": 100 * max(0, 1 - speech / duration) if duration else 0,
            "long_pauses": int(sum(p > 0.3 for p in pauses)),
            "avg_word_sec": float(np.mean([x["end"] - x["start"] for x in w])) if w else 0,
            "confidence": 100 * res["confidence"]}

def keywords(text, k=8):
    toks = [t.strip(".,!?;:\"'()") for t in text.lower().split()]
    toks = [t for t in toks if t not in STOP and len(t) > 2]
    return [t for t, _ in Counter(toks).most_common(k)]
