"""Our own greedy (best-path) CTC decoder: argmax -> collapse repeats -> remove blanks."""
import numpy as np

def greedy_ctc_decode(probs, id2tok, blank_id, frame_sec):
    ids = probs.argmax(1); conf = probs.max(1)
    def is_sep(t): return t in ("|", " ")
    def valid(t): return is_sep(t) or len(t) == 1

    raw = []
    for i in ids:
        t = id2tok[i]
        raw.append("_" if i == blank_id else "·" if is_sep(t) else t if len(t) == 1 else "?")

    chars, cur, prev = [], None, -1
    for f, i in enumerate(ids):
        t = id2tok[i]
        if i != prev:                                   # new run
            cur = None
            if i != blank_id and valid(t):
                cur = {"ch": " " if is_sep(t) else t, "start": f, "end": f, "conf": [float(conf[f])]}
                chars.append(cur)
        elif cur is not None:                           # repeated symbol -> merge (CTC collapse)
            cur["end"] = f; cur["conf"].append(float(conf[f]))
        prev = i

    words, buf = [], []
    def flush():
        if buf:
            words.append({"word": "".join(c["ch"] for c in buf).lower(),
                          "start": buf[0]["start"] * frame_sec, "end": (buf[-1]["end"] + 1) * frame_sec,
                          "conf": float(np.mean([x for c in buf for x in c["conf"]]))})
            buf.clear()
    for c in chars:
        if c["ch"] == " ": flush()
        else: buf.append(c)
    flush()

    text = " ".join(w["word"] for w in words)
    text = text[:1].upper() + text[1:]
    all_conf = [x for c in chars for x in c["conf"]]
    return {"text": text, "words": words, "raw": raw, "chars": chars,
            "confidence": float(np.mean(all_conf)) if all_conf else 0.0,
            "frame_conf": conf, "frame_ids": ids}
