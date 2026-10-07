#!/usr/bin/env python3
"""Intelligibility check: SenseVoice ASR on original vs rendered clip pairs.

python3 eval_asr.py ../out/<render>_asr [more dirs...]
Writes <dir>/asr.json and prints mean CER (rendered transcript vs original transcript).
"""
import json
import re
import sys
from pathlib import Path

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

_model = None


def model():
    global _model
    if _model is None:
        _model = AutoModel(model="iic/SenseVoiceSmall", disable_update=True, device="cpu")
    return _model


def asr(path):
    r = model().generate(input=str(path), language="auto", use_itn=False)
    return rich_transcription_postprocess(r[0]["text"]) if r else ""


def norm(t):
    return re.sub(r"[\W_]+", "", t.lower())


def cer(ref, hyp):
    a, b = norm(ref), norm(hyp)
    if not a:
        return None
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
            prev = cur
    return d[len(b)] / len(a)


def run(dirp):
    dirp = Path(dirp)
    rows = []
    man = dirp / "manifest.json"
    cues = dirp.parent / (dirp.name[:-len("_asr")] + ".cues.json")
    keep = None
    if man.exists():
        keep = set(json.load(open(man)))
    elif cues.exists():
        t_end = cues.stat().st_mtime
        keep = {o.name.split("_")[0] for o in dirp.glob("*_rend.wav")
                if t_end - 900 <= o.stat().st_mtime <= t_end + 5}
    for o in sorted(dirp.glob("*_orig.wav")):
        rnd = o.with_name(o.name.replace("_orig", "_rend"))
        if not rnd.exists() or (keep is not None and o.name.split("_")[0] not in keep):
            continue
        to, tr = asr(o), asr(rnd)
        c = cer(to, tr)
        rows.append({"id": o.stem.split("_")[0], "orig": to, "rend": tr, "cer": c})
    valid = [r["cer"] for r in rows if r["cer"] is not None]
    summary = {"n": len(rows), "n_valid": len(valid),
               "mean_cer": round(sum(valid) / len(valid), 3) if valid else None,
               "frac_cer_le_0.3": round(sum(1 for v in valid if v <= 0.3) / len(valid), 3) if valid else None}
    json.dump({"summary": summary, "rows": rows}, open(dirp / "asr.json", "w"),
              ensure_ascii=False, indent=1)
    print(dirp.name, summary, flush=True)
    for r in rows[:6]:
        print(f"   orig: {r['orig'][:30]:30s} | rend: {r['rend'][:30]:30s} | cer={r['cer']}")
    return summary


if __name__ == "__main__":
    for d in sys.argv[1:]:
        run(d)
