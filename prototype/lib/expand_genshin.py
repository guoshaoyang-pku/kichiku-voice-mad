#!/usr/bin/env python3
"""Expand genshin library with iconic characters from the full_cn pack.

Random subsample (capped) per character -> trim -> 16k slices -> CREPE analyze
-> merge into library_genshin.json (dedup by path).
"""
import json
import random
from pathlib import Path

import librosa
import soundfile as sf

import build_library as B

FULL = B.MAT / "genshin" / "full_cn" / "中文 - Chinese"
CHARS = ["胡桃", "钟离", "雷电将军", "芙宁娜", "温迪", "纳西妲", "可莉", "荒泷一斗"]
CAP = 600
SEED = 7


def collect():
    rng = random.Random(SEED)
    out = []
    for char in CHARS:
        d = FULL / char
        wavs = sorted(d.rglob("*.wav"))
        if not wavs:
            print(f"  {char}: MISSING")
            continue
        pick = rng.sample(wavs, min(CAP, len(wavs)))
        dst_dir = B.ensure_dir(B.LIB / "samples" / "genshin" / char)
        kept = 0
        for w in pick:
            dst = dst_dir / w.name
            if dst.exists():
                kept += 1
            else:
                try:
                    y, _ = librosa.load(w, sr=B.SR, mono=True)
                except Exception:
                    continue
                y, _ = librosa.effects.trim(y, top_db=B.TRIM_DB)
                if not (B.MIN_DUR <= len(y) / B.SR <= B.MAX_DUR):
                    continue
                sf.write(dst, y, B.SR)
                kept += 1
            text = ""
            lab = w.with_suffix(".lab")
            if lab.exists():
                text = lab.read_text(errors="ignore").strip()
            out.append({"path": str(dst), "work": "genshin", "char": char,
                        "text": text, "src": str(w)})
        print(f"  {char}: {kept}/{len(pick)}", flush=True)
    return out


def main():
    items = collect()
    items = [it for it in items if Path(it["path"]).exists()]
    items = B.analyze(items)
    lib_path = B.LIB / "library_genshin.json"
    old = json.load(open(lib_path)) if lib_path.exists() else []
    have = {it["path"] for it in old}
    merged = old + [it for it in items if it["path"] not in have]
    B.write_and_report(merged, "genshin")


if __name__ == "__main__":
    main()
