#!/usr/bin/env python3
"""Recover high-sample-rate audio for library slices.

Library slices were cut at 16 kHz (consonants above 8 kHz lost). This module
re-derives each slice's time range in its 44.1/48 kHz source and reads it back.

- genshin: slice = librosa.effects.trim(src@16k, top_db=35)
- anime:   slice = deterministic replay of build_library.slice_anime() on src@16k

Anime maps are cached in hires_map.json (built once per source file).
"""
import json
import re
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

import build_library as B

CACHE = B.LIB / "hires_map.json"
PRE_PAD, POST_PAD = 0.02, 0.03
_map = None


def _load_map():
    global _map
    if _map is None:
        _map = json.load(open(CACHE)) if CACHE.exists() else {}
    return _map


def _save_map():
    json.dump(_map, open(CACHE, "w"), ensure_ascii=False)


def _anime_ranges(src):
    """Replay slice_anime() for one source -> {slice_name: (t0, t1)} at 16k."""
    SR = B.SR
    wav = Path(src)
    stem = re.sub(r"[^\w]", "_", wav.stem)[:60]
    y, _ = librosa.load(wav, sr=SR, mono=True)
    intervals = librosa.effects.split(y, top_db=B.SLICE_TOP_DB)
    merged = []
    for s, e in intervals:
        if merged and (s - merged[-1][1]) / SR < B.SLICE_MIN_GAP:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    out = {}
    for i, (s, e) in enumerate(merged):
        seg = y[s:e]
        if len(seg) / SR < B.MIN_DUR:
            continue
        if len(seg) / SR > B.MAX_DUR:
            hop = int(B.MAX_DUR * SR)
            starts = list(range(0, len(seg) - int(B.MIN_DUR * SR), hop))
            chunks = [(cs, min(cs + hop, len(seg))) for cs in starts]
        else:
            chunks = [(0, len(seg))]
        for j, (cs, ce) in enumerate(chunks):
            c = seg[cs:ce]
            _, idx = librosa.effects.trim(c, top_db=B.TRIM_DB)
            n = idx[1] - idx[0]
            if not (B.MIN_DUR <= n / SR <= B.MAX_DUR + 0.5):
                continue
            a0 = s + cs + idx[0]
            out[f"{stem}_{i:04d}_{j}.wav"] = (a0 / SR, (a0 + n) / SR)
    return out


def ensure_anime_maps(srcs):
    m = _load_map()
    todo = [s for s in sorted(set(srcs)) if s not in m]
    for k, s in enumerate(todo):
        print(f"  hires map {k+1}/{len(todo)}: {Path(s).name[:40]}", flush=True)
        m[s] = _anime_ranges(s)
        _save_map()
    return m


def time_range(sample):
    """(t0, t1) seconds in source for a library sample."""
    src = sample["src"]
    if sample["work"].startswith("anime"):
        rng = _load_map().get(src, {}).get(Path(sample["path"]).name)
        if rng is None:
            return None
        return tuple(rng)
    y, _ = librosa.load(src, sr=B.SR, mono=True)
    _, idx = librosa.effects.trim(y, top_db=B.TRIM_DB)
    return idx[0] / B.SR, idx[1] / B.SR


def load(sample, sr_out=32000):
    """High-res mono audio of a sample (with small pre/post pad). Falls back to 16k slice."""
    rng = time_range(sample)
    if rng is not None:
        info = sf.info(sample["src"])
        t0 = max(0.0, rng[0] - PRE_PAD)
        t1 = min(info.duration, rng[1] + POST_PAD)
        y, sr = sf.read(sample["src"], start=int(t0 * info.samplerate),
                        stop=int(t1 * info.samplerate), always_2d=True, dtype="float32")
        y = y.mean(axis=1)
        if sr != sr_out:
            y = librosa.resample(y, orig_sr=sr, target_sr=sr_out)
        return y.astype(np.float64), True
    y, _ = librosa.load(sample["path"], sr=sr_out, mono=True)
    return y.astype(np.float64), False


if __name__ == "__main__":
    lib = json.load(open(B.LIB / "library_anime_full.json"))
    ensure_anime_maps([s["src"] for s in lib])
    m = _load_map()
    names = {Path(s["path"]).name for s in lib}
    mapped = sum(1 for s in lib if Path(s["path"]).name in m.get(s["src"], {}))
    print(f"anime slices mapped: {mapped}/{len(lib)}")
