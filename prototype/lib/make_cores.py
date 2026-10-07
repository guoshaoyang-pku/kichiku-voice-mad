#!/usr/bin/env python3
"""Add stable "core" windows to library samples (UTAU-style single-pitch regions).

For each sample: per-frame CREPE f0 (argmax), then find the window (~0.26s) of
continuously voiced frames with minimal f0 variance. Store:
  core_t0, core_t1 (sec), core_f0_semi (median), core_iqr_cents.

python3 make_cores.py ../lib/library_genshin.json ../lib/library_anime.json
(rewrites jsons in place; safe to re-run, skips samples that already have cores)
"""
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torchcrepe

SR = 16000
HOP = 160
CORE_FRAMES = 26          # 260ms core window
MIN_CORE_FRAMES = 12      # accept shorter cores for short samples
device = "mps" if torch.backends.mps.is_available() else "cpu"


def frame_f0(y):
    f0, per = torchcrepe.predict(
        torch.tensor(y).unsqueeze(0).to(device), SR, model="full",
        decoder=torchcrepe.decode.argmax, return_periodicity=True,
        device=device, batch_size=1)
    return f0.cpu().numpy().ravel(), per.cpu().numpy().ravel()


def best_core(cents, voiced, win):
    """Sliding-window minimal-variance fully-voiced run. Returns (i0, i1)."""
    n = len(cents)
    if n < win:
        win = max(min(n, MIN_CORE_FRAMES + 2), 2)
        if n < 4:
            return None
    ok = np.convolve(voiced.astype(np.int16), np.ones(win, np.int16), "valid") == win
    if not ok.any():
        # relax: allow up to 15% unvoiced frames inside window
        thresh = int(win * 0.85)
        ok = np.convolve(voiced.astype(np.int16), np.ones(win, np.int16), "valid") >= thresh
        if not ok.any():
            return None
    c = np.where(voiced, cents, np.nan)
    idx = np.where(ok)[0]
    cf = np.where(np.isnan(c), 0.0, c)
    cs1 = np.concatenate([[0], np.cumsum(cf)])
    cs2 = np.concatenate([[0], np.cumsum(cf ** 2)])
    s1 = (cs1[idx + win] - cs1[idx]) / win
    s2 = (cs2[idx + win] - cs2[idx]) / win
    var = s2 - s1 ** 2
    best_i = idx[int(np.argmin(var))]
    return best_i, best_i + win


def add_cores(lib_path):
    items = json.load(open(lib_path))
    todo = [it for it in items if "core_f0_semi" not in it]
    print(f"{Path(lib_path).name}: {len(todo)}/{len(items)} need cores", flush=True)
    for n, it in enumerate(todo):
        try:
            y, _ = librosa.load(it["path"], sr=SR, mono=True)
        except Exception:
            it["core_f0_semi"] = None
            continue
        f0, per = frame_f0(y)
        voiced = (per > 0.5) & (f0 > 60) & (f0 < 1500)
        if voiced.sum() < 6:
            it["core_f0_semi"] = None
            continue
        cents = np.where(voiced, 1200 * np.log2(np.maximum(f0, 1e-6) / 440.0) + 6900, np.nan)
        res = best_core(cents, voiced, CORE_FRAMES)
        if res is None:
            it["core_f0_semi"] = None
            continue
        i0, i1 = res
        seg = cents[i0:i1]
        seg = seg[~np.isnan(seg)]
        if len(seg) < 4:
            it["core_f0_semi"] = None
            continue
        it["core_t0"] = round(i0 * HOP / SR, 4)
        it["core_t1"] = round(min(i1 * HOP / SR + 0.02, len(y) / SR), 4)
        it["core_f0_semi"] = round(float(np.median(seg) / 100.0), 3)  # cents->semitones(A440)=/100... see note
        it["core_iqr_cents"] = round(float(np.percentile(seg, 75) - np.percentile(seg, 25)), 1)
        if (n + 1) % 200 == 0:
            print(f"  {n+1}/{len(todo)}", flush=True)
        if (n + 1) % 300 == 0:  # checkpoint for resume
            with open(lib_path, "w") as f:
                json.dump(items, f, ensure_ascii=False)
    with open(lib_path, "w") as f:
        json.dump(items, f, ensure_ascii=False)
    have = sum(1 for it in items if it.get("core_f0_semi") is not None)
    print(f"{Path(lib_path).name}: cores for {have}/{len(items)}", flush=True)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        add_cores(p)
