#!/usr/bin/env python3
"""Build a bank of voiced "nuclei" (stable vowel-like segments) from the clip library.

Speech only carries usable pitch in its vowels. Each nucleus is a 100 ms+ stretch of
stable voiced signal inside a real character line; the sampler later re-pitches and
sustains these to carry the melody. Analysis runs CREPE batched on the Apple GPU.
"""
import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torchcrepe

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
sys.path.insert(0, str(HERE))
import hires  # noqa: E402
import phrase_match as PM  # noqa: E402

SR16 = 16000
HOP = 160
OUT = HERE.parent / "materials" / "nucleus_bank.json"


def runs_from_track(st, voiced, min_frames=10, split_semi=1.0):
    idx = np.where(voiced)[0]
    if len(idx) == 0:
        return []
    groups, s = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > 2:
            groups.append((s, a))
            s = b
    groups.append((s, idx[-1]))
    out = []
    for a, b in groups:
        seg_start = a
        for i in range(a + 1, b + 2):
            if i > b or abs(st[i] - np.median(st[seg_start:i])) > split_semi:
                if i - seg_start >= min_frames:
                    out.append((seg_start, i - 1))
                seg_start = i
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", nargs="+", default=[str(HERE.parent / "lib" / "library_anime_full.json")])
    ap.add_argument("--min-periodicity", type=float, default=0.6)
    ap.add_argument("--min-ms", type=int, default=110)
    args = ap.parse_args()

    clips = PM.load_library(args.lib, None, None, 0.0, 99.0, 0.15)
    hires.ensure_anime_maps([c["src"] for c in clips if c["work"].startswith("anime")])
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"{len(clips)} clips on {dev}", flush=True)

    bank = []
    for k, clip in enumerate(clips):
        try:
            y32, _ = hires.load(clip, 32000)
        except Exception as e:  # unreadable slice
            print("skip", clip["path"], e, flush=True)
            continue
        y = librosa.resample(y32.astype(np.float32), orig_sr=32000, target_sr=SR16)
        if len(y) < SR16 // 4:
            continue
        x = torch.from_numpy(y).float().unsqueeze(0).to(dev)
        f0, per = torchcrepe.predict(x, SR16, HOP, 80.0, 900.0, model="full",
                                     decoder=torchcrepe.decode.argmax, return_periodicity=True,
                                     device=dev, batch_size=512, pad=False)
        f0 = f0.squeeze(0).cpu().numpy()
        per = per.squeeze(0).cpu().numpy()
        rms = librosa.feature.rms(y=y, frame_length=640, hop_length=HOP)[0][:len(f0)]
        n = min(len(f0), len(rms))
        f0, per, rms = f0[:n], per[:n], rms[:n]
        st = 69 + 12 * np.log2(np.maximum(f0, 1e-3) / 440.0)
        thr = max(rms.max() * 0.08, 1e-4)
        voiced = (per > args.min_periodicity) & (rms > thr)
        for a, b in runs_from_track(st, voiced, args.min_ms // 10):
            seg = st[a:b + 1]
            bank.append({
                "ci": k, "path": clip["path"], "char": clip["char"], "work": clip["work"],
                "t0": round(a * HOP / SR16, 3), "t1": round((b + 1) * HOP / SR16, 3),
                "st": round(float(np.median(seg)), 2),
                "cents_std": round(float(np.std(seg) * 100), 1),
                "per": round(float(np.mean(per[a:b + 1])), 3),
                "rms": round(float(np.mean(rms[a:b + 1]) / (rms.max() + 1e-9)), 3),
                "text": clip.get("text", ""),
            })
        if (k + 1) % 250 == 0:
            print(f"  {k + 1}/{len(clips)} clips, {len(bank)} nuclei", flush=True)
    OUT.parent.mkdir(exist_ok=True)
    json.dump(bank, open(OUT, "w"), ensure_ascii=False)
    d = np.array([b["t1"] - b["t0"] for b in bank])
    print(f"bank: {len(bank)} nuclei; dur median {np.median(d):.2f}s, p90 {np.percentile(d, 90):.2f}s, "
          f"pitch range {min(b['st'] for b in bank):.0f}-{max(b['st'] for b in bank):.0f} semitone, "
          f"{len({b['char'] for b in bank})} characters", flush=True)


if __name__ == "__main__":
    main()
