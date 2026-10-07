#!/usr/bin/env python3
"""Diagnostics for v2 renders: compare against the ORIGINAL audio, not a synth reference."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC",
                                          "Hiragino Sans GB", "DejaVu Sans"]
import matplotlib.pyplot as plt
import librosa
import numpy as np

SR = 32000
HOP = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--melody-cache", required=True, help="npz with times/f0/conf")
    ap.add_argument("--window", type=float, nargs=2, required=True)
    ap.add_argument("--out-dir", default="out")
    args = ap.parse_args()

    out = Path(args.out_dir)
    name = args.name
    t0, t1 = args.window
    load = lambda p: librosa.load(p, sr=SR, mono=True)[0]
    orig_v = load(out / f"{name}_orig_vocal_excerpt.wav")
    orig_m = load(out / f"{name}_orig_mix_excerpt.wav")
    rend_v = load(out / f"{name}_vocal.wav")
    rend_m = load(out / f"{name}.wav")
    n = min(len(orig_v), len(rend_v))
    orig_v, orig_m, rend_v, rend_m = orig_v[:n], orig_m[:n], rend_v[:n], rend_m[:n]

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    for ax, y, title in [(axes[0], orig_m, f"原曲混音 ({t0:.0f}-{t1:.0f}s)"),
                         (axes[1], rend_m, "v2 渲染混音")]:
        S = librosa.feature.melspectrogram(y=y, sr=SR, hop_length=HOP, n_mels=128)
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=SR, hop_length=HOP,
                                 x_axis="time", y_axis="mel", ax=ax, cmap="magma")
        ax.set_title(title)

    z = np.load(args.melody_cache)
    tt, f0o, conf = z["times"], z["f0"], z.get("conf", z.get("per"))
    m = (tt >= t0) & (tt <= t1)
    ax = axes[2]
    ax.plot(tt[m] - t0, np.where((conf[m] > 0.15) & (f0o[m] > 0), f0o[m], np.nan),
            lw=1.6, label="原曲主旋律 (Melodia)", color="tab:blue", alpha=0.9)
    y16 = librosa.resample(rend_v, orig_sr=SR, target_sr=16000)
    f0r = librosa.yin(y16, fmin=60, fmax=1000, sr=16000, frame_length=1024,
                      hop_length=160, trough_threshold=0.08)
    tr = np.arange(len(f0r)) * 160 / 16000
    f0r[f0r <= 60] = np.nan
    ax.plot(tr, f0r, lw=1.0, label="渲染人声 F0 (YIN)", color="tab:red", alpha=0.7)
    ax.set_ylim(50, 600)
    ax.set_ylabel("Hz")
    ax.legend()
    ax.set_title("旋律轮廓对比")
    ax.grid(alpha=0.3)

    co = librosa.feature.chroma_cqt(y=orig_v, sr=SR, hop_length=HOP)
    cr = librosa.feature.chroma_cqt(y=rend_v, sr=SR, hop_length=HOP)
    mo = librosa.feature.mfcc(y=orig_v, sr=SR, hop_length=HOP, n_mfcc=20)
    mr = librosa.feature.mfcc(y=rend_v, sr=SR, hop_length=HOP, n_mfcc=20)
    nrm = lambda x: np.nan_to_num((x - x.mean()) / (x.std() + 1e-9))
    D, wp = librosa.sequence.dtw(X=nrm(co), Y=nrm(cr), metric="euclidean")
    wp = wp[::-1]
    ax = axes[3]
    ax.plot(wp[:, 0] * HOP / SR, wp[:, 1] * HOP / SR, lw=1)
    ax.plot([0, n / SR], [0, n / SR], "k--", lw=0.8)
    ax.set_xlabel("原曲时间 (s)")
    ax.set_ylabel("渲染时间 (s)")
    D2, wp2 = librosa.sequence.dtw(X=nrm(mo), Y=nrm(mr), metric="euclidean")
    chroma_cost = float(D[-1, -1] / len(wp))
    mfcc_cost = float(D2[-1, -1] / len(wp2) / 20)
    ax.set_title(f"chroma DTW cost={chroma_cost:.2f}   MFCC DTW cost={mfcc_cost:.2f}")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    p = out / "diagnostics" / f"{name}_vs_original.png"
    p.parent.mkdir(exist_ok=True)
    plt.savefig(p, dpi=110)
    json.dump({"chroma_dtw": round(chroma_cost, 3), "mfcc_dtw": round(mfcc_cost, 3),
               "note": "DTW vs original vocal-stem excerpt, normalized features"},
              open(out / "diagnostics" / f"{name}.diagnostics.json", "w"))
    print("saved", p)
    print(json.dumps({"chroma_dtw": round(chroma_cost, 3),
                      "mfcc_dtw": round(mfcc_cost, 3)}))


if __name__ == "__main__":
    main()
