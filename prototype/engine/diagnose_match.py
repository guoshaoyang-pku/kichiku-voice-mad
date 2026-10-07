#!/usr/bin/env python3
"""Diagnostics for v1 phrase renders: spectrogram, MFCC/chroma DTW match."""
import argparse
import json
from pathlib import Path

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC",
                                          "Hiragino Sans GB", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

SR = 22050
HOP = 512


def load(path):
    return librosa.load(path, sr=SR, mono=True)[0]


def mfcc(y):
    m = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20, hop_length=HOP)
    return np.nan_to_num(librosa.util.normalize(m, axis=0), nan=0.0,
                         posinf=0.0, neginf=0.0)


def chroma(y):
    c = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP, bins_per_octave=36)
    return np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0) + 1e-6


def aligned_cosine(x, y, wp):
    xs = x[:, wp[:, 0]]
    ys = y[:, wp[:, 1]]
    num = np.sum(xs * ys, axis=0)
    den = np.linalg.norm(xs, axis=0) * np.linalg.norm(ys, axis=0) + 1e-9
    return float(np.mean(num / den))


def dtw_metrics(ref, got):
    ref_m, got_m = mfcc(ref), mfcc(got)
    ref_c, got_c = chroma(ref), chroma(got)
    dm, wp_m = librosa.sequence.dtw(X=ref_m, Y=got_m, metric="cosine")
    dc, wp_c = librosa.sequence.dtw(X=ref_c, Y=got_c, metric="cosine")
    return {
        "mfcc_dtw_cosine_distance_per_step": round(float(dm[-1, -1] / len(wp_m)), 4),
        "mfcc_aligned_cosine": round(aligned_cosine(ref_m, got_m, wp_m), 4),
        "chroma_dtw_cosine_distance_per_step": round(float(dc[-1, -1] / len(wp_c)), 4),
        "chroma_aligned_cosine": round(aligned_cosine(ref_c, got_c, wp_c), 4),
        "frames_ref": int(ref_m.shape[1]),
        "frames_render": int(got_m.shape[1]),
    }, ref_c, got_c


def plot_melody(out_png, ref, got, mix, cues, start, dur):
    fig, axes = plt.subplots(4, 1, figsize=(16, 11), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1.35, 1.2]})
    ref_c = chroma(ref)
    got_c = chroma(got)
    extent = [0, dur, 0, 12]
    for ax, data, title in [
        (axes[0], ref_c, "Target melody reference - chroma"),
        (axes[1], got_c, "Rendered vocal - chroma"),
    ]:
        img = librosa.display.specshow(data, x_axis="time", y_axis="chroma", sr=SR,
                                       hop_length=HOP, ax=ax, cmap="magma")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlim(0, dur)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(
        y=got, sr=SR, n_mels=128, hop_length=HOP, fmax=8000), ref=np.max)
    librosa.display.specshow(mel, x_axis="time", y_axis="mel", sr=SR,
                             hop_length=HOP, ax=axes[2], cmap="viridis")
    axes[2].set_title("Rendered vocal - mel spectrogram", loc="left", fontsize=10)
    axes[2].set_xlim(0, dur)

    # F0 overlay against target note steps.
    f0_ref = librosa.yin(ref, fmin=70, fmax=1000, sr=SR, frame_length=2048,
                         hop_length=HOP, trough_threshold=0.08)
    f0 = librosa.yin(got, fmin=70, fmax=1000, sr=SR, frame_length=2048,
                     hop_length=HOP, trough_threshold=0.08)
    t = np.arange(len(f0)) * HOP / SR
    t_ref = np.arange(len(f0_ref)) * HOP / SR
    semi_ref = 69 + 12 * np.log2(np.maximum(f0_ref, 1e-6) / 440.0)
    semi_ref[f0_ref <= 0] = np.nan
    semi = 69 + 12 * np.log2(np.maximum(f0, 1e-6) / 440.0)
    semi[f0 <= 0] = np.nan
    axes[3].plot(t_ref, semi_ref, lw=1.2, color="#d62728", alpha=0.65,
                 label="target reference F0")
    axes[3].plot(t, semi, lw=1.0, color="#1f77b4", label="rendered F0")
    for c in cues:
        axes[3].axvspan(c["t0"], min(c["t1"], dur), color="gray", alpha=0.08, lw=0)
    axes[3].set_ylim(45, 85)
    axes[3].set_title("Target vs rendered F0 with selected clip spans", loc="left", fontsize=10)
    axes[3].set_xlabel("seconds")
    axes[3].set_ylabel("MIDI semitone")
    axes[3].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_clips(out_png, base, cues, vocal, max_clips=8):
    ref_dir = base.parent / (base.name + "_refs")
    rows = min(max_clips, len(cues))
    fig, axes = plt.subplots(rows, 2, figsize=(13, max(4, rows * 1.7)), sharex=True)
    if rows == 1:
        axes = np.array([axes])
    for r in range(rows):
        c = cues[r]
        refs = sorted(ref_dir.glob(f"{r + 1:02d}_*.wav"))
        orig = load(refs[0]) if refs else np.zeros(SR // 2)
        got = vocal[int(c["t0"] * SR):int(c["t1"] * SR)]
        for col, (sig, title) in enumerate([
            (orig, f"original {r + 1}: {c['char']}"),
            (got, f"rendered {r + 1}"),
        ]):
            sig = librosa.util.fix_length(sig, size=int(4.2 * SR))
            mel = librosa.power_to_db(librosa.feature.melspectrogram(
                y=sig, sr=SR, n_mels=96, hop_length=256, fmax=8000), ref=np.max)
            librosa.display.specshow(mel, x_axis="time", y_axis="mel", sr=SR,
                                     hop_length=256, ax=axes[r, col], cmap="viridis")
            axes[r, col].set_title(title, loc="left", fontsize=9)
            axes[r, col].set_xlim(0, 4.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="output base, e.g. out/v1_roundabout_anime")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, required=True)
    ap.add_argument("--max-clips", type=int, default=8)
    args = ap.parse_args()

    base = Path(args.base)
    diag = base.parent / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    vocal = load(base.with_name(base.name + "_vocal.wav"))
    ref = load(base.with_name(base.name + "_melody_reference.wav"))
    original = load(base.with_name(base.name + "_reference.wav"))
    mix = load(base.with_suffix(".wav"))
    cues = json.load(open(base.with_suffix(".cues.json")))

    metrics, _, _ = dtw_metrics(ref, vocal)
    metrics.update({
        "base": base.name,
        "duration_seconds": args.dur,
        "reference_original_vs_render_mfcc_cosine": None,
    })
    orig_metrics, _, _ = dtw_metrics(ref, original)
    metrics["original_sequence_mfcc_aligned_cosine"] = orig_metrics["mfcc_aligned_cosine"]
    metrics["original_sequence_chroma_aligned_cosine"] = orig_metrics["chroma_aligned_cosine"]

    stem = diag / base.name
    json.dump(metrics, open(stem.with_suffix(".diagnostics.json"), "w"), indent=1)
    plot_melody(stem.with_name(stem.name + "_melody_compare.png"), ref, vocal, mix, cues,
                args.start, args.dur)
    plot_clips(stem.with_name(stem.name + "_clip_compare.png"), base, cues, vocal,
               args.max_clips)
    print(json.dumps(metrics, indent=1))
    print(stem.with_name(stem.name + "_melody_compare.png"))
    print(stem.with_name(stem.name + "_clip_compare.png"))


if __name__ == "__main__":
    main()
