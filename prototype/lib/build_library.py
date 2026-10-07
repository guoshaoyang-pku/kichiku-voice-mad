#!/usr/bin/env python3
"""Build the kichiku sample library.

Sources:
- materials/genshin/voice/<char>/*.wav (+ .lab text)   [AI-Hobbyist packs]
- materials/anime/<work>/dry_voice/*.wav               [long clean voice -> silence-sliced]

Output:
- prototype/lib/samples/<work>/<char>/*.wav  (16k mono, silence-trimmed slices)
- prototype/lib/library_<which>.json         (per-sample analysis)

Modes: python3 build_library.py [genshin|anime|all|re-genshin|re-anime]
re-* modes re-analyze existing library json in place (audio slices must exist).

NOTE: uses torchcrepe argmax decode in length-sorted batches. (Batched viterbi
decode is corrupted by zero padding -> deflated voiced ratios; argmax matches
single-file results exactly.)
"""
import json
import os
import re
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torchcrepe

ROOT = Path(__file__).resolve().parents[2]
MAT = ROOT / "materials"
LIB = ROOT / "prototype" / "lib"
SR = 16000

MIN_DUR, MAX_DUR = 0.25, 6.0      # keep slices in this range (seconds)
TRIM_DB = 35                       # silence trim threshold
SLICE_TOP_DB = 32                  # anime long-file slicing threshold
SLICE_MIN_GAP = 0.09               # merge gaps shorter than this (s)

# BV -> (work, character) for anime dry-voice files
ANIME_MAP = {
    "BV11pYpzrEJw": ("mygo", "千早爱音"), "BV19XQ9BTEtP": ("mygo", "高松灯"),
    "BV19yQDYiEQe": ("mygo", "椎名立希"), "BV1ct421E7Rj": ("mygo", "要乐奈"),
    "BV1GVkhY9Eu7": ("mygo", "要乐奈"), "BV1JAcLz8ErB": ("mygo", "高松灯"),
    "BV1RGKLerEKb": ("mygo", "长崎爽世"), "BV1WX4y1j71B": ("mygo", "长崎爽世"),
    "BV1yaKLeZEDG": ("mygo", "长崎爽世"), "BV1Zu411F7og": ("mygo", "若叶睦"),
    "BV1zjbK6oESw": ("mygo", "要乐奈"),
    "BV1gGcdeTEy4": ("ave_mujica", "若叶睦"), "BV1GwRZYFE2c": ("ave_mujica", "墨缇丝"),
    "BV1gxwmeEEZG": ("ave_mujica", "丰川祥子"), "BV1iPRXYpEmD": ("ave_mujica", "若叶睦"),
    "BV1iWryY8EAh": ("ave_mujica", "祐天寺喵梦"), "BV1j4VizmEh5": ("ave_mujica", "三角初华"),
    "BV1s65fzcEi2": ("ave_mujica", "八幡海铃"),
    "BV1hd6fBnEHn": ("bocchi", "后藤一里"), "BV1Wr421s761": ("bocchi", "混合"),
    "BV1zJJ56sE7o": ("bocchi", "未确认"),
}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def collect_genshin():
    """Trim AI-Hobbyist genshin wavs into lib; pair .lab text."""
    out = []
    for char_dir in sorted((MAT / "genshin" / "voice").iterdir()):
        if not char_dir.is_dir():
            continue
        wavs = sorted(char_dir.rglob("*.wav"))
        if not wavs:
            continue
        dst_dir = ensure_dir(LIB / "samples" / "genshin" / char_dir.name)
        kept = 0
        for w in wavs:
            try:
                y, _ = librosa.load(w, sr=SR, mono=True)
            except Exception:
                continue
            y, _ = librosa.effects.trim(y, top_db=TRIM_DB)
            dur = len(y) / SR
            if not (MIN_DUR <= dur <= MAX_DUR):
                continue
            text = ""
            lab = w.with_suffix(".lab")
            if lab.exists():
                text = lab.read_text(errors="ignore").strip()
            dst = dst_dir / w.name
            sf.write(dst, y, SR)
            out.append({"path": str(dst), "work": "genshin", "char": char_dir.name,
                        "text": text, "src": str(w)})
            kept += 1
        print(f"  genshin/{char_dir.name}: {kept}/{len(wavs)} kept", flush=True)
    return out


def slice_anime():
    """Silence-slice long dry-voice wavs into short clips."""
    out = []
    for wav in sorted(MAT.glob("anime/*/dry_voice/*.wav")):
        m = re.search(r"(BV[0-9A-Za-z]+)", wav.name)
        work, char = ANIME_MAP.get(m.group(1), (wav.parent.parent.name, "未知")) if m \
            else (wav.parent.parent.name, "未知")
        dst_dir = ensure_dir(LIB / "samples" / "anime" / work / char)
        stem = re.sub(r"[^\w]", "_", wav.stem)[:60]
        print(f"  slicing {wav.name} ({work}/{char})", flush=True)
        y, _ = librosa.load(wav, sr=SR, mono=True)
        intervals = librosa.effects.split(y, top_db=SLICE_TOP_DB)
        merged = []
        for s, e in intervals:
            if merged and (s - merged[-1][1]) / SR < SLICE_MIN_GAP:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        kept = 0
        for i, (s, e) in enumerate(merged):
            seg = y[s:e]
            dur = len(seg) / SR
            if dur < MIN_DUR:
                continue
            chunks = []
            if dur > MAX_DUR:
                hop = int(MAX_DUR * SR)
                for cs in range(0, len(seg) - int(MIN_DUR * SR), hop):
                    chunks.append(seg[cs:cs + hop])
            else:
                chunks.append(seg)
            for j, c in enumerate(chunks):
                c, _ = librosa.effects.trim(c, top_db=TRIM_DB)
                if not (MIN_DUR <= len(c) / SR <= MAX_DUR + 0.5):
                    continue
                dst = dst_dir / f"{stem}_{i:04d}_{j}.wav"
                sf.write(dst, c, SR)
                out.append({"path": str(dst), "work": f"anime:{work}", "char": char,
                            "text": "", "src": str(wav)})
                kept += 1
        print(f"    -> {kept} slices", flush=True)
    return out


def analyze(items):
    """CREPE f0 + rms. argmax decode, length-sorted batches (padding-safe)."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"analyzing {len(items)} samples on {device} ...", flush=True)
    ys = [None] * len(items)
    for i, it in enumerate(items):
        y, _ = librosa.load(it["path"], sr=SR, mono=True)
        ys[i] = y
    order = sorted(range(len(items)), key=lambda i: len(ys[i]))
    B = 64
    for b0 in range(0, len(order), B):
        idxs = order[b0:b0 + B]
        lens = [len(ys[i]) for i in idxs]
        pad = np.zeros((len(idxs), max(lens)), dtype=np.float32)
        for k, i in enumerate(idxs):
            pad[k, :lens[k]] = ys[i]
        audio = torch.tensor(pad).to(device)
        f0, periodicity = torchcrepe.predict(
            audio, SR, model="full", decoder=torchcrepe.decode.argmax,
            return_periodicity=True, device=device, batch_size=len(idxs))
        f0 = f0.cpu().numpy()
        per = periodicity.cpu().numpy()
        for k, i in enumerate(idxs):
            y = ys[i]
            n_frames = 1 + lens[k] // 160
            fk, pk = f0[k].ravel()[:n_frames], per[k].ravel()[:n_frames]
            voiced = fk[(pk > 0.5) & (fk > 60) & (fk < 1500)]
            rms = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
            it = items[i]
            if len(voiced) >= 3:
                semis = 12 * np.log2(voiced / 440.0) + 69
                it["f0_semi"] = float(np.median(semis))
                it["f0_iqr_cents"] = float((np.percentile(semis, 75) - np.percentile(semis, 25)) * 100)
                it["voiced_ratio"] = float(len(voiced) / max(n_frames, 1))
            else:
                it["f0_semi"] = None
                it["f0_iqr_cents"] = 999.0
                it["voiced_ratio"] = 0.0
            it["dur"] = round(lens[k] / SR, 3)
            it["rms_db"] = round(20 * np.log10(rms), 1)
            ys[i] = None  # free memory
        if (b0 // B) % 10 == 0:
            print(f"  {min(b0 + B, len(order))}/{len(order)}", flush=True)
    return items


def repair_anime_chars(items):
    """Re-derive work/char from src filename (BV id) using fixed regex."""
    fixed = 0
    for it in items:
        m = re.search(r"(BV[0-9A-Za-z]+)", os.path.basename(it.get("src", "")))
        if m and m.group(1) in ANIME_MAP:
            work, char = ANIME_MAP[m.group(1)]
            it["work"] = f"anime:{work}"
            it["char"] = char
            fixed += 1
    print(f"  repaired chars for {fixed}/{len(items)} anime items", flush=True)
    return items


def write_and_report(items, which):
    usable = [it for it in items if it.get("f0_semi") is not None
              and 40 <= it["f0_semi"] <= 95 and it["voiced_ratio"] > 0.3]
    print(f"total {len(items)}, usable(pitched&voiced) {len(usable)}")
    out_path = LIB / (f"library_{which}.json" if which != "all" else "library.json")
    with open(out_path, "w") as f:
        json.dump(items, f, ensure_ascii=False)
    by_work = {}
    for it in usable:
        by_work.setdefault(it["work"], []).append(it)
    for w, its in sorted(by_work.items()):
        chars = sorted({i["char"] for i in its})
        semis = [i["f0_semi"] for i in its]
        print(f"  {w}: {len(its)} samples, f0 {min(semis):.1f}-{max(semis):.1f} "
              f"(median {np.median(semis):.1f}) chars={','.join(chars[:8])}")


def reanalyze(which):
    path = LIB / f"library_{which}.json"
    items = json.load(open(path))
    if which == "anime":
        items = repair_anime_chars(items)
    items = [it for it in items if Path(it["path"]).exists()]
    items = analyze(items)
    write_and_report(items, which)


def main():
    ensure_dir(LIB)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which.startswith("re-"):
        reanalyze(which[3:])
        return
    items = []
    if which in ("all", "genshin"):
        print("[1/3] genshin packs", flush=True)
        items += collect_genshin()
    if which in ("all", "anime"):
        print("[2/3] anime slicing", flush=True)
        items += slice_anime()
        items = repair_anime_chars(items)
    if not items:
        print("no items?!")
        return
    print("[3/3] analysis", flush=True)
    items = analyze(items)
    write_and_report(items, which)


if __name__ == "__main__":
    main()
