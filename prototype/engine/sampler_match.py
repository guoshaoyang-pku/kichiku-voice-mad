#!/usr/bin/env python3
"""v3 engine: layered sampler driven by the ORIGINAL recording.

Layers
  melody : the original lead melody (Melodia on the mix) is cut into legato notes. Each note
           is sung by a real vowel nucleus from a character line, re-pitched with WORLD (the
           spectral envelope / formants are kept, so the voice stays itself) and sustained by
           ping-pong looping the stable part of the vowel. One "singer" character per phrase.
  meme   : whole, unmodified character lines dropped into the gaps between phrases.
  backing: quiet synth following the original bass stem, -10 dB under the vocals.
No original audio is mixed into the output.
"""
import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import pyworld as pw
import soundfile as sf
import torch
import torchcrepe

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
sys.path.insert(0, str(HERE))
import hires  # noqa: E402
import phrase_match as PM  # noqa: E402
import audio_match as AM  # noqa: E402
from nucleus_bank import runs_from_track  # noqa: E402

FS = 32000
FP = 5.0                      # WORLD frame period, ms
FPS = FP / 1000.0


# ------------------------------------------------------------ target melody
def target_track(times, f0, conf, t0, t1, vth=0.15, fill_gap=0.12, min_run=0.08):
    m = (times >= t0 - 0.3) & (times <= t1 + 0.3)
    t, f, c = times[m], f0[m], conf[m]
    dt = t[1] - t[0]
    v = (f > 0) & (c > vth)
    st = np.full(len(f), np.nan)
    st[v] = 69 + 12 * np.log2(f[v] / 440.0)
    idx = np.flatnonzero(v)
    if len(idx) < 4:
        return None
    # fill short gaps
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= int(fill_gap / dt):
            st[a:b + 1] = np.linspace(st[a], st[b], b - a + 1)
            v[a:b + 1] = True
    # drop very short voiced runs
    idx = np.flatnonzero(v)
    runs, s = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > 1:
            runs.append((s, a)); s = b
    runs.append((s, idx[-1]))
    for a, b in runs:
        if (b - a + 1) * dt < min_run:
            v[a:b + 1] = False
            st[a:b + 1] = np.nan
    # median filter + octave-jump repair
    idx = np.flatnonzero(v)
    vals = st[idx].copy()
    k = 5
    padded = np.pad(vals, (k // 2, k // 2), mode="edge")
    vals = np.array([np.median(padded[i:i + k]) for i in range(len(vals))])
    for i in range(1, len(vals)):
        ref = np.median(vals[max(0, i - 12):i])
        d = vals[i] - ref
        for sh in (12, -12):
            if abs(d) > 8 and abs(d - sh) < 2.5:
                vals[i] -= sh
                break
    st[idx] = vals
    grid = np.arange(t0, t1 + 0.4, FPS)
    vg = np.interp(grid, t, v.astype(float)) > 0.5
    stg = np.full(len(grid), np.nan)
    stg[vg] = np.interp(grid[vg], t[idx], st[idx])
    return grid, stg, vg


def make_notes(grid, stg, vg, legato=0.35, split_semi=0.9, min_ms=90):
    step = 2                                    # 10 ms
    st10 = stg[::step]
    v10 = vg[::step]
    t10 = grid[::step]
    st_f = np.where(v10, st10, 0.0)
    runs = runs_from_track(st_f, v10, min_frames=max(2, min_ms // 10), split_semi=split_semi)
    notes = []
    for a, b in runs:
        notes.append({"t0": float(t10[a]), "t1": float(t10[b] + 0.01),
                      "st": float(np.median(st10[a:b + 1]))})
    for i in range(len(notes) - 1):
        gap = notes[i + 1]["t0"] - notes[i]["t1"]
        notes[i]["gap_after"] = gap
        if gap < legato:
            notes[i]["t1"] = notes[i + 1]["t0"]
    if notes:
        notes[-1]["gap_after"] = 9.0
    # phrases: gap >= 0.45 s or length cap
    ph, start = 0, notes[0]["t0"]
    for i, n in enumerate(notes):
        if i > 0:
            gap = n["t0"] - notes[i - 1]["t1"]
            if gap >= 0.45 or (n["t0"] - start > 3.8 and gap >= 0.1):
                ph += 1
                start = n["t0"]
        n["phrase"] = ph
        n["first"] = i == 0 or notes[i - 1]["phrase"] != ph
    return notes


# ------------------------------------------------------------ nucleus selection
class Bank:
    def __init__(self, path, n_chars):
        self.items = json.load(open(path))
        good = [b for b in self.items if b["per"] > 0.75 and b["cents_std"] < 90]
        cnt = {}
        for b in good:
            cnt[b["char"]] = cnt.get(b["char"], 0) + 1
        self.chars = [c for c, _ in sorted(cnt.items(), key=lambda x: -x[1])][:n_chars]
        self.items = [b for b in good if b["char"] in self.chars]
        self.st = np.array([b["st"] for b in self.items])
        self.dur = np.array([b["t1"] - b["t0"] for b in self.items])
        self.std = np.array([b["cents_std"] for b in self.items])
        self.per = np.array([b["per"] for b in self.items])
        self.rms = np.array([b["rms"] for b in self.items])
        self.char_idx = {c: np.flatnonzero([b["char"] == c for b in self.items]) for c in self.chars}
        self.uses = np.zeros(len(self.items))
        print(f"bank: {len(self.items)} nuclei, singers {self.chars}", flush=True)

    def cost(self, note, idx, D):
        dp = np.abs(self.st[idx] - note["st"])
        cp = (dp / 2.5) ** 1.5 + 3.0 * (dp > 5.0)
        r = D / self.dur[idx]
        cd = np.where(r <= 1, 0.2 * (1 - r), 0.45 * np.log2(np.maximum(r, 1)))
        return (cp + cd + 0.3 * self.std[idx] / 100.0 + 0.3 * (1 - self.per[idx]) +
                0.2 * (1 - self.rms[idx]) + 0.35 * self.uses[idx])

    def pick_singer(self, phrase_notes):
        best, best_c = None, 1e9
        for c in self.chars:
            idx = self.char_idx[c]
            tot = np.mean([self.cost(n, idx, n["t1"] - n["t0"]).min() for n in phrase_notes])
            if tot < best_c:
                best, best_c = c, tot
        return best

    def pick(self, note, singer):
        idx = self.char_idx[singer]
        c = self.cost(note, idx, note["t1"] - note["t0"])
        j = idx[int(np.argmin(c))]
        self.uses[j] += 1
        return j, float(c.min())


# ------------------------------------------------------------ synthesis
_audio_cache = {}
_lib_by_path = {}


def clip_audio(path):
    if path not in _audio_cache:
        y, _ = hires.load(_lib_by_path[path], FS)
        _audio_cache[path] = y
    return _audio_cache[path]


def interp_frames(sp, ap, pos):
    lo = np.clip(np.floor(pos).astype(int), 0, len(sp) - 1)
    hi = np.clip(lo + 1, 0, len(sp) - 1)
    fr = (pos - lo)[:, None]
    sp_o = np.exp((1 - fr) * np.log(sp[lo] + 1e-16) + fr * np.log(sp[hi] + 1e-16))
    ap_o = (1 - fr) * ap[lo] + fr * ap[hi]
    return sp_o, ap_o


def src_index_map(Np, Nn, Tn, total):
    """Map each of `total` output frames to a source frame (pre-context 1:1, then attack /
    ping-pong sustain / release for the nucleus)."""
    out = np.zeros(total)
    out[:Np] = np.arange(Np)
    if Tn <= Nn:
        out[Np:Np + Tn] = Np + np.arange(Tn)
    else:
        A = max(1, int(0.25 * Nn))
        R = max(1, int(0.2 * Nn))
        Z = max(2, Nn - A - R)
        for k in range(Tn):
            if k < A:
                s = k
            elif k >= Tn - R:
                s = Nn - R + (k - (Tn - R))
            else:
                t = k - A
                m = t % (2 * Z)
                s = A + min(m if m < Z else 2 * Z - m, Z - 1)
            out[Np + k] = Np + min(s, Nn - 1)
    return out


def sing_note(bank, j, note, grid, stg, vg, pre, tail):
    b = bank.items[j]
    y = clip_audio(b["path"])
    a = max(0, int((b["t0"] - pre) * FS))
    pre_real = b["t0"] - a / FS
    e = min(len(y), int((b["t1"] + 0.02) * FS))
    seg = y[a:e].astype(np.float64)
    f0, t = pw.dio(seg, FS, f0_floor=70.0, f0_ceil=1000.0, frame_period=FP)
    f0 = pw.stonemask(seg, f0, t, FS)
    sp = pw.cheaptrick(seg, f0, t, FS)
    ap = pw.d4c(seg, f0, t, FS)
    Np = int(round(pre_real / FPS))
    Nn = max(2, len(f0) - Np - 1)
    D = (note["t1"] - note["t0"]) + tail
    Tn = max(2, int(round(D / FPS)))
    total = Np + Tn
    pos = src_index_map(Np, Nn, Tn, total)
    sp_o, ap_o = interp_frames(sp, ap, pos)
    src_v = f0[np.clip(np.round(pos).astype(int), 0, len(f0) - 1)] > 0
    nuc_slice = slice(Np, total)
    if src_v[nuc_slice].mean() < 0.5:
        src_v[nuc_slice] = True
    t_abs = note["t0"] - Np * FPS + np.arange(total) * FPS
    valid = vg
    st_t = np.interp(t_abs, grid[valid], stg[valid])
    f0_o = np.where(src_v, 440.0 * 2 ** ((st_t - 69) / 12.0), 0.0)
    wav = pw.synthesize(np.ascontiguousarray(f0_o), np.ascontiguousarray(sp_o),
                        np.ascontiguousarray(ap_o), FS, FP)
    act = np.abs(wav) > 1e-4
    rms = np.sqrt(np.mean(wav[act] ** 2)) if act.any() else 1.0
    wav = wav * (0.1 / (rms + 1e-9))
    fi = min(int((0.004 if pre_real > 0.01 else 0.012) * FS), len(wav) // 4)
    fo = min(int(0.03 * FS), len(wav) // 3)
    if fi:
        wav[:fi] *= np.linspace(0, 1, fi)
    if fo:
        wav[-fo:] *= np.linspace(1, 0, fo)
    return wav.astype(np.float32), note["t0"] - Np * FPS, pre_real


def _varispeed(y, semi):
    """Plain resampling pitch change: no vocoder, only the raw waveform (tiny shifts)."""
    if abs(semi) < 0.03:
        return y
    r = 2 ** (semi / 12.0)
    return librosa.resample(y, orig_sr=int(round(FS * r)), target_sr=FS, res_type="soxr_hq")


def splice_note(bank, j, note, singer, pre, tail, max_parts=4, xfade=0.02):
    """Raw-waveform note: real nucleus (+ optional natural onset), varispeed to the note pitch,
    lengthened by crossfading further real nuclei of the same singer (never loop / vocode)."""
    target = (note["t1"] - note["t0"]) + tail
    used = {j}
    parts, onset_len = [], 0.0
    cur = j
    total = 0.0
    while True:
        b = bank.items[cur]
        y = clip_audio(b["path"])
        a = max(0, int((b["t0"] - (pre if not parts else 0.0)) * FS))
        e = min(len(y), int(b["t1"] * FS))
        seg = _varispeed(y[a:e].astype(np.float32), note["st"] - b["st"])
        if not parts:
            onset_len = (b["t0"] - a / FS) * 2 ** ((note["st"] - b["st"]) / -12.0)
        seg = seg / (np.sqrt(np.mean(seg ** 2)) + 1e-9) * 0.1
        parts.append(seg)
        total += len(seg) / FS - (xfade if len(parts) > 1 else 0.0)
        if total >= target or len(parts) >= max_parts:
            break
        idx = bank.char_idx[singer]
        dp = np.abs(bank.st[idx] - note["st"]) + 4.0 * np.array([int(i in used) for i in idx])
        cur = int(idx[int(np.argmin(dp))])
        used.add(cur)
        bank.uses[cur] += 1
    xf = int(xfade * FS)
    out = parts[0]
    for seg in parts[1:]:
        n = min(xf, len(out), len(seg))
        w = np.linspace(0, 1, n)
        out = np.concatenate([out[:-n], out[-n:] * np.cos(w * np.pi / 2) + seg[:n] * np.sin(w * np.pi / 2), seg[n:]])
    want = int((onset_len + target) * FS)
    out = out[:want]
    fi = min(int((0.004 if onset_len > 0.01 else 0.01) * FS), len(out) // 4)
    fo = min(int(0.03 * FS), len(out) // 3)
    if fi:
        out[:fi] *= np.linspace(0, 1, fi)
    if fo:
        out[-fo:] *= np.linspace(1, 0, fo)
    return out.astype(np.float32), note["t0"] - onset_len, onset_len, len(parts)


def shift_clip(y, semi):
    f0, t = pw.dio(y, FS, f0_floor=70.0, f0_ceil=1000.0, frame_period=FP)
    f0 = pw.stonemask(y, f0, t, FS)
    sp = pw.cheaptrick(y, f0, t, FS)
    ap = pw.d4c(y, f0, t, FS)
    return pw.synthesize(f0 * 2 ** (semi / 12.0), sp, ap, FS, FP)


def place(buf, y, t_start):
    i0 = int(round(t_start * FS))
    if i0 < 0:
        y = y[-i0:]
        i0 = 0
    if i0 >= len(buf):
        return
    seg = y[:len(buf) - i0]
    buf[i0:i0 + len(seg)] += seg


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True)
    ap.add_argument("--vocal-stem", required=True)
    ap.add_argument("--bass-stem", default=None)
    ap.add_argument("--bank", default=str(HERE.parent / "materials" / "nucleus_bank.json"))
    ap.add_argument("--lib", nargs="+", required=True)
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--dur-limit", type=float, default=100.0)
    ap.add_argument("--singers", type=int, default=8)
    ap.add_argument("--legato", type=float, default=0.35)
    ap.add_argument("--min-meme-gap", type=float, default=0.8)
    ap.add_argument("--meme-gain-db", type=float, default=-1.0)
    ap.add_argument("--backing-gain-db", type=float, default=-10.0)
    ap.add_argument("--transpose", type=float, default=0.0, help="semitones applied to melody target")
    ap.add_argument("--engine", choices=("raw", "world"), default="raw")
    ap.add_argument("--env-depth", type=float, default=0.8, help="0 disables macro envelope gain")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0, t1 = args.start, args.start + args.dur_limit
    times, f0, conf, rms_stem = AM.extract_melody(args.mix, args.vocal_stem)
    grid, stg, vg = target_track(times, f0, conf, t0, t1)
    stg = stg + args.transpose
    notes = make_notes(grid, stg, vg, legato=args.legato)
    notes = [n for n in notes if n["t0"] < t1]
    n_ph = notes[-1]["phrase"] + 1
    print(f"window {t0:.0f}-{t1:.0f}s: {len(notes)} notes, {n_ph} phrases", flush=True)

    lib = PM.load_library(args.lib, None, None, 0.0, 99.0, 0.15)
    for c in lib:
        _lib_by_path[c["path"]] = c
    hires.ensure_anime_maps([c["src"] for c in lib if c["work"].startswith("anime")])
    bank = Bank(args.bank, args.singers)

    n_samp = int((args.dur_limit + 2.5) * FS)
    melody = np.zeros(n_samp, dtype=np.float32)
    cues, raw_nuclei = [], []
    for p in range(n_ph):
        pn = [n for n in notes if n["phrase"] == p]
        if not pn:
            continue
        singer = bank.pick_singer(pn)
        for n in pn:
            j, cost = bank.pick(n, singer)
            pre = 0.07 if n["first"] else 0.0
            tail = min(n["gap_after"], 0.0) + 0.04 if n["gap_after"] < args.legato + 0.05 else 0.02
            tail = 0.04 if n["gap_after"] < args.legato + 0.05 else 0.02
            n_parts = 1
            if args.engine == "raw":
                wav, ts, pre_real, n_parts = splice_note(bank, j, n, singer, pre, tail)
            else:
                wav, ts, pre_real = sing_note(bank, j, n, grid, stg, vg, pre, tail)
            place(melody, wav, ts - t0)
            b = bank.items[j]
            cues.append({"layer": "melody", "engine": args.engine, "n_parts": n_parts,
                         "nuc_path": b["path"], "nuc_t0": b["t0"], "nuc_t1": b["t1"],
                         "t0": round(n["t0"] - t0, 3), "t1": round(n["t1"] - t0, 3),
                         "char": b["char"], "clip": Path(b["path"]).name, "text": b.get("text", ""),
                         "note_st": round(n["st"], 2), "nucleus_st": b["st"],
                         "shift": round(n["st"] - b["st"], 2), "nucleus_dur": round(b["t1"] - b["t0"], 3),
                         "phrase": p, "singer": singer, "cost": round(cost, 3)})
            raw_nuclei.append(b)
        print(f"  phrase {p + 1}/{n_ph} singer={singer} notes={len(pn)}", flush=True)

    # macro dynamics from the original vocal-stem envelope (smoothed ~150 ms)
    m = (times >= t0) & (times < t0 + len(melody) / FS)
    env = PM.smooth(rms_stem[m], 25)
    env_t = times[m] - t0
    ref = np.percentile(env[env > 1e-5], 90) if (env > 1e-5).any() else 1.0
    env_db = 20 * np.log10(np.maximum(env, 1e-5) / ref)
    gain = np.clip(10 ** (args.env_depth * env_db / 20), 0.3, 1.3)
    melody *= np.interp(np.arange(len(melody)) / FS, env_t, gain).astype(np.float32)

    # meme layer: whole lines, unmodified, in the gaps between phrases
    memes = np.zeros(n_samp, dtype=np.float32)
    cands = PM.load_library(args.lib, None, None, 1.0, 4.0, 0.35)
    used = set()
    ph_bounds = []
    for p in range(n_ph):
        pn = [n for n in notes if n["phrase"] == p]
        if pn:
            ph_bounds.append((p, pn[0]["t0"], pn[-1]["t1"], float(np.median([n["st"] for n in pn]))))
    for (p, a, b, st_m), nxt in zip(ph_bounds[:-1], ph_bounds[1:]):
        gap = nxt[1] - b
        if gap < args.min_meme_gap:
            continue
        room = gap + 0.35
        best, best_c = None, 1e9
        for c in cands:
            if c["path"] in used or c["dur"] > room + 0.5:
                continue
            cost = abs(c["f0_semi"] - st_m) / 6.0 + 0.8 * abs(c["dur"] - gap) / max(gap, 0.5) + \
                0.15 * min(c.get("f0_iqr_cents", 999), 1200) / 1200.0
            if cost < best_c:
                best, best_c = c, cost
        if best is None:
            continue
        used.add(best["path"])
        y, _ = hires.load(best, FS)
        y = y.astype(np.float32)
        y *= 0.1 / (np.sqrt(np.mean(y ** 2)) + 1e-9)
        g = min(int(0.006 * FS), len(y) // 4)
        y[:g] *= np.linspace(0, 1, g)
        y[-g:] *= np.linspace(1, 0, g)
        place(memes, y, b + 0.08 - t0)
        cues.append({"layer": "meme", "t0": round(b + 0.08 - t0, 3), "t1": round(b + 0.08 - t0 + len(y) / FS, 3),
                     "char": best["char"], "clip": Path(best["path"]).name, "text": best.get("text", ""),
                     "dur": round(len(y) / FS, 3), "phrase_center": round(st_m, 2),
                     "clip_center": round(best["f0_semi"], 2)})
    meme_gain = 10 ** (args.meme_gain_db / 20)
    mel_rms = np.sqrt(np.mean(melody[np.abs(melody) > 1e-5] ** 2)) if (np.abs(melody) > 1e-5).any() else 0.05
    mem_rms = np.sqrt(np.mean(memes[np.abs(memes) > 1e-5] ** 2)) if (np.abs(memes) > 1e-5).any() else 1.0
    memes *= (mel_rms / mem_rms) * meme_gain

    vocal = melody + memes

    backing = np.zeros_like(vocal)
    if args.bass_stem:
        bt, bf, bp, br = AM.extract_contour(args.bass_stem, fmin=35, fmax=350)
        backing = AM.synth_backing(bt, bf, bp, br, t0, t1 + 2.5, n_samp)
    av, ab = np.abs(vocal) > 1e-5, np.abs(backing) > 1e-6
    if av.any() and ab.any():
        backing *= np.sqrt(np.mean(vocal[av] ** 2)) * 10 ** (args.backing_gain_db / 20) / np.sqrt(np.mean(backing[ab] ** 2))
    end = int((args.dur_limit + 1.5) * FS)
    melody, memes, vocal, backing = melody[:end], memes[:end], vocal[:end], backing[:end]
    mix = vocal + backing
    master = 0.93 / max(np.max(np.abs(mix)), 1e-9)
    out = Path(args.out)
    if out.suffix == ".wav":
        out = out.with_suffix("")
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix, sig in (("_melody_layer", melody), ("_meme_layer", memes), ("_vocal", vocal),
                        ("_backing", backing), ("", mix)):
        sf.write(out.with_name(out.name + suffix + ".wav"), (sig * master).astype(np.float32), FS)

    def excerpt(src, name):
        info = sf.info(src)
        y, sr = sf.read(src, start=int(t0 * info.samplerate),
                        stop=int(min(t0 + args.dur_limit + 1.5, info.duration) * info.samplerate),
                        always_2d=True, dtype="float32")
        y = y.mean(axis=1)
        if sr != FS:
            y = librosa.resample(y, orig_sr=sr, target_sr=FS)
        sf.write(out.with_name(out.name + name), (y / (np.max(np.abs(y)) + 1e-9) * 0.85).astype(np.float32), FS)
    excerpt(args.vocal_stem, "_orig_vocal_excerpt.wav")
    excerpt(args.mix, "_orig_mix_excerpt.wav")

    # references: raw (unmodified) nuclei in order, and raw meme lines in order
    gap = np.zeros(int(0.08 * FS), dtype=np.float32)
    parts = []
    for b in raw_nuclei:
        y = clip_audio(b["path"])
        parts += [y[int(b["t0"] * FS):int(b["t1"] * FS)].astype(np.float32), gap]
    if parts:
        sf.write(out.with_name(out.name + "_nuclei_reference.wav"), np.concatenate(parts) * 0.8, FS)
    ref_dir = out.with_name(out.name + "_refs")
    ref_dir.mkdir(exist_ok=True)
    parts = []
    for k, c in enumerate([c for c in cues if c["layer"] == "meme"]):
        y, _ = hires.load(_lib_by_path[next(p for p in _lib_by_path if Path(p).name == c["clip"])], FS)
        sf.write(ref_dir / f"{k:02d}_{c['char']}_{c['clip'][:24]}.wav", y.astype(np.float32), FS)
        parts += [y.astype(np.float32), gap]
    if parts:
        sf.write(out.with_name(out.name + "_reference.wav"), np.concatenate(parts) * 0.8, FS)
    json.dump(cues, open(out.with_suffix(".cues.json"), "w"), ensure_ascii=False, indent=1)

    # metrics: CREPE (viterbi) on the melody layer vs the target contour
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    y16 = librosa.resample(melody.astype(np.float32), orig_sr=FS, target_sr=16000)
    x = torch.from_numpy(y16).float().unsqueeze(0).to(dev)
    fr, pr = torchcrepe.predict(x, 16000, 160, 80.0, 900.0, model="full",
                                decoder=torchcrepe.decode.viterbi, return_periodicity=True,
                                device=dev, batch_size=512, pad=False)
    fr, pr = fr.squeeze(0).cpu().numpy(), pr.squeeze(0).cpu().numpy()
    tr = np.arange(len(fr)) * 0.01 + t0
    tgt = np.interp(tr, grid, np.where(vg, stg, np.nan))
    tv = np.interp(tr, grid, vg.astype(float)) > 0.5
    rv = pr > 0.5
    both = tv & rv & np.isfinite(tgt)
    dev_c = np.abs((69 + 12 * np.log2(np.maximum(fr[both], 1e-3) / 440.0)) - tgt[both]) * 100
    seg = np.abs(vocal[:int(args.dur_limit * FS)])
    win = 3200
    duty = float(np.mean([np.max(seg[i:i + win]) > 3e-3 for i in range(0, len(seg) - win, win)]))
    shifts = np.array([c["shift"] for c in cues if c["layer"] == "melody"])
    m_ = {
        "version": "v3", "target": "original recording melody (Melodia) + layered sampler",
        "orig_window": [round(t0, 2), round(t1, 2)], "transpose": args.transpose,
        "n_notes": len(notes), "n_phrases": n_ph, "n_memes": sum(c["layer"] == "meme" for c in cues),
        "note_dur_median": round(float(np.median([n["t1"] - n["t0"] for n in notes])), 3),
        "melody_coverage_of_target": round(float(np.mean(rv[tv])) if tv.any() else 0.0, 3),
        "melody_median_abs_cents": round(float(np.median(dev_c)), 1),
        "melody_acc_50c": round(float(np.mean(dev_c < 50)), 3),
        "melody_acc_100c": round(float(np.mean(dev_c < 100)), 3),
        "render_duty_cycle": round(duty, 3),
        "pitch_shift_semi_median_abs": round(float(np.median(np.abs(shifts))), 2),
        "pitch_shift_semi_p90_abs": round(float(np.percentile(np.abs(shifts), 90)), 2),
        "time_stretch": "none for nuclei<=note (crop); ping-pong sustain loop for longer notes; memes untouched",
        "backing_gain_db": args.backing_gain_db, "meme_gain_db": args.meme_gain_db,
    }
    json.dump(m_, open(out.with_suffix(".metrics.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(m_, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
