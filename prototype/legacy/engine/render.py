#!/usr/bin/env python3
"""Kichiku render engine: MIDI melody -> voice-sample mapping -> pitched render.

Versions:
  v1 baseline: nearest-pitch sample, phase-vocoder shift, truncate to fit, velocity gain.
  v2 musical:  phrase-consistent character, stability/velocity-aware selection,
               time-stretch fit with legato, per-sample loudness normalize.
  v3 rich:     anchor-sample key-range reuse ("one sample covers a phrase"),
               chord layers from accompaniment tracks (one character each).

Usage:
  python3 render.py --midi midis/il_vento_doro.mid --track 0 --transpose -12 \
      --lib ../lib/library_genshin.json --version v2 --out ../out/v2_x.wav
"""
import argparse
import json
import math
import random
import sys
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import soundfile as sf

SR = 16000
SEED = 42


# ---------------------------------------------------------------- library
class Library:
    def __init__(self, paths, palette=None):
        self.samples = []
        for p in paths:
            self.samples += json.load(open(p))
        self.samples = [s for s in self.samples
                        if s.get("f0_semi") is not None
                        and 45 <= s["f0_semi"] <= 95
                        and s.get("voiced_ratio", 0) > 0.3
                        and s.get("dur", 0) >= 0.22]
        if palette:  # e.g. "genshin" or "anime"
            self.samples = [s for s in self.samples if s["work"].startswith(palette)]
        if not self.samples:
            raise SystemExit("empty library after filtering")
        by_char = {}
        for s in self.samples:
            by_char.setdefault(s.get("char", "?"), []).append(s)
        self.by_char = by_char
        print(f"library: {len(self.samples)} samples, {len(by_char)} characters", flush=True)

    @staticmethod
    @lru_cache(maxsize=6000)
    def audio(path):
        y, _ = librosa.load(path, sr=SR, mono=True)
        return y


# ---------------------------------------------------------------- midi
def pick_track(pm, idx):
    if idx is not None and idx >= 0:
        return pm.instruments[idx]
    best, best_score = None, -1
    for ins in pm.instruments:
        if ins.is_drum or not ins.notes:
            continue
        pitches = np.array([n.pitch for n in ins.notes])
        in_range = float(np.mean((pitches >= 55) & (pitches <= 88)))
        total = sum(n.end - n.start for n in ins.notes)
        score = in_range * 2 + min(total, 600) / 600
        if "melody" in (ins.name or "").lower():
            score += 3
        if score > best_score:
            best, best_score = ins, score
    return best


def get_notes(ins, transpose=0):
    notes = sorted(((n.start, n.end, n.pitch + transpose, n.velocity) for n in ins.notes),
                   key=lambda x: x[0])
    # collapse exact-duplicate pitches at same time (keep strongest)
    out = []
    for n in notes:
        if out and abs(n[0] - out[-1][0]) < 0.012 and n[2] == out[-1][2]:
            if n[3] > out[-1][3]:
                out[-1] = n
            continue
        out.append(n)
    return out


def phrases(notes, gap=0.32, jump=14):
    """Split note list into phrases at rests / big leaps."""
    ph, cur = [], []
    for i, n in enumerate(notes):
        if cur:
            prev = cur[-1]
            if n[0] - prev[1] > gap or abs(n[2] - prev[2]) > jump:
                ph.append(cur)
                cur = []
        cur.append(n)
    if cur:
        ph.append(cur)
    return ph


# ---------------------------------------------------------------- selection
def candidates_near(lib, target, max_shift=9):
    c = [s for s in lib.samples if abs(round(s["f0_semi"]) - target) <= max_shift]
    return c or lib.samples


def sel_v1(lib, note, rng, ctx):
    start, end, pitch, vel = note
    cands = candidates_near(lib, pitch)
    def cost(s):
        sh = pitch - s["f0_semi"]
        return abs(sh) + 4 * max(0, abs(sh) - 7) + rng.random() * 0.5
    return min(cands, key=cost)


def sel_v2(lib, note, rng, ctx):
    start, end, pitch, vel = note
    dur = end - start
    cands = candidates_near(lib, pitch)
    target_rms = -38 + 24 * (vel / 127.0)          # -38..-14 dB by velocity
    long_note = dur > 0.75
    def cost(s):
        sh = pitch - s["f0_semi"]
        c = abs(sh) + 4 * max(0, abs(sh) - 7) + 2 * max(0, abs(sh) - 12)
        if long_note:
            c += 3.0 * max(0, 0.45 - s["dur"])     # need body for long notes
            c += s["f0_iqr_cents"] / 140.0          # prefer stable pitch
        c += abs(s["rms_db"] - target_rms) / 7.0    # loudness matches dynamics
        if vel >= 100:
            c -= 1.2 * (s["f0_iqr_cents"] > 120)    # expressive for accents
        if ctx.get("char") and s.get("char") == ctx["char"]:
            c -= 4.0                                # phrase consistency
        if ctx.get("need_core"):
            if s.get("core_f0_semi") is None:
                c += 8.0                            # strong preference for cored samples
            else:
                c += min(s.get("core_iqr_cents", 999), 400) / 400.0  # flatter core = better
        return c + rng.random() * 0.3
    return min(cands, key=cost)


def make_sel_v3(lib):
    """Anchor-based: per phrase pick one anchor sample; its key-range covers notes."""
    def sel(note, rng, ctx):
        pitch = note[2]
        anchors = ctx.get("anchors")
        if anchors:
            for a in anchors:
                if abs(pitch - a["f0_semi"]) <= 7.5:
                    return a
        s = sel_v2(lib, note, rng, ctx)
        if anchors is not None and s not in anchors:
            anchors.append(s)
        return s
    return sel


# ---------------------------------------------------------------- dsp
@lru_cache(maxsize=8000)
def shifted(path, shift_semi, mode):
    y = Library.audio(path)
    if mode == "rs":   # classic resample: pitch up = fewer samples at same SR
        rate = 2.0 ** (shift_semi / 12.0)
        if rate == 1.0:
            return y
        return librosa.resample(y, orig_sr=SR, target_sr=max(1000, int(round(SR / rate))))
    if abs(shift_semi) < 0.05:
        return y
    return librosa.effects.pitch_shift(y, sr=SR, n_steps=shift_semi)


def fit_length(y, note_dur, version, rng):
    n = len(y)
    target = int(note_dur * SR)
    if version == "v1":
        allow = int(target * 1.25)
        if n > allow:
            y = y[:allow]
    else:
        hi = int(target * 1.3)
        if n > hi and n > 0:
            y = librosa.effects.time_stretch(y, rate=n / (target * 1.12))
        elif n < int(target * 0.6) and target > int(0.45 * SR):
            y = librosa.effects.time_stretch(y, rate=max(0.35, n / (target * 0.92)))
    return y


def envelope(y, vel, version):
    n = len(y)
    if n == 0:
        return y
    rms = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
    if version != "v1":                       # normalize then apply velocity
        y = y * (10 ** (-21 / 20) / rms)
        g = (vel / 127.0) ** 1.4
    else:
        g = (vel / 127.0) ** 1.5 * min(1.6, (10 ** (-21 / 20)) / rms)
    y = y * g
    a = min(int(0.004 * SR), n // 2)
    r = min(int(0.025 * SR), n // 2)
    if a:
        y[:a] *= np.linspace(0, 1, a)
    if r:
        y[-r:] *= np.linspace(1, 0, r)
    return y


# ---------------------------------------------------------------- core mode
@lru_cache(maxsize=6000)
def shifted_core(path, t0, t1, shift):
    """Pitch-shift only the stable vowel core; float shift, no grid rounding."""
    y = Library.audio(path)
    i0, i1 = int(t0 * SR), int(t1 * SR)
    core = y[i0:i1]
    if len(core) < int(0.05 * SR):
        return None
    if abs(shift) < 0.01:
        return core
    return librosa.effects.pitch_shift(core, sr=SR, n_steps=shift)


def loop_to(y, target_len):
    """Sustain a short core to target length by crossfaded tiling."""
    if len(y) >= target_len:
        return y[:target_len]
    xf = min(int(0.02 * SR), max(len(y) // 4, 1))
    out = y.copy()
    while len(out) < target_len:
        head = out[:-xf] if xf and len(out) > xf else out[:0]
        fade = (out[-xf:] * np.linspace(1, 0, xf) + y[:xf] * np.linspace(0, 1, xf)) \
            if xf and len(out) > xf else y[:xf]
        out = np.concatenate([head, fade, y[xf:]])
    return out[:target_len]


# ---------------------------------------------------------------- render
def render_track(buf, notes, lib, version, rng, shift_mode, gain=1.0,
                 char_lock=None, anchor_mode=False):
    use_core = version in ("v4", "v5", "v6")
    anchor = version in ("v3", "v5") or anchor_mode
    sel = make_sel_v3(lib) if anchor else (sel_v2 if version != "v1" else sel_v1)
    for ph in phrases(notes):
        ctx = {"char": char_lock, "anchors": [] if anchor else None,
               "need_core": use_core}
        for note in ph:
            start, end, pitch, vel = note
            accent = (version == "v6" and vel >= 100 and (end - start) >= 0.22)
            ctx["need_core"] = use_core and not accent
            s = sel(lib, note, rng, ctx) if not anchor else sel(note, rng, ctx)
            if version != "v1" and not char_lock:
                ctx["char"] = s.get("char")
            y = None
            if use_core and not accent and s.get("core_f0_semi") is not None:
                shift = pitch - s["core_f0_semi"]
                if -24 <= shift <= 24:
                    yc = shifted_core(s["path"], s["core_t0"], s["core_t1"],
                                      round(shift * 4) / 4)
                    if yc is not None:
                        y = loop_to(yc, int((end - start) * SR * 1.02))
            if y is None:
                shift = pitch - s["f0_semi"]
                shift = round(shift * 2) / 2                 # half-semitone grid
                y = shifted(s["path"], shift, shift_mode)
                y = fit_length(y, end - start, "v2" if use_core else version, rng)
            y = envelope(y, vel, "v2" if version in ("v4", "v5", "v6") else version) * gain
            i0 = int(start * SR)
            avail = len(buf) - i0
            if avail <= 0:
                continue
            if len(y) > avail:
                y = y[:avail]
            buf[i0:i0 + len(y)] += y
    return buf


# ---------------------------------------------------------------- metrics
def write_metrics(path, args, notes, buf, melody_ins, pm):
    import torch
    import torchcrepe
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    m = {"song": Path(args.midi).name, "version": args.version,
         "palette": args.palette, "shift_mode": args.shift_mode,
         "n_notes": len(notes)}
    # expected f0 per 10ms frame from melody notes
    n_frames = 1 + len(buf) // 160
    exp = np.zeros(n_frames, dtype=np.float32)
    for s, e, p, v in notes:
        i0, i1 = int(s * 100), min(int(e * 100), n_frames)
        exp[max(i0, 0):max(i1, 0)] = 440.0 * 2 ** ((p - 69) / 12.0)
    # measured f0 (chunked to bound memory)
    f0_all, per_all = [], []
    CH = int(30 * SR)
    for c0 in range(0, len(buf) - 1600, CH):
        seg = torch.tensor(buf[c0:c0 + CH]).unsqueeze(0).to(device)
        f0, per = torchcrepe.predict(seg, SR, model="full", device=device,
                                     return_periodicity=True,
                                     decoder=torchcrepe.decode.argmax, batch_size=1)
        f0_all.append(f0.squeeze(0).cpu().numpy())
        per_all.append(per.squeeze(0).cpu().numpy())
    f0 = np.concatenate(f0_all)
    per = np.concatenate(per_all)
    nf = min(len(f0), len(exp))
    voiced = (per[:nf] > 0.5) & (exp[:nf] > 0) & (f0[:nf] > 50)
    if voiced.sum() > 30:
        cents = np.abs(1200 * np.log2(f0[:nf][voiced] / exp[:nf][voiced]))
        m["pitch_acc_50c"] = round(float(np.mean(cents < 50)), 3)
        m["pitch_acc_100c"] = round(float(np.mean(cents < 100)), 3)
        m["median_abs_cents"] = round(float(np.median(cents)), 1)
        m["voiced_frame_ratio"] = round(float(voiced.sum() / max((exp[:nf] > 0).sum(), 1)), 3)
    # coverage: energy present near each note onset
    win = np.abs(buf)
    cov = 0
    for s, e, p, v in notes:
        i0 = max(int((s - 0.03) * SR), 0)
        i1 = min(int((s + 0.12) * SR), len(win))
        if i1 > i0 and win[i0:i1].max() > 0.02:
            cov += 1
    m["onset_coverage"] = round(cov / max(len(notes), 1), 3)
    # dynamics correlation: per-note measured RMS vs velocity
    rms_db, vels = [], []
    for s, e, p, v in notes:
        i0, i1 = int(s * SR), min(int(e * SR), len(buf))
        if i1 - i0 > int(0.05 * SR):
            seg = buf[i0:i1]
            rms_db.append(20 * math.log10(float(np.sqrt(np.mean(seg ** 2))) + 1e-9))
            vels.append(v)
    if len(vels) > 20 and np.std(vels) > 1e-6 and np.std(rms_db) > 1e-6:
        m["dyn_corr"] = round(float(np.corrcoef(rms_db, vels)[0, 1]), 3)
    else:
        m["dyn_corr"] = None  # constant velocity in this MIDI
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(m, open(path, "w"), ensure_ascii=False, indent=1)
    print("metrics:", json.dumps(m, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi", required=True)
    ap.add_argument("--track", type=int, default=-1)
    ap.add_argument("--transpose", type=int, default=0)
    ap.add_argument("--fold-range", default=None,
                    help="fold pitches into [lo,hi] by octaves, e.g. 52-83")
    ap.add_argument("--lib", nargs="+", required=True)
    ap.add_argument("--palette", default=None)
    ap.add_argument("--version", default="v1", choices=["v1", "v2", "v3", "v4", "v5", "v6"])
    ap.add_argument("--shift-mode", default="pv", choices=["pv", "rs"])
    ap.add_argument("--chords", type=int, default=0, help="v3: number of backing tracks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dur-limit", type=float, default=None,
                    help="only render notes starting before this second")
    ap.add_argument("--metrics", default=None)
    args = ap.parse_args()

    rng = random.Random(SEED)
    np.random.seed(SEED)
    lib = Library(args.lib, args.palette)
    pm = pretty_midi.PrettyMIDI(args.midi)
    melody_ins = pick_track(pm, args.track)
    notes = get_notes(melody_ins, args.transpose)
    if args.fold_range:
        lo, hi = (int(x) for x in args.fold_range.split("-"))
        folded = []
        for s, e, p, v in notes:
            while p > hi:
                p -= 12
            while p < lo:
                p += 12
            folded.append((s, e, p, v))
        notes = folded
    dur = pm.get_end_time() + 8.0
    if args.dur_limit:
        notes = [n for n in notes if n[0] < args.dur_limit]
        dur = min(dur, args.dur_limit + 8.0)
    print(f"melody track: {melody_ins.name!r} notes={len(notes)} "
          f"pitch {min(n[2] for n in notes)}-{max(n[2] for n in notes)} song {dur:.0f}s",
          flush=True)

    buf = np.zeros(int(dur * SR), dtype=np.float32)
    render_track(buf, notes, lib, args.version, rng, args.shift_mode)

    if args.chords and args.version in ("v3", "v5", "v6"):
        used = {melody_ins.program}
        backs = [i for i in pm.instruments
                 if not i.is_drum and i.notes and i is not melody_ins]
        backs.sort(key=lambda i: -sum(n.end - n.start for n in i.notes))
        chars = sorted(lib.by_char, key=lambda c: -len(lib.by_char[c]))
        for k, ins in enumerate(backs[:args.chords]):
            bn = get_notes(ins, args.transpose)
            if args.dur_limit:
                bn = [n for n in bn if n[0] < args.dur_limit]
            bn = [(s, e, max(48, min(84, p)), v) for s, e, p, v in bn]  # fold to vocal range
            char = chars[(k + 1) % len(chars)] if len(chars) > 1 else None
            sub = Library.__new__(Library)
            sub.samples = [s for s in lib.samples if not char or s.get("char") == char]
            sub.by_char = lib.by_char
            print(f"chord layer {k}: {ins.name!r} {len(bn)} notes char={char}", flush=True)
            render_track(buf, bn, sub, args.version, rng, args.shift_mode,
                         gain=0.34, char_lock=char, anchor_mode=True)

    # master: gentle normalize
    peak = float(np.max(np.abs(buf)) + 1e-9)
    buf = buf / peak * 0.95
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, buf, SR)
    print(f"written {out} ({len(buf)/SR:.0f}s, peak {peak:.2f})", flush=True)

    if args.metrics:
        write_metrics(args.metrics, args, notes, buf, melody_ins, pm)


if __name__ == "__main__":
    main()
