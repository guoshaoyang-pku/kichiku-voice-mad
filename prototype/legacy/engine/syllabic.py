#!/usr/bin/env python3
"""v7 syllabic engine: every clip plays in full; its syllables land on melody notes.

Pipeline per song:
  MIDI melody (skyline, octave-centred) -> phrases
  -> DP chunking: each chunk = k consecutive notes covered by ONE clip whose
     syllable count / duration / register fit (one clip covers several notes)
  -> WORLD vocoder per clip: piecewise time-warp so syllable onsets hit note
     onsets (consonant lead), with selectable melody or source F0; spectral
     envelope kept (words stay intelligible); nothing is truncated
  -> macro dynamics: smoothed phrase-level velocity curve on the vocal bus;
     clip-internal envelopes untouched (only 6 ms click guards)
  -> optional vowel pad layers from backing tracks (quiet, one character each)

Outputs: <out>.wav (32 kHz), <out>.cues.json, <out>.srt, <out>.metrics.json,
         <out>_asr/ (sampled original/rendered clip pairs for intelligibility eval)
"""
import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import librosa
import numpy as np
import pretty_midi
import pyworld as pw
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))
import render as R  # noqa: E402
import hires  # noqa: E402

FS = 32000
FP = 2.5                       # WORLD frame period (ms), 400 frames/s
LEAD = 0.035                  # consonant lead before note onset (s)
KEEP_MICRO = 0.15             # retain only a small, smoothed trace of source intonation
WARP_LO, WARP_HI = 0.6, 1.7   # allowed local time-stretch per segment
MAX_K = 8
RANGE_DB = 12.0               # macro dynamic range mapped from song velocity spread
EXCLUDE_CHARS = {"混合", "未确认", "未知", "?"}
SEED = 7


# ---------------------------------------------------------------- syllables
def detect_syllables(path):
    y, sr = librosa.load(path, sr=16000, mono=True)
    hop = 160
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    on = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=hop,
                                    backtrack=True, units="time", wait=9, delta=0.07)
    rms = librosa.feature.rms(y=y, frame_length=400, hop_length=hop)[0]
    thr = rms.max() * 0.08
    keep = []
    for t in sorted(set([0.0] + [float(x) for x in on])):
        i = min(int(t * 100) + 3, len(rms) - 1)
        if keep and t - keep[-1] < 0.09:
            continue
        if t == 0.0 or rms[i] > thr:
            keep.append(round(t, 3))
    return keep


def load_candidates(lib_paths, palette, works):
    items = []
    for p in lib_paths:
        items += json.load(open(p))
    cache_p = HERE.parent / "lib" / "syllables_cache.json"
    cache = json.load(open(cache_p)) if cache_p.exists() else {}
    out = []
    for s in items:
        if s.get("f0_semi") is None or s.get("voiced_ratio", 0) < 0.35:
            continue
        if not (0.25 <= s.get("dur", 0) <= 4.0) or not (44 <= s["f0_semi"] <= 86):
            continue
        if s.get("char") in EXCLUDE_CHARS:
            continue
        if palette and not s["work"].startswith(palette):
            continue
        if works and s["work"].split(":")[-1] not in works:
            continue
        out.append(s)
    todo = [s for s in out if s["path"] not in cache]
    if todo:
        print(f"detecting syllables for {len(todo)} clips ...", flush=True)
        for n, s in enumerate(todo):
            cache[s["path"]] = detect_syllables(s["path"])
            if (n + 1) % 1000 == 0:
                print(f"  {n+1}/{len(todo)}", flush=True)
                json.dump(cache, open(cache_p, "w"))
        json.dump(cache, open(cache_p, "w"))
    for s in out:
        s["syl"] = cache[s["path"]]
    out = [s for s in out if 1 <= len(s["syl"]) <= 12]
    print(f"candidates: {len(out)} clips, {len({s['char'] for s in out})} characters", flush=True)
    return out


# ---------------------------------------------------------------- melody
def melody_notes(pm, track, lib_median):
    ins = R.pick_track(pm, track)
    raw = sorted(((n.start, n.end, n.pitch, n.velocity) for n in ins.notes), key=lambda x: (x[0], -x[2]))
    sky = []
    for n in raw:                                   # skyline: keep top voice
        if sky and n[0] - sky[-1][0] < 0.03:
            if n[2] > sky[-1][2]:
                sky[-1] = n
            continue
        sky.append(n)
    notes = []
    for i, n in enumerate(sky):
        end = n[1] if i + 1 == len(sky) else min(n[1], sky[i + 1][0])
        if end - n[0] > 0.03:
            notes.append((n[0], end, n[2], n[3]))
    med = float(np.median([n[2] for n in notes]))
    octs = round((lib_median - med) / 12.0)
    lo, hi = lib_median - 10, lib_median + 12          # comfortable speaking-voice register
    out = []
    for s, e, p, v in notes:
        p = p + 12 * octs
        while p < lo:
            p += 12
        while p > hi:
            p -= 12
        out.append((s, e, p, v))
    return ins, out, 12 * octs


# ---------------------------------------------------------------- chunking
class Selector:
    def __init__(self, cands, rng):
        self.c = cands
        self.rng = rng
        self.f0 = np.array([s["f0_semi"] for s in cands])
        self.dur = np.array([s["dur"] for s in cands])
        self.m = np.array([len(s["syl"]) for s in cands], dtype=float)
        self.vr = np.array([s["voiced_ratio"] for s in cands])
        self.chars = sorted({s["char"] for s in cands})
        self.char_idx = {ch: np.array([i for i, s in enumerate(cands) if s["char"] == ch])
                         for ch in self.chars}
        self.used = np.zeros(len(cands))

    def chunk_cost(self, idx, notes, nxt_start):
        k = len(notes)
        span_end = nxt_start if nxt_start is not None else notes[-1][1] + 0.12
        D = max(span_end - notes[0][0], 0.12)
        w = np.array([n[1] - n[0] for n in notes]) + 1e-3
        P = float(np.sum(w * np.array([n[2] for n in notes])) / w.sum())
        m, dur, f0 = self.m[idx], self.dur[idx], self.f0[idx]
        dp = np.abs(P - f0)
        cost = (0.6 * np.maximum(0, k - m) + 1.0 * np.maximum(0, m - k)
                + 5.0 * np.abs(np.log2(D / dur))
                + 0.35 * np.maximum(0, dp - 3) + 0.8 * np.maximum(0, dp - 8)
                + 1.5 * (1 - self.vr[idx]) + 3.5 * self.used[idx]
                + 0.2 * self.rng.random())
        top = np.argsort(cost)[:40]
        best_c, best_j = np.inf, int(top[0])
        for t in top:
            s = self.c[idx[t]]
            _, _, alpha, dist = build_warp(s["syl"], s["dur"], notes, nxt_start)
            c = cost[t] + 6.0 * dist + 2.0 * (1.0 - alpha)
            if c < best_c:
                best_c, best_j = c, int(t)
        return float(best_c) + 1.2, int(idx[best_j])

    def plan_phrase(self, ph, nxt_phrase_start, prev_char):
        best = None
        P = float(np.median([n[2] for n in ph]))
        fit = sorted(self.chars, key=lambda ch: abs(np.median(self.f0[self.char_idx[ch]]) - P))
        for ch in fit[:6]:
            idx = self.char_idx[ch]
            if len(idx) < 25:
                continue
            n = len(ph)
            dp = [0.0] + [np.inf] * n
            back = [None] * (n + 1)
            for i in range(1, n + 1):
                for k in range(1, min(MAX_K, i) + 1):
                    seg = ph[i - k:i]
                    nxt = ph[i][0] if i < n else nxt_phrase_start
                    c, clip = self.chunk_cost(idx, seg, nxt)
                    if dp[i - k] + c < dp[i]:
                        dp[i], back[i] = dp[i - k] + c, (i - k, clip)
            total = dp[n] / max(n, 1) + (0.8 if ch == prev_char else 0.0)
            if best is None or total < best[0]:
                best = (total, ch, back)
        _, ch, back = best
        chunks, i = [], len(ph)
        while i > 0:
            j, clip = back[i]
            chunks.append((j, i, clip))
            i = j
        chunks.reverse()
        for _, _, clip in chunks:
            self.used[clip] += 1
        return ch, chunks


# ---------------------------------------------------------------- WORLD render
def world_analyze(y):
    f0, t = pw.dio(y, FS, f0_floor=65.0, f0_ceil=1100.0, frame_period=FP)
    f0 = pw.stonemask(y, f0, t, FS)
    sp = pw.cheaptrick(y, f0, t, FS)
    ap = pw.d4c(y, f0, t, FS)
    return f0, sp, ap


def smooth(x, n):
    if n <= 1:
        return x
    k = np.ones(n) / n
    return np.convolve(np.pad(x, (n // 2, n - 1 - n // 2), mode="edge"), k, mode="valid")


def build_warp(syl, clip_len, notes, nxt_start):
    """Anchors orig->out. Syllables pulled toward note onsets only as far as the
    local stretch stays within [WARP_LO, WARP_HI]; otherwise blend toward uniform."""
    k, m = len(notes), len(syl)
    a = min(k, m)
    si = np.unique(np.round(np.linspace(0, m - 1, a)).astype(int))
    ni = np.unique(np.round(np.linspace(0, k - 1, len(si))).astype(int))
    a = min(len(si), len(ni))
    orig = np.array([syl[i] for i in si[:a]])
    tgt = np.array([notes[j][0] - LEAD for j in ni[:a]])
    target_end = (nxt_start if nxt_start is not None else notes[-1][1] + 0.15) + 0.04
    start_out = tgt[0]
    S = np.clip((target_end - start_out) / max(clip_len - orig[0], 1e-3), 0.6, 1.6)
    uni = start_out + (orig - orig[0]) * S
    end_uni = start_out + (clip_len - orig[0]) * S
    best = None
    for alpha in (1.0, 0.8, 0.6, 0.4, 0.2, 0.0):
        out = uni + alpha * (tgt - uni)
        tail_o = clip_len - orig[-1]
        tail_out = float(np.clip(end_uni - out[-1], WARP_LO * tail_o, WARP_HI * tail_o))
        orig_a = np.concatenate([[0.0], orig, [clip_len]])
        out_a = np.concatenate([[out[0] - orig[0]], out, [out[-1] + tail_out]])
        d_o, d_n = np.diff(orig_a), np.diff(out_a)
        mask = d_o > 0.02
        r = d_n[mask] / d_o[mask]
        if np.all(d_n > 0) and (len(r) == 0 or (r.min() >= WARP_LO and r.max() <= WARP_HI)):
            best = (orig_a, out_a, alpha, r)
            break
    if best is None:
        orig_a = np.array([0.0, clip_len])
        out_a = np.array([start_out - orig[0] * S, start_out + (clip_len - orig[0]) * S])
        best = (orig_a, out_a, 0.0, np.array([S]))
    orig_a, out_a, alpha, r = best
    distortion = float(np.mean(np.abs(np.log2(r)))) if len(r) else 0.0
    return orig_a, out_a, alpha, distortion


def render_chunk(sample, notes, nxt_start, prev_end, f0_mode="melody"):
    y, is_hi = hires.load(sample, FS)
    pre = hires.PRE_PAD if is_hi else 0.0
    f0, sp, ap = world_analyze(y)
    clip_len = len(y) / FS
    syl = [pre + s for s in sample["syl"]]
    k, m = len(notes), len(syl)
    orig_a, out_a, alpha, distortion = build_warp(syl, clip_len, notes, nxt_start)
    t0, t1 = out_a[0], out_a[-1]
    n_out = max(int((t1 - t0) * 1000 / FP), 4)
    t_abs = t0 + np.arange(n_out) * FP / 1000.0
    t_orig = np.interp(t_abs, out_a, orig_a)
    fi_float = np.clip(t_orig * 1000 / FP, 0, len(f0) - 1)
    fi = np.floor(fi_float).astype(int)
    fj = np.minimum(fi + 1, len(f0) - 1)
    fw = (fi_float - fi)[:, None]
    sp_out = sp[fi] * (1.0 - fw) + sp[fj] * fw
    ap_out = ap[fi] * (1.0 - fw) + ap[fj] * fw
    f0_source = f0.copy()
    voiced_source = f0_source > 0
    max_gap = max(1, int(round(10.0 / FP)))
    changes = np.diff(np.r_[True, voiced_source, True].astype(np.int8))
    starts = np.flatnonzero(changes == -1)
    ends = np.flatnonzero(changes == 1)
    for gap_start, gap_end in zip(starts, ends):
        if gap_start > 0 and gap_end < len(f0_source) and gap_end - gap_start <= max_gap:
            f0_source[gap_start:gap_end] = np.linspace(
                f0_source[gap_start - 1], f0_source[gap_end], gap_end - gap_start + 2
            )[1:-1]
    vuv = f0_source[fi] > 0
    onsets = np.array([n[0] for n in notes])
    j = np.clip(np.searchsorted(onsets, t_abs + LEAD * 0.5, side="right") - 1, 0, k - 1)
    semi = smooth(np.array([notes[x][2] for x in j], dtype=float), max(1, int(round(40.0 / FP))))
    nat = np.where(f0_source > 0, 12 * np.log2(np.maximum(f0_source, 1e-6) / 440.0) + 69, np.nan)
    nat_f = nat[fi]
    if f0_mode == "source":
        f0_out = np.where(vuv, f0_source[fi], 0.0)
    else:
        if np.isfinite(nat_f).sum() > 8:
            filled = np.where(np.isfinite(nat_f), nat_f, np.nanmedian(nat_f))
            micro = np.clip(filled - smooth(filled, max(1, int(round(120.0 / FP)))), -1.0, 1.0)
            semi = semi + KEEP_MICRO * np.where(vuv, micro, 0.0)
        f0_out = np.where(vuv, 440.0 * 2 ** ((semi - 69) / 12.0), 0.0)
    yo = pw.synthesize(np.ascontiguousarray(f0_out), np.ascontiguousarray(sp_out),
                       np.ascontiguousarray(ap_out), FS, FP)
    # per-clip loudness (keep internal envelope)
    fr = librosa.feature.rms(y=yo, frame_length=1024, hop_length=256)[0]
    loud = np.sqrt(np.mean(np.sort(fr)[len(fr) // 2:] ** 2)) + 1e-9
    yo = yo * (10 ** (-18 / 20) / loud)
    g = int(0.006 * FS)
    if len(yo) > 2 * g:
        yo[:g] *= np.linspace(0, 1, g)
        yo[-g:] *= np.linspace(1, 0, g)
    stretch = (t1 - t0) / clip_len
    return yo.astype(np.float32), t0, {"stretch": stretch, "syl": m, "notes": k,
                                       "hires": is_hi, "dur_out": t1 - t0,
                                       "align": alpha, "warp_dist": distortion,
                                       "f0_mode": f0_mode,
                                       "pitches": [int(n[2]) for n in notes]}


# ---------------------------------------------------------------- pad
def make_pad_voice(sample):
    y, _ = hires.load(sample, FS)
    f0, sp, ap = world_analyze(y)
    v = f0 > 0
    semis = np.where(v, 12 * np.log2(np.maximum(f0, 1e-6) / 440) + 69, np.nan)
    W = 40
    best, bi = np.inf, None
    for i in range(0, len(f0) - W):
        seg = semis[i:i + W]
        if np.isnan(seg).any():
            continue
        var = np.var(seg)
        if var < best:
            best, bi = var, i
    if bi is None:
        return None
    return sp[bi:bi + W], ap[bi:bi + W]


def render_pad_note(voice, pitch, dur):
    sp_c, ap_c = voice
    n = max(int(dur * 1000 / FP), 6)
    W = len(sp_c)
    pp = np.concatenate([np.arange(W), np.arange(W - 2, 0, -1)])
    fi = pp[np.arange(n) % len(pp)]
    t = np.arange(n) * FP / 1000
    semi = pitch + 0.02 * np.sin(2 * np.pi * 0.7 * t)
    f0 = 440.0 * 2 ** ((semi - 69) / 12.0)
    y = pw.synthesize(f0, np.ascontiguousarray(sp_c[fi]), np.ascontiguousarray(ap_c[fi]), FS, FP)
    a = min(int(0.006 * FS), len(y) // 4)
    r = min(int(0.006 * FS), len(y) // 4)
    y[:a] *= np.linspace(0, 1, a)
    y[-r:] *= np.linspace(1, 0, r)
    return y


# ---------------------------------------------------------------- dynamics
def macro_curve(notes, n_samples, win_s=1.5):
    n = n_samples // 320 + 2                       # 100 Hz control rate
    v = np.full(n, np.nan)
    for s, e, p, vel in notes:
        v[int(s * 100):int(e * 100) + 1] = vel
    last = np.nanmean(v) if np.isfinite(v).any() else 100
    for i in range(n):                             # hold through rests
        if np.isnan(v[i]):
            v[i] = last
        last = v[i]
    v = smooth(v, int(win_s * 100))
    vels = np.array([n[3] for n in notes], dtype=float)
    lo, hi = np.percentile(vels, 5), np.percentile(vels, 95)
    if hi - lo < 4:                                 # flat-velocity MIDI: keep level
        return np.ones(n_samples, dtype=np.float32)
    x = np.clip((v - lo) / (hi - lo), 0, 1)
    g = 10 ** (-RANGE_DB * (1 - x) / 20)
    return np.interp(np.arange(n_samples) / FS * 100, np.arange(n), g).astype(np.float32)


def srt_time(t):
    h, rem = divmod(max(t, 0), 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}".replace(".", ",")


# ---------------------------------------------------------------- main
def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--midi", required=True)
    ap_.add_argument("--track", type=int, default=-1)
    ap_.add_argument("--lib", nargs="+", required=True)
    ap_.add_argument("--palette", default=None)
    ap_.add_argument("--works", default=None, help="comma list, e.g. mygo,ave_mujica")
    ap_.add_argument("--pad", type=int, default=0, help="number of backing pad layers; -1 renders all non-drum backing tracks")
    ap_.add_argument("--f0-mode", choices=("melody", "source"), default="melody",
                     help="use melody pitch or preserve the source speech F0 contour")
    ap_.add_argument("--backing-gain-db", type=float, default=-10.0,
                     help="backing level relative to its unattenuated render, in dB")
    ap_.add_argument("--dur-limit", type=float, default=None, help="window length (s)")
    ap_.add_argument("--start", type=float, default=0.0, help="window start (s) in the MIDI")
    ap_.add_argument("--asr-samples", type=int, default=30)
    ap_.add_argument("--out", required=True)
    args = ap_.parse_args()

    rng = random.Random(SEED)
    np.random.seed(SEED)
    works = set(args.works.split(",")) if args.works else None
    cands = load_candidates(args.lib, args.palette, works)
    srcs = [s["src"] for s in cands if s["work"].startswith("anime")]
    if srcs:
        hires.ensure_anime_maps(srcs)
    lib_med = float(np.median([s["f0_semi"] for s in cands]))

    pm = pretty_midi.PrettyMIDI(args.midi)
    ins, notes, transp = melody_notes(pm, args.track, lib_med)
    T0 = args.start
    T1 = T0 + args.dur_limit if args.dur_limit else pm.get_end_time()
    notes = [(s - T0, e - T0, p, v) for s, e, p, v in notes if T0 <= s < T1]
    song_end = (T1 - T0) + 4.0
    print(f"melody {ins.name!r}: {len(notes)} notes, transpose {transp:+d}, "
          f"pitch {min(n[2] for n in notes)}-{max(n[2] for n in notes)}, lib median {lib_med:.1f}",
          flush=True)

    sel = Selector(cands, rng)
    phs = R.phrases(notes)
    buf = np.zeros(int(song_end * FS) + FS * 4, dtype=np.float32)
    cues, stats = [], []
    asr_dir = Path(args.out + "_asr")
    plan = []
    prev_char = None
    for pi, ph in enumerate(phs):
        nxt = phs[pi + 1][0][0] if pi + 1 < len(phs) else None
        ch, chunks = sel.plan_phrase(ph, nxt, prev_char)
        prev_char = ch
        for j, i, clip in chunks:
            nn = ph[i][0] if i < len(ph) else nxt
            plan.append((ph[j:i], nn, clip))
    print(f"planned {len(plan)} chunks over {len(phs)} phrases "
          f"(avg {len(notes)/max(len(plan),1):.2f} notes/clip)", flush=True)
    asr_idx = set(rng.sample(range(len(plan)), min(args.asr_samples, len(plan))))
    if asr_idx:
        asr_dir.mkdir(parents=True, exist_ok=True)
    prev_end = 0.0
    for ci, (seg, nn, clip) in enumerate(plan):
        s = cands[clip]
        yo, t0, st = render_chunk(s, seg, nn, prev_end, args.f0_mode)
        i0 = int(round(t0 * FS))
        if i0 < 0:
            yo, i0 = yo[-i0:], 0
        buf[i0:i0 + len(yo)] += yo[:len(buf) - i0]
        prev_end = t0 + len(yo) / FS
        cue = {"t0": round(t0, 3), "t1": round(prev_end, 3), "char": s["char"],
               "work": s["work"], "text": s.get("text", ""), "clip": Path(s["path"]).name,
               "notes": [n[2] for n in seg], **{k: (round(v, 3) if isinstance(v, float) else v)
                                                  for k, v in st.items()}}
        cues.append(cue)
        stats.append(st)
        if ci in asr_idx:
            yh, _ = hires.load(s, FS)
            sf.write(asr_dir / f"{ci:04d}_orig.wav", yh.astype(np.float32), FS)
            sf.write(asr_dir / f"{ci:04d}_rend.wav", yo, FS)
        if (ci + 1) % 50 == 0:
            print(f"  rendered {ci+1}/{len(plan)}", flush=True)

    if asr_idx:
        json.dump([f"{i:04d}" for i in sorted(asr_idx)], open(asr_dir / "manifest.json", "w"))
    gain = macro_curve(notes, len(buf))
    vocal = buf * gain

    pad_bus = np.zeros_like(buf)
    pad_track_names = []
    if args.pad:
        backs = [i for i in pm.instruments if i.notes and i is not ins]
        backs.sort(key=lambda i: -sum(n.end - n.start for n in i.notes))
        back_layers = backs if args.pad < 0 else backs[:args.pad]
        used_chars = {c["char"] for c in cues}
        pool = sorted([s for s in cands if s["dur"] > 1.0 and s["voiced_ratio"] > 0.6],
                      key=lambda s: s.get("f0_iqr_cents", 999))
        for L, b in enumerate(back_layers):
            pick = next((s for s in pool if s["char"] not in used_chars), pool[L % len(pool)])
            used_chars.add(pick["char"])
            voice = make_pad_voice(pick)
            if voice is None:
                continue
            pad_track_names.append(b.name)
            cnt = 0
            for n in b.notes:
                if not (T0 <= n.start < T1):
                    continue
                p = n.pitch + transp
                while p > 76:
                    p -= 12
                while p < 50:
                    p += 12
                y = render_pad_note(voice, p, n.end - n.start)
                rr = librosa.feature.rms(y=y)[0].max() + 1e-9
                y = y * (10 ** (-31 / 20) / rr) * (n.velocity / 127.0)
                k0 = int((n.start - T0) * FS)
                pad_bus[k0:k0 + len(y)] += y[:len(pad_bus) - k0]
                cnt += 1
            print(f"pad layer {L}: {b.name!r} {cnt} notes, voice={pick['char']}", flush=True)
        pad_bus *= gain * (10 ** (args.backing_gain_db / 20.0))
    mix = vocal + pad_bus
    end = int(min(len(mix), (max(c["t1"] for c in cues) + 1.0) * FS))
    mix = mix[:end]
    master_gain = 0.93 / (np.max(np.abs(mix)) + 1e-9)
    mix *= master_gain
    vocal_out = vocal[:end] * master_gain
    backing_out = pad_bus[:end] * master_gain
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out.with_suffix(".wav"), mix, FS)
    if args.pad:
        sf.write(out.with_name(out.name + "_vocal.wav"), vocal_out, FS)
        sf.write(out.with_name(out.name + "_backing.wav"), backing_out, FS)
    json.dump(cues, open(out.with_suffix(".cues.json"), "w"), ensure_ascii=False, indent=0)
    with open(out.with_suffix(".srt"), "w") as f:
        for i, c in enumerate(cues):
            label = f"[{c['char']}] {c['text']}" if c["text"] else f"[{c['char']}]"
            f.write(f"{i+1}\n{srt_time(c['t0'])} --> {srt_time(c['t1'])}\n{label}\n\n")

    # metrics: reuse v1-v6 measurement on the vocal bus (16 kHz)
    v16 = librosa.resample(vocal[:end], orig_sr=FS, target_sr=16000)
    ns = SimpleNamespace(midi=args.midi, version="v7", palette=args.palette, shift_mode="world")
    # metrics need note times on the same clock as the rendered buffer (already shifted)
    mpath = str(out.with_suffix(".metrics.json"))
    R.write_metrics(mpath, ns, notes, v16, ins, pm)
    m = json.load(open(mpath))
    # macro dynamics: 1.5 s smoothed velocity vs 1.5 s smoothed vocal loudness, on note-active frames
    cr = 100
    n_fr = len(vocal[:end]) // (FS // cr)
    vel = np.full(n_fr, np.nan)
    for s_, e_, p_, v_ in notes:
        vel[int(s_ * cr):min(int(e_ * cr) + 1, n_fr)] = v_
    act = np.isfinite(vel)
    vel_f = smooth(np.where(act, vel, np.nanmean(vel)), 150)
    fr = vocal[:n_fr * (FS // cr)].reshape(n_fr, -1)
    ldb = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-6)
    sounding = act & (ldb > ldb.max() - 30)
    lin = np.where(sounding, 10 ** (ldb / 20), np.nan)
    num = smooth(np.nan_to_num(lin), 300)
    den = smooth(sounding.astype(float), 300) + 1e-6
    loud = 20 * np.log10(num / den + 1e-6)
    ok = sounding & (den > 0.2)
    if ok.sum() > 300 and np.std(vel_f[ok]) > 1e-3:
        m["macro_dyn_corr"] = round(float(np.corrcoef(vel_f[ok], loud[ok])[0, 1]), 3)
    st = np.array([s["stretch"] for s in stats])
    m.update({
        "n_chunks": len(stats),
        "notes_per_clip": round(len(notes) / max(len(stats), 1), 2),
        "clip_played_full": 1.0,
        "stretch_median": round(float(np.median(st)), 3),
        "stretch_within_0.7_1.4": round(float(np.mean((st > 0.7) & (st < 1.4))), 3),
        "syllable_note_match": round(float(np.mean([s["syl"] == s["notes"] for s in stats])), 3),
        "rhythm_align_mean": round(float(np.mean([s["align"] for s in stats])), 3),
        "warp_distortion_mean": round(float(np.mean([s["warp_dist"] for s in stats])), 3),
        "hires_fraction": round(float(np.mean([s["hires"] for s in stats])), 3),
        "characters": sorted({c["char"] for c in cues}),
        "pad_layers": args.pad,
        "pad_tracks": pad_track_names,
        "f0_mode": args.f0_mode,
        "sample_rate": FS,
    })
    json.dump(m, open(mpath, "w"), ensure_ascii=False, indent=1)
    print("v7 metrics:", json.dumps({k: m.get(k) for k in ("pitch_acc_50c", "median_abs_cents",
          "dyn_corr", "macro_dyn_corr", "notes_per_clip", "stretch_median", "stretch_within_0.7_1.4",
          "syllable_note_match", "rhythm_align_mean", "warp_distortion_mean")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
