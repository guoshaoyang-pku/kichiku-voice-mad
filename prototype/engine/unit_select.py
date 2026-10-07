#!/usr/bin/env python3
"""v5: unit selection directly against the ORIGINAL vocal (beam search + regularisers).

Target  : original vocal stem -> syllable segments (onsets + note changes), each with a
          regularised pitch contour (scale-snapped main notes, grace notes < min_note removed,
          a little of the singer's own deviation kept), vowel timbre (MFCC) and loudness.
Units   : real syllables (onset consonant + stable vowel) from character lines.
Search  : beam search over segments. cost = target cost (pitch shift, stretch, timbre,
          quality) + join cost (speaker change, source contiguity bonus, reuse penalty).
Render  : PSOLA each unit onto its segment (formants kept), vowel onset on the original
          syllable onset, short crossfades. Instrumental (demucs no-vocals) kept as background.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import parselmouth
import soundfile as sf
import torch
import torchcrepe
from parselmouth.praat import call
from scipy.signal import fftconvolve

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
sys.path.insert(0, str(HERE))
import hires  # noqa: E402
import phrase_match as PM  # noqa: E402
import sampler_match as SM  # noqa: E402
import sing_melody as SG  # noqa: E402

FS = 32000
SR16 = 16000
HOP = 160
DT = HOP / SR16
DEV = SG.DEV
CACHE = HERE.parent / "materials" / "contour_cache"
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR = [0, 2, 4, 5, 7, 9, 11]
DEG = {"1": 0, "2": 2, "3": 4, "4": 5, "5": 7, "6": 9, "7": 11}

# 简谱 (jianpu) reference, bars 9-16 of the MyGO 春日影 lead sheet; "'" = octave up, "," = down
JIANPU_9_16 = ("3 3 2 4 3 2 | 2 2 1 1 4 3 2 | 2 1 2 3 | 3 5 1' | 7 1' 7 1' | 7 6 5 5 2 4 | "
               "4 3 3 5, | 4 3 2 3 5,")


def jianpu_semitones(s):
    out = []
    for tok in s.replace("|", " ").split():
        v = DEG[tok[0]] + 12 * tok.count("'") - 12 * tok.count(",")
        out.append(v)
    return out


# ------------------------------------------------------------ target
def key_from_frames(st, voiced):
    h = np.bincount(np.round(st[voiced]).astype(int) % 12, minlength=12).astype(float)
    maj = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    k = int(np.argmax([np.corrcoef(h, np.roll(maj, i))[0, 1] for i in range(12)]))
    return k


def regularise_pitch(st, voiced, tonic, min_note=0.12, keep_dev=0.25, glide=0.03):
    """Scale-snapped main notes; short runs (grace notes / scoops) absorbed by neighbours."""
    n = len(st)
    sm = st.copy()
    for i in np.flatnonzero(voiced):
        lo, hi = max(0, i - 4), min(n, i + 5)
        sm[i] = np.median(st[lo:hi][voiced[lo:hi]])
    scale = np.array([tonic + d + 12 * o for o in range(-2, 9) for d in MAJOR])
    q = np.full(n, np.nan)
    cur = None
    for i in range(n):
        if not voiced[i]:
            cur = None
            continue
        near = scale[np.argmin(np.abs(scale - sm[i]))]
        if cur is not None and abs(sm[i] - cur) < 0.75:   # hysteresis
            near = cur
        q[i] = near
        cur = near
    # absorb short runs into the longer neighbour (inside one voiced region)
    min_f = int(min_note / DT)
    for _ in range(3):
        runs, i = [], 0
        while i < n:
            if np.isnan(q[i]):
                i += 1
                continue
            j = i
            while j + 1 < n and q[j + 1] == q[i]:
                j += 1
            runs.append([i, j, q[i]])
            i = j + 1
        changed = False
        for r, (a, b, v) in enumerate(runs):
            if b - a + 1 >= min_f:
                continue
            prv = runs[r - 1] if r > 0 and runs[r - 1][1] == a - 1 else None
            nxt = runs[r + 1] if r + 1 < len(runs) and runs[r + 1][0] == b + 1 else None
            cands = [x for x in (prv, nxt) if x is not None]
            if not cands:
                continue
            best = max(cands, key=lambda x: x[1] - x[0])
            q[a:b + 1] = best[2]
            changed = True
        if not changed:
            break
    dev = np.clip(sm - q, -0.35, 0.35)
    tgt = q + keep_dev * np.nan_to_num(dev)
    # glides at note steps
    g = max(1, int(glide / DT))
    steps = np.flatnonzero(np.isfinite(q[1:]) & np.isfinite(q[:-1]) & (q[1:] != q[:-1])) + 1
    for s in steps:
        a, b = max(0, s - g // 2), min(n - 1, s + g // 2)
        if np.isfinite(tgt[a]) and np.isfinite(tgt[b]):
            tgt[a:b + 1] = np.linspace(tgt[a], tgt[b], b - a + 1)
    tgt[~voiced] = np.nan
    return q, tgt


def segment_syllables(y16, voiced, q, min_seg=0.09):
    on = librosa.onset.onset_detect(y=y16, sr=SR16, hop_length=HOP, units="frames",
                                    backtrack=False, wait=8, delta=0.06)
    n = len(voiced)
    bounds = set(int(x) for x in on if x < n)
    # voiced-region starts
    starts = np.flatnonzero(voiced & ~np.r_[False, voiced[:-1]])
    bounds |= set(int(x) for x in starts)
    # note changes without a nearby onset (legato syllables)
    ch = np.flatnonzero(np.isfinite(q[1:]) & np.isfinite(q[:-1]) & (q[1:] != q[:-1])) + 1
    arr = np.array(sorted(bounds))
    for c in ch:
        if len(arr) == 0 or np.min(np.abs(arr - c)) > int(0.08 / DT):
            bounds.add(int(c))
    b = sorted(x for x in bounds if voiced[min(x, n - 1)] or voiced[min(x + 2, n - 1)])
    segs = []
    for i, s in enumerate(b):
        e = b[i + 1] if i + 1 < len(b) else n
        idx = np.arange(s, e)
        v = idx[voiced[idx]]
        if len(v) * DT < min_seg:
            continue
        last = v[-1]
        # stop at first unvoiced gap > 60 ms
        gaps = np.flatnonzero(np.diff(v) > int(0.06 / DT))
        if len(gaps):
            last = v[gaps[0]]
        segs.append([int(v[0]), int(last) + 1])
    return segs


# ------------------------------------------------------------ units
def unit_table(bank_path, singers, cache_name):
    items = [b for b in json.load(open(bank_path))
             if b["char"] in singers and b["per"] > 0.7 and b["cents_std"] < 80 and b["rms"] > 0.3]
    by_clip = {}
    for b in items:
        by_clip.setdefault(b["path"], []).append(b)
    units = []
    for path, bs in by_clip.items():
        bs.sort(key=lambda x: x["t0"])
        for i, b in enumerate(bs):
            prev_end = bs[i - 1]["t1"] if i > 0 else 0.0
            u = dict(b)
            u["on"] = max(prev_end, b["t0"] - 0.08, 0.0)
            u["next"] = None
            units.append(u)
        base = len(units) - len(bs)
        for i in range(len(bs) - 1):
            if bs[i + 1]["t0"] - bs[i]["t1"] < 0.15:
                units[base + i]["next"] = base + i + 1
    cache = CACHE / f"unit_mfcc_{cache_name}.npy"
    if cache.exists() and len(np.load(cache)) == len(units):
        mf = np.load(cache)
    else:
        print(f"unit MFCC for {len(units)} units ...", flush=True)
        mf = np.zeros((len(units), 12), dtype=np.float32)
        for k, u in enumerate(units):
            y = SM.clip_audio(u["path"])
            seg = y[int(u["t0"] * FS):int(u["t1"] * FS)].astype(np.float32)
            seg = librosa.resample(seg, orig_sr=FS, target_sr=SR16)
            if len(seg) < 400:
                continue
            m = librosa.feature.mfcc(y=seg, sr=SR16, n_mfcc=13, n_fft=512, hop_length=HOP)[1:]
            mf[k] = m.mean(axis=1)
        np.save(cache, mf)
    return units, mf


# ------------------------------------------------------------ search
def beam_search(segs, units, cost_t, cand, args):
    """segs: list of dict; cost_t[i]: array over cand[i]; returns unit index per segment."""
    char = [u["char"] for u in units]
    nxt = [u["next"] for u in units]
    beam = [(0.0, None, ())]          # (cost, backpointer node, recent tuple)
    nodes = []                        # (unit, parent_node)
    for i, s in enumerate(segs):
        new = {}
        for cost, node, recent in beam:
            last = nodes[node][0] if node is not None else None
            for ci, u in enumerate(cand[i]):
                c = cost + cost_t[i][ci]
                if last is not None:
                    if char[u] != char[last]:
                        c += args.w_char if not s["phrase_start"] else args.w_char * 0.2
                    if nxt[last] == u and not s["phrase_start"]:
                        c -= args.w_contig
                if u in recent:
                    c += args.w_reuse
                key = u
                if key not in new or c < new[key][0]:
                    new[key] = (c, node, u, recent)
        best = sorted(new.values(), key=lambda x: x[0])[:args.beam]
        beam = []
        for c, node, u, recent in best:
            nodes.append((u, node))
            beam.append((c, len(nodes) - 1, (recent + (u,))[-args.reuse_window:]))
    node = beam[0][1]
    path = []
    while node is not None:
        u, node = nodes[node]
        path.append(u)
    return path[::-1], beam[0][0]


# ------------------------------------------------------------ render
def psola_unit(y, u, seg_t0, seg_t1, tgt_fn, prev_tail=0.0):
    a = int(u["on"] * FS)
    e = min(len(y), int(u["t1"] * FS))
    x = y[a:e].astype(np.float64)
    src = len(x) / FS
    on = u["t0"] - u["on"]
    vow = max(src - on, 0.04)
    out_vow = max(seg_t1 - seg_t0, 0.06)
    k = out_vow / vow
    snd = parselmouth.Sound(x, sampling_frequency=FS)
    manip = call(snd, "To Manipulation", 0.005, 75, 1000)
    pt = call("Create PitchTier", "p", 0, src)
    f_on = tgt_fn(seg_t0)
    call(pt, "Add point", 0.0, f_on)
    npts = max(4, int(out_vow / 0.01))
    for q in range(npts + 1):
        tau = out_vow * q / npts
        call(pt, "Add point", on + tau / k, tgt_fn(seg_t0 + tau))
    call([pt, manip], "Replace pitch tier")
    dt = call("Create DurationTier", "d", 0, src)
    call(dt, "Add point", 0.0, 1.0)
    call(dt, "Add point", max(on - 0.002, 0.0), 1.0)
    call(dt, "Add point", on + 0.002, k)
    call(dt, "Add point", src, k)
    call([dt, manip], "Replace duration tier")
    w = call(manip, "Get resynthesis (overlap-add)").values[0].astype(np.float32)
    return w, on


def onset_f1(ref_t, est_t, tol=0.05):
    ref_t, est_t = np.asarray(ref_t), np.asarray(est_t)
    if len(ref_t) == 0 or len(est_t) == 0:
        return 0.0
    used, hit = set(), 0
    for r in ref_t:
        d = np.abs(est_t - r)
        j = int(np.argmin(d))
        if d[j] <= tol and j not in used:
            used.add(j)
            hit += 1
    p, rc = hit / len(est_t), hit / len(ref_t)
    return 2 * p * rc / (p + rc + 1e-9)


def semiglobal_acc(ref, hyp):
    """Best alignment of ref inside hyp (free start/end in hyp). Returns matched/len(ref)."""
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1))
    D[1:, 0] = np.arange(1, n + 1)
    M = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = D[i - 1, j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1)
            de, ins = D[i - 1, j] + 1, D[i, j - 1] + 1
            D[i, j] = min(sub, de, ins)
            if D[i, j] == sub:
                M[i, j] = M[i - 1, j - 1] + (ref[i - 1] == hyp[j - 1])
            elif D[i, j] == de:
                M[i, j] = M[i - 1, j]
            else:
                M[i, j] = M[i, j - 1]
    j = int(np.argmin(D[n, 1:])) + 1
    return float(M[n, j] / n), int(D[n, j]), j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True)
    ap.add_argument("--stems", required=True, help="demucs stem dir (vocals/drums/bass/other)")
    ap.add_argument("--lib", nargs="+", default=[str(HERE.parent / "lib" / "library_anime_full.json")])
    ap.add_argument("--bank", default=str(HERE.parent / "materials" / "nucleus_bank.json"))
    ap.add_argument("--singers", default="长崎爽世")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur-limit", type=float, default=None)
    ap.add_argument("--min-note", type=float, default=0.12)
    ap.add_argument("--keep-dev", type=float, default=0.25)
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--beam", type=int, default=32)
    ap.add_argument("--w-pitch", type=float, default=1.0)
    ap.add_argument("--w-stretch", type=float, default=0.6)
    ap.add_argument("--w-timbre", type=float, default=0.5)
    ap.add_argument("--w-char", type=float, default=1.5)
    ap.add_argument("--w-contig", type=float, default=0.5)
    ap.add_argument("--w-reuse", type=float, default=0.8)
    ap.add_argument("--reuse-window", type=int, default=24)
    ap.add_argument("--env-depth", type=float, default=0.6)
    ap.add_argument("--inst-db", type=float, default=-6.0, help="instrumental level vs rendered vocal")
    ap.add_argument("--wet-db", type=float, default=-14.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out = out.with_suffix("") if out.suffix == ".wav" else out
    out.parent.mkdir(parents=True, exist_ok=True)
    stems = Path(args.stems)

    # ---- target
    t, f0, per, rms = SG.crepe_track(str(stems / "vocals.wav"))
    y16, _ = librosa.load(stems / "vocals.wav", sr=SR16, mono=True)
    n = min(len(t), len(y16) // HOP)
    t, f0, per, rms = t[:n], f0[:n], per[:n], rms[:n]
    rdb = 20 * np.log10(rms + 1e-9)
    voiced = (per > 0.45) & (rdb > np.percentile(rdb, 95) - 30) & (f0 > 70)
    st_raw = 69 + 12 * np.log2(np.maximum(f0, 1e-3) / 440.0)
    frac = st_raw[voiced] - np.round(st_raw[voiced])
    tune = float(np.angle(np.mean(np.exp(2j * np.pi * frac))) / (2 * np.pi))
    st = st_raw - tune
    tonic_pc = key_from_frames(st, voiced)
    q, tgt = regularise_pitch(st, voiced, tonic_pc, args.min_note, args.keep_dev)
    t0 = args.start
    t1 = t0 + args.dur_limit if args.dur_limit else t[-1]
    win = (t >= t0) & (t < t1)
    segs_f = segment_syllables(y16[:n * HOP], voiced, q)
    segs_f = [s for s in segs_f if t0 <= t[s[0]] < t1]
    mf_t = librosa.feature.mfcc(y=y16, sr=SR16, n_mfcc=13, n_fft=512, hop_length=HOP)[1:, :n]
    segs = []
    for a, b in segs_f:
        qa = q[a:b][np.isfinite(q[a:b])]
        if len(qa) == 0:
            continue
        vals, cnt = np.unique(qa, return_counts=True)
        segs.append({"a": a, "b": b, "t0": float(t[a]), "t1": float(t[b - 1] + DT),
                     "st": float(vals[np.argmax(cnt)]), "db": float(np.mean(rdb[a:b])),
                     "mfcc": mf_t[:, a:b].mean(axis=1)})
    for i, s in enumerate(segs):
        s["phrase_start"] = i == 0 or s["t0"] - segs[i - 1]["t1"] > 0.35
        s["legato_next"] = i + 1 < len(segs) and segs[i + 1]["t0"] - s["t1"] < 0.06
    tonic_midi = 48 + tonic_pc + (12 if 48 + tonic_pc < 54 else 0)
    print(f"target: key {NAMES[tonic_pc]} major, tuning {tune * 100:+.0f}c, {len(segs)} syllable segments, "
          f"median {np.median([s['t1'] - s['t0'] for s in segs]):.2f}s", flush=True)

    # ---- jianpu check (bars 9-16) on segment note sequence
    # compare interval sequences (transposition invariant); repeated notes kept as 0
    ref_n = jianpu_semitones(JIANPU_9_16)
    ref = list(np.diff(ref_n))
    hyp_n = [int(round(s["st"])) for s in segs if s["t0"] < t0 + 120]
    hyp = list(np.diff(hyp_n))
    acc, ed, j = semiglobal_acc(ref, hyp)
    best = (acc, ed, j)
    print(f"jianpu bars 9-16: {best[0] * 100:.0f}% notes matched (edit distance {best[1]} / {len(ref)})",
          flush=True)

    # ---- units
    singers = [s.strip() for s in args.singers.split(",")]
    lib = PM.load_library(args.lib, None, None, 0.0, 99.0, 0.15)
    SM._lib_by_path.update({c["path"]: c for c in lib})
    hires.ensure_anime_maps([c["src"] for c in lib if c["work"].startswith("anime")])
    units, mf_u = unit_table(args.bank, singers, "_".join(singers))
    u_st = np.array([u["st"] for u in units])
    u_vow = np.array([u["t1"] - u["t0"] for u in units])
    u_q = np.array([0.3 * (1 - u["per"]) + u["cents_std"] / 200.0 for u in units])
    # CMVN per corpus, so stem vs dry-speech channel differences cancel
    T = np.stack([s["mfcc"] for s in segs])
    zt = (T - T.mean(0)) / (T.std(0) + 1e-6)
    ok = np.abs(mf_u).sum(1) > 0
    zu = (mf_u - mf_u[ok].mean(0)) / (mf_u[ok].std(0) + 1e-6)
    print(f"units: {len(units)} syllables from {len({u['path'] for u in units})} lines "
          f"({', '.join(singers)}), contiguous pairs {sum(u['next'] is not None for u in units)}", flush=True)

    cand, cost_t = [], []
    for i, s in enumerate(segs):
        dp = np.abs(u_st - s["st"])
        r = (s["t1"] - s["t0"]) / u_vow
        lr = np.log2(r)
        c_p = (dp / 3.0) ** 2 + 2.0 * (dp > 6)
        c_s = np.maximum(0, lr - 1.0) + 0.5 * np.maximum(0, -lr - 1.0)
        c_m = np.sqrt(np.mean((zu - zt[i]) ** 2, axis=1))
        c = args.w_pitch * c_p + args.w_stretch * c_s + args.w_timbre * c_m + u_q + 3.0 * (~ok)
        idx = np.argpartition(c, args.k)[:args.k]
        # make sure contiguous successors of the previous segment's candidates are reachable
        if i > 0:
            succ = [units[u]["next"] for u in cand[-1] if units[u]["next"] is not None]
            idx = np.unique(np.r_[idx, np.array(succ, dtype=int)])
        cand.append(idx)
        cost_t.append(c[idx])
    path, total = beam_search(segs, units, cost_t, cand, args)
    print(f"beam search done, total cost {total:.1f}", flush=True)

    # ---- render
    span = t1 - t0
    N = int((span + 3) * FS)
    voc = np.zeros(N, dtype=np.float32)
    grid_t = t
    tgt_f = np.where(np.isfinite(tgt), tgt, np.nan)
    good = np.isfinite(tgt_f)

    def tgt_fn(time):
        v = np.interp(time, grid_t[good], tgt_f[good])
        return float(440.0 * 2 ** ((v - 69) / 12.0))

    cues = []
    for i, (s, u) in enumerate(zip(segs, path)):
        un = units[u]
        y = SM.clip_audio(un["path"])
        tail = 0.03 if s["legato_next"] else 0.05
        w, on = psola_unit(y, un, s["t0"], s["t1"] + tail, tgt_fn)
        act = np.abs(w) > 1e-4
        w *= 0.12 / (np.sqrt(np.mean(w[act] ** 2)) + 1e-9) if act.any() else 1.0
        g = 10 ** (args.env_depth * np.clip(s["db"] - np.percentile([x["db"] for x in segs], 90), -15, 3) / 20)
        w *= g
        fi = min(int(0.004 * FS), len(w) // 4)
        fo = min(int(tail * FS), len(w) // 3)
        w[:fi] *= np.linspace(0, 1, fi)
        w[-fo:] *= np.linspace(1, 0, fo) ** 0.7
        SM.place(voc, w, s["t0"] - on - t0)
        prev = path[i - 1] if i > 0 else None
        cues.append({"t0": round(s["t0"] - t0, 3), "t1": round(s["t1"] - t0, 3),
                     "note": f"{NAMES[int(round(s['st'])) % 12]}{int(round(s['st'])) // 12 - 1}",
                     "midi": int(round(s["st"])), "unit": int(u), "char": un["char"],
                     "unit_st": un["st"], "shift": round(s["st"] - un["st"], 2),
                     "stretch": round((s["t1"] - s["t0"]) / (un["t1"] - un["t0"]), 2),
                     "contiguous": bool(prev is not None and units[prev]["next"] == u and not s["phrase_start"]),
                     "text": un.get("text", ""), "path": un["path"], "on": un["on"], "u_t1": un["t1"]})

    wet = fftconvolve(voc, SG.reverb_ir())[:N].astype(np.float32)
    voc_w = voc + wet * (np.sqrt(np.mean(voc ** 2)) / (np.sqrt(np.mean(wet ** 2)) + 1e-9)) * 10 ** (args.wet_db / 20)

    inst = None
    for name in ("drums", "bass", "other"):
        p = stems / f"{name}.wav"
        info = sf.info(p)
        x, sr = sf.read(p, start=int(t0 * info.samplerate),
                        stop=int(min(t0 + span + 3, info.duration) * info.samplerate),
                        always_2d=True, dtype="float32")
        x = x.mean(axis=1)
        x = librosa.resample(x, orig_sr=sr, target_sr=FS) if sr != FS else x
        inst = x if inst is None else inst[:len(x)] + x[:len(inst)]
    inst = np.pad(inst, (0, max(0, N - len(inst))))[:N]
    va = np.abs(voc_w) > 1e-4
    inst *= np.sqrt(np.mean(voc_w[va] ** 2)) * 10 ** (args.inst_db / 20) / (np.sqrt(np.mean(inst ** 2)) + 1e-9)
    mix = voc_w + inst
    master = 0.95 / max(np.max(np.abs(mix)), 1e-9)
    end = int((span + 1.5) * FS)

    def save(suf, sig, norm=False):
        p = out.with_name(out.name + suf + ".wav")
        s_ = sig[:end] / (np.max(np.abs(sig[:end])) + 1e-9) * 0.85 if norm else sig[:end] * master
        sf.write(p, s_.astype(np.float32), FS)
        SG.mp3(p)

    save("", mix)
    save("_vocal_dry", voc)
    save("_vocal", voc_w)
    save("_instrumental", inst)
    for suf, src in (("_orig_mix", args.mix), ("_orig_vocal", str(stems / "vocals.wav"))):
        info = sf.info(src)
        x, sr = sf.read(src, start=int(t0 * info.samplerate),
                        stop=int(min(t0 + span + 1.5, info.duration) * info.samplerate),
                        always_2d=True, dtype="float32")
        x = x.mean(axis=1)
        x = librosa.resample(x, orig_sr=sr, target_sr=FS) if sr != FS else x
        save(suf, np.pad(x, (0, max(0, end - len(x)))), norm=True)
    # guide: smooth sine following the regularised target contour (checks the melody target)
    gt = np.arange(end) / FS + t0
    gv = np.interp(gt, t, np.where(np.isfinite(tgt), 1.0, 0.0)) > 0.5
    gs = np.interp(gt, t[good], tgt_f[good])
    ph = np.cumsum(2 * np.pi * 440 * 2 ** ((gs - 69) / 12) / FS)
    env = PM.smooth(gv.astype(float), int(0.01 * FS))
    save("_target_guide", (np.sin(ph) + 0.25 * np.sin(2 * ph)) * env * 0.3, norm=True)

    # fragments: first 24 syllables, raw unit vs rendered syllable vs original syllable
    frag = out.with_name(out.name + "_fragments")
    frag.mkdir(exist_ok=True)
    yv, _ = librosa.load(stems / "vocals.wav", sr=FS, mono=True, offset=t0, duration=span + 2)
    rows = []
    for k, c in enumerate(cues[:24]):
        y = SM.clip_audio(c["path"])
        raw = y[int(c["on"] * FS):int(c["u_t1"] * FS)].astype(np.float32)
        a, b = int(max(0, c["t0"] - 0.06) * FS), int((c["t1"] + 0.06) * FS)
        for suf, sig in (("A_unit_raw", raw), ("B_orig_syllable", yv[a:b]), ("C_rendered", voc[a:b])):
            p = frag / f"{k:02d}_{suf}.wav"
            sf.write(p, sig / (np.max(np.abs(sig)) + 1e-9) * 0.7, FS)
            SG.mp3(p)
        rows.append((k, c))
    h = ['<!doctype html><meta charset=utf-8><title>v5 碎片</title><style>body{font-family:-apple-system,"PingFang SC";max-width:960px;margin:2rem auto}td,th{border:1px solid #ddd;padding:.3rem .5rem;font-size:.85rem}table{border-collapse:collapse;width:100%}audio{width:100%;height:32px}</style>',
         '<h1>v5 碎片：素材原声 / 原曲这个音节 / 渲染结果</h1><table><tr><th>#</th><th>音节</th><th>A 素材原声（未处理）</th><th>B 原曲人声这个音节</th><th>C 渲染（干声）</th></tr>']
    for k, c in rows:
        h.append(f'<tr><td>{k:02d}</td><td>{c["note"]} · {c["t1"] - c["t0"]:.2f}s<br>{c["char"]} · 移调 {c["shift"]:+.1f} · 延长 ×{c["stretch"]}{" · 接上一段" if c["contiguous"] else ""}<br>{c["text"][:20]}</td>'
                 + "".join(f'<td><audio controls preload=none src="{k:02d}_{s}.mp3"></audio></td>'
                           for s in ("A_unit_raw", "B_orig_syllable", "C_rendered")) + "</tr>")
    h.append("</table>")
    (frag / "index.html").write_text("\n".join(h))
    json.dump(cues, open(out.with_suffix(".cues.json"), "w"), ensure_ascii=False, indent=1)

    # ---- metrics ("sounds right" checklist)
    yr = librosa.resample(voc[:end], orig_sr=FS, target_sr=SR16)
    x = torch.from_numpy(yr).float().unsqueeze(0).to(DEV)
    fr, pr = torchcrepe.predict(x, SR16, HOP, 70.0, 1000.0, model="full",
                                decoder=torchcrepe.decode.viterbi, return_periodicity=True,
                                device=DEV, batch_size=1024, pad=True)
    fr, pr = fr.squeeze(0).cpu().numpy(), pr.squeeze(0).cpu().numpy()
    tr = np.arange(len(fr)) * DT + t0
    i0 = int(t0 / DT)
    L = min(len(fr), n - i0)
    fr, pr, tr = fr[:L], pr[:L], tr[:L]
    tv = voiced[i0:i0 + L]
    rv = pr > 0.4
    obs = 69 + 12 * np.log2(np.maximum(fr, 1e-3) / 440.0)
    qq = q[i0:i0 + L]
    both = tv & rv & np.isfinite(qq)
    d_note = np.abs(obs[both] - qq[both]) * 100
    d_raw = np.abs(obs[both] - st[i0:i0 + L][both]) * 100
    stab = []
    for s in segs:
        a = int((s["t0"] + 0.2 * (s["t1"] - s["t0"])) / DT) - i0
        b = int((s["t1"] - 0.2 * (s["t1"] - s["t0"])) / DT) - i0
        if b - a >= 5 and b <= L:
            v = obs[a:b][rv[a:b]]
            if len(v) >= 5:
                stab.append(np.std(v) * 100)
    on_ref = librosa.onset.onset_detect(y=y16[i0 * HOP:(i0 + L) * HOP], sr=SR16, hop_length=HOP, units="time")
    on_est = librosa.onset.onset_detect(y=yr, sr=SR16, hop_length=HOP, units="time")
    r_rms = librosa.feature.rms(y=yr, frame_length=1024, hop_length=HOP)[0][:L]
    sm = lambda v: PM.smooth(v, 51)
    e_r = sm(20 * np.log10(r_rms + 1e-6))
    e_o = sm(rdb[i0:i0 + len(e_r)])
    m_ = tv[:len(e_r)]
    dyn = float(np.corrcoef(e_r[m_], e_o[m_])[0, 1]) if m_.sum() > 10 else float("nan")
    joins = [c["contiguous"] for c in cues[1:]]
    metrics = {
        "version": "v5", "song": Path(args.mix).stem, "singers": singers,
        "window": [round(t0, 1), round(t1, 1)], "key": f"{NAMES[tonic_pc]} major",
        "n_syllables": len(segs), "syllable_dur_median": round(float(np.median([s["t1"] - s["t0"] for s in segs])), 3),
        "jianpu_bars9_16_match": round(best[0], 3),
        "M1_melody_acc50_vs_main_notes": round(float(np.mean(d_note < 50)), 3),
        "M1_melody_cents_median_vs_main_notes": round(float(np.median(d_note)), 1),
        "M1_melody_acc50_vs_raw_vocal": round(float(np.mean(d_raw < 50)), 3),
        "M2_voicing_recall": round(float(np.mean(rv[tv])), 3),
        "M2_voicing_false_alarm": round(float(np.mean(rv[~tv])), 3),
        "M3_onset_f1_50ms": round(onset_f1(on_ref, on_est), 3),
        "M4_note_stability_cents_median": round(float(np.median(stab)), 1) if stab else None,
        "M5_dynamics_corr": round(dyn, 3),
        "M6_shift_semi_median_abs": round(float(np.median(np.abs([c["shift"] for c in cues]))), 2),
        "M6_shift_gt4_ratio": round(float(np.mean(np.abs([c["shift"] for c in cues]) > 4)), 3),
        "M6_stretch_median": round(float(np.median([c["stretch"] for c in cues])), 2),
        "M7_contiguous_join_ratio": round(float(np.mean(joins)), 3) if joins else 0.0,
        "M7_distinct_lines": len({c["path"] for c in cues}),
        "M7_distinct_units": len({c["unit"] for c in cues}),
        "inst_db": args.inst_db,
    }
    json.dump(metrics, open(out.with_suffix(".metrics.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(metrics, ensure_ascii=False), flush=True)

    # ---- piano roll
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "DejaVu Sans"]
    import matplotlib.pyplot as plt
    show = min(span, 50.0)
    fig, ax = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    mm = (t >= t0) & (t < t0 + show)
    for a_ in ax:
        for c in cues:
            if c["t0"] < show:
                a_.add_patch(plt.Rectangle((c["t0"], c["midi"] - 0.4), c["t1"] - c["t0"], 0.8,
                                           color="tab:orange", alpha=0.3, lw=0))
        a_.grid(alpha=0.3)
        a_.set_ylabel("MIDI")
    ax[0].plot(t[mm] - t0, np.where(voiced[mm], st[mm], np.nan), color="tab:blue", lw=0.8, label="原曲人声 F0")
    ax[0].plot(t[mm] - t0, tgt[mm], color="k", lw=1.2, label="正则化目标")
    ax[0].legend(loc="upper right")
    ax[0].set_title("音节分段（橙）· 原曲人声 F0（蓝）· 正则化目标（黑）")
    rs = np.where(rv & (tr - t0 < show), obs, np.nan)
    ax[1].plot(tr - t0, rs, color="tab:red", lw=1)
    ax[1].set_title("渲染人声 F0（红）")
    lo = np.nanpercentile(st[voiced], 1) - 2
    hi = np.nanpercentile(st[voiced], 99) + 2
    for a_ in ax:
        a_.set_ylim(lo, hi)
    ax[1].set_xlim(0, show)
    plt.tight_layout()
    d = out.parent / "diagnostics"
    d.mkdir(exist_ok=True)
    plt.savefig(d / f"{out.name}_pianoroll.png", dpi=100)


if __name__ == "__main__":
    main()
