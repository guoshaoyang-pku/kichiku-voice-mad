#!/usr/bin/env python3
"""v1 sparse phrase-matching renderer.

Uses a constrained codebook of 1-4 s voice clips: each melody phrase picks one
clip whose natural pitch contour/rhythm is already close, then applies only a
small amount of correction if needed. Vocal and sparse backing stems are
rendered separately; backing is mixed 10 dB below its own stem level.
"""
import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import pyworld as pw
import soundfile as sf

try:
    import torch
    import torchcrepe
except Exception:  # pragma: no cover - CPU fallback
    torch = None
    torchcrepe = None

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
import hires  # noqa: E402

FS = 32000
F0_HOP = 160                       # 5 ms at 32 kHz
MAX_SHIFT_SEMI = 3.0
MAX_RATE = 1.25
EXCLUDE_CHARS = {"混合", "未确认", "未知", "?"}
SEED = 7
CONTOUR_CACHE = HERE.parent / "lib" / "phrase_f0_cache.json"


# ------------------------------------------------------------ features
def smooth(x, n):
    if n <= 1 or len(x) < n:
        return x
    k = np.ones(n) / n
    return np.convolve(np.pad(x, (n // 2, n - 1 - n // 2), mode="edge"), k, mode="valid")


def resample_seq(x, n):
    if len(x) == n:
        return x.astype(float)
    if len(x) < 2:
        return np.full(n, float(x[0]) if len(x) else 0.0)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)


def notes_from_track(pm, track):
    ins = pm.instruments[track] if track >= 0 else max(
        (i for i in pm.instruments if i.notes and not i.is_drum),
        key=lambda i: sum(n.end - n.start for n in i.notes),
    )
    notes = sorted(((n.start, n.end, n.pitch, n.velocity) for n in ins.notes),
                   key=lambda x: (x[0], -x[2]))
    out = []
    for n in notes:
        if out and n[0] - out[-1][0] < 0.025 and n[2] <= out[-1][2]:
            continue
        out.append(n)
    return ins, out


def phrases(notes, gap=0.38, jump=16):
    out, cur = [], []
    for n in notes:
        if cur and (n[0] - cur[-1][1] > gap or abs(n[2] - cur[-1][2]) > jump):
            out.append(cur)
            cur = []
        cur.append(n)
    if cur:
        out.append(cur)
    return out


def melody_segments(notes, target=2.9, min_dur=1.0, max_dur=4.2):
    """Partition the melody into sparse 1-4 s chunks, preferring rests."""
    n = len(notes)
    dp = [0.0] + [float("inf")] * n
    back = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(max(0, i - 9), i):
            dur = notes[i - 1][1] - notes[j][0]
            if dur < min_dur or dur > max_dur:
                continue
            gap = notes[i][0] - notes[i - 1][1] if i < n else 0.6
            jump = abs(notes[i][2] - notes[i - 1][2]) if i < n else 0
            cost = ((dur - target) / 1.15) ** 2
            cost += 0.35 if gap < 0.14 else (-0.45 if gap > 0.28 else 0.0)
            cost += min(jump, 18) / 60.0
            if dp[j] + cost < dp[i]:
                dp[i], back[i] = dp[j] + cost, j
    out, i = [], n
    while i > 0:
        j = back[i]
        out.append(notes[j:i])
        i = j
    return list(reversed(out))


def phrase_feature(notes):
    dur = max(notes[-1][1] - notes[0][0], 1e-3)
    ts, ps, vs = [], [], []
    for s, e, p, v in notes:
        ts += [s, e]
        ps += [p, p]
        vs += [v, v]
    ts = np.array(ts)
    grid = np.linspace(notes[0][0], notes[0][0] + dur, 24)
    p = np.interp(grid, ts, ps)
    v = np.interp(grid, ts, vs)
    center = np.median(ps)
    return {
        "start": notes[0][0], "end": notes[-1][1], "dur": dur,
        "n_notes": len(notes), "center": center,
        "contour": (p - center) / 12.0,
        "vel": float(np.mean([n[3] for n in notes])),
        "onsets": len(notes) / dur,
        "notes": notes,
    }


_contours = None


def _contour_map():
    global _contours
    if _contours is None:
        _contours = json.load(open(CONTOUR_CACHE)) if CONTOUR_CACHE.exists() else {}
    return _contours


def _save_contours():
    CONTOUR_CACHE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(_contours, open(CONTOUR_CACHE, "w"))


def _contour_from_f0(f0, good):
    good = np.asarray(good, dtype=bool) & np.isfinite(f0) & (f0 > 0)
    if good.sum() < 4:
        return [np.zeros(24).tolist(), 0.0, 0.0]
    semi = 69 + 12 * np.log2(f0[good] / 440.0)
    med = float(np.median(semi))
    contour = resample_seq(semi - med, 24) / 12.0
    return [np.round(contour, 5).tolist(), med, float(np.mean(good))]


def clip_f0(path):
    cached = _contour_map().get(path)
    if cached is not None:
        return cached
    y, _ = librosa.load(path, sr=16000, mono=True)
    hop = 160
    f0 = librosa.yin(y, fmin=70, fmax=1000, sr=16000, frame_length=1024,
                     hop_length=hop, trough_threshold=0.08)
    f0 = np.asarray(f0, dtype=float)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0]
    good = rms[:len(f0)] > np.percentile(rms, 60) * 0.55
    out = _contour_from_f0(f0, good)
    _contour_map()[path] = out
    if len(_contours) % 40 == 0:
        _save_contours()
    return out


def prepare_contours(paths, device="auto", batch_size=32):
    """Batch CREPE contour extraction on GPU; falls back to per-clip YIN."""
    missing = [p for p in dict.fromkeys(paths) if p not in _contour_map()]
    if not missing:
        return
    use_gpu = (torch is not None and torchcrepe is not None and
               ((device == "auto" and torch.backends.mps.is_available()) or device == "mps"))
    if not use_gpu:
        for p in missing:
            clip_f0(p)
        _save_contours()
        return
    dev = "mps"
    print(f"contours: CREPE full on {dev}, {len(missing)} clips", flush=True)
    for i in range(0, len(missing), batch_size):
        batch = missing[i:i + batch_size]
        ys = [librosa.load(p, sr=16000, mono=True)[0] for p in batch]
        maxlen = max(len(y) for y in ys)
        x = torch.stack([
            torch.nn.functional.pad(torch.from_numpy(y).float(), (0, maxlen - len(y)))
            for y in ys
        ]).to(dev)
        f0, periodicity = torchcrepe.predict(
            x, 16000, 160, 70, 1000, model="full",
            decoder=torchcrepe.decode.argmax, return_periodicity=True,
            device=dev, batch_size=len(batch), pad=False,
        )
        if dev == "mps":
            torch.mps.synchronize()
        f0 = f0.detach().cpu().numpy()
        periodicity = periodicity.detach().cpu().numpy()
        for p, f, per in zip(batch, f0, periodicity):
            n = min(len(f), len(per))
            _contour_map()[p] = _contour_from_f0(f[:n], per[:n] > 0.35)
        if (i // batch_size + 1) % 4 == 0 or i + batch_size >= len(missing):
            print(f"  contours {min(i + batch_size, len(missing))}/{len(missing)}", flush=True)
            _save_contours()
    _save_contours()


# ------------------------------------------------------------ library
def load_library(paths, palette, works, min_dur, max_dur, min_voiced=0.35):
    items = []
    for p in paths:
        items += json.load(open(p))
    works = set(works.split(",")) if works else None
    out = []
    for s in items:
        if s.get("char") in EXCLUDE_CHARS or s.get("f0_semi") is None:
            continue
        if not (min_dur <= s.get("dur", 0) <= max_dur):
            continue
        if not (40 <= s["f0_semi"] <= 86):
            continue
        if s.get("voiced_ratio", 0) < min_voiced:
            continue
        if palette and not s["work"].startswith(palette):
            continue
        if works and s["work"].split(":")[-1] not in works:
            continue
        out.append(s)
    print(f"library: {len(out)} long clips, {len({s['char'] for s in out})} characters", flush=True)
    return out


def rough_score(clip, ph, prev_char):
    center_cost = abs(clip["f0_semi"] - ph["center"]) / 12.0
    dur_cost = abs(np.log2(clip["dur"] / ph["dur"]))
    stability = min(clip.get("f0_iqr_cents", 999), 1200) / 1200.0
    voiced_bonus = -0.35 * (clip.get("voiced_ratio", 0.0) - 0.45)
    repeat_cost = 0.4 if clip.get("used", 0) else 0.0
    char_cost = -0.08 if prev_char == clip["char"] else 0.0
    return 0.9 * center_cost + 0.8 * dur_cost + 0.15 * stability + voiced_bonus + repeat_cost + char_cost


def score_clip(clip, ph, prev_char):
    contour, center, voiced = clip_f0(clip["path"])
    contour = np.asarray(contour, dtype=float)
    target = np.asarray(ph["contour"], dtype=float)
    # Allow a small constant register offset, then compare shape; this favors
    # clips whose natural intonation already moves like the melody.
    shifted = contour + np.clip(ph["center"] - center, -0.25, 0.25) / 12.0
    shape_cost = float(np.mean(np.abs(shifted - target)))
    trend_cost = float(np.mean(np.abs(np.diff(shifted) - np.diff(target))))
    center_cost = abs(center - ph["center"]) / 12.0 if center else 2.0
    dur_cost = abs(np.log2(clip["dur"] / ph["dur"]))
    stability = min(clip.get("f0_iqr_cents", 999), 1200) / 1200.0
    rhythm_cost = abs(voiced - min(0.9, 0.35 + 0.08 * ph["n_notes"]))
    repeat_cost = 1.2 if clip.get("used", 0) else 0.0
    char_cost = -0.04 if prev_char == clip["char"] else 0.0
    return (2.6 * shape_cost + 0.9 * trend_cost + 1.1 * center_cost +
            0.65 * dur_cost + 0.25 * rhythm_cost + 0.15 * stability +
            repeat_cost + char_cost)


def choose_clip(cands, ph, prev_char, top_n=300):
    rough = sorted(((rough_score(c, ph, prev_char), c) for c in cands), key=lambda x: x[0])[:max(top_n, 1)]
    scored = sorted(((score_clip(c, ph, prev_char), c) for _, c in rough), key=lambda x: x[0])
    return scored[:top_n]


# ------------------------------------------------------------ render
def fit_clip(y, target_dur):
    if target_dur <= 0 or len(y) == 0:
        return y
    cur = len(y) / FS
    rate = np.clip(cur / target_dur, 1 / MAX_RATE, MAX_RATE)
    if abs(rate - 1) > 0.025:
        y = librosa.effects.time_stretch(y, rate=rate)
    return y.astype(np.float64)


def correct_pitch(y, target_center, source_center):
    shift = float(np.clip(target_center - source_center, -MAX_SHIFT_SEMI, MAX_SHIFT_SEMI))
    if abs(shift) < 0.08:
        return y, 0.0
    return librosa.effects.pitch_shift(y, sr=FS, n_steps=shift), shift


def guide_pitch_to_phrase(y, ph, strength):
    """Blend source F0 toward the phrase contour, preserving spectral envelope."""
    if strength <= 0 or len(y) < FS // 8:
        return y
    fp = 5.0
    f0, t = pw.dio(y, FS, f0_floor=70.0, f0_ceil=1000.0, frame_period=fp)
    f0 = pw.stonemask(y, f0, t, FS)
    sp = pw.cheaptrick(y, f0, t, FS)
    ap = pw.d4c(y, f0, t, FS)
    dur = len(y) / FS
    scale = dur / max(ph["dur"], 1e-3)
    frames = np.arange(len(f0)) * fp / 1000.0
    target = np.full(len(f0), np.nan)
    for s, e, p, _v in ph["notes"]:
        a = (s - ph["start"]) * scale
        b = (e - ph["start"]) * scale
        target[(frames >= a) & (frames < b)] = p
    if not np.isfinite(target).any():
        return y
    good_t = np.isfinite(target)
    target = np.interp(frames, frames[good_t], target[good_t])
    target = smooth(target, 5)
    src = np.where(f0 > 0, 69 + 12 * np.log2(np.maximum(f0, 1e-6) / 440.0), np.nan)
    if np.isfinite(src).sum() < 4:
        return y
    src_f = np.where(np.isfinite(src), src, np.nanmedian(src))
    out_semi = src_f + strength * (target - src_f)
    f0_out = np.where(f0 > 0, 440.0 * 2 ** ((out_semi - 69) / 12.0), 0.0)
    return pw.synthesize(np.ascontiguousarray(f0_out), np.ascontiguousarray(sp),
                         np.ascontiguousarray(ap), FS, fp)


def place(buf, y, t0, gain=1.0):
    i0 = int(round(t0 * FS))
    if i0 < 0:
        y = y[-i0:]
        i0 = 0
    if i0 >= len(buf):
        return
    buf[i0:i0 + len(y)] += y[:len(buf) - i0] * gain


def render_sample(clip, ph, start_offset, pitch_strength=0.7):
    y, _ = hires.load(clip, FS)
    target = min(ph["dur"] * 1.05, max(0.6, clip["dur"] * 1.2))
    y = fit_clip(y, target)
    _, center, _ = clip_f0(clip["path"])
    if not center:
        center = clip.get("f0_semi", 0.0)
    shift = float(np.clip(ph["center"] - center, -MAX_SHIFT_SEMI, MAX_SHIFT_SEMI)) if center else 0.0
    if pitch_strength > 0:
        y = guide_pitch_to_phrase(y, ph, pitch_strength)
    elif abs(shift) >= 0.08:
        y = librosa.effects.pitch_shift(y, sr=FS, n_steps=shift)
    rms = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
    y *= 10 ** (-20 / 20) / rms
    g = min(int(0.006 * FS), len(y) // 4)
    if g:
        y[:g] *= np.linspace(0, 1, g)
        y[-g:] *= np.linspace(1, 0, g)
    if center and abs(shift) > 1e-6:
        center += shift
    t0 = ph["start"] - start_offset
    return y.astype(np.float32), t0, shift, center


def backing_notes(pm, melody_ins, t0, t1):
    tracks = []
    for ins in pm.instruments:
        if not ins.notes or ins is melody_ins:
            continue
        total = sum(n.end - n.start for n in ins.notes if t0 <= n.start < t1)
        if ins.is_drum:
            total *= 0.2
        tracks.append((total, ins))
    tracks.sort(key=lambda x: -x[0])
    out = []
    for total, ins in tracks[:3]:
        ns = [n for n in ins.notes if t0 <= n.start < t1]
        if not ns:
            continue
        step = max(1, int(np.ceil(len(ns) / 12)))
        sparse = ns[::step]
        for n in sparse:
            p = n.pitch
            while p > 64:
                p -= 12
            while p < 40:
                p += 12
            out.append((n.start, n.end, p, n.velocity, ins.name))
    return sorted(out)


def render_backing(cands, notes, t0):
    buf = np.zeros(1, dtype=np.float32)
    return buf


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)


def melody_accuracy(vocal, notes, t0, sr):
    y = librosa.resample(vocal, orig_sr=sr, target_sr=16000)
    hop = 160
    f0 = librosa.yin(y, fmin=70, fmax=1000, sr=16000, frame_length=1024,
                     hop_length=hop, trough_threshold=0.08)
    t = np.arange(len(f0)) * hop / 16000.0
    exp = np.full(len(t), np.nan)
    for s, e, p, _v in notes:
        exp[(t >= s - t0) & (t < e - t0)] = p
    mask = np.isfinite(exp) & np.isfinite(f0) & (f0 > 0)
    if mask.sum() < 20:
        return {}
    obs = 69 + 12 * np.log2(f0[mask] / 440.0)
    d = np.abs(obs - exp[mask]) * 100.0
    return {
        "melody_median_abs_cents": round(float(np.median(d)), 1),
        "melody_acc_50c": round(float(np.mean(d < 50)), 3),
        "melody_acc_100c": round(float(np.mean(d < 100)), 3),
    }


def synth_melody(notes, t0, n):
    y = np.zeros(n, dtype=np.float32)
    for s, e, p, v in notes:
        i0 = max(0, int((s - t0) * FS))
        i1 = min(n, int((e - t0) * FS))
        if i1 <= i0:
            continue
        tt = np.arange(i1 - i0) / FS
        f = 440.0 * 2 ** ((p - 69) / 12.0)
        tone = (np.sin(2 * np.pi * f * tt) +
                0.35 * np.sin(4 * np.pi * f * tt) +
                0.12 * np.sin(6 * np.pi * f * tt))
        a = min(int(0.006 * FS), len(tone) // 4)
        r = min(int(0.035 * FS), len(tone) // 4)
        if a:
            tone[:a] *= np.linspace(0, 1, a)
        if r:
            tone[-r:] *= np.linspace(1, 0, r)
        y[i0:i1] += tone * (v / 127.0) * 0.16
    return y


def write_references(out, cues, cands, n, melody, t0):
    by_clip = {Path(s["path"]).name: s for s in cands}
    ref_dir = out.parent / (out.name + "_refs")
    ref_dir.mkdir(parents=True, exist_ok=True)
    reference = np.zeros(n, dtype=np.float32)
    ref_files = []
    for i, c in enumerate(cues, 1):
        sample = by_clip[c["clip"]]
        y, _ = hires.load(sample, FS)
        y = y[:min(len(y), int(4.2 * FS))].astype(np.float32)
        rms = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
        y *= 10 ** (-20 / 20) / rms
        name = f"{i:02d}_{c['char']}_{c['clip']}"
        sf.write(ref_dir / name, y, FS)
        place(reference, y, c["t0"])
        ref_files.append(str(ref_dir / name))
    peak = np.max(np.abs(reference)) + 1e-9
    reference = reference / peak * 0.82
    sf.write(out.with_name(out.name + "_reference.wav"), reference, FS)
    melody_ref = synth_melody(melody, t0, n)
    sf.write(out.with_name(out.name + "_melody_reference.wav"), melody_ref, FS)
    return ref_dir, ref_files


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi", required=True)
    ap.add_argument("--track", type=int, default=-1)
    ap.add_argument("--lib", nargs="+", required=True)
    ap.add_argument("--palette", default=None)
    ap.add_argument("--works", default=None)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur-limit", type=float, required=True)
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=4.0)
    ap.add_argument("--min-voiced", type=float, default=0.45)
    ap.add_argument("--candidate-pool", type=int, default=220)
    ap.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--backing-gain-db", type=float, default=-10.0)
    ap.add_argument("--pitch-strength", type=float, default=0.7,
                    help="0 keeps source F0; 1 follows the phrase contour")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pm = pretty_midi.PrettyMIDI(args.midi)
    melody_ins, notes = notes_from_track(pm, args.track)
    t0, t1 = args.start, args.start + args.dur_limit
    melody = [n for n in notes if t0 <= n[0] < t1]
    phs = [phrase_feature(p) for p in melody_segments(melody)]
    cands = load_library(args.lib, args.palette, args.works, args.min_dur,
                         args.max_dur, args.min_voiced)
    srcs = [s["src"] for s in cands if s["work"].startswith("anime")]
    if srcs:
        hires.ensure_anime_maps(srcs)

    bnotes = backing_notes(pm, melody_ins, t0, t1)
    pool_paths = []
    for ph in phs:
        rough = sorted(cands, key=lambda c: rough_score(c, ph, None))[:args.candidate_pool]
        pool_paths.extend(c["path"] for c in rough)
    for s, e, p, v, _tr in bnotes:
        ph = {"contour": np.zeros(24), "center": p,
              "dur": min(max(e - s, 0.35), 2.0), "n_notes": 1,
              "notes": [(s, e, p, v)], "start": s, "vel": v}
        rough = sorted(cands, key=lambda c: rough_score(c, ph, None))[:48]
        pool_paths.extend(c["path"] for c in rough)
    prepare_contours(pool_paths, args.device, args.batch_size)

    n = int((args.dur_limit + 6) * FS)
    vocal = np.zeros(n, dtype=np.float32)
    cues = []
    prev_char = None
    for ph in phs:
        best_score, clip = choose_clip(cands, ph, prev_char, top_n=args.candidate_pool)[0]
        clip["used"] = clip.get("used", 0) + 1
        y, pos, shift, center = render_sample(clip, ph, t0, args.pitch_strength)
        place(vocal, y, pos)
        prev_char = clip["char"]
        cues.append({
            "t0": round(float(pos), 3), "t1": round(float(pos + len(y) / FS), 3),
            "char": clip["char"], "work": clip["work"], "clip": Path(clip["path"]).name,
            "text": clip.get("text", ""), "dur": round(float(len(y) / FS), 3),
            "notes": len(ph["notes"]), "phrase_center": round(float(ph["center"]), 3),
            "clip_center": round(float(center), 3), "shift_semi": round(float(shift), 3),
            "match_score": round(float(best_score), 4),
            "pitch_strength": args.pitch_strength,
        })

    backing = np.zeros_like(vocal)
    for i, (s, e, p, v, tr) in enumerate(bnotes):
        ph = {"contour": np.zeros(24), "center": p, "dur": min(max(e - s, 0.35), 2.0),
              "n_notes": 1, "notes": [(s, e, p, v)], "start": s, "vel": v}
        _, clip = choose_clip(cands, ph, None, top_n=24)[0]
        y, pos, shift, center = render_sample(clip, ph, t0, args.pitch_strength)
        place(backing, y, pos, gain=(v / 127.0) ** 0.8)
    active_v = np.abs(vocal) > 1e-5
    active_b = np.abs(backing) > 1e-6
    v_rms = np.sqrt(np.mean(vocal[active_v] ** 2)) if active_v.any() else 0.0
    b_rms = np.sqrt(np.mean(backing[active_b] ** 2)) if active_b.any() else 0.0
    if v_rms > 0 and b_rms > 0:
        target = 10 ** (args.backing_gain_db / 20.0)
        backing *= np.clip((v_rms * target) / b_rms, 0.0, 1.0)

    vocal_end = int((args.dur_limit + 2) * FS)
    vocal = vocal[:vocal_end]
    backing = backing[:vocal_end]
    mix = vocal + backing
    master = 0.93 / max(np.max(np.abs(mix)), 1e-9)
    vocal *= master
    backing *= master
    mix *= master

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out.with_name(out.name + "_vocal.wav"), vocal, FS)
    sf.write(out.with_name(out.name + "_backing.wav"), backing, FS)
    sf.write(out.with_suffix(".wav"), mix, FS)
    ref_dir, ref_files = write_references(out, cues, cands, len(vocal), melody, t0)
    json.dump(cues, open(out.with_suffix(".cues.json"), "w"), ensure_ascii=False, indent=1)
    _save_contours()

    active = (np.abs(vocal) > 1e-5) & (np.abs(backing) > 1e-6)
    rel_db = rms_db(backing[active]) - rms_db(vocal[active]) if active.any() else None
    shifts = [abs(c["shift_semi"]) for c in cues]
    m = {
        "version": "v1", "song": Path(args.midi).name, "palette": args.palette,
        "start": args.start, "dur_limit": args.dur_limit,
        "n_vocal_clips": len(cues),
        "vocal_clip_dur_median": round(float(np.median([c["dur"] for c in cues])), 3) if cues else None,
        "vocal_clips_per_second": round(len(cues) / max(args.dur_limit, 1e-9), 3),
        "notes_covered": int(sum(c["notes"] for c in cues)),
        "shift_abs_median_semi": round(float(np.median(shifts)), 3) if shifts else None,
        "shift_abs_max_semi": round(float(max(shifts)), 3) if shifts else None,
        "backing_notes": len(bnotes),
        "backing_tracks": sorted({x[4] for x in bnotes}),
        "backing_gain_db": args.backing_gain_db,
        "pitch_strength": args.pitch_strength,
        "backing_vs_vocal_active_db": round(float(rel_db), 2) if rel_db is not None else None,
        "sample_rate": FS,
        "reference_clips": len(ref_files),
        "reference_dir": ref_dir.name,
        "device": args.device,
        "candidate_pool": args.candidate_pool,
    }
    m.update(melody_accuracy(vocal, melody, t0, FS))
    json.dump(m, open(out.with_suffix(".metrics.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(m, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
