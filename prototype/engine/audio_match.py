#!/usr/bin/env python3
"""v2 renderer: match voice clips to the ORIGINAL RECORDING, not the MIDI score.

The original song's vocal stem provides a continuous melody contour (F0 track
+ energy envelope). Voice clips are matched to 1-4 s phrase segments of that
contour and guided toward the continuous target, so the render keeps the
original's duty cycle instead of collapsing into point-like MIDI notes.

Backing is a quiet synth following the original bass stem contour (-10 dB).
No original audio is mixed into the render itself.
"""
import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import pyworld as pw
import soundfile as sf

try:
    import torch
    import torchcrepe
except Exception:  # pragma: no cover
    torch = None
    torchcrepe = None

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
import hires  # noqa: E402
import phrase_match as PM  # noqa: E402  reuse library loading + clip contour cache

FS = 32000
CREPE_SR = 16000
HOP = 160                      # 10 ms at 16 kHz
MAX_RATE = 1.35
CACHE_DIR = HERE.parent / "materials" / "contour_cache"


# ------------------------------------------------------------ contour extraction
def extract_contour(path, fmin=50.0, fmax=1000.0, chunk_s=30.0):
    """CREPE F0 + periodicity + RMS for a long audio file; cached as npz."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = f"{Path(path).parent.parent.name}_{Path(path).stem}_{int(fmin)}_{int(fmax)}"
    cache = CACHE_DIR / f"{key}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["times"], z["f0"], z["per"], z["rms"]
    y, _ = librosa.load(path, sr=CREPE_SR, mono=True)
    n = len(y)
    f0_all, per_all = [], []
    use_gpu = torch is not None and torchcrepe is not None and torch.backends.mps.is_available()
    dev = "mps" if use_gpu else "cpu"
    print(f"contour {Path(path).name}: {n / CREPE_SR:.1f}s on {dev}", flush=True)
    step = int(chunk_s * CREPE_SR)
    for i in range(0, n, step):
        seg = y[i:i + step]
        x = torch.from_numpy(seg).float().unsqueeze(0).to(dev)
        f0, per = torchcrepe.predict(
            x, CREPE_SR, HOP, fmin, fmax, model="full",
            decoder=torchcrepe.decode.argmax, return_periodicity=True,
            device=dev, batch_size=1, pad=False)
        if dev == "mps":
            torch.mps.synchronize()
        f0_all.append(f0.squeeze(0).cpu().numpy())
        per_all.append(per.squeeze(0).cpu().numpy())
    f0 = np.concatenate(f0_all)
    per = np.concatenate(per_all)
    times = np.arange(len(f0)) * HOP / CREPE_SR
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0]
    rms = np.interp(times, np.arange(len(rms)) * HOP / CREPE_SR, rms)
    np.savez(cache, times=times, f0=f0, per=per, rms=rms)
    return times, f0, per, rms


def extract_melody(mix_path, stem_path, fmin=150.0, fmax=800.0):
    """Predominant melody (Melodia) on the mix + stem energy envelope.

    Melodia tracks the lead line through polyphonic accompaniment, unlike a
    monophonic tracker on a leaky vocal stem. Returns the same tuple shape as
    extract_contour, with Melodia voicing confidence in place of periodicity.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = f"melodia_{Path(mix_path).stem}_{int(fmin)}_{int(fmax)}"
    cache = CACHE_DIR / f"{key}.npz"
    if cache.exists():
        z = np.load(cache)
        times, f0, conf = z["times"], z["f0"], z["conf"]
    else:
        import essentia.standard as es
        y, _ = librosa.load(mix_path, sr=44100, mono=True)
        ext = es.PredominantPitchMelodia(frameSize=2048, hopSize=128, sampleRate=44100,
                                         minFrequency=fmin, maxFrequency=fmax,
                                         voicingTolerance=0.2, voiceVibrato=True,
                                         filterIterations=3)
        f0, conf_raw = ext(y)
        times = np.arange(len(f0)) * 128 / 44100.0
        conf = np.interp(times, np.linspace(0, times[-1], len(conf_raw)), conf_raw)
        np.savez(cache, times=times, f0=f0, conf=conf)
        print(f"melody {Path(mix_path).name}: {times[-1]:.1f}s melodia", flush=True)
    ys, _ = librosa.load(stem_path, sr=CREPE_SR, mono=True)
    rms = librosa.feature.rms(y=ys, frame_length=1024, hop_length=HOP)[0]
    rms = np.interp(times, np.arange(len(rms)) * HOP / CREPE_SR, rms)
    return times, f0, conf, rms


# ------------------------------------------------------------ phrase segmentation
def voiced_runs(times, voiced, min_gap=0.22):
    """Voiced frame runs; unvoiced gaps >= min_gap are hard boundaries."""
    idx = np.where(voiced)[0]
    runs = []
    if len(idx) == 0:
        return runs
    s = idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if times[b] - times[a] > min_gap:
            runs.append((s, a))
            s = b
    runs.append((s, idx[-1]))
    return runs


def segment_phrases(times, f0, per, rms, min_dur=1.0, max_dur=4.2, target=2.9, vth=0.45):
    """Segment continuous contour into 1-4 s phrases on the original timeline."""
    voiced = (per > vth) & (f0 > 40)
    runs = voiced_runs(times, voiced)
    phrases = []
    for a, b in runs:
        dur = times[b] - times[a]
        if dur < 0.3:
            continue
        if dur <= max_dur:
            phrases.append((a, b))
            continue
        # DP split long run: prefer target length + low-energy split points
        frames = np.arange(a, b + 1)
        n = len(frames)
        e_smooth = PM.smooth(rms[a:b + 1], 15)
        norm = e_smooth / (e_smooth.max() + 1e-9)
        dp = [0.0] + [float("inf")] * n
        back = [0] * (n + 1)
        fps = 1.0 / (times[1] - times[0])
        for i in range(1, n + 1):
            for j in range(max(0, i - int(max_dur * fps)), i):
                dur = (i - j) / fps
                if dur < min_dur and i < n:
                    continue
                cost = ((dur - target) / 1.2) ** 2
                cost += 0.8 * float(np.mean(norm[max(j - 2, 0):j + 3])) if 0 < j < n else 0.0
                if dp[j] + cost < dp[i]:
                    dp[i], back[i] = dp[j] + cost, j
        segs, i = [], n
        while i > 0:
            j = back[i]
            segs.append((frames[j], frames[i - 1]))
            i = j
        phrases.extend(reversed(segs))
    out = []
    for a, b in phrases:
        dur = times[b] - times[a]
        if out and dur < min_dur and times[a] - out[-1][1] < 0.6 and \
                out[-1][1] - out[-1][0] + dur + (times[a] - out[-1][1]) <= max_dur:
            pa = np.searchsorted(times, out[-1][0])
            pb = b
            out[-1] = (times[pa], times[b])
            continue
        out.append((times[a], times[b]))
    return out


def phrase_features(times, f0, per, rms, span, vth=0.45):
    t0, t1 = span
    m = (times >= t0) & (times <= t1)
    tt = times[m] - t0
    ff = f0[m].astype(float)
    vv = (per[m] > vth) & (ff > 40)
    rr = rms[m]
    semi = np.full(len(ff), np.nan)
    semi[vv] = 69 + 12 * np.log2(ff[vv] / 440.0)
    if not np.isfinite(semi).any():
        return None
    good = np.isfinite(semi)
    semi_i = np.interp(tt, tt[good], semi[good])          # continuous, legato-filled
    semi_i = PM.smooth(semi_i, 5)
    center = float(np.nanmedian(semi))
    contour = PM.resample_seq(semi_i - center, 24) / 12.0
    return {
        "start": float(t0), "end": float(t1), "dur": float(t1 - t0),
        "center": center, "contour": contour,
        "times_rel": tt, "semi": semi_i, "voiced": vv,
        "target_rms": float(np.sqrt(np.mean(rr ** 2)) + 1e-12),
        "voiced_ratio": float(np.mean(vv)),
        "n_notes": int(max(1, np.sum(~vv[1:] & vv[:-1]))),  # syllable-ish onsets
    }


def pick_window(times, per, f0, rms, dur_limit, t_lo=0.0, t_hi=None, vth=0.45):
    voiced = ((per > vth) & (f0 > 40)).astype(float)
    weight = voiced * rms                    # ignore near-silent bleed
    t_hi = (times[-1] - dur_limit - 1) if t_hi is None else t_hi
    step = 2.0
    best, best_s = -1.0, t_lo
    win = max(1, int(dur_limit / (times[1] - times[0])))
    cs = np.concatenate([[0.0], np.cumsum(weight)])
    t = t_lo
    while t < t_hi:
        i = int(np.searchsorted(times, t))
        j = min(len(times), i + win)
        score = cs[j] - cs[i]
        if score > best:
            best, best_s = score, t
        t += step
    return best_s


# ------------------------------------------------------------ matching
def score_clip(clip, ph, prev_char):
    contour, center, voiced = PM.clip_f0(clip["path"])
    contour = np.asarray(contour, dtype=float)
    target = np.asarray(ph["contour"], dtype=float)
    shifted = contour + np.clip(ph["center"] - center, -0.25, 0.25) / 12.0
    shape_cost = float(np.mean(np.abs(shifted - target)))
    trend_cost = float(np.mean(np.abs(np.diff(shifted) - np.diff(target))))
    center_cost = abs(center - ph["center"]) / 12.0 if center else 2.0
    dur_cost = abs(np.log2(clip["dur"] / ph["dur"]))
    stability = min(clip.get("f0_iqr_cents", 999), 1200) / 1200.0
    rhythm_cost = abs(voiced - ph["voiced_ratio"])
    repeat_cost = 1.2 if clip.get("used", 0) else 0.0
    char_cost = -0.04 if prev_char == clip["char"] else 0.0
    return (2.6 * shape_cost + 0.9 * trend_cost + 1.1 * center_cost +
            0.65 * dur_cost + 0.35 * rhythm_cost + 0.15 * stability +
            repeat_cost + char_cost)


def rough_score(clip, ph):
    center_cost = abs(clip["f0_semi"] - ph["center"]) / 12.0
    dur_cost = abs(np.log2(clip["dur"] / ph["dur"]))
    stability = min(clip.get("f0_iqr_cents", 999), 1200) / 1200.0
    repeat_cost = 0.4 if clip.get("used", 0) else 0.0
    return 0.9 * center_cost + 0.8 * dur_cost + 0.15 * stability + repeat_cost


def choose_clip(cands, ph, prev_char, top_n):
    rough = sorted(((rough_score(c, ph), c) for c in cands), key=lambda x: x[0])[:max(top_n, 1)]
    scored = sorted(((score_clip(c, ph, prev_char), c) for _, c in rough), key=lambda x: x[0])
    return scored[0]


# ------------------------------------------------------------ render
def guide_pitch_continuous(y, ph, strength):
    """Blend clip F0 toward the phrase's continuous absolute F0 contour."""
    if strength <= 0 or len(y) < FS // 8:
        return y
    fp = 5.0
    f0, t = pw.dio(y, FS, f0_floor=60.0, f0_ceil=1000.0, frame_period=fp)
    f0 = pw.stonemask(y, f0, t, FS)
    sp = pw.cheaptrick(y, f0, t, FS)
    ap = pw.d4c(y, f0, t, FS)
    frames = np.arange(len(f0)) * fp / 1000.0
    scale = (len(y) / FS) / max(ph["dur"], 1e-3)
    tgt = np.interp(frames / scale, ph["times_rel"], ph["semi"])
    tgt = PM.smooth(tgt, 5)
    src = np.where(f0 > 0, 69 + 12 * np.log2(np.maximum(f0, 1e-6) / 440.0), np.nan)
    if np.isfinite(src).sum() < 4:
        return y
    src_f = np.where(np.isfinite(src), src, np.nanmedian(src))
    out_semi = src_f + strength * (tgt - src_f)
    f0_out = np.where(f0 > 0, 440.0 * 2 ** ((out_semi - 69) / 12.0), 0.0)
    return pw.synthesize(np.ascontiguousarray(f0_out), np.ascontiguousarray(sp),
                         np.ascontiguousarray(ap), FS, fp)


def render_sample(clip, ph, pitch_strength, energy_clamp_db=5.0):
    y, _ = hires.load(clip, FS)
    cur = len(y) / FS
    rate = np.clip(cur / ph["dur"], 1 / MAX_RATE, MAX_RATE)
    if abs(rate - 1) > 0.03:
        y = librosa.effects.time_stretch(y, rate=rate)
    if pitch_strength > 0:
        y = guide_pitch_continuous(y, ph, pitch_strength)
    rms = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
    y *= 10 ** (-20 / 20) / rms
    # macro dynamics: phrase-level gain from the original vocal energy
    gain = float(np.clip(ph["target_rms"] / 0.05, 10 ** (-energy_clamp_db / 20),
                         10 ** (energy_clamp_db / 20)))
    y *= gain
    g = min(int(0.008 * FS), len(y) // 4)
    if g:
        y[:g] *= np.linspace(0, 1, g)
        y[-g:] *= np.linspace(1, 0, g)
    return y.astype(np.float32), float(np.log2(rate))


def synth_backing(times, f0, per, rms, t0, t1, n):
    """Quiet synth following the original bass stem contour."""
    m = (times >= t0) & (times < t1)
    tt = times[m] - t0
    ff = f0[m].astype(float)
    vv = (per[m] > 0.4) & (ff > 30)
    rr = rms[m]
    if not vv.any():
        return np.zeros(n, dtype=np.float32)
    f_i = np.interp(np.arange(n) / FS, tt, np.where(vv, ff, np.nan),
                    left=np.nan, right=np.nan)
    good = np.isfinite(f_i)
    if not good.any():
        return np.zeros(n, dtype=np.float32)
    f_i[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), f_i[good])
    f_i = PM.smooth(f_i, 21)
    phase = np.cumsum(2 * np.pi * f_i / FS)
    tone = np.sin(phase) + 0.25 * np.sin(2 * phase)
    amp = np.interp(np.arange(n) / FS, tt, rr)
    amp = PM.smooth(amp, 31)
    amp = amp / (amp.max() + 1e-9)
    gate = np.interp(np.arange(n) / FS, tt, vv.astype(float)) > 0.5
    tone = tone * amp * gate
    sos = None
    try:
        from scipy.signal import butter, sosfilt
        sos = butter(2, 700, "lowpass", fs=FS, output="sos")
        tone = sosfilt(sos, tone)
    except Exception:
        pass
    return (tone * 0.5).astype(np.float32)


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocal-stem", required=True)
    ap.add_argument("--melody", choices=("melodia", "crepe"), default="melodia")
    ap.add_argument("--voiced-th", type=float, default=None)
    ap.add_argument("--bass-stem", default=None)
    ap.add_argument("--mix", default=None, help="original full mix, for review excerpt")
    ap.add_argument("--lib", nargs="+", required=True)
    ap.add_argument("--palette", default=None)
    ap.add_argument("--works", default=None)
    ap.add_argument("--start", type=float, default=None, help="original timeline seconds")
    ap.add_argument("--dur-limit", type=float, default=100.0)
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=4.0)
    ap.add_argument("--min-voiced", type=float, default=0.45)
    ap.add_argument("--candidate-pool", type=int, default=220)
    ap.add_argument("--pitch-strength", type=float, default=1.0)
    ap.add_argument("--backing-gain-db", type=float, default=-10.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    vth = args.voiced_th
    if args.melody == "melodia":
        if not args.mix:
            ap.error("--melody melodia requires --mix (original full mix)")
        times, f0, per, rms = extract_melody(args.mix, args.vocal_stem)
        vth = 0.15 if vth is None else vth
    else:
        times, f0, per, rms = extract_contour(args.vocal_stem, fmin=60, fmax=1000)
        vth = 0.45 if vth is None else vth
    start = args.start if args.start is not None else \
        pick_window(times, per, f0, rms, args.dur_limit, vth=vth)
    t0, t1 = float(start), float(start + args.dur_limit)
    spans = segment_phrases(times, f0, per, rms, args.min_dur, args.max_dur + 0.2, vth=vth)
    spans = [s for s in spans if s[1] > t0 and s[0] < t1]
    phs = [phrase_features(times, f0, per, rms, s, vth=vth) for s in spans]
    phs = [p for p in phs if p is not None and p["voiced_ratio"] > 0.25]
    print(f"window {t0:.1f}-{t1:.1f}s, {len(phs)} phrases, "
          f"median dur {np.median([p['dur'] for p in phs]):.2f}s", flush=True)

    cands = PM.load_library(args.lib, args.palette, args.works, args.min_dur,
                            args.max_dur, args.min_voiced)
    srcs = [s["src"] for s in cands if s["work"].startswith("anime")]
    if srcs:
        hires.ensure_anime_maps(srcs)
    pool = []
    for ph in phs:
        rough = sorted(cands, key=lambda c: rough_score(c, ph))[:args.candidate_pool]
        pool.extend(c["path"] for c in rough)
    PM.prepare_contours(pool)

    n = int((args.dur_limit + 4) * FS)
    vocal = np.zeros(n, dtype=np.float32)
    cues = []
    prev_char = None
    for ph in phs:
        best_score, clip = choose_clip(cands, ph, prev_char, args.candidate_pool)
        clip["used"] = clip.get("used", 0) + 1
        y, rate_log = render_sample(clip, ph, args.pitch_strength)
        PM.place(vocal, y, ph["start"] - t0)
        prev_char = clip["char"]
        cues.append({
            "t0": round(ph["start"] - t0, 3), "t1": round(ph["start"] - t0 + len(y) / FS, 3),
            "orig_t0": round(ph["start"], 2), "orig_t1": round(ph["end"], 2),
            "char": clip["char"], "work": clip["work"], "clip": Path(clip["path"]).name,
            "text": clip.get("text", ""), "dur": round(float(len(y) / FS), 3),
            "phrase_center": round(ph["center"], 2),
            "clip_center": round(float(PM.clip_f0(clip["path"])[1]), 2),
            "stretch_log2": round(rate_log, 3),
            "phrase_voiced_ratio": round(ph["voiced_ratio"], 3),
            "match_score": round(float(best_score), 4),
        })

    backing = np.zeros(n, dtype=np.float32)
    if args.bass_stem:
        bt, bf, bp, br = extract_contour(args.bass_stem, fmin=35, fmax=350)
        backing = synth_backing(bt, bf, bp, br, t0, t1, n)
    active_v = np.abs(vocal) > 1e-5
    active_b = np.abs(backing) > 1e-6
    v_rms = np.sqrt(np.mean(vocal[active_v] ** 2)) if active_v.any() else 0.0
    b_rms = np.sqrt(np.mean(backing[active_b] ** 2)) if active_b.any() else 0.0
    if v_rms > 0 and b_rms > 0:
        backing *= np.clip(v_rms * 10 ** (args.backing_gain_db / 20) / b_rms, 0, 1)

    end = int((args.dur_limit + 2) * FS)
    vocal, backing = vocal[:end], backing[:end]
    mix = vocal + backing
    master = 0.93 / max(np.max(np.abs(mix)), 1e-9)
    vocal *= master
    backing *= master
    mix *= master

    out = Path(args.out)
    if out.suffix == ".wav":
        out = out.with_suffix("")
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out.with_name(out.name + "_vocal.wav"), vocal, FS)
    sf.write(out.with_name(out.name + "_backing.wav"), backing, FS)
    sf.write(out.with_suffix(".wav"), mix, FS)

    # original-audio references (excerpts on the same window)
    def excerpt(src, name):
        y, sr = sf.read(src, start=int(t0 * sf.info(src).samplerate),
                        stop=int(min(t1 + 1, sf.info(src).duration) * sf.info(src).samplerate),
                        always_2d=True, dtype="float32")
        y = y.mean(axis=1)
        if sr != FS:
            y = librosa.resample(y, orig_sr=sr, target_sr=FS)
        peak = np.max(np.abs(y)) + 1e-9
        sf.write(out.with_name(out.name + name), (y / peak * 0.85).astype(np.float32), FS)
    excerpt(args.vocal_stem, "_orig_vocal_excerpt.wav")
    if args.mix:
        excerpt(args.mix, "_orig_mix_excerpt.wav")
    PM.write_references(out, cues, cands, len(vocal), [], 0.0)
    # remove the now-meaningless point-note melody reference
    stale = out.with_name(out.name + "_melody_reference.wav")
    if stale.exists():
        stale.unlink()
    json.dump(cues, open(out.with_suffix(".cues.json"), "w"), ensure_ascii=False, indent=1)

    # metrics: rendered vocal F0 vs original continuous contour
    y16 = librosa.resample(vocal, orig_sr=FS, target_sr=CREPE_SR)
    f0r = librosa.yin(y16, fmin=60, fmax=1000, sr=CREPE_SR, frame_length=1024,
                      hop_length=HOP, trough_threshold=0.08)
    tr = np.arange(len(f0r)) * HOP / CREPE_SR
    devs, tgt_voiced = [], 0
    for ph in phs:
        a = int((ph["start"] - t0) * CREPE_SR / HOP)
        b = int((ph["end"] - t0) * CREPE_SR / HOP)
        obs = f0r[max(0, a):b]
        tm = tr[max(0, a):b] - (ph["start"] - t0)
        tgt = np.interp(tm, ph["times_rel"], ph["semi"])
        tv = np.interp(tm, ph["times_rel"], ph["voiced"].astype(float)) > 0.5
        ok = tv & np.isfinite(obs) & (obs > 0)
        tgt_voiced += int(tv.sum())
        if ok.sum() > 3:
            d = np.abs((69 + 12 * np.log2(obs[ok] / 440.0)) - tgt[ok]) * 100
            devs.append(d)
    devs = np.concatenate(devs) if devs else np.array([np.nan])
    duty = float(np.mean(np.abs(vocal[:int(args.dur_limit * FS)]) > 1e-4))
    m = {
        "version": "v2", "target": "original recording (vocal stem contour)",
        "vocal_stem": Path(args.vocal_stem).name,
        "orig_window": [round(t0, 2), round(t1, 2)],
        "n_phrases": len(phs), "n_vocal_clips": len(cues),
        "phrase_dur_median": round(float(np.median([p["dur"] for p in phs])), 3),
        "target_voiced_ratio": round(float(np.mean([p["voiced_ratio"] for p in phs])), 3),
        "render_duty_cycle": round(duty, 3),
        "melody_median_abs_cents_vs_orig": round(float(np.nanmedian(devs)), 1),
        "melody_acc_50c_vs_orig": round(float(np.nanmean(devs < 50)), 3),
        "melody_acc_100c_vs_orig": round(float(np.nanmean(devs < 100)), 3),
        "backing": "synth follows original bass stem" if args.bass_stem else None,
        "backing_gain_db": args.backing_gain_db,
        "pitch_strength": args.pitch_strength,
        "stretch_log2_median": round(float(np.median([c["stretch_log2"] for c in cues])), 3),
    }
    json.dump(m, open(out.with_suffix(".metrics.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(m, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
