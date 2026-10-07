#!/usr/bin/env python3
"""v4: melody-first renderer. One singer, a small vowel codebook, PSOLA notes.

1. Transcribe the lead vocal of the original song (CREPE viterbi on the demucs vocal stem)
   into semitone-quantised notes, legato inside phrases.
2. Build a restricted codebook: K long, stable vowel nuclei of ONE character spread over the
   song's range. Every note is sung by one codebook entry (consistent timbre = clear melody).
3. Each note = natural onset (consonant) + vowel, pitch flattened to the exact note and
   lengthened with Praat PSOLA (overlap-add, formants kept), short glide from the previous
   note, notes overlapped so the line never breaks inside a phrase.
Outputs dry/wet melody, a pure-tone transcription reference, original excerpts, a quiet
bass backing (-10 dB) and a piano-roll diagnostic.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
sys.path.insert(0, str(HERE))
import hires  # noqa: E402
import phrase_match as PM  # noqa: E402
import sampler_match as SM  # noqa: E402
import audio_match as AM  # noqa: E402

FS = 32000
SR16 = 16000
HOP = 160
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
CACHE = HERE.parent / "materials" / "contour_cache"


# ------------------------------------------------------------ transcription
def crepe_track(path):
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"crepe_viterbi_{Path(path).parent.name}_{Path(path).stem}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["t"], z["f0"], z["per"], z["rms"]
    y, _ = librosa.load(path, sr=SR16, mono=True)
    f0s, pers = [], []
    step = 30 * SR16
    for i in range(0, len(y), step):
        x = torch.from_numpy(y[i:i + step]).float().unsqueeze(0).to(DEV)
        f0, per = torchcrepe.predict(x, SR16, HOP, 70.0, 1000.0, model="full",
                                     decoder=torchcrepe.decode.viterbi, return_periodicity=True,
                                     device=DEV, batch_size=1024, pad=True)
        n = (min(step, len(y) - i)) // HOP
        f0s.append(f0.squeeze(0).cpu().numpy()[:n])
        pers.append(per.squeeze(0).cpu().numpy()[:n])
    f0, per = np.concatenate(f0s), np.concatenate(pers)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP, center=True)[0][:len(f0)]
    f0, per = f0[:len(rms)], per[:len(rms)]
    t = np.arange(len(f0)) * HOP / SR16
    np.savez(cache, t=t, f0=f0, per=per, rms=rms)
    return t, f0, per, rms


def transcribe(t, f0, per, rms, per_th=0.45, min_note=0.09, split=0.75, legato=0.3,
               phrase_gap=0.45):
    rdb = 20 * np.log10(rms + 1e-9)
    loud = rdb > (np.percentile(rdb, 95) - 30)
    voiced = (per > per_th) & loud & (f0 > 70)
    st = 69 + 12 * np.log2(np.maximum(f0, 1e-3) / 440.0)
    # global tuning offset (cents) from stable frames
    frac = (st[voiced] - np.round(st[voiced]))
    tune = float(np.angle(np.mean(np.exp(2j * np.pi * frac))) / (2 * np.pi))
    stc = st - tune
    # median-smooth inside voiced runs
    sm = stc.copy()
    k = 7
    for i in np.flatnonzero(voiced):
        lo, hi = max(0, i - k // 2), min(len(sm), i + k // 2 + 1)
        w = stc[lo:hi][voiced[lo:hi]]
        sm[i] = np.median(w)
    notes = []
    i, n = 0, len(sm)
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        j = i
        cur = [sm[i]]
        while j + 1 < n and voiced[j + 1]:
            ref = np.median(cur[-15:])
            if abs(sm[j + 1] - ref) > split:
                # require the change to persist 40 ms
                ahead = sm[j + 1:j + 5][voiced[j + 1:j + 5]]
                if len(ahead) >= 3 and np.all(np.abs(ahead - ref) > split):
                    break
            j += 1
            cur.append(sm[j])
        dur = (j - i + 1) * HOP / SR16
        if dur >= min_note:
            core = np.array(cur[len(cur) // 5: max(len(cur) // 5 + 1, len(cur) * 4 // 5)])
            notes.append({"t0": float(t[i]), "t1": float(t[j] + HOP / SR16),
                          "midi": int(np.round(np.median(core))),
                          "rms": float(np.mean(rms[i:j + 1]))})
        i = j + 1
    # merge glitches: very short notes ±1 semitone stuck between same-pitch notes
    merged = []
    for nt in notes:
        if merged and nt["midi"] == merged[-1]["midi"] and nt["t0"] - merged[-1]["t1"] < 0.05:
            merged[-1]["t1"] = nt["t1"]
            continue
        merged.append(nt)
    notes = merged
    # octave-error repair against neighbours
    for a in range(len(notes)):
        nb = [notes[b]["midi"] for b in range(max(0, a - 3), min(len(notes), a + 4)) if b != a]
        if nb:
            med = np.median(nb)
            for sh in (12, -12):
                if abs(notes[a]["midi"] - sh - med) < abs(notes[a]["midi"] - med) - 6:
                    notes[a]["midi"] -= sh
    # phrases + legato
    ph = 0
    for a, nt in enumerate(notes):
        if a > 0 and nt["t0"] - notes[a - 1]["t1"] >= phrase_gap:
            ph += 1
        nt["phrase"] = ph
    for a in range(len(notes) - 1):
        gap = notes[a + 1]["t0"] - notes[a]["t1"]
        notes[a]["legato"] = bool(gap < legato and notes[a + 1]["phrase"] == notes[a]["phrase"])
        if notes[a]["legato"]:
            notes[a]["t1"] = notes[a + 1]["t0"]
    if notes:
        notes[-1]["legato"] = False
    return notes, tune


def key_estimate(notes):
    maj = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    mnr = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    h = np.zeros(12)
    for nt in notes:
        h[nt["midi"] % 12] += nt["t1"] - nt["t0"]
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    best = max(((np.corrcoef(h, np.roll(p, k))[0, 1], f"{names[k]} {q}")
                for p, q in ((maj, "major"), (mnr, "minor")) for k in range(12)))
    return best[1]


# ------------------------------------------------------------ codebook
def build_codebook(bank_path, singer, lo, hi, k, min_dur=0.22):
    items = [b for b in json.load(open(bank_path))
             if b["char"] == singer and b["per"] > 0.8 and b["cents_std"] < 50 and b["rms"] > 0.45
             and b["t1"] - b["t0"] >= min_dur]
    if not items:
        raise SystemExit(f"no nuclei for {singer}")
    for b in items:
        d = b["t1"] - b["t0"]
        b["q"] = (min(d, 0.6) / 0.6) + 0.5 * b["per"] + 0.3 * b["rms"] - b["cents_std"] / 150.0
    centers = np.linspace(lo, hi, k)
    book, used = [], set()
    for c in centers:
        cand = sorted((x for x in items if id(x) not in used),
                      key=lambda x: -(x["q"] - abs(x["st"] - c) / 2.5))
        if cand:
            used.add(id(cand[0]))
            book.append(cand[0])
    book.sort(key=lambda b: b["st"])
    return book


# ------------------------------------------------------------ PSOLA note
def psola_note(y, nuc, pre, midi, prev_midi, out_dur, glide=0.04, vibrato=True):
    a = max(0, int((nuc["t0"] - pre) * FS))
    e = min(len(y), int(nuc["t1"] * FS))
    seg = y[a:e].astype(np.float64)
    src_dur = len(seg) / FS
    on = nuc["t0"] - a / FS
    vow = max(src_dur - on, 0.05)
    snd = parselmouth.Sound(seg, sampling_frequency=FS)
    manip = call(snd, "To Manipulation", 0.005, 75, 1000)
    f_t = 440.0 * 2 ** ((midi - 69) / 12.0)
    f_p = 440.0 * 2 ** (((prev_midi if prev_midi is not None else midi) - 69) / 12.0)
    k = max((out_dur - on) / vow, 0.3)
    pt = call("Create PitchTier", "p", 0, src_dur)
    gl = min(glide / max(k, 1e-3), vow * 0.3)
    call(pt, "Add point", 0.0, f_p if prev_midi is not None else f_t)
    call(pt, "Add point", on + gl, f_t)
    if vibrato and out_dur > 0.45:
        # gentle, delayed vibrato (5.5 Hz, ±18 cents) in source time
        n_pts = 40
        for q in range(n_pts + 1):
            ts = on + gl + (src_dur - on - gl) * q / n_pts
            tout = on + (ts - on) * k
            depth = 0.18 * min(1.0, max(0.0, (tout - 0.35) / 0.3))
            call(pt, "Add point", ts, f_t * 2 ** (depth * np.sin(2 * np.pi * 5.5 * tout) / 12.0))
    else:
        call(pt, "Add point", src_dur, f_t)
    call([pt, manip], "Replace pitch tier")
    dt = call("Create DurationTier", "d", 0, src_dur)
    call(dt, "Add point", 0.0, 1.0)
    call(dt, "Add point", max(on - 0.002, 0.0), 1.0)
    call(dt, "Add point", on + 0.002, k)
    call(dt, "Add point", src_dur, k)
    call([dt, manip], "Replace duration tier")
    res = call(manip, "Get resynthesis (overlap-add)")
    w = res.values[0].astype(np.float32)
    return w, on


def reverb_ir(dur=1.1, decay=0.35, seed=0):
    rng = np.random.default_rng(seed)
    n = int(dur * FS)
    t = np.arange(n) / FS
    ir = rng.standard_normal(n) * np.exp(-t / decay)
    ir[:int(0.012 * FS)] = 0
    from scipy.signal import butter, sosfilt
    ir = sosfilt(butter(2, 5000, "lowpass", fs=FS, output="sos"), ir)
    return (ir / np.sqrt(np.sum(ir ** 2))).astype(np.float32)


def tone(midi, dur):
    t = np.arange(int(dur * FS)) / FS
    f = 440.0 * 2 ** ((midi - 69) / 12.0)
    x = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)
    env = np.minimum(1, t / 0.01) * np.minimum(1, (dur - t) / 0.03)
    return (x * np.clip(env, 0, 1) * 0.25).astype(np.float32)


def mp3(path):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-codec:a", "libmp3lame",
                    "-q:a", "2", str(path.with_suffix(".mp3"))], check=True)


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True)
    ap.add_argument("--vocal-stem", required=True)
    ap.add_argument("--bass-stem", default=None)
    ap.add_argument("--lib", nargs="+", default=[str(HERE.parent / "lib" / "library_anime_full.json")])
    ap.add_argument("--bank", default=str(HERE.parent / "materials" / "nucleus_bank.json"))
    ap.add_argument("--singer", default="长崎爽世")
    ap.add_argument("--codebook", type=int, default=16)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur-limit", type=float, default=None)
    ap.add_argument("--transpose", type=int, default=0)
    ap.add_argument("--pre", type=float, default=0.05, help="onset consonant kept before each vowel")
    ap.add_argument("--overlap", type=float, default=0.03)
    ap.add_argument("--wet-db", type=float, default=-13.0)
    ap.add_argument("--env-depth", type=float, default=0.5)
    ap.add_argument("--backing-gain-db", type=float, default=-10.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out = out.with_suffix("") if out.suffix == ".wav" else out
    out.parent.mkdir(parents=True, exist_ok=True)

    t, f0, per, rms = crepe_track(args.vocal_stem)
    notes, tune = transcribe(t, f0, per, rms)
    t0 = args.start
    t1 = args.start + args.dur_limit if args.dur_limit else t[-1]
    notes = [n for n in notes if n["t0"] >= t0 and n["t0"] < t1]
    for n in notes:
        n["midi"] += args.transpose
    key = key_estimate(notes)
    mids = np.array([n["midi"] for n in notes])
    print(f"{len(notes)} notes, {notes[-1]['phrase'] - notes[0]['phrase'] + 1} phrases, "
          f"tuning {tune * 100:+.0f} cents, key {key}, range {mids.min()}-{mids.max()}", flush=True)

    lib = PM.load_library(args.lib, None, None, 0.0, 99.0, 0.15)
    SM._lib_by_path.update({c["path"]: c for c in lib})
    hires.ensure_anime_maps([c["src"] for c in lib if c["work"].startswith("anime")])
    book = build_codebook(args.bank, args.singer, mids.min() + 1, mids.max() - 1, args.codebook)
    print(f"codebook ({args.singer}): " + ", ".join(f"{b['st']:.1f}/{b['t1'] - b['t0']:.2f}s" for b in book),
          flush=True)
    book_st = np.array([b["st"] for b in book])
    book_dur = np.array([b["t1"] - b["t0"] for b in book])

    span = t1 - t0
    n_samp = int((span + 3) * FS)
    dry = np.zeros(n_samp, dtype=np.float32)
    ref = np.zeros(n_samp, dtype=np.float32)
    cues = []
    prev_midi, prev_phrase, last_k = None, None, None
    for idx, n in enumerate(notes):
        dur = (n["t1"] - n["t0"]) + (args.overlap if n["legato"] else 0.06)
        new_phrase = n["phrase"] != prev_phrase
        cost = np.abs(book_st - n["midi"]) / 2.0 + 0.35 * np.maximum(0, np.log2(dur / book_dur) - 1)
        if last_k is not None:
            cost[last_k] += 0.15             # vary the vowel a little
        kk = int(np.argmin(cost))
        nuc = book[kk]
        y = SM.clip_audio(nuc["path"])
        pre = args.pre if (new_phrase or not notes[idx - 1]["legato"] or idx % 1 == 0) else 0.0
        w, on = psola_note(y, nuc, pre, n["midi"], None if new_phrase else prev_midi, dur)
        act = np.abs(w) > 1e-4
        w *= 0.12 / (np.sqrt(np.mean(w[act] ** 2)) + 1e-9) if act.any() else 1.0
        fi = min(int(0.004 * FS), len(w) // 4)
        fo = min(int((args.overlap if n["legato"] else 0.05) * FS), len(w) // 3)
        w[:fi] *= np.linspace(0, 1, fi)
        w[-fo:] *= np.linspace(1, 0, fo) ** 0.7
        SM.place(dry, w, n["t0"] - on - t0)
        SM.place(ref, tone(n["midi"], n["t1"] - n["t0"]), n["t0"] - t0)
        cues.append({"t0": round(n["t0"] - t0, 3), "t1": round(n["t1"] - t0, 3), "midi": n["midi"],
                     "phrase": n["phrase"], "legato": n["legato"], "code": kk,
                     "nuc_st": nuc["st"], "shift": round(n["midi"] - nuc["st"], 2),
                     "nuc_dur": round(book_dur[kk], 3), "stretch": round(dur / book_dur[kk], 2),
                     "text": nuc.get("text", ""), "nuc_path": nuc["path"], "nuc_t0": nuc["t0"],
                     "nuc_t1": nuc["t1"]})
        prev_midi, prev_phrase, last_k = n["midi"], n["phrase"], kk

    # macro dynamics from the original vocal (smoothed ~400 ms)
    m = (t >= t0) & (t < t0 + n_samp / FS)
    env = PM.smooth(rms[m], 41)
    edb = 20 * np.log10(np.maximum(env, 1e-6) / np.percentile(env[env > 1e-5], 90))
    gain = np.clip(10 ** (args.env_depth * np.maximum(edb, -12) / 20), 0.5, 1.2)
    dry *= np.interp(np.arange(n_samp) / FS, t[m] - t0, gain).astype(np.float32)
    from scipy.signal import fftconvolve
    wet = fftconvolve(dry, reverb_ir())[:n_samp].astype(np.float32)
    wet_mel = dry + wet * (np.sqrt(np.mean(dry ** 2)) / (np.sqrt(np.mean(wet ** 2)) + 1e-9)) * 10 ** (args.wet_db / 20)

    backing = np.zeros(n_samp, dtype=np.float32)
    if args.bass_stem:
        bt, bf, bp, br = AM.extract_contour(args.bass_stem, fmin=35, fmax=350)
        backing = AM.synth_backing(bt, bf, bp, br, t0, t0 + span + 3, n_samp)
        av, ab = np.abs(wet_mel) > 1e-5, np.abs(backing) > 1e-6
        if av.any() and ab.any():
            backing *= np.sqrt(np.mean(wet_mel[av] ** 2)) * 10 ** (args.backing_gain_db / 20) / \
                np.sqrt(np.mean(backing[ab] ** 2))
    mix = wet_mel + backing
    master = 0.93 / max(np.max(np.abs(mix)), 1e-9)
    end = int((span + 1.5) * FS)
    files = {"": mix, "_melody_dry": dry * master, "_melody": wet_mel * master,
             "_backing": backing * master}
    for suf, sig in files.items():
        p = out.with_name(out.name + suf + ".wav")
        sf.write(p, (sig if suf else sig * master)[:end].astype(np.float32), FS)
        mp3(p)
    p = out.with_name(out.name + "_transcription_tone.wav")
    sf.write(p, (ref / (np.max(np.abs(ref)) + 1e-9) * 0.8)[:end], FS)
    mp3(p)

    def excerpt(src, suf):
        info = sf.info(src)
        yy, sr = sf.read(src, start=int(t0 * info.samplerate),
                         stop=int(min(t0 + span + 1.5, info.duration) * info.samplerate),
                         always_2d=True, dtype="float32")
        yy = yy.mean(axis=1)
        yy = librosa.resample(yy, orig_sr=sr, target_sr=FS) if sr != FS else yy
        pth = out.with_name(out.name + suf + ".wav")
        sf.write(pth, yy / (np.max(np.abs(yy)) + 1e-9) * 0.85, FS)
        mp3(pth)
    excerpt(args.mix, "_orig_mix")
    excerpt(args.vocal_stem, "_orig_vocal")

    # codebook audition: each raw codeword, then the same codeword PSOLA'd to 3 notes
    parts, gap = [], np.zeros(int(0.15 * FS), dtype=np.float32)
    for b in book:
        y = SM.clip_audio(b["path"])
        raw = y[max(0, int((b["t0"] - args.pre) * FS)):int(b["t1"] * FS)].astype(np.float32)
        parts += [raw / (np.max(np.abs(raw)) + 1e-9) * 0.6, gap]
    p = out.with_name(out.name + "_codebook_raw.wav")
    sf.write(p, np.concatenate(parts), FS)
    mp3(p)

    # fragments: first 2 phrases, note by note: raw codeword | rendered note (dry)
    frag = out.with_name(out.name + "_fragments")
    frag.mkdir(exist_ok=True)
    rows = []
    for c in cues[:24]:
        y = SM.clip_audio(c["nuc_path"])
        raw = y[max(0, int((c["nuc_t0"] - args.pre) * FS)):int(c["nuc_t1"] * FS)].astype(np.float32)
        a, b = int(max(0, c["t0"] - args.pre) * FS), int((c["t1"] + 0.05) * FS)
        rend = (dry * master)[a:b]
        tag = f"{len(rows):02d}"
        for suf, sig in (("A_raw", raw), ("C_note", rend)):
            pth = frag / f"{tag}_{suf}.wav"
            sf.write(pth, sig / (np.max(np.abs(sig)) + 1e-9) * 0.7, FS)
            mp3(pth)
        rows.append((tag, c))
    h = ['<!doctype html><meta charset=utf-8><title>v4 碎片</title><style>body{font-family:-apple-system,"PingFang SC";max-width:860px;margin:2rem auto}td,th{border:1px solid #ddd;padding:.3rem .5rem;font-size:.85rem}table{border-collapse:collapse;width:100%}audio{width:100%;height:32px}</style>',
         '<h1>v4 碎片：码本原声 vs 渲染出的音符</h1><p>前 24 个音符。A 是码本里的原始元音（带起音辅音，未处理）；C 是 PSOLA 改到目标音高/时长后的音符（干声，不含混响）。</p>',
         '<table><tr><th>#</th><th>音符</th><th>A 原声码字</th><th>C 渲染音符</th></tr>']
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    for tag, c in rows:
        h.append(f'<tr><td>{tag}</td><td>{names[c["midi"] % 12]}{c["midi"] // 12 - 1} · {c["t1"] - c["t0"]:.2f}s<br>码字 #{c["code"]} 移调 {c["shift"]:+.1f} 半音 · 延长 ×{c["stretch"]}</td>'
                 f'<td><audio controls preload=none src="{tag}_A_raw.mp3"></audio></td><td><audio controls preload=none src="{tag}_C_note.mp3"></audio></td></tr>')
    h.append("</table>")
    (frag / "index.html").write_text("\n".join(h))

    json.dump(cues, open(out.with_suffix(".cues.json"), "w"), ensure_ascii=False, indent=1)

    # metrics: CREPE on the dry melody vs transcribed notes
    y16 = librosa.resample((dry * master)[:end].astype(np.float32), orig_sr=FS, target_sr=SR16)
    x = torch.from_numpy(y16).float().unsqueeze(0).to(DEV)
    fr, pr = torchcrepe.predict(x, SR16, HOP, 70.0, 1000.0, model="full",
                                decoder=torchcrepe.decode.viterbi, return_periodicity=True,
                                device=DEV, batch_size=1024, pad=True)
    fr, pr = fr.squeeze(0).cpu().numpy(), pr.squeeze(0).cpu().numpy()
    tr = np.arange(len(fr)) * HOP / SR16
    tgt = np.full(len(fr), np.nan)
    for c in cues:
        tgt[(tr >= c["t0"] + 0.04) & (tr < c["t1"])] = c["midi"]
    ok = np.isfinite(tgt)
    obs = 69 + 12 * np.log2(np.maximum(fr, 1e-3) / 440.0)
    sung = ok & (pr > 0.4)
    dev = np.abs(obs[sung] - tgt[sung]) * 100
    # transcription vs original vocal (cents, on voiced frames inside notes)
    to = np.interp(tr + t0, t, 69 + 12 * np.log2(np.maximum(f0, 1e-3) / 440.0) - tune)
    tv = np.interp(tr + t0, t, per) > 0.45
    tdev = np.abs(to[ok & tv] - tgt[ok & tv]) * 100
    metrics = {
        "version": "v4", "song": Path(args.mix).stem, "singer": args.singer, "codebook_size": len(book),
        "window": [round(t0, 1), round(t1, 1)], "key": key, "tuning_cents": round(tune * 100, 1),
        "n_notes": len(notes), "n_phrases": len({n["phrase"] for n in notes}),
        "note_dur_median": round(float(np.median([n["t1"] - n["t0"] for n in notes])), 3),
        "legato_ratio": round(float(np.mean([n["legato"] for n in notes])), 3),
        "codewords_used": len({c["code"] for c in cues}),
        "shift_semi_median_abs": round(float(np.median(np.abs([c["shift"] for c in cues]))), 2),
        "stretch_median": round(float(np.median([c["stretch"] for c in cues])), 2),
        "render_voiced_in_notes": round(float(sung.sum() / max(ok.sum(), 1)), 3),
        "render_cents_median": round(float(np.median(dev)), 1),
        "render_acc_50c": round(float(np.mean(dev < 50)), 3),
        "transcription_vs_orig_cents_median": round(float(np.median(tdev)), 1),
        "transcription_vs_orig_acc_50c": round(float(np.mean(tdev < 50)), 3),
    }
    json.dump(metrics, open(out.with_suffix(".metrics.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(metrics, ensure_ascii=False), flush=True)

    # piano roll diagnostic
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "DejaVu Sans"]
    import matplotlib.pyplot as plt
    show = min(span, 60.0)
    fig, ax = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    mm = (t >= t0) & (t < t0 + show)
    orig_st = np.where(per[mm] > 0.45, 69 + 12 * np.log2(np.maximum(f0[mm], 1e-3) / 440) - tune, np.nan)
    for a_ in ax:
        for c in cues:
            if c["t0"] < show:
                a_.add_patch(plt.Rectangle((c["t0"], c["midi"] - 0.4), c["t1"] - c["t0"], 0.8,
                                           color="tab:orange", alpha=0.35, lw=0))
    ax[0].plot(t[mm] - t0, orig_st, color="tab:blue", lw=1, label="原曲人声 F0")
    ax[0].set_title("转写音符（橙）vs 原曲人声 F0（蓝）")
    rs = np.where((pr > 0.4) & (tr < show), obs, np.nan)
    ax[1].plot(tr, rs, color="tab:red", lw=1, label="渲染旋律 F0")
    ax[1].set_title("转写音符（橙）vs 渲染旋律 F0（红）")
    for a_ in ax:
        a_.set_ylim(mids.min() - 3, mids.max() + 3)
        a_.set_ylabel("MIDI")
        a_.grid(alpha=0.3)
    ax[1].set_xlim(0, show)
    ax[1].set_xlabel("秒")
    plt.tight_layout()
    d = out.parent / "diagnostics"
    d.mkdir(exist_ok=True)
    plt.savefig(d / f"{out.name}_pianoroll.png", dpi=100)


if __name__ == "__main__":
    main()
