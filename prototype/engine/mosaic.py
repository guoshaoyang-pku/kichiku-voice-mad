#!/usr/bin/env python3
"""v6: direct placement optimisation against the original vocal. No notes, no codebook,
no vocoder.

Decision variables: a sequence of placements (clip span, start time, speed shift).
  clip span  = 1..4 consecutive syllables of a real character line (leading consonant kept)
  start time = any 20 ms grid point / onset of the original vocal
  speed shift= integer semitones via plain resampling ("tape speed"), |s| <= max-shift
Objective (summed over 10 ms frames of the original vocal):
  both voiced          : stability-weighted |unit pitch + s - original pitch| (capped)
  unit voiced, orig not: sounding over silence
  orig voiced, uncovered: skip cost
Regularisers: per-placement cost (prefer long, few clips), |shift| cost, reuse penalty
(iterated), crossfaded overlaps allowed so the line never breaks.
Solved exactly by segmental dynamic programming over time; candidate costs on the GPU.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torchcrepe

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
sys.path.insert(0, str(HERE))
import hires  # noqa: E402
import phrase_match as PM  # noqa: E402
import sampler_match as SM  # noqa: E402
import sing_melody as SG  # noqa: E402
import unit_select as US  # noqa: E402

FS = 32000
SR16 = 16000
HOP = 160
DT = HOP / SR16
DEV = SG.DEV
CACHE = HERE.parent / "materials" / "contour_cache"


# ------------------------------------------------------------ clip contours
def clip_contours(clips, tag):
    path = CACHE / f"clip_crepe_{tag}.pkl"
    have = pickle.load(open(path, "rb")) if path.exists() else {}
    todo = [c for c in clips if c["path"] not in have]
    if todo:
        print(f"CREPE on {len(todo)} clips ({DEV}) ...", flush=True)
    for k, c in enumerate(todo):
        try:
            y = SM.clip_audio(c["path"])
        except Exception:
            continue
        y16 = librosa.resample(y.astype(np.float32), orig_sr=FS, target_sr=SR16)
        if len(y16) < SR16 // 5:
            continue
        x = torch.from_numpy(y16).float().unsqueeze(0).to(DEV)
        f0, per = torchcrepe.predict(x, SR16, HOP, 70.0, 1000.0, model="full",
                                     decoder=torchcrepe.decode.viterbi, return_periodicity=True,
                                     device=DEV, batch_size=1024, pad=True)
        f0, per = f0.squeeze(0).cpu().numpy(), per.squeeze(0).cpu().numpy()
        rms = librosa.feature.rms(y=y16, frame_length=640, hop_length=HOP)[0][:len(f0)]
        have[c["path"]] = (f0[:len(rms)].astype(np.float32), per[:len(rms)].astype(np.float32),
                           rms.astype(np.float32))
        if (k + 1) % 300 == 0:
            print(f"  {k + 1}/{len(todo)}", flush=True)
            pickle.dump(have, open(path, "wb"))
    pickle.dump(have, open(path, "wb"))
    return have


def build_units(clips, contours, max_syl=4, max_len=1.8, pre=0.06):
    units = []
    for c in clips:
        if c["path"] not in contours:
            continue
        f0, per, rms = contours[c["path"]]
        st = 69 + 12 * np.log2(np.maximum(f0, 1e-3) / 440.0)
        thr = max(float(rms.max()) * 0.08, 1e-4)
        v = (per > 0.55) & (rms > thr) & (f0 > 70)
        # syllable nuclei = voiced runs (gaps <= 30 ms bridged), >= 80 ms
        idx = np.flatnonzero(v)
        if len(idx) == 0:
            continue
        runs, s = [], idx[0]
        for a, b in zip(idx[:-1], idx[1:]):
            if b - a > 3:
                runs.append((s, a + 1))
                s = b
        runs.append((s, idx[-1] + 1))
        runs = [r for r in runs if r[1] - r[0] >= 8]
        for i in range(len(runs)):
            for j in range(i, min(len(runs), i + max_syl)):
                if j > i and runs[j][0] - runs[j - 1][1] > 25:      # >250 ms pause inside
                    break
                a, b = runs[i][0], runs[j][1]
                if (b - a) * DT > max_len:
                    break
                seg_st = st[a:b].copy()
                seg_v = v[a:b].copy()
                if seg_v.mean() < 0.5:
                    continue
                units.append({"path": c["path"], "char": c["char"], "text": c.get("text", ""),
                              "a": a, "b": b, "n_syl": j - i + 1,
                              "on": max(0.0, a * DT - pre), "st": seg_st, "v": seg_v,
                              "med": float(np.median(seg_st[seg_v]))})
    return units


# ------------------------------------------------------------ cost tensors
def unit_tensor(units, shift, lmax):
    r = 2 ** (shift / 12.0)
    n = len(units)
    U = torch.zeros((n, lmax))
    V = torch.zeros((n, lmax), dtype=torch.bool)
    L = torch.zeros(n, dtype=torch.long)
    for k, u in enumerate(units):
        l0 = len(u["st"])
        l1 = max(2, int(round(l0 / r)))
        l1 = min(l1, lmax)
        src = np.linspace(0, l0 - 1, l1)
        st = np.interp(src, np.arange(l0), u["st"]) + shift
        vv = np.interp(src, np.arange(l0), u["v"].astype(float)) > 0.5
        U[k, :l1] = torch.from_numpy(st.astype(np.float32))
        V[k, :l1] = torch.from_numpy(vv)
        L[k] = l1
    return U, V, L


def candidate_costs(T, TV, W, starts, units, shifts, args):
    """Return per start: list of (cost, end_frame, unit, shift)."""
    lmax = int(args.max_len / DT) + 2
    Tp = torch.cat([T, torch.zeros(lmax)])
    TVp = torch.cat([TV, torch.zeros(lmax, dtype=torch.bool)])
    Wp = torch.cat([W, torch.zeros(lmax)])
    j = torch.arange(lmax)
    out = [[] for _ in starts]
    starts_t = torch.tensor(starts)
    for s in shifts:
        U, V, L = unit_tensor(units, s, lmax)
        U, V, L = U.to(DEV), V.to(DEV), L.to(DEV)
        inlen = (j.to(DEV)[None, :] < L[:, None])                  # [Nu, L]
        for b0 in range(0, len(starts), args.batch):
            sb = starts_t[b0:b0 + args.batch]
            gi = sb[:, None] + j[None, :]
            Tt = Tp[gi].to(DEV)[:, None, :]                        # [B,1,L]
            TVt = TVp[gi].to(DEV)[:, None, :]
            Wt = Wp[gi].to(DEV)[:, None, :]
            Vu = V[None] & inlen[None]                             # [1,Nu,L]
            both = Vu & TVt
            d = torch.clamp(torch.abs(U[None] - Tt), max=args.pitch_cap)
            c_both = (Wt * d * both).sum(-1)
            c_sil = (Vu & ~TVt).float().sum(-1) * args.w_silence
            c_hole = ((~Vu) & inlen[None] & TVt).float().sum(-1) * args.c_skip * 0.5
            cover = (inlen[None] & TVt).float() * Wt
            gain_ = (cover.sum(-1)) * args.c_skip                  # skip cost saved
            c = c_both + c_sil + c_hole - gain_ + args.w_place + args.w_shift * abs(s)
            nfr = L.float()[None].expand_as(c)
            # keep top-k per start by cost
            kk = min(args.topk, c.shape[1])
            val, idx = torch.topk(-c, kk, dim=1)
            val, idx = (-val).cpu().numpy(), idx.cpu().numpy()
            Ln = L.cpu().numpy()
            for bi in range(len(sb)):
                st0 = int(sb[bi])
                for v_, u_ in zip(val[bi], idx[bi]):
                    out[b0 + bi].append((float(v_), st0 + int(Ln[u_]), int(u_), s))
    for i in range(len(out)):
        out[i].sort()
        out[i] = out[i][:args.topk]
    return out


def solve(cands, starts, TV, W, units, penalty, args):
    n = len(TV)
    INF = 1e18
    best = np.full(n + 1, INF)
    back = [None] * (n + 1)
    best[0] = 0.0
    start_set = {s: i for i, s in enumerate(starts)}
    for t in range(n):
        if best[t] >= INF:
            continue
        # advance one frame (leave it uncovered); covered-frame savings live in candidate costs
        if best[t] < best[t + 1]:
            best[t + 1], back[t + 1] = best[t], ("skip", t)
        # allow a unit to start up to `overlap` frames before t (crossfaded overlap)
        for o in range(0, args.overlap_frames + 1):
            s0 = t - o
            if s0 not in start_set:
                continue
            for c, e, u, s in cands[start_set[s0]]:
                e = min(e, n)
                if e <= t:
                    continue
                cc = best[t] + c + penalty[units[u]["path"]]
                if o:
                    cc += args.c_skip * o
                if cc < best[e]:
                    best[e], back[e] = cc, ("unit", t, s0, u, s)
    seq, t = [], n
    while t > 0:
        bp = back[t]
        if bp[0] == "skip":
            t = bp[1]
        else:
            _, tp, s0, u, s = bp
            seq.append((s0, t, u, s))
            t = tp
    return seq[::-1], best[n]


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True)
    ap.add_argument("--stems", required=True)
    ap.add_argument("--lib", nargs="+", default=[str(HERE.parent / "lib" / "library_anime_full.json")])
    ap.add_argument("--singers", default="长崎爽世")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur-limit", type=float, default=None)
    ap.add_argument("--max-shift", type=int, default=4)
    ap.add_argument("--max-len", type=float, default=1.8)
    ap.add_argument("--pitch-cap", type=float, default=3.0)
    ap.add_argument("--c-skip", type=float, default=1.0)
    ap.add_argument("--w-silence", type=float, default=1.2)
    ap.add_argument("--w-place", type=float, default=2.0)
    ap.add_argument("--w-shift", type=float, default=0.6)
    ap.add_argument("--w-reuse", type=float, default=1.5)
    ap.add_argument("--reuse-iters", type=int, default=3)
    ap.add_argument("--stride", type=int, default=2, help="start grid in 10 ms frames")
    ap.add_argument("--overlap-frames", type=int, default=6)
    ap.add_argument("--topk", type=int, default=24)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--inst-db", type=float, default=-6.0)
    ap.add_argument("--wet-db", type=float, default=-16.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out = out.with_suffix("") if out.suffix == ".wav" else out
    out.parent.mkdir(parents=True, exist_ok=True)
    stems = Path(args.stems)

    # ---- target: original vocal, frame level
    t, f0, per, rms = SG.crepe_track(str(stems / "vocals.wav"))
    y16, _ = librosa.load(stems / "vocals.wav", sr=SR16, mono=True)
    n_all = min(len(t), len(y16) // HOP)
    i0 = int(args.start / DT)
    i1 = n_all if args.dur_limit is None else min(n_all, i0 + int(args.dur_limit / DT))
    rdb_all = 20 * np.log10(rms[:n_all] + 1e-9)
    rdb = rdb_all[i0:i1]
    f0w, perw = f0[i0:i1], per[i0:i1]
    TV = (perw > 0.45) & (rdb > np.percentile(rdb_all, 95) - 30) & (f0w > 70)
    st = 69 + 12 * np.log2(np.maximum(f0w, 1e-3) / 440.0)
    frac = st[TV] - np.round(st[TV])
    tune = float(np.angle(np.mean(np.exp(2j * np.pi * frac))) / (2 * np.pi))
    st = st - tune
    sm = st.copy()
    for i in np.flatnonzero(TV):
        lo, hi = max(0, i - 3), min(len(st), i + 4)
        sm[i] = np.median(st[lo:hi][TV[lo:hi]])
    slope = np.abs(np.gradient(sm)) / DT                        # semitones / s
    W = np.where(TV, 0.25 + 0.75 * np.exp(-slope / 25.0), 0.0)  # glides / grace notes count less
    n = len(TV)
    print(f"target: {n * DT:.0f}s, voiced {TV.mean() * 100:.0f}%, tuning {tune * 100:+.0f}c", flush=True)

    # ---- units
    singers = [s.strip() for s in args.singers.split(",")]
    lib = PM.load_library(args.lib, None, None, 0.0, 99.0, 0.15)
    SM._lib_by_path.update({c["path"]: c for c in lib})
    hires.ensure_anime_maps([c["src"] for c in lib if c["work"].startswith("anime")])
    clips = [c for c in lib if c["char"] in singers]
    contours = clip_contours(clips, "_".join(singers))
    units = build_units(clips, contours, max_len=args.max_len)
    print(f"units: {len(units)} spans (1-4 syllables) from {len({u['path'] for u in units})} lines; "
          f"median {np.median([len(u['st']) for u in units]) * DT:.2f}s", flush=True)

    on = librosa.onset.onset_detect(y=y16[i0 * HOP:i1 * HOP], sr=SR16, hop_length=HOP, units="frames",
                                    wait=6, delta=0.05)
    vstart = np.flatnonzero(TV & ~np.r_[False, TV[:-1]])
    grid = np.flatnonzero(TV)[::args.stride]
    starts = sorted(set(int(x) for x in np.r_[on, vstart, grid] if 0 <= x < n and TV[min(int(x) + 2, n - 1)]))
    Tt = torch.from_numpy(np.where(TV, sm, 0).astype(np.float32))
    TVt = torch.from_numpy(TV)
    Wt = torch.from_numpy(W.astype(np.float32))
    shifts = list(range(-args.max_shift, args.max_shift + 1))
    print(f"{len(starts)} candidate start times x {len(units)} spans x {len(shifts)} shifts", flush=True)
    cands = candidate_costs(Tt, TVt, Wt, starts, units, shifts, args)

    penalty = {u["path"]: 0.0 for u in units}
    for it in range(args.reuse_iters):
        seq, total = solve(cands, starts, TVt, Wt, units, penalty, args)
        uses = {}
        for _, _, u, _ in seq:
            uses[units[u]["path"]] = uses.get(units[u]["path"], 0) + 1
        over = sum(max(0, v - 1) for v in uses.values())
        print(f"  solve {it + 1}: {len(seq)} placements, cost {total:.0f}, repeated lines {over}", flush=True)
        for p, v in uses.items():
            if v > 1:
                penalty[p] += args.w_reuse * (v - 1)

    # ---- render: raw audio + varispeed only
    span = n * DT
    N = int((span + 3) * FS)
    voc = np.zeros(N, dtype=np.float32)
    cues = []
    ref_db = np.percentile(rdb[TV], 90)
    for k, (s0, e, u, s) in enumerate(seq):
        un = units[u]
        y = SM.clip_audio(un["path"])
        # fine shift: median residual on matched frames, kept within +-0.5 of s
        r = 2 ** (s / 12.0)
        l1 = e - s0
        src = np.linspace(0, len(un["st"]) - 1, max(2, int(round(len(un["st"]) / r))))[:l1]
        ust = np.interp(src, np.arange(len(un["st"])), un["st"]) + s
        uv = np.interp(src, np.arange(len(un["v"])), un["v"].astype(float)) > 0.5
        tseg, tv = sm[s0:s0 + len(ust)], TV[s0:s0 + len(ust)]
        m = uv[:len(tseg)] & tv
        fine = float(np.clip(np.median(tseg[m] - ust[:len(tseg)][m]), -0.5, 0.5)) if m.sum() > 3 else 0.0
        shift = s + fine
        a = int(un["on"] * FS)
        b_ = min(len(y), int((un["b"] * DT + 0.05) * FS))
        seg = SM._varispeed(y[a:b_].astype(np.float32), shift)
        pre = (un["a"] * DT - un["on"]) / 2 ** (shift / 12.0)
        act = np.abs(seg) > 1e-4
        seg = seg * (0.1 / (np.sqrt(np.mean(seg[act] ** 2)) + 1e-9)) if act.any() else seg
        tdb = float(np.mean(rdb[s0:e][TV[s0:e]])) if TV[s0:e].any() else ref_db - 10
        seg *= 10 ** (np.clip(0.7 * (tdb - ref_db), -12, 2) / 20)
        fi = min(int(0.005 * FS), len(seg) // 4)
        fo = min(int(0.03 * FS), len(seg) // 3)
        seg[:fi] *= np.linspace(0, 1, fi)
        seg[-fo:] *= np.linspace(1, 0, fo)
        t_place = s0 * DT - pre
        SM.place(voc, seg, t_place)
        cues.append({"k": k, "t0": round(s0 * DT, 3), "t1": round(e * DT, 3), "shift": round(shift, 2),
                     "char": un["char"], "n_syl": un["n_syl"], "text": un["text"], "path": un["path"],
                     "src_t0": round(un["on"], 3), "src_t1": round(un["b"] * DT + 0.05, 3),
                     "dur": round(len(seg) / FS, 3)})

    from scipy.signal import fftconvolve
    wet = fftconvolve(voc, SG.reverb_ir())[:N].astype(np.float32)
    voc_w = voc + wet * (np.sqrt(np.mean(voc ** 2)) / (np.sqrt(np.mean(wet ** 2)) + 1e-9)) * 10 ** (args.wet_db / 20)
    t0s = args.start
    inst = None
    for name in ("drums", "bass", "other"):
        p = stems / f"{name}.wav"
        info = sf.info(p)
        x, sr = sf.read(p, start=int(t0s * info.samplerate),
                        stop=int(min(t0s + span + 3, info.duration) * info.samplerate),
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
    for suf, srcp in (("_orig_mix", args.mix), ("_orig_vocal", str(stems / "vocals.wav"))):
        info = sf.info(srcp)
        x, sr = sf.read(srcp, start=int(t0s * info.samplerate),
                        stop=int(min(t0s + span + 1.5, info.duration) * info.samplerate),
                        always_2d=True, dtype="float32")
        x = x.mean(axis=1)
        x = librosa.resample(x, orig_sr=sr, target_sr=FS) if sr != FS else x
        save(suf, np.pad(x, (0, max(0, end - len(x)))), norm=True)
    # placements in order, unprocessed, for "is it really her voice" checks
    gap = np.zeros(int(0.12 * FS), dtype=np.float32)
    parts = []
    for c in cues[:60]:
        y = SM.clip_audio(c["path"])
        r_ = y[int(c["src_t0"] * FS):int(c["src_t1"] * FS)].astype(np.float32)
        parts += [r_ / (np.max(np.abs(r_)) + 1e-9) * 0.6, gap]
    save("_raw_units_first60", np.concatenate(parts), norm=True) if parts else None

    frag = out.with_name(out.name + "_fragments")
    frag.mkdir(exist_ok=True)
    yv, _ = librosa.load(stems / "vocals.wav", sr=FS, mono=True, offset=t0s, duration=span + 2)
    h = ['<!doctype html><meta charset=utf-8><title>v6 碎片</title><style>body{font-family:-apple-system,"PingFang SC";max-width:980px;margin:2rem auto}td,th{border:1px solid #ddd;padding:.3rem .5rem;font-size:.85rem}table{border-collapse:collapse;width:100%}audio{width:100%;height:32px}</style>',
         '<h1>v6 碎片：素材原声 / 原曲这一段 / 放进去的结果</h1><p>C 只做了整体变速（移调多少就快/慢多少），没有任何声码器处理；C 截取的是整首人声干声里这一段，包含前后相邻素材的交叠。</p>',
         '<table><tr><th>#</th><th>放置</th><th>A 素材原声</th><th>B 原曲人声</th><th>C 渲染</th></tr>']
    for c in cues[:30]:
        y = SM.clip_audio(c["path"])
        r_ = y[int(c["src_t0"] * FS):int(c["src_t1"] * FS)].astype(np.float32)
        a, b = int(max(0, c["t0"] - 0.08) * FS), int((c["t1"] + 0.08) * FS)
        for suf, sig in (("A", r_), ("B", yv[a:b]), ("C", voc[a:b])):
            p = frag / f"{c['k']:02d}_{suf}.wav"
            sf.write(p, sig / (np.max(np.abs(sig)) + 1e-9) * 0.7, FS)
            SG.mp3(p)
        h.append(f'<tr><td>{c["k"]:02d}</td><td>{c["t0"]:.2f}–{c["t1"]:.2f}s · {c["n_syl"]} 音节<br>变速 {c["shift"]:+.2f} 半音<br>{c["text"][:22]}</td>'
                 + "".join(f'<td><audio controls preload=none src="{c["k"]:02d}_{s}.mp3"></audio></td>' for s in "ABC") + "</tr>")
    h.append("</table>")
    (frag / "index.html").write_text("\n".join(h))
    json.dump(cues, open(out.with_suffix(".cues.json"), "w"), ensure_ascii=False, indent=1)

    # ---- metrics
    yr = librosa.resample(voc[:end], orig_sr=FS, target_sr=SR16)
    x = torch.from_numpy(yr).float().unsqueeze(0).to(DEV)
    fr, pr = torchcrepe.predict(x, SR16, HOP, 70.0, 1000.0, model="full",
                                decoder=torchcrepe.decode.viterbi, return_periodicity=True,
                                device=DEV, batch_size=1024, pad=True)
    fr, pr = fr.squeeze(0).cpu().numpy()[:n], pr.squeeze(0).cpu().numpy()[:n]
    L = len(fr)
    rv = pr > 0.4
    obs = 69 + 12 * np.log2(np.maximum(fr, 1e-3) / 440.0)
    both = TV[:L] & rv
    d = np.abs(obs[both] - sm[:L][both]) * 100
    dw = W[:L][both]
    on_ref = librosa.onset.onset_detect(y=y16[i0 * HOP:i1 * HOP], sr=SR16, hop_length=HOP, units="time")
    on_est = librosa.onset.onset_detect(y=yr, sr=SR16, hop_length=HOP, units="time")
    r_rms = librosa.feature.rms(y=yr, frame_length=1024, hop_length=HOP)[0][:L]
    e_r = PM.smooth(20 * np.log10(r_rms + 1e-6), 51)
    e_o = PM.smooth(rdb[:len(e_r)], 51)
    mm = TV[:len(e_r)]
    durs = np.array([c["t1"] - c["t0"] for c in cues])
    metrics = {
        "version": "v6", "song": Path(args.mix).stem, "singers": singers,
        "window": [round(args.start, 1), round(args.start + span, 1)],
        "method": "segmental DP over (clip span, start time, speed shift); raw audio + varispeed only",
        "n_placements": len(cues), "placement_dur_median": round(float(np.median(durs)), 3),
        "syllables_per_placement_mean": round(float(np.mean([c["n_syl"] for c in cues])), 2),
        "M1_pitch_acc50": round(float(np.mean(d < 50)), 3),
        "M1_pitch_acc50_stable_frames": round(float(np.sum((d < 50) * dw) / np.sum(dw)), 3),
        "M1_pitch_cents_median": round(float(np.median(d)), 1),
        "M2_voicing_recall": round(float(np.mean(rv[TV[:L]])), 3),
        "M2_voicing_false_alarm": round(float(np.mean(rv[~TV[:L]])), 3),
        "M3_onset_f1_50ms": round(US.onset_f1(on_ref, on_est), 3),
        "M5_dynamics_corr": round(float(np.corrcoef(e_r[mm], e_o[mm])[0, 1]), 3),
        "M6_vocoder": "none",
        "M6_shift_semi_median_abs": round(float(np.median(np.abs([c["shift"] for c in cues]))), 2),
        "M6_shift_gt3_ratio": round(float(np.mean(np.abs([c["shift"] for c in cues]) > 3)), 3),
        "M7_distinct_lines": len({c["path"] for c in cues}),
        "inst_db": args.inst_db,
    }
    json.dump(metrics, open(out.with_suffix(".metrics.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(metrics, ensure_ascii=False), flush=True)

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "DejaVu Sans"]
    import matplotlib.pyplot as plt
    show = min(span, 50.0)
    k1 = int(show / DT)
    tt = np.arange(n) * DT
    fig, ax = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    ax[0].plot(tt[:k1], np.where(TV[:k1], sm[:k1], np.nan), color="tab:blue", lw=1.2, label="原曲人声")
    ax[0].plot(tt[:min(k1, L)], np.where(rv[:k1], obs[:k1], np.nan), color="tab:red", lw=0.9, alpha=0.8, label="渲染")
    ax[0].legend(loc="upper right")
    ax[0].set_title("音高：原曲人声（蓝）vs 渲染（红）")
    for c in cues:
        if c["t0"] < show:
            ax[1].add_patch(plt.Rectangle((c["t0"], 0), c["t1"] - c["t0"], 1, color=f"C{c['k'] % 10}", alpha=0.5))
            ax[1].text(c["t0"], 1.05, f"{c['shift']:+.0f}", fontsize=7)
    ax[1].set_ylim(0, 1.3)
    ax[1].set_title("每段素材的放置位置（色块）与变速半音数")
    lo, hi = np.percentile(sm[TV], 1) - 2, np.percentile(sm[TV], 99) + 2
    ax[0].set_ylim(lo, hi)
    ax[1].set_xlim(0, show)
    plt.tight_layout()
    dd = out.parent / "diagnostics"
    dd.mkdir(exist_ok=True)
    plt.savefig(dd / f"{out.name}_placement.png", dpi=100)


if __name__ == "__main__":
    main()
