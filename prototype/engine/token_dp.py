#!/usr/bin/env python3
"""v7: strict token formulation (agreed with the user).

Codebook C : raw lines of ONE character (Soyo), never pitch-shifted or time-stretched.
Token z_i  : (clip k, start tau, crop [a, b] at syllable boundaries keeping >= rho of the
             line's sounding length, constant gain g). Leading/trailing silence trim is free.
Decoder    : y = sum_i g_i * c_k[a:b] shifted to tau, 5 ms fades only.
Constraints: tokens ordered in time; overlap <= min(0.1 s, 10% of the token).
Loss       : per 10 ms frame of the ORIGINAL vocal
               pitch   capped |cents| on frames voiced in both (glides weigh less)
               loudness |dB residual| after the closed-form token gain
               voicing  sounding over silence / leaving sung frames uncovered
               vowel    (optional) WavLM-unit distance (k-means codes of layer-6 features)
             + lambda_N * N + lambda_crop * sum(crop fraction)
Solver     : segmental Viterbi DP over time (exact for this additive loss); candidate token
             costs computed on the GPU. lambda_N is swept without recomputing candidates.
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
import mosaic as MO  # noqa: E402

FS = 32000
SR16 = 16000
HOP = 160
DT = HOP / SR16
DEV = SG.DEV
CACHE = HERE.parent / "materials" / "contour_cache"


# ------------------------------------------------------------ vowel units (WavLM k-means)
def wavlm_frames(y16, model, layer=6, chunk=20 * SR16):
    feats = []
    for i in range(0, len(y16), chunk):
        x = torch.from_numpy(y16[i:i + chunk]).float()[None].to(DEV)
        if x.shape[1] < 800:
            x = torch.nn.functional.pad(x, (0, 800 - x.shape[1]))
        with torch.no_grad():
            h = model(x, output_hidden_states=True).hidden_states[layer][0]
        feats.append(h.float().cpu().numpy())
    return np.concatenate(feats)


def vowel_codes(target_y16, clips, tag, k=64):
    path = CACHE / f"wavlm_codes_{tag}_k{k}.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    from transformers import WavLMModel
    from sklearn.cluster import MiniBatchKMeans
    model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(DEV).eval()
    print("WavLM features: target ...", flush=True)
    ft = wavlm_frames(target_y16, model)
    fc = {}
    print(f"WavLM features: {len(clips)} clips ...", flush=True)
    for c in clips:
        y = SM.clip_audio(c["path"])
        fc[c["path"]] = wavlm_frames(librosa.resample(y.astype(np.float32), orig_sr=FS, target_sr=SR16), model)
    # per-source mean/var normalisation removes most of the speaker / channel offset
    mt, sdt = ft.mean(0), ft.std(0) + 1e-6
    allc = np.concatenate(list(fc.values()))
    mc, sdc = allc.mean(0), allc.std(0) + 1e-6
    zt = (ft - mt) / sdt
    zc = {p: (f - mc) / sdc for p, f in fc.items()}
    rng = np.random.default_rng(0)
    pool = np.concatenate([zt[rng.choice(len(zt), min(len(zt), 40000), replace=False)],
                           np.concatenate(list(zc.values()))[rng.choice(len(allc), min(len(allc), 40000), replace=False)]])
    km = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=4096, n_init=3).fit(pool)
    cen = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    dtab = (1 - cen @ cen.T).astype(np.float32)
    res = {"target": km.predict(zt).astype(np.int16),
           "clips": {p: km.predict(z).astype(np.int16) for p, z in zc.items()},
           "dtab": dtab}
    pickle.dump(res, open(path, "wb"))
    return res


def up2(lab, n):
    """50 Hz codes -> 100 Hz frames."""
    out = np.repeat(lab, 2)[:n]
    return np.pad(out, (0, max(0, n - len(out))), mode="edge")


# ------------------------------------------------------------ attacks (key frames)
def impulse_train(frames, heights, n, sigma=2.0):
    x = np.zeros(n, dtype=np.float32)
    for f, h in zip(frames, heights):
        if 0 <= f < n:
            x[f] = max(x[f], h)
    k = np.arange(-int(4 * sigma), int(4 * sigma) + 1)
    g = np.exp(-0.5 * (k / sigma) ** 2)
    return np.minimum(np.convolve(x, g, mode="same"), 1.0).astype(np.float32)


def onset_peaks(y16):
    env = librosa.onset.onset_strength(y=y16, sr=SR16, hop_length=HOP)
    pk = librosa.onset.onset_detect(onset_envelope=env, sr=SR16, hop_length=HOP, units="frames",
                                    backtrack=False, wait=5, delta=0.07)
    return pk, (env[pk] if len(pk) else np.zeros(0))


def pitch_steps(sm, voiced, thr=0.8, gap=8):
    out, last = [], -99
    for i in range(4, len(sm) - 4):
        if not (voiced[i - 4:i].all() and voiced[i:i + 4].all()):
            continue
        d = abs(np.median(sm[i:i + 4]) - np.median(sm[i - 4:i]))
        if d > thr and i - last >= gap:
            out.append(i)
            last = i
    return np.array(out, dtype=int)


def clip_attacks(clips, contours, tag):
    path = CACHE / f"clip_onsets_{tag}.pkl"
    have = pickle.load(open(path, "rb")) if path.exists() else {}
    todo = [c for c in clips if c["path"] in contours and c["path"] not in have]
    if todo:
        print(f"onsets for {len(todo)} clips ...", flush=True)
    for c in todo:
        y = SM.clip_audio(c["path"])
        have[c["path"]] = onset_peaks(librosa.resample(y.astype(np.float32), orig_sr=FS, target_sr=SR16))
    if todo:
        pickle.dump(have, open(path, "wb"))
    allh = np.concatenate([h for _, h in have.values() if len(h)])
    ref = np.percentile(allh, 90)
    peaks = {p: v[0] for p, v in have.items()}
    trains = {p: impulse_train(v[0], np.clip(v[1] / ref, 0.3, 1.0), len(contours[p][0]))
              for p, v in have.items() if p in contours}
    return trains, peaks


def _rs(x, l1, nearest=False):
    l0 = len(x)
    src = np.linspace(0, l0 - 1, l1)
    if nearest:
        return x[np.clip(np.round(src).astype(int), 0, l0 - 1)]
    return np.interp(src, np.arange(l0), x.astype(float))


# ------------------------------------------------------------ tokens
def build_tokens(clips, contours, rho, max_len, pad=0.06, codes=None, attacks=None, peaks=None,
                 primary=None, shifts=(0,), low_cap=63.0, char_cost=0.0, w_shift=0.0, low_chars_cap=63.0):
    toks = []
    pf = int(pad / DT)
    for c in clips:
        if c["path"] not in contours:
            continue
        f0, per, rms = contours[c["path"]]
        n = len(f0)
        att = attacks[c["path"]] if attacks is not None else np.zeros(n, dtype=np.float32)
        is_primary = primary is None or c["char"] == primary
        st = 69 + 12 * np.log2(np.maximum(f0, 1e-3) / 440.0)
        thr = max(float(rms.max()) * 0.08, 1e-4)
        v = (per > 0.55) & (rms > thr) & (f0 > 70)
        act = rms > max(float(rms.max()) * 0.03, 1e-5)
        idx = np.flatnonzero(v)
        if len(idx) < 8:
            continue
        runs, s = [], idx[0]
        for a, b in zip(idx[:-1], idx[1:]):
            if b - a > 3:
                runs.append((s, a + 1))
                s = b
        runs.append((s, idx[-1] + 1))
        aidx = np.flatnonzero(act)
        lo = max(0, min(runs[0][0], aidx[0]) - pf)
        hi = min(n, max(runs[-1][1], aidx[-1] + 1) + pf)
        sound_len = hi - lo
        cuts = [lo] + [(runs[i][1] + runs[i + 1][0]) // 2 for i in range(len(runs) - 1)
                       if runs[i + 1][0] - runs[i][1] >= 4] + [hi]
        if peaks is not None and c["path"] in peaks:
            # "start on the strong attack": a cut 20 ms before the first attack after each cut
            pk = np.asarray(peaks[c["path"]])
            extra = []
            for q in cuts[:-1]:
                nx = pk[pk > q + 3]
                if len(nx) and nx[0] - 2 < hi - 10:
                    extra.append(int(nx[0]) - 2)
            cuts = sorted(set(cuts + extra))
        db = 20 * np.log10(rms + 1e-9)
        db = db - np.percentile(db[v], 90)
        lab = up2(codes["clips"][c["path"]], n) if codes else None
        for i in range(len(cuts)):
            for j in range(len(cuts) - 1, i, -1):
                a, b = cuts[i], cuts[j]
                if b - a < rho * sound_len:
                    break
                if (b - a) * DT > max_len or v[a:b].mean() < 0.2:
                    continue
                base = {"path": c["path"], "char": c["char"], "text": c.get("text", ""), "a": a, "b": b,
                        "la": lo, "lb": hi,
                        "crop": 1 - (b - a) / sound_len, "line_len": round(sound_len * DT, 3)}
                med = float(np.median(st[a:b][v[a:b]]))
                for sh in shifts:
                    if not is_primary and (sh != 0 or med > low_chars_cap):
                        continue                       # other characters: only unshifted low lines
                    if sh < 0 and med + sh > low_cap:
                        continue                       # downward speed only where it buys low notes
                    if sh > 0 and med + sh < 66:
                        continue
                    r = 2 ** (sh / 12.0)
                    l1 = max(2, int(round((b - a) / r)))
                    if l1 * DT > max_len:
                        continue
                    t_ = dict(base)
                    att_r = _rs(att[a:b], l1)
                    head = att_r[:max(4, min(l1, 25))]
                    fa = int(np.argmax(head)) if head.max() > 0.5 else int(np.argmax(_rs(v[a:b].astype(float), l1) > 0.5))
                    offs = np.flatnonzero(att_r > 0.4)
                    if len(offs):
                        ramp = np.minimum(np.abs(np.arange(l1)[:, None] - offs[None]).min(1), 15) / 15.0
                    else:
                        ramp = np.ones(l1)
                    t_.update({"shift": sh, "st": _rs(st[a:b], l1) + sh, "fa": fa,
                               "v": _rs(v[a:b].astype(float), l1) > 0.5, "db": _rs(db[a:b], l1),
                               "att": att_r.astype(np.float32), "ramp": ramp.astype(np.float32),
                               "lab": _rs(lab[a:b], l1, nearest=True) if lab is not None else None,
                               "pen": (0.0 if is_primary else char_cost) + w_shift * abs(sh)})
                    toks.append(t_)
    return toks


def token_tensors(toks, lmax, with_lab):
    n = len(toks)
    U = torch.zeros((n, lmax))
    V = torch.zeros((n, lmax), dtype=torch.bool)
    D = torch.zeros((n, lmax))
    LB = torch.zeros((n, lmax), dtype=torch.long)
    A = torch.zeros((n, lmax))
    R = torch.ones((n, lmax))
    L = torch.zeros(n, dtype=torch.long)
    for k, t in enumerate(toks):
        l = len(t["st"])
        A[k, :l] = torch.from_numpy(t["att"])
        R[k, :l] = torch.from_numpy(t.get("ramp", np.ones(l, dtype=np.float32)))
        U[k, :l] = torch.from_numpy(t["st"].astype(np.float32))
        V[k, :l] = torch.from_numpy(t["v"])
        D[k, :l] = torch.from_numpy(t["db"].astype(np.float32))
        if with_lab:
            LB[k, :l] = torch.from_numpy(t["lab"].astype(np.int64))
        L[k] = l
    return U, V, D, LB, L, A, R


def candidates(T, TV, TD, TL, W, TA, starts, toks, dtab, args):
    lmax = max(len(t["st"]) for t in toks)
    with_lab = dtab is not None
    U, V, D, LB, L, A, R = [x.to(DEV) for x in token_tensors(toks, lmax, with_lab)]
    crop = torch.tensor([float(t["crop"]) for t in toks], dtype=torch.float32, device=DEV)
    pen = torch.tensor([float(t["pen"]) for t in toks], dtype=torch.float32, device=DEV)
    fa = torch.tensor([int(t["fa"]) for t in toks], dtype=torch.long)
    KM = getattr(args, "_keymask", None)
    KMp = torch.cat([torch.from_numpy(KM), torch.zeros(lmax + 64, dtype=torch.bool)]) if KM is not None else None
    WKA = getattr(args, "_wka", None)
    WKAp = torch.cat([torch.from_numpy(WKA), torch.zeros(lmax + 64)]) if WKA is not None else None
    inlen = torch.arange(lmax, device=DEV)[None] < L[:, None]
    Vu = (V & inlen)[None]
    pad = lambda x, z: torch.cat([x, torch.full((lmax,), z, dtype=x.dtype)])
    Tp, TVp, TDp, TLp, Wp, TAp = pad(T, 0.), pad(TV, False), pad(TD, -60.), pad(TL, 0), pad(W, 0.), pad(TA, 0.)
    j = torch.arange(lmax)
    dt_ = torch.from_numpy(dtab).to(DEV) if with_lab else None
    out = []
    st_t = torch.tensor(starts)
    for b0 in range(0, len(starts), args.batch):
        sb = st_t[b0:b0 + args.batch]
        gi = sb[:, None] + j[None]
        Tt, TVt, TDt, Wt = (Tp[gi].to(DEV)[:, None], TVp[gi].to(DEV)[:, None],
                            TDp[gi].to(DEV)[:, None], Wp[gi].to(DEV)[:, None])
        both = Vu & TVt
        bf = both.float()
        diff = TDt - D[None]
        cnt = bf.sum(-1).clamp(min=1)
        g = ((diff * bf).sum(-1) / cnt).clamp(args.gain_min, args.gain_max)
        il = inlen[None].float()
        Cf = (args.w_pitch * Wt * torch.clamp(torch.abs(U[None] - Tt), max=args.pitch_cap) * bf
              + args.w_db * torch.clamp(torch.abs(diff - g[..., None]), max=12.0) / 6.0 * bf
              + args.w_sil * (Vu & ~TVt).float()
              + args.c_skip * (((~Vu) & inlen[None] & TVt).float() * Wt - (inlen[None] & TVt).float() * Wt))
        if with_lab:
            TLt = TLp[gi].to(DEV)[:, None]
            Cf = Cf + args.w_vowel * dt_[LB[None].expand(len(sb), -1, -1), TLt.expand(-1, LB.shape[0], -1)] * bf
        if args.w_onset > 0:
            TAt = TAp[gi].to(DEV)[:, None]
            Cf = Cf + args.w_onset * ((TAt - A[None]) ** 2) * il
        if getattr(args, "w_keyalign", 0.0) > 0:
            WKAt = WKAp[gi].to(DEV)[:, None]
            Cf = Cf + args.w_keyalign * WKAt * R[None] * il
        if not args.choke:
            c = Cf.sum(-1) + args.l_crop * crop[None] + pen[None]
            if KMp is not None:
                c = torch.where(KMp[sb[:, None] + fa[None]].to(DEV), c, torch.full_like(c, 1e9))
            kk = min(args.topk, c.shape[1])
            val, idx = torch.topk(-c, kk, dim=1)
            val, idx, gg = (-val).cpu().numpy(), idx.cpu().numpy(), torch.gather(g, 1, idx).cpu().numpy()
            Ln = L.cpu().numpy()
            for bi in range(len(sb)):
                s0 = int(sb[bi])
                out.append([(float(val[bi, q]), s0 + int(Ln[idx[bi, q]]), int(idx[bi, q]), float(gg[bi, q]))
                            for q in range(kk)])
        else:
            # ends: full length, or choked at a later key frame (keep >= choke_keep of the token)
            CS = torch.cumsum(Cf, dim=-1)
            K = args._key_offsets[b0:b0 + len(sb)].to(DEV)                    # [B, Km] offsets, 0 = none
            Lf = L.float()[None, :, None]
            Kf = K.float()[:, None, :]
            valid = (K[:, None, :] > 0) & (Kf < Lf) & (Kf >= args.choke_keep * Lf)
            idxk = (K.clamp(min=1) - 1)[:, None, :].expand(-1, CS.shape[1], -1)
            ck = torch.gather(CS, 2, idxk)                                   # [B, Nu, Km]
            frac = Kf / Lf
            ck = ck + args.l_crop * (1 - frac * (1 - crop[None, :, None])) + pen[None, :, None]
            ck = torch.where(valid, ck, torch.full_like(ck, 1e9))
            cfull = (CS[..., -1] + args.l_crop * crop[None] + pen[None])[..., None]
            allc = torch.cat([cfull, ck], dim=-1)                             # [B, Nu, 1+Km]
            if KMp is not None:
                allc = torch.where(KMp[sb[:, None] + fa[None]].to(DEV)[..., None], allc, torch.full_like(allc, 1e9))
            flat = allc.reshape(len(sb), -1)
            kk = min(args.topk, flat.shape[1])
            val, idx = torch.topk(-flat, kk, dim=1)
            val, idx = (-val).cpu().numpy(), idx.cpu().numpy()
            g_np = g.cpu().numpy()
            Ln = L.cpu().numpy()
            Kn = K.cpu().numpy()
            m1 = allc.shape[-1]
            for bi in range(len(sb)):
                s0 = int(sb[bi])
                row = []
                for q in range(kk):
                    u, e_i = divmod(int(idx[bi, q]), m1)
                    eo = int(Ln[u]) if e_i == 0 else int(Kn[bi, e_i - 1])
                    row.append((float(val[bi, q]), s0 + eo, u, float(g_np[bi, u])))
                out.append(row)
        if (b0 // args.batch) % 200 == 0:
            print(f"  candidates {b0}/{len(starts)}", flush=True)
    return out


def solve(cands, starts, n, toks, l_n, args):
    INF = 1e18
    best = np.full(n + 1, INF)
    back = [None] * (n + 1)
    best[0] = 0.0
    sidx = {s: i for i, s in enumerate(starts)}
    omax = int(args.max_overlap / DT)
    for t in range(n):
        if best[t] >= INF:
            continue
        if best[t] < best[t + 1]:
            best[t + 1], back[t + 1] = best[t], ("skip", t)
        for o in range(0, omax + 1):
            s0 = t - o
            if s0 not in sidx:
                continue
            for c, e, u, g in cands[sidx[s0]]:
                if c >= 1e8:
                    continue
                if o > 0.1 * (e - s0):
                    continue
                e = min(e, n)
                if e <= t:
                    continue
                cc = best[t] + c + l_n + args.c_skip * o
                if cc < best[e]:
                    best[e], back[e] = cc, ("tok", t, s0, u, g)
    seq, t = [], n
    while t > 0:
        bp = back[t]
        if bp[0] == "skip":
            t = bp[1]
        else:
            seq.append(bp[2:] + (t,))
            t = bp[1]
    return seq[::-1], float(best[n])


def predicted_metrics(seq, toks, TV, sm, n, keys=None):
    pst = np.full(n, np.nan)
    pv = np.zeros(n, dtype=bool)
    pa = np.zeros(n, dtype=np.float32)
    for s0, u, g, e in seq:
        t = toks[u]
        l = min(len(t["st"]), n - s0, e - s0)
        pa[s0:s0 + l] = np.maximum(pa[s0:s0 + l], t["att"][:l])
        pst[s0:s0 + l] = np.where(t["v"][:l], t["st"][:l], np.nan)
        pv[s0:s0 + l] = t["v"][:l]
    both = TV & pv
    d = np.abs(pst[both] - sm[both]) * 100
    durs = [(e - s0) * DT for s0, u, g, e in seq]
    return {"N": len(seq), "dur_median": round(float(np.median(durs)), 2) if durs else 0,
            "pitch_acc50": round(float(np.mean(d < 50)), 3) if len(d) else 0.0,
            "pitch_acc100": round(float(np.mean(d < 100)), 3) if len(d) else 0.0,
            "voicing_recall": round(float(both.sum() / max(TV.sum(), 1)), 3),
            "false_alarm": round(float((pv & ~TV).sum() / max((~TV).sum(), 1)), 3),
            "keyframe_hit_30ms": round(float(np.mean([pa[max(0, k - 3):k + 4].max() > 0.5 for k in keys])), 3)
            if keys is not None and len(keys) else None}


# ------------------------------------------------------------ render
def write_fragments(cues, voc, yv, out, tag_note):
    """逐音效碎片页：A0 完整原句（裁剪前）/ A 素材原声（裁剪后）/ B 原曲人声 / C 渲染结果。
    只需 cues + 渲染干声 + 原曲人声即可重建，可离线复用（regen_fragments.py）。"""
    frag = out.with_name(out.name + "_fragments")
    frag.mkdir(exist_ok=True)
    h = ['<!doctype html><meta charset=utf-8><title>碎片</title><style>body{font-family:-apple-system,"PingFang SC";max-width:1080px;margin:2rem auto}td,th{border:1px solid #ddd;padding:.3rem .5rem;font-size:.85rem}table{border-collapse:collapse;width:100%}audio{width:100%;height:32px}</style>',
         f'<h1>{out.name} 碎片：完整原句 / 素材原声 / 原曲这一段 / 放进去的结果</h1><p>{tag_note}。A0 是裁剪前的完整台词（素材原始音高），A 是实际使用的裁剪片段；C 与 A 只差常数音量、5 ms 淡入淡出（以及变速 token 的音高/速度）。</p>',
         '<table><tr><th>#</th><th>token</th><th>A0 完整原句（裁剪前）</th><th>A 素材原声（裁剪后）</th><th>B 原曲人声这一段</th><th>C 渲染干声这一段</th></tr>']
    for c in cues[:30]:
        y = SM.clip_audio(c["path"])
        r0 = y.astype(np.float32)
        r_ = y[int(c["src_t0"] * FS):int(c["src_t1"] * FS)].astype(np.float32)
        a, b = int(c["t0"] * FS), int(c["t1"] * FS)
        for suf, sig in (("A0", r0), ("A", r_), ("B", yv[a:b]), ("C", voc[a:b])):
            p = frag / f"{c['k']:02d}_{suf}.wav"
            sf.write(p, sig / (np.max(np.abs(sig)) + 1e-9) * 0.7, FS)
            SG.mp3(p)
        h.append(f'<tr><td>{c["k"]:02d}</td><td>{c["t0"]:.2f}–{c["t1"]:.2f}s · 增益 {c["gain_db"]:+.1f} dB · 裁掉 {round(c["crop"] * 100)}% · 整句 {c.get("line_len", 0):.1f}s'
                 + (f' · 变速 {c["shift"]:+g} 半音' if c.get("shift") else '')
                 + (' · 被掐断' if c.get("choked") else '')
                 + f'<br>{c["text"][:26]}</td>'
                 + "".join(f'<td><audio controls preload=none src="{c["k"]:02d}_{s}.mp3"></audio></td>' for s in ("A0", "A", "B", "C")) + "</tr>")
    h.append("</table>")
    (frag / "index.html").write_text("\n".join(h))


def render(seq, toks, args, stems, span, rdb, TV, sm, out, i0, i1, y16, tag_note):
    N = int((span + 3) * FS)
    voc = np.zeros(N, dtype=np.float32)
    cues = []
    for k, (s0, u, g, e) in enumerate(seq):
        t = toks[u]
        y = SM.clip_audio(t["path"])
        choked = (e - s0) < len(t["st"])
        if getattr(args, "play_full", False):
            # 完整播放：拟合照常（choke/crop 参与搜索），播放时放完整原句，拒绝一切裁剪。
            # token 帧 0 = 素材帧 a；变速比 r 下，句首 [la,a) 提前 (a-la)*DT/r 秒播出。
            r = 2 ** (t.get("shift", 0) / 12.0)
            seg = y[int(t["la"] * DT * FS):int(t["lb"] * DT * FS)].astype(np.float32).copy()
            if t.get("shift", 0):
                seg = SM._varispeed(seg, t["shift"]).astype(np.float32)
            head = (t["a"] - t["la"]) * DT / r
            f0, per, rms = MO_CONT[t["path"]]
            ref = np.percentile(rms[(per > 0.55) & (f0 > 70)], 90) if ((per > 0.55) & (f0 > 70)).any() else rms.max()
            seg *= 0.1 / (ref * np.sqrt(2) + 1e-9) * 10 ** (g / 20)
            fi = min(int(0.005 * FS), len(seg) // 4)
            seg[:fi] *= np.linspace(0, 1, fi)
            seg[-fi:] *= np.linspace(1, 0, fi)
            place_at = s0 * DT - head
            if place_at < 0:
                seg = seg[int(-place_at * FS):]
                place_at = 0.0
            SM.place(voc, seg, place_at)
            cues.append({"k": k, "t0": round(place_at, 3), "t1": round(place_at + len(seg) / FS, 3),
                         "gain_db": round(g, 1), "char": t.get("char", ""), "shift": t.get("shift", 0),
                         "choked": False, "play_full": True, "crop": 0, "fit_t1": round(e * DT, 3),
                         "line_len": t["line_len"], "text": t["text"], "path": t["path"],
                         "src_t0": round(t["la"] * DT, 3), "src_t1": round(t["lb"] * DT, 3)})
            continue
        seg = y[int(t["a"] * DT * FS):int(t["b"] * DT * FS)].astype(np.float32).copy()
        if t.get("shift", 0):
            seg = SM._varispeed(seg, t["shift"]).astype(np.float32)
        choked = (e - s0) < len(t["st"])
        if choked:
            seg = seg[:int((e - s0) * DT * FS) + int(0.005 * FS)]
        f0, per, rms = MO_CONT[t["path"]]
        ref = np.percentile(rms[(per > 0.55) & (f0 > 70)], 90) if ((per > 0.55) & (f0 > 70)).any() else rms.max()
        seg *= 0.1 / (ref * np.sqrt(2) + 1e-9) * 10 ** (g / 20)
        fi = min(int(0.005 * FS), len(seg) // 4)
        seg[:fi] *= np.linspace(0, 1, fi)
        seg[-fi:] *= np.linspace(1, 0, fi)
        SM.place(voc, seg, s0 * DT)
        cues.append({"k": k, "t0": round(s0 * DT, 3), "t1": round(e * DT, 3), "gain_db": round(g, 1),
                     "char": t.get("char", ""), "shift": t.get("shift", 0), "choked": bool(choked),
                     "crop": round(t["crop"], 2), "line_len": t["line_len"], "text": t["text"],
                     "path": t["path"], "src_t0": round(t["a"] * DT, 3), "src_t1": round(t["b"] * DT, 3)})
    from scipy.signal import fftconvolve
    wet = fftconvolve(voc, SG.reverb_ir())[:N].astype(np.float32)
    voc_w = voc + wet * (np.sqrt(np.mean(voc ** 2)) / (np.sqrt(np.mean(wet ** 2)) + 1e-9)) * 10 ** (args.wet_db / 20)
    t0s = args.start
    inst = None
    for name in {"full": ("drums", "bass", "other"), "nobass": ("drums", "other"),
                 "none": ()}[getattr(args, "backing", "full")]:
        p = stems / f"{name}.wav"
        info = sf.info(p)
        x, sr = sf.read(p, start=int(t0s * info.samplerate),
                        stop=int(min(t0s + span + 3, info.duration) * info.samplerate),
                        always_2d=True, dtype="float32")
        x = x.mean(axis=1)
        x = librosa.resample(x, orig_sr=sr, target_sr=FS) if sr != FS else x
        inst = x if inst is None else inst[:len(x)] + x[:len(inst)]
    if inst is None:
        inst = np.zeros(N, dtype=np.float32)
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
    if not (out.parent / f"{args.ref_name}_orig_mix.mp3").exists():
        refo = out.with_name(args.ref_name)
        for suf, srcp in (("_orig_mix", args.mix), ("_orig_vocal", str(stems / "vocals.wav")), ("_instrumental", None)):
            if srcp is None:
                x = inst
            else:
                info = sf.info(srcp)
                x, sr = sf.read(srcp, start=int(t0s * info.samplerate),
                                stop=int(min(t0s + span + 1.5, info.duration) * info.samplerate),
                                always_2d=True, dtype="float32")
                x = x.mean(axis=1)
                x = librosa.resample(x, orig_sr=sr, target_sr=FS) if sr != FS else x
            x = np.pad(x, (0, max(0, end - len(x))))[:end]
            p = refo.with_name(refo.name + suf + ".wav")
            sf.write(p, (x / (np.max(np.abs(x)) + 1e-9) * 0.85).astype(np.float32), FS)
            SG.mp3(p)

    frag = out.with_name(out.name + "_fragments")
    yv, _ = librosa.load(stems / "vocals.wav", sr=FS, mono=True, offset=t0s, duration=span + 2)
    write_fragments(cues, voc, yv, out, tag_note)
    json.dump(cues, open(out.parent / (out.name + ".cues.json"), "w"), ensure_ascii=False, indent=1)

    yr = librosa.resample(voc[:end], orig_sr=FS, target_sr=SR16)
    x = torch.from_numpy(yr).float().unsqueeze(0).to(DEV)
    fr, pr = torchcrepe.predict(x, SR16, HOP, 70.0, 1000.0, model="full",
                                decoder=torchcrepe.decode.viterbi, return_periodicity=True,
                                device=DEV, batch_size=1024, pad=True)
    n = len(TV)
    fr, pr = fr.squeeze(0).cpu().numpy()[:n], pr.squeeze(0).cpu().numpy()[:n]
    L = len(fr)
    rv = pr > 0.4
    obs = 69 + 12 * np.log2(np.maximum(fr, 1e-3) / 440.0)
    both = TV[:L] & rv
    d = np.abs(obs[both] - sm[:L][both]) * 100
    on_ref = librosa.onset.onset_detect(y=y16[i0 * HOP:i1 * HOP], sr=SR16, hop_length=HOP, units="time")
    on_est = librosa.onset.onset_detect(y=yr, sr=SR16, hop_length=HOP, units="time")
    lags = []
    for o in on_ref:
        dd_ = on_est - o
        if len(dd_):
            jj = int(np.argmin(np.abs(dd_)))
            if abs(dd_[jj]) < 0.15:
                lags.append(dd_[jj] * 1000)
    lags = np.array(lags) if lags else np.array([np.nan])
    r_rms = librosa.feature.rms(y=yr, frame_length=1024, hop_length=HOP)[0][:L]
    e_r = PM.smooth(20 * np.log10(r_rms + 1e-6), 51)
    e_o = PM.smooth(rdb[:len(e_r)], 51)
    mm = TV[:len(e_r)]
    bands = {}
    for lo_, hi_ in ((0, 61), (61, 68), (68, 99)):
        mb = TV[:L] & (sm[:L] >= lo_) & (sm[:L] < hi_)
        okb = mb & rv
        db_ = np.abs(obs[okb] - sm[:L][okb]) * 100
        bands[f"band_{lo_}_{hi_}"] = {"share": round(float(mb.sum() / max(TV[:L].sum(), 1)), 3),
                                      "covered": round(float(okb.sum() / max(mb.sum(), 1)), 3),
                                      "acc50": round(float(np.mean(db_ < 50)), 3) if len(db_) else 0.0}
    return cues, {
        "M3_onsets_hit_150ms": round(float(np.isfinite(lags).sum() / max(len(on_ref), 1)), 3),
        "M3_onset_lag_median_ms": round(float(np.nanmedian(lags)), 1),
        "M3_onset_within_30ms": round(float(np.nanmean(np.abs(lags) < 30) * np.isfinite(lags).sum() / max(len(on_ref), 1)), 3),
        "M3_onset_late_gt30ms": round(float(np.nanmean(lags > 30)), 3),
        "M3_onset_early_gt30ms": round(float(np.nanmean(lags < -30)), 3),
        "pitch_bands": bands,
        "M1_pitch_acc50": round(float(np.mean(d < 50)), 3),
        "M1_pitch_acc100": round(float(np.mean(d < 100)), 3),
        "M1_pitch_cents_median": round(float(np.median(d)), 1),
        "M2_voicing_recall": round(float(np.mean(rv[TV[:L]])), 3),
        "M2_voicing_false_alarm": round(float(np.mean(rv[~TV[:L]])), 3),
        "M3_onset_f1_50ms": round(US.onset_f1(on_ref, on_est), 3),
        "M5_dynamics_corr": round(float(np.corrcoef(e_r[mm], e_o[mm])[0, 1]), 3),
    }, obs, rv


MO_CONT = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True)
    ap.add_argument("--stems", required=True)
    ap.add_argument("--lib", nargs="+", default=[str(HERE.parent / "lib" / "library_anime_full.json")])
    ap.add_argument("--singer", default="长崎爽世")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur-limit", type=float, default=None)
    ap.add_argument("--rho", type=float, default=0.7)
    ap.add_argument("--max-len", type=float, default=6.0)
    ap.add_argument("--pitch-cap", type=float, default=3.0)
    ap.add_argument("--w-pitch", type=float, default=1.0)
    ap.add_argument("--w-db", type=float, default=0.3)
    ap.add_argument("--w-sil", type=float, default=0.8)
    ap.add_argument("--c-skip", type=float, default=1.2)
    ap.add_argument("--w-vowel", type=float, default=0.0)
    ap.add_argument("--w-onset", type=float, default=0.0)
    ap.add_argument("--w-keyalign", type=float, default=0.0,
                    help="penalize distance from target key frames to nearest internal attack of token")
    ap.add_argument("--keyframe-hard", action="store_true", help="first strong attack of every token on a key frame")
    ap.add_argument("--choke", action="store_true", help="a token may be cut by the next one at a key frame")
    ap.add_argument("--play-full", action="store_true",
                    help="拟合照常，播放时放完整原句（拒绝裁剪/掐断，允许自然叠加）")
    ap.add_argument("--choke-keep", type=float, default=0.4)
    ap.add_argument("--key-tol", type=int, default=1, help="frames (10 ms) of tolerance for the hard key-frame rule")
    ap.add_argument("--shifts", default="0", help="allowed varispeed shifts in semitones (floats ok), e.g. -4,-3,-2,-1.73,0")
    ap.add_argument("--w-shift", type=float, default=0.4)
    ap.add_argument("--low-chars", default="", help="extra characters, used only for unshifted low lines")
    ap.add_argument("--low-cap", type=float, default=63.0)
    ap.add_argument("--char-cost", type=float, default=1.5)
    ap.add_argument("--l-crop", type=float, default=3.0)
    ap.add_argument("--gain-min", type=float, default=-12.0)
    ap.add_argument("--gain-max", type=float, default=6.0)
    ap.add_argument("--max-overlap", type=float, default=0.1)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--lambdas", default="0,2,5,10,20,40,80")
    ap.add_argument("--render", default="", help="comma list of lambda_N values to render")
    ap.add_argument("--inst-db", type=float, default=-6.0)
    ap.add_argument("--target-mode", default="vocal", choices=["vocal", "bass", "joint"],
                    help="vocal: 主旋律; bass: 贝斯声部（升高 --accomp-octave 个八度唱）; joint: 主旋律 + 间隙贝斯填充")
    ap.add_argument("--accomp-octave", type=int, default=2)
    ap.add_argument("--gap-min", type=float, default=0.4, help="joint 模式下，人声间隙多長才填贝斯（秒）")
    ap.add_argument("--backing", default="full", choices=["full", "nobass", "none"],
                    help="混音打底：full=drums+bass+other; nobass=drums+other; none=只出干声")
    ap.add_argument("--wet-db", type=float, default=-16.0)
    ap.add_argument("--ref-name", default="v7_haruhikage")
    ap.add_argument("--version", default="v7")
    ap.add_argument("--variant", default="")
    ap.add_argument("--out", required=True, help="output name prefix, e.g. out/v7_haruhikage_soyo")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stems = Path(args.stems)

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
    sm = st.copy()
    for i in np.flatnonzero(TV):
        lo, hi = max(0, i - 3), min(len(st), i + 4)
        sm[i] = np.median(st[lo:hi][TV[lo:hi]])
    slope = np.abs(np.gradient(sm)) / DT
    W = np.where(TV, 0.25 + 0.75 * np.exp(-slope / 25.0), 0.0)
    TD = rdb - np.percentile(rdb[TV], 90)
    n = len(TV)

    # ---- 伴奏目标：贝斯轨升高 --accomp-octave 个八度（鬼畜式"哼贝斯"）----
    pk_extra = None
    if args.target_mode != "vocal":
        bt, bf0, bper, brms = SG.crepe_track(str(stems / "bass.wav"), fmin=35.0)
        bn_all = min(len(bt), n_all)
        brdb_all = 20 * np.log10(brms[:bn_all] + 1e-9)
        bf, bp, br = bf0[:bn_all][i0:i1], bper[:bn_all][i0:i1], brdb_all[i0:i1]
        BV = (bp > 0.45) & (br > np.percentile(brdb_all, 95) - 30) & (bf > 30)
        bst = 69 + 12 * np.log2(np.maximum(bf, 1e-3) / 440.0) + 12 * args.accomp_octave
        bsm = bst.copy()
        for i in np.flatnonzero(BV):
            lo, hi = max(0, i - 3), min(len(bst), i + 4)
            bsm[i] = np.median(bst[lo:hi][BV[lo:hi]])
        by16, _ = librosa.load(stems / "bass.wav", sr=SR16, mono=True)
        if args.target_mode == "bass":
            TV, sm, rdb, y16 = BV, bsm, br, by16
            W = np.where(BV, 0.25 + 0.75 * np.exp(-np.abs(np.gradient(bsm)) / DT / 25.0), 0.0)
            TD = br - np.percentile(br[BV], 90)
            print(f"bass target: voiced {BV.mean() * 100:.0f}%, sung +{args.accomp_octave} oct", flush=True)
        else:  # joint：人声长间隙填贝斯
            gap = np.zeros(n, dtype=bool)
            i = 0
            while i < n:
                if TV[i]:
                    i += 1
                    continue
                j = i
                while j < n and not TV[j]:
                    j += 1
                if (j - i) * DT >= args.gap_min:
                    gap[i:j] = True
                i = j
            JV = gap & BV
            TV = TV | JV
            sm = np.where(JV, bsm, sm)
            rdb = np.where(JV, br, rdb)
            TD = rdb - np.percentile(rdb[TV], 90)
            Wb = 0.8 * (0.25 + 0.75 * np.exp(-np.abs(np.gradient(bsm)) / DT / 25.0))
            W = np.where(JV, Wb, W)
            pk_extra = onset_peaks(by16[i0 * HOP:i1 * HOP])
            print(f"joint target: vocal + bass-filled gaps {JV.sum() * DT:.0f}s "
                  f"(voiced {TV.mean() * 100:.0f}%)", flush=True)

    lib = PM.load_library(args.lib, None, None, 0.0, 99.0, 0.15)
    SM._lib_by_path.update({c["path"]: c for c in lib})
    hires.ensure_anime_maps([c["src"] for c in lib if c["work"].startswith("anime")])
    clips = [c for c in lib if c["char"] == args.singer]
    contours = MO.clip_contours(clips, args.singer)
    trains, peaks = clip_attacks(clips, contours, args.singer)
    for ch in [x.strip() for x in args.low_chars.split(",") if x.strip()]:
        cc = [c for c in lib if c["char"] == ch]
        ct = MO.clip_contours(cc, ch)
        tr_, pk_ = clip_attacks(cc, ct, ch)
        contours.update(ct)
        trains.update(tr_)
        peaks.update(pk_)
        clips += cc
    MO_CONT.update(contours)
    codes, dtab, TL = None, None, np.zeros(n, dtype=np.int64)
    if args.w_vowel > 0:
        codes = vowel_codes(y16[:n_all * HOP], [c for c in clips if c["path"] in contours], args.singer)
        TL = up2(codes["target"], n_all)[i0:i1].astype(np.int64)
        dtab = codes["dtab"]
    toks = build_tokens(clips, contours, args.rho, args.max_len, codes=codes, attacks=trains, peaks=peaks,
                        primary=args.singer, shifts=[float(x) for x in args.shifts.split(",")],
                        low_cap=args.low_cap, char_cost=args.char_cost, w_shift=args.w_shift,
                        low_chars_cap=args.low_cap)
    # target key frames: energy onsets of the original vocal + note changes
    pk_t, h_t = onset_peaks(y16[i0 * HOP:i1 * HOP])
    if pk_extra is not None:
        allpk = np.r_[pk_t, pk_extra[0]]
        allh = np.r_[h_t, pk_extra[1]]
        order = np.argsort(allpk)
        pk_t, h_t = allpk[order].astype(int), allh[order]
    ref_h = np.percentile(h_t, 90) if len(h_t) else 1.0
    steps = pitch_steps(sm, TV)
    TA = np.maximum(impulse_train(pk_t, np.clip(h_t / ref_h, 0.3, 1.0), n),
                    impulse_train(steps, np.full(len(steps), 0.6), n))
    lens = np.array([len(x["st"]) * DT for x in toks])
    print(f"target {n * DT:.0f}s voiced {TV.mean() * 100:.0f}% | codebook: {len({x['path'] for x in toks})} lines, "
          f"{len(toks)} tokens (crops), token length median {np.median(lens):.2f}s p90 {np.percentile(lens, 90):.2f}s",
          flush=True)

    near = np.convolve(TV.astype(float), np.ones(61), mode="same") > 0
    keys = np.unique(np.r_[pk_t, steps]).astype(int)
    if args.w_keyalign > 0:
        wka = np.zeros(n, dtype=np.float32)
        for k_ in keys:
            wka[max(0, k_ - 5):k_ + 6] = 1.0
        args._wka = wka
    if args.keyframe_hard:
        km = np.zeros(n, dtype=bool)
        for k_ in keys:
            km[max(0, k_ - args.key_tol):k_ + args.key_tol + 1] = True
        args._keymask = km
        fas = sorted({t["fa"] for t in toks})
        cand_s = set()
        for k_ in keys:
            for f_ in fas:
                for d_ in range(-args.key_tol, args.key_tol + 1):
                    if 0 <= k_ + d_ - f_ < n:
                        cand_s.add(k_ + d_ - f_)
        starts = sorted(s_ for s_ in cand_s if near[s_])
    else:
        key = np.convolve((TA > 0.5).astype(float), np.ones(9), mode="same") > 0
        starts = [int(s) for s in range(0, n) if near[s] and (s % args.stride == 0 or key[s])]
    if args.choke:
        lmax_ = max(len(t["st"]) for t in toks)
        ko = []
        for s_ in starts:
            rel = keys[(keys > s_) & (keys <= s_ + lmax_)] - s_
            ko.append(rel[:32])
        km_ = max(1, max(len(r) for r in ko))
        args._key_offsets = torch.zeros((len(starts), km_), dtype=torch.long)
        for i_, r in enumerate(ko):
            args._key_offsets[i_, :len(r)] = torch.from_numpy(r.astype(np.int64))
    print(f"{len(toks)} tokens ({sum(t['shift'] != 0 for t in toks)} speed-shifted, "
          f"{sum(t['char'] != args.singer for t in toks)} other-character); {len(pk_t)} onsets + {len(steps)} note "
          f"changes as key frames; {len(starts)} start times", flush=True)
    cands = candidates(torch.from_numpy(np.where(TV, sm, 0).astype(np.float32)), torch.from_numpy(TV),
                       torch.from_numpy(TD.astype(np.float32)), torch.from_numpy(TL), torch.from_numpy(W.astype(np.float32)),
                       torch.from_numpy(TA.astype(np.float32)), starts, toks, dtab, args)

    sweep = []
    sols = {}
    for l_n in [float(x) for x in args.lambdas.split(",")]:
        seq, cost = solve(cands, starts, n, toks, l_n, args)
        m = predicted_metrics(seq, toks, TV, sm, n, keys=np.unique(np.r_[pk_t, steps]).astype(int))
        m["lambda_N"] = l_n
        m["distinct_lines"] = len({toks[u]["path"] for _, u, _, _ in seq})
        m["crop_mean"] = round(float(np.mean([toks[u]["crop"] for _, u, _, _ in seq])), 3) if seq else 0
        sweep.append(m)
        sols[l_n] = seq
        print(f"  lambda_N={l_n:>5}: {m}", flush=True)
    json.dump(sweep, open(out.with_name(out.name + "_sweep.json"), "w"), ensure_ascii=False, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "DejaVu Sans"]
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    xs = [m["N"] for m in sweep]
    ax.plot(xs, [m["pitch_acc50"] for m in sweep], "o-", label="音高 ±50 音分（两边都有声的帧）")
    ax.plot(xs, [m["voicing_recall"] for m in sweep], "s-", label="该唱的地方在唱")
    ax.plot(xs, [m["false_alarm"] for m in sweep], "^-", label="不该唱却在响")
    for m in sweep:
        ax.annotate(f"λ={m['lambda_N']:g}\n{m['dur_median']}s", (m["N"], m["pitch_acc50"]), fontsize=7,
                    textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel("token 数 N（用了多少个音效）")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"{out.name}: token 数 vs 还原度（预测值，按 λ_N 扫描）")
    plt.tight_layout()
    dd = out.parent / "diagnostics"
    dd.mkdir(exist_ok=True)
    plt.savefig(dd / f"{out.name}_tradeoff.png", dpi=110)

    for l_n in [float(x) for x in args.render.split(",") if x]:
        seq = sols[l_n]
        name = out.with_name(f"{out.name}_L{l_n:g}")
        note = (f"λ_N={l_n:g}，{len(seq)} 个 token，素世原声整句（只在音节边界裁剪，保留 ≥{round(args.rho * 100)}%），"
                f"不变调不变速，只有常数增益" + ("，含 WavLM 元音匹配项" if args.w_vowel > 0 else ""))
        cues, mr, obs, rv = render(seq, toks, args, stems, n * DT, rdb, TV, sm, name, i0, i1, y16, note)
        pm = [m for m in sweep if m["lambda_N"] == l_n][0]
        metrics = {"version": args.version, "song": Path(args.mix).stem, "singer": args.singer, "lambda_N": l_n,
                   "w_vowel": args.w_vowel, "w_onset": args.w_onset, "w_keyalign": getattr(args, "w_keyalign", 0.0), "shifts": args.shifts,
                   "w_pitch": args.w_pitch, "c_skip": args.c_skip, "topk": args.topk,
                   "low_chars": args.low_chars, "variant": args.variant,
                   "n_shifted": sum(1 for c in cues if c["shift"]), "n_choked": sum(1 for c in cues if c["choked"]),
                   "keyframe_hard": args.keyframe_hard, "choke": args.choke, "play_full": bool(getattr(args, "play_full", False)), "n_other_char": sum(1 for c in cues if c["char"] != args.singer), "rho": args.rho, "window": [args.start, round(args.start + n * DT, 1)],
                   "n_tokens": len(seq), "token_dur_median": pm["dur_median"],
                   "distinct_lines": pm["distinct_lines"], "crop_mean": pm["crop_mean"],
                   "gain_db_median_abs": round(float(np.median(np.abs([c["gain_db"] for c in cues]))), 1),
                   "processing": "none (constant gain + 5 ms fades)", **mr, "inst_db": args.inst_db,
                   "ref_name": args.ref_name}
        json.dump(metrics, open(name.parent / (name.name + ".metrics.json"), "w"), ensure_ascii=False, indent=1)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
        show = min(n * DT, 50.0)
        k1 = int(show / DT)
        tt = np.arange(n) * DT
        fig, ax = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
        ax[0].plot(tt[:k1], np.where(TV[:k1], sm[:k1], np.nan), color="tab:blue", lw=1.2, label="原曲人声")
        ax[0].plot(tt[:k1], np.where(rv[:k1], obs[:k1], np.nan), color="tab:red", lw=0.9, alpha=0.8, label="渲染")
        ax[0].legend(loc="upper right")
        ax[0].set_ylim(np.percentile(sm[TV], 1) - 3, np.percentile(sm[TV], 99) + 3)
        ax[0].set_title(f"{name.name}: 音高 原曲（蓝）vs 渲染（红）")
        for c in cues:
            if c["t0"] < show:
                ax[1].add_patch(plt.Rectangle((c["t0"], 0), c["t1"] - c["t0"], 1, color=f"C{c['k'] % 10}", alpha=0.5))
                ax[1].text(c["t0"], 1.04, f"{c['gain_db']:+.0f}dB", fontsize=7)
        ax[1].set_ylim(0, 1.3)
        ax[1].set_xlim(0, show)
        ax[1].set_title("token 放置（色块）与常数增益")
        plt.tight_layout()
        plt.savefig(dd / f"{name.name}_placement.png", dpi=100)
        plt.close("all")


if __name__ == "__main__":
    main()
