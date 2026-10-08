#!/usr/bin/env python3
"""对照学习 v2：BV1PNAcegEe7《春日影，但是用素世扣出来》vs 我们的 v10 joint。
对齐用旋律线 DTW（对变速+整曲移调都鲁棒），然后同口径比：音准/覆盖/片段/电平/铺法。
"""
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torchcrepe

HERE = Path(__file__).parent
MAT = HERE.parent / "materials"
OUT = HERE.parent / "out"
FS = 44100
HOP = 160
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def crepe(y, sr):
    x = librosa.resample(y, orig_sr=sr, target_sr=16000)
    t = torch.from_numpy(x).float()[None].to(DEV)
    f0, per = torchcrepe.predict(t, 16000, HOP, 70.0, 1000.0, model="full",
                                 return_periodicity=True, device=DEV, batch_size=2048)
    return torchcrepe.filter.median(f0, 3)[0].cpu().numpy(), per[0].cpu().numpy()


def load(p, sr=FS):
    y, _ = librosa.load(str(p), sr=sr, mono=True)
    return y


def midi(f):
    return 69 + 12 * np.log2(np.maximum(f, 1e-3) / 440.0)


def contour10(f0, per, thr=0.5):
    """10fps MIDI 轮廓，有声帧原值、无声帧线性插值 + 有效掩码。"""
    hop_s = HOP / 16000
    n10 = int(len(f0) * hop_s * 10)
    m = midi(np.interp(np.arange(n10) / 10, np.arange(len(f0)) * hop_s, f0))
    v = np.interp(np.arange(n10) / 10, np.arange(len(per)) * hop_s, per) > thr
    if v.sum() > 2:
        idx = np.flatnonzero(v)
        m = np.interp(np.arange(n10), idx, m[idx])
    return m, v


def subseq_dtw(a, b):
    """a 作为 b 的子序列（a=ref, b=orig）。返回 (cost, path_a_to_b)。"""
    na, nb = len(a), len(b)
    D = np.full((na + 1, nb + 1), np.inf, dtype=np.float32)
    C = np.abs(a[:, None] - b[None, :]).astype(np.float32)
    D[0, :] = 0.0
    D[1:, 0] = np.inf
    for i in range(1, na + 1):
        d0 = D[i - 1, :-1]          # match
        d1 = D[i - 1, 1:]           # a advance (insertion in b skipped... )
        d2 = D[i, :-1]              # b advance
        D[i, 1:] = C[i - 1] + np.minimum(np.minimum(d0, d1), d2)
    j = int(np.argmin(D[na]))
    path = np.zeros(na, dtype=int)
    i = na
    while i > 0:
        path[i - 1] = j
        opts = (D[i - 1, j], D[i - 1, j + 1] if j + 1 <= nb else np.inf, D[i, j - 1] if j > 0 else np.inf)
        k = int(np.argmin(opts))
        if k == 0:
            i, j = i - 1, j - 1
        elif k == 1:
            i = i - 1
        else:
            j = j - 1
        j = max(j, 0)
    return float(D[na, np.argmin(D[na])] / na), path


ref_mix = load(MAT / "ref_soyokiru.wav")
ref_voc = load(MAT / "stems_ft/htdemucs_ft/ref_soyokiru/vocals.wav")
ref_acc = sum(load(MAT / f"stems_ft/htdemucs_ft/ref_soyokiru/{s}.wav") for s in ("drums", "bass", "other"))
n = min(len(ref_mix), len(ref_voc), len(ref_acc))
ref_mix, ref_voc, ref_acc = ref_mix[:n], ref_voc[:n], ref_acc[:n]
orig_mix = load(MAT / "haruhikage_original.wav")
orig_voc = load(MAT / "stems_ft/htdemucs_ft/haruhikage_original/vocals.wav")

print("[1] CREPE", flush=True)
f_ref, p_ref = crepe(ref_voc, FS)
f_orig, p_orig = crepe(orig_voc, FS)
mr, vr = contour10(f_ref, p_ref, 0.45)
mo, vo = contour10(f_orig, p_orig, 0.5)
print(f"  ref 人声基频中位 {np.median(mr[vr]):.1f} MIDI，原曲人声 {np.median(mo[vo]):.1f} MIDI")

print("[2] 网格对齐：速度比 × 移调 × 偏移（chroma 互相关）", flush=True)
HP = 1024
C_r = librosa.feature.chroma_cqt(y=ref_mix, sr=FS, hop_length=HP)
C_o = librosa.feature.chroma_cqt(y=orig_mix, sr=FS, hop_length=HP)
fps = FS / HP
N_r, N_o = C_r.shape[1], C_o.shape[1]
best = None
# orig_frame = ref_frame / r + off_frames（r = 原曲速度/ref速度；ref 快则 r < 1）
for r in np.arange(0.85, 1.19, 0.02):
    M = min(int(N_r / r), N_o - 2)
    if M < 20:
        continue
    Crs = np.stack([np.interp(np.arange(M) * r, np.arange(N_r), C_r[b]) for b in range(12)])
    for sh in range(-7, 8):
        Cr = np.roll(Crs, sh, axis=0)
        num = np.zeros(N_o - M + 1)
        for b in range(12):
            num += np.correlate(C_o[b], Cr[b], mode="valid")
        k = int(np.argmax(num))
        score = float(num[k] / (np.linalg.norm(Crs) * np.linalg.norm(C_o[:, k:k + M]) + 1e-9))
        if best is None or score > best[0]:
            best = (score, r, sh, k, M)
score, r_ratio, semi, k, nfr = best
off = k / fps
dur_o = nfr / fps
print(f"  速度比 原曲/ref = {r_ratio:.3f}，移调 {semi:+d} 半音，ref 0s ≈ 原曲 {off:.1f}s，覆盖 {dur_o:.1f}s（相关 {score:.3f}）")
t_o0, t_o1 = off, off + dur_o
n10 = len(mr)
map_o = np.clip(np.round((np.arange(n10) / 10 / r_ratio + off) * 10), 0, len(mo) - 1).astype(int)
dd = (mr - semi - mo[map_o])
vv0 = vr & vo[map_o]

print("[3] 音准（网格对齐后）", flush=True)
d = dd * 100
print(f"  标杆: ±50 {np.mean(np.abs(d[vv0]) < 50) * 100:.0f}%  ±100 {np.mean(np.abs(d[vv0]) < 100) * 100:.0f}%  中位偏差 {np.median(np.abs(d[vv0])):.0f} 音分  覆盖 {vv0.sum() / max(vo[map_o].sum(), 1) * 100:.0f}%")

print("[3b] 我们 v10 joint（同一原曲窗口）", flush=True)
our = load(OUT / "v10_haruhikage_joint_L2.5_vocal_dry.wav")
f_our, p_our = crepe(our, FS)
mou, vou = contour10(f_our, p_our, 0.45)
ia, ib = int(t_o0 * 10), min(int(t_o1 * 10), len(mou) - 1)
d2 = (mou[ia:ib] - mo[ia:ib]) * 100
vv2 = vou[ia:ib] & vo[ia:ib]
print(f"  我们: ±50 {np.mean(np.abs(d2[vv2]) < 50) * 100:.0f}%  ±100 {np.mean(np.abs(d2[vv2]) < 100) * 100:.0f}%  中位偏差 {np.median(np.abs(d2[vv2])):.0f} 音分  覆盖 {vv2.sum() / max(vo[ia:ib].sum(), 1) * 100:.0f}%")

print("[4] 片段统计（能量门）", flush=True)


def segments(y):
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    act = rms > max(rms.max() * 0.06, 1e-4)
    segs, s = [], None
    for i, a in enumerate(act):
        if a and s is None:
            s = i
        elif not a and s is not None:
            if i - s > 3:
                segs.append((s * 512 / FS, i * 512 / FS))
            s = None
    if s is not None:
        segs.append((s * 512 / FS, len(act) * 512 / FS))
    return np.array([(b - a) for a, b in segs]), segs, act.mean()


d_ref, segs_ref, cov_ref = segments(ref_voc)
d_our, segs_our, cov_our = segments(our[:int(99.45 * FS)])
print(f"  标杆: {len(segs_ref)} 段, 时长 p10/50/90 = {np.percentile(d_ref, [10, 50, 90]).round(2)}s, 有声占比 {cov_ref * 100:.0f}%")
print(f"  我们: {len(segs_our)} 段, 时长 p10/50/90 = {np.percentile(d_our, [10, 50, 90]).round(2)}s, 有声占比 {cov_our * 100:.0f}%")

print("[5] 电平配比", flush=True)
va = np.abs(ref_voc) > 1e-4
r_ref = np.sqrt(np.mean(ref_voc[va] ** 2)) / (np.sqrt(np.mean(ref_acc ** 2)) + 1e-9)
inst = sum(load(MAT / f"stems_ft/htdemucs_ft/haruhikage_original/{s}.wav") for s in ("drums", "other"))
seg = inst[int(t_o0 * FS):int(t_o1 * FS)]
our_seg = our[:len(seg)]
va2 = np.abs(our_seg) > 1e-4
r_our = np.sqrt(np.mean(our_seg[va2] ** 2)) / (np.sqrt(np.mean(seg ** 2)) + 1e-9)
print(f"  标杆 人声/伴奏 = {20 * np.log10(r_ref):+.1f} dB   我们 人声/伴奏 = {20 * np.log10(r_our):+.1f} dB")

print("[6] 片段内音高跨度（一个音效跨几个音）", flush=True)


def spans(f0, per, segs):
    out = []
    hop_s = HOP / 16000
    for a, b in segs:
        ia, ib = int(a / hop_s), int(b / hop_s)
        vv = per[ia:ib] > 0.5
        if vv.sum() > 4:
            m = midi(f0[ia:ib])
            out.append(float(np.percentile(m[vv], 90) - np.percentile(m[vv], 10)))
    return np.array(out)


sp_ref = spans(f_ref, p_ref, segs_ref)
sp_our = spans(f_our, p_our, segs_our)
print(f"  标杆 片段内跨度 p50/p90 = {np.percentile(sp_ref, [50, 90]).round(1)} 半音")
print(f"  我们 片段内跨度 p50/p90 = {np.percentile(sp_our, [50, 90]).round(1)} 半音")

print("[7] 标杆的移调手法：逐段中位基频 vs 对齐后旋律", flush=True)
devs = []
for a, b in segs_ref:
    ia, ib = int(a * 10), int(b * 10)
    if ib - ia < 3:
        continue
    vv = vr[ia:ib]
    if vv.sum() < 2:
        continue
    clip_med = float(np.median(mr[ia:ib][vv]))
    msel = mo[map_o[ia:ib]][vo[map_o[ia:ib]] & vv]
    mel_med = float(np.median(msel)) if len(msel) else np.nan
    if not np.isnan(mel_med):
        devs.append(clip_med - semi - mel_med)
devs = np.array(devs)
print(f"  片段中位基频 - 旋律音高（半音，已扣整曲移调）: p10/50/90 = {np.percentile(devs, [10, 50, 90]).round(1)}")
print(f"  |偏差|<1 半音的片段占 {np.mean(np.abs(devs) < 1) * 100:.0f}%，<2 半音 {np.mean(np.abs(devs) < 2) * 100:.0f}%（≈他们为贴旋律变调的幅度）")
