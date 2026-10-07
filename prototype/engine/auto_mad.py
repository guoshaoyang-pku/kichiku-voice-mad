#!/usr/bin/env python3
"""auto_mad — 全自动化鬼畜调音管线（VQ-VAE 思想的落地：不训练的 VQ + 自动搜索超参）。

码本固定 = 真实台词；编码器 = 分段 Viterbi DP 精确搜索；解码器 = 相加。
这套系统里唯一值得"学"的是结构选择（码本变体）和 λ_N——本脚本把它们变成
自动调参闭环：跑变体 × λ 扫描，用"听起来对劲"指标清单打分，选胜者渲染。

子命令：
  prepare  歌曲（URL / yt-dlp 搜索串 / 本地文件）→ wav + demucs htdemucs_ft stems（幂等）
  tune     结构变体 × λ_N 扫描 → 打分 → 最优配置渲染 → tune 报告
  render   用显式配置渲染一次

用户的两个技巧作为开关：
  --song-speed d   歌曲与音效"同时放慢再缩回"的诚实等价式：λ_N、c_skip 除以 d，
                   key_tol 乘以 d（纯重参数化，不改变 formulation）。
  --len-scale      允许音效长度缩放的变体（±~1.7 半音重采样 ≈ 时长 ×1.106 / ×0.902）。
                   默认关闭（用户拍板：默认别用）。
"""
import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
PROTO = HERE.parent
MAT = PROTO / "materials"
OUT = PROTO / "out"
ENGINE = HERE / "token_dp.py"

LOW_CHARS = "椎名立希,八幡海铃,若叶睦"
BASE_LAMBDAS = [5.0, 10.0, 20.0]


def song_paths(name):
    mix = MAT / f"{name}_original.wav"
    stems = MAT / "stems_ft" / "htdemucs_ft" / f"{name}_original"
    return mix, stems


def prepare(args):
    mix, stems = song_paths(args.name)
    if not mix.exists():
        src = args.src
        if src is None:
            sys.exit(f"{mix} 不存在，且未给 --src")
        if Path(src).exists():
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, str(mix)], check=True)
        else:
            spec = src if re.match(r"^(https?://|ytsearch)", src) else f"ytsearch1:{src}"
            subprocess.run(["yt-dlp", "--no-playlist", "-x", "--audio-format", "wav",
                            "--audio-quality", "0", "-o", str(mix).replace(".wav", ".%(ext)s"),
                            spec], check=True, cwd=MAT)
    if not (stems / "vocals.wav").exists() or not (stems / "bass.wav").exists():
        stems.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["demucs", "-n", "htdemucs_ft", "-o", str(MAT / "stems_ft"), str(mix)], check=True)
    print(f"prepare ok: mix={mix} stems={stems}")


# ---- “听起来对劲”的标量化（预测指标，无需渲染即可比较配置）----------------
# 维度来自用户认可的 checklist：音高（主）、该唱在唱/不该唱在响、起音卡点。
# 门槛：recall ≥ 0.70 且 false_alarm ≤ 0.15；过门槛的配置之间按 score 比大小。
def score(m):
    s = (1.0 * m["pitch_acc50"] + 0.5 * m["pitch_acc100"]
         + 0.8 * m["voicing_recall"] - 0.8 * m["false_alarm"]
         + 0.6 * m.get("keyframe_hit_30ms", 0.0))
    gated = m["voicing_recall"] < 0.70 or m["false_alarm"] > 0.15
    return s, gated


def variants(args):
    vs = [("raw", "0", ""),
          ("down", "-4,-3,-2,-1,0", ""),
          ("down+low", "-4,-3,-2,-1,0", LOW_CHARS)]
    if args.len_scale:
        vs.append(("down+low+scale", "-4,-3,-2,-1.73,-1,0,1.78", LOW_CHARS))
    return vs


def run_engine(args, vtag, shifts, low_chars, render, extra=()):
    mix, stems = song_paths(args.name)
    d = args.song_speed
    lambdas = ",".join(f"{x / d:g}" for x in BASE_LAMBDAS)
    cmd = [sys.executable, str(ENGINE), "--mix", str(mix), "--stems", str(stems),
           "--out", str(OUT / f"v9_{args.name}_{vtag}"), "--version", "v9", "--variant", vtag,
           "--ref-name", f"v9_{args.name}", "--singer", args.singer,
           "--keyframe-hard", "--choke", "--w-onset", "2", "--c-skip", f"{2.0 / d:g}",
           "--key-tol", str(max(1, round(d))), "--lambdas", lambdas,
           f"--shifts={shifts}", "--render", render,
           "--start", str(args.start)]
    if low_chars:
        cmd += ["--low-chars", low_chars]
    if args.dur_limit:
        cmd += ["--dur-limit", str(args.dur_limit)]
    cmd += list(extra)
    print("  $ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        sys.exit(f"engine failed for variant {vtag}")
    return r.stdout


def tune(args):
    OUT.mkdir(exist_ok=True)
    table = []
    for vtag, shifts, low in variants(args):
        print(f"[tune] variant {vtag}", flush=True)
        out = run_engine(args, vtag, shifts, low, render="")
        sweep = json.load(open(OUT / f"v9_{args.name}_{vtag}_sweep.json"))
        for m in sweep:
            s, gated = score(m)
            table.append({"variant": vtag, "lambda_N": m["lambda_N"], "score": round(s, 4),
                          "gated": gated, **{k: m[k] for k in
                          ("N", "dur_median", "pitch_acc50", "pitch_acc100",
                           "voicing_recall", "false_alarm", "keyframe_hit_30ms",
                           "distinct_lines", "crop_mean")}})
    ok = [t for t in table if not t["gated"]] or table
    best = max(ok, key=lambda t: (t["score"], t["voicing_recall"]))
    print(f"[tune] winner: {best['variant']} λ_N={best['lambda_N']:g} score={best['score']}", flush=True)
    vtag = best["variant"]
    shifts, low = next((s, l) for v, s, l in variants(args) if v == vtag)
    run_engine(args, vtag, shifts, low, render=f"{best['lambda_N']:g}")
    report = {"song": args.name, "singer": args.singer, "song_speed": args.song_speed,
              "len_scale": args.len_scale, "winner": best, "table": table,
              "score_def": "acc50 + 0.5*acc100 + 0.8*recall - 0.8*false_alarm + 0.6*keyframe30; "
                           "gates: recall>=0.70, false_alarm<=0.15"}
    json.dump(report, open(OUT / f"v9_{args.name}_tune.json", "w"), ensure_ascii=False, indent=1)
    print(f"[tune] report -> out/v9_{args.name}_tune.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "tune", "render"):
        p = sub.add_parser(name)
        p.add_argument("--name", required=True, help="歌曲名（materials/<name>_original.wav）")
        if name == "prepare":
            p.add_argument("--src", default=None, help="本地文件路径 / URL / yt-dlp 搜索串")
        else:
            p.add_argument("--singer", default="长崎爽世")
            p.add_argument("--start", type=float, default=0.0)
            p.add_argument("--dur-limit", type=float, default=None)
            p.add_argument("--song-speed", type=float, default=1.0,
                           help="同时放慢歌曲与音效的等价系数：λ_N、c_skip 除以它")
            p.add_argument("--len-scale", action="store_true", help="允许音效长度缩放变体（默认关）")
        if name == "render":
            p.add_argument("--variant", default="down+low")
            p.add_argument("--lambda", dest="lam", type=float, default=10.0)
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args)
    elif args.cmd == "tune":
        tune(args)
    elif args.cmd == "render":
        vtag = args.variant
        shifts, low = next((s, l) for v, s, l in variants(args) if v == vtag)
        run_engine(args, vtag, shifts, low, render=f"{args.lam:g}",
                   extra=("--lambdas", f"{args.lam:g}"))


if __name__ == "__main__":
    main()
