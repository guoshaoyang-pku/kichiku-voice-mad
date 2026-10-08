#!/usr/bin/env python3
"""用已有的 cues.json + vocal_dry.wav + 原曲人声重建碎片页（含 A0 裁剪前完整原句），不重跑 DP。

用法：python3 engine/regen_fragments.py out/v9_haruhikage_down+low+wp2.5cs4_L2.5 --stems materials/stems_ft/htdemucs_ft/haruhikage_original
"""
import argparse
import json
import sys
from pathlib import Path

import librosa

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))

import token_dp as TD
import phrase_match as PM
import sampler_match as SM
import hires

ap = argparse.ArgumentParser()
ap.add_argument("prefix", help="渲染输出前缀，如 out/v9_haruhikage_..._L2.5")
ap.add_argument("--stems", required=True)
ap.add_argument("--lib", nargs="+", default=[str(HERE.parent / "lib" / "library_anime_full.json")])
args = ap.parse_args()

out = Path(args.prefix)
cues = json.load(open(out.parent / (out.name + ".cues.json")))
voc, _ = librosa.load(out.parent / (out.name + "_vocal_dry.wav"), sr=TD.FS, mono=True)
lib = PM.load_library(args.lib, None, None, 0.0, 99.0, 0.15)
SM._lib_by_path.update({c["path"]: c for c in lib})
hires.ensure_anime_maps([c["src"] for c in lib if str(c["work"]).startswith("anime")])
yv, _ = librosa.load(Path(args.stems) / "vocals.wav", sr=TD.FS, mono=True)
TD.write_fragments(cues, voc, yv, out, "素世原声整句（只在音节边界裁剪，保留 ≥70%），不变调不变速，只有常数增益")
print("fragments regenerated:", out.name)
