#!/usr/bin/env python3
"""Cap libraries to a small diverse subset (per user request: keep it small).

Per character: quality filter, then evenly spaced picks over f0 range.
Overwrites library_{genshin,anime}.json (full versions kept as *_full.json).
"""
import json
from pathlib import Path

import numpy as np

LIB = Path(__file__).parent
CAPS = {"genshin": 70, "anime": 110}


def cap(lib_name, per_char):
    path = LIB / f"library_{lib_name}.json"
    items = json.load(open(path))
    usable = [s for s in items if s.get("f0_semi") and s.get("voiced_ratio", 0) > 0.4
              and s.get("dur", 0) >= 0.35]
    by_char = {}
    for s in usable:
        by_char.setdefault(s.get("char", "?"), []).append(s)
    out = []
    for char, ss in sorted(by_char.items()):
        ss.sort(key=lambda s: s["f0_semi"])
        if len(ss) <= per_char:
            out += ss
        else:
            idx = np.linspace(0, len(ss) - 1, per_char).round().astype(int)
            out += [ss[i] for i in sorted(set(idx))]
    json.dump(out, open(path, "w"), ensure_ascii=False)
    chars = sorted({s["char"] for s in out})
    f0s = [s["f0_semi"] for s in out]
    print(f"{lib_name}: {len(items)} -> {len(out)} ({len(chars)} chars, "
          f"f0 {min(f0s):.1f}-{max(f0s):.1f})")


for name, n in CAPS.items():
    cap(name, n)
