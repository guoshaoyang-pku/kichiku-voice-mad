#!/usr/bin/env python3
"""Generate prototype/REPORT.md from metrics + library stats."""
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "out"
LIB = ROOT / "lib"


def lib_stats(path):
    d = json.load(open(path))
    usable = [s for s in d if s.get("f0_semi") and s.get("voiced_ratio", 0) > 0.3]
    chars = {}
    for s in usable:
        chars.setdefault(s.get("char", "?"), 0)
        chars[s.get("char", "?")] += 1
    cored = sum(1 for s in usable if s.get("core_f0_semi") is not None)
    return len(d), len(usable), cored, chars


def main():
    lines = [f"# Prototype 报告 · {date.today()}", ""]
    # libraries
    lines += ["## 素材库", "", "| 库 | 样本 | 可用(有稳定音高) | 含元音核 | 角色数 |", "|---|---|---|---|---|"]
    for name, p in [("原神", LIB / "library_genshin.json"), ("番剧", LIB / "library_anime.json")]:
        if p.exists():
            tot, ok, cored, chars = lib_stats(p)
            lines.append(f"| {name} | {tot} | {ok} | {cored} | {len(chars)} |")
    lines.append("")
    # renders
    rows = []
    for mf in sorted(OUT.glob("*.metrics.json")):
        n = mf.stem.replace(".metrics", "")
        if n.startswith("smoke") or not (OUT / f"{n}.mp3").exists():
            continue
        rows.append((n, json.load(open(mf))))
    lines += ["## 渲染矩阵", "",
              "| 渲染 | 版本 | 素材 | pitch_acc±50c | 中位偏差(音分) | dyn_corr | coverage |",
              "|---|---|---|---|---|---|---|"]
    for n, m in rows:
        pa = m.get("pitch_acc_50c")
        lines.append(f"| {n} | {m.get('version')} | {m.get('palette') or '混合'} | "
                     f"{pa*100:.0f}% | {m.get('median_abs_cents','—')} | "
                     f"{m.get('dyn_corr') if m.get('dyn_corr') is not None else '—'} | "
                     f"{(m.get('onset_coverage') or 0)*100:.0f}% |")
    lines.append("")
    (ROOT / "REPORT.md").write_text("\n".join(lines))
    print("REPORT.md written:", len(rows), "renders")


if __name__ == "__main__":
    main()
