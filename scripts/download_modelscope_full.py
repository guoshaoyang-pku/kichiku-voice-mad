#!/usr/bin/env python3
"""Download full CN voice packs from ModelScope (fast CDN), extract, cleanup.

- StarRail4.2_CN.7z (~24GB) -> materials/starrail/full_cn/
- Genshin7.0_CN.7z (~40GB) -> materials/genshin/full_cn/
Resumable via curl -C -. Extracts with 7zz, verifies, then removes archive.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "materials"
SEVENZIP = "/opt/homebrew/bin/7zz"
LOG = ROOT.parent / "scripts" / "modelscope_download.log"

JOBS = [
    {
        "name": "starrail",
        "url": "https://modelscope.cn/datasets/aihobbyist/StarRail_Dataset/resolve/master/StarRail4.2_CN.7z",
        "archive": ROOT / "starrail" / "StarRail4.2_CN.7z",
        "dest": ROOT / "starrail" / "full_cn",
    },
    {
        "name": "genshin",
        "url": "https://modelscope.cn/datasets/aihobbyist/Genshin_Dataset/resolve/master/Genshin7.0_CN.7z",
        "archive": ROOT / "genshin" / "Genshin7.0_CN.7z",
        "dest": ROOT / "genshin" / "full_cn",
    },
]


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def download(job):
    arc = job["archive"]
    arc.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(20):
        r = subprocess.run(
            ["curl", "-sSL", "-C", "-", "--connect-timeout", "30",
             "-o", str(arc), job["url"]])
        if r.returncode == 0:
            return True
        log(f"{job['name']}: curl exit {r.returncode}, retry {attempt+1} "
            f"(have {arc.stat().st_size/1e9:.2f}GB)" if arc.exists() else "no file")
        time.sleep(10)
    return False


def main():
    for job in JOBS:
        name, arc, dest = job["name"], job["archive"], job["dest"]
        marker = dest / ".extract_done"
        if marker.exists():
            log(f"{name}: already done, skip")
            continue
        t0 = time.time()
        log(f"{name}: downloading {job['url']}")
        if not download(job):
            log(f"{name}: DOWNLOAD FAILED, giving up")
            continue
        size_gb = arc.stat().st_size / 1e9
        log(f"{name}: downloaded {size_gb:.1f}GB in {(time.time()-t0)/60:.0f}min, testing archive")
        test = subprocess.run([SEVENZIP, "t", "-y", str(arc)], capture_output=True)
        if test.returncode != 0:
            log(f"{name}: ARCHIVE CORRUPT: {test.stderr.decode(errors='ignore')[-300:]}")
            continue
        log(f"{name}: extracting to {dest}")
        dest.mkdir(parents=True, exist_ok=True)
        ex = subprocess.run([SEVENZIP, "x", "-y", f"-o{dest}", str(arc)],
                            capture_output=True)
        if ex.returncode != 0:
            log(f"{name}: EXTRACT FAILED: {ex.stderr.decode(errors='ignore')[-300:]}")
            continue
        n_wav = sum(1 for _ in dest.rglob("*.wav"))
        marker.touch()
        arc.unlink(missing_ok=True)
        log(f"{name}: DONE {n_wav} wavs, archive removed, "
            f"total {(time.time()-t0)/60:.0f}min")
    log("ALL JOBS FINISHED")


if __name__ == "__main__":
    main()
