#!/usr/bin/env python3
"""Download AI-Hobbyist per-character voice packs (Genshin / Star Rail CN) via res.ai-lab.top API.

Downloads 7z -> extracts to materials/<game>/voice/<character>/ -> deletes 7z.
Resumable: skips characters whose target dir already has files.
"""
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "materials"
GAMES = {
    "genshin": ("原神", ROOT / "genshin" / "character_list.json"),
    "starrail": ("星穹铁道", ROOT / "starrail" / "character_list.json"),
}


def refresh_lists():
    """Re-fetch fresh signed download links (signatures expire)."""
    for game, (cat, listfile) in GAMES.items():
        body = json.dumps({"root_path": "/datasets", "repo": cat,
                           "category": cat, "subcategory": "中文"})
        r = subprocess.run(
            ["curl", "-sS", "-m", "30", "-X", "POST",
             "https://res.ai-lab.top/api/acgnailib/models",
             "-H", "Content-Type: application/json", "-d", body],
            capture_output=True, text=True,
        )
        try:
            data = json.loads(r.stdout)
            if isinstance(data, list) and data and "dl_link" in data[0]:
                listfile.write_text(json.dumps(data, ensure_ascii=False, indent=1))
                print(f"refreshed {game}: {len(data)} characters", flush=True)
        except json.JSONDecodeError:
            print(f"WARN: refresh failed for {game}, using cached list", flush=True)
SEVENZIP = "/opt/homebrew/bin/7zz"
MAX_WORKERS = 3
LOG = ROOT.parent / "scripts" / "download.log"


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def process(game: str, item: dict) -> tuple[str, str, str]:
    name = item["dataname"]
    url = item["dl_link"]
    voice_dir = ROOT / game / "voice" / name
    pack = ROOT / game / "packs" / f"{name}.7z"
    if voice_dir.exists() and any(voice_dir.rglob("*.wav")):
        return (name, "skip", "already extracted")
    pack.parent.mkdir(parents=True, exist_ok=True)
    voice_dir.mkdir(parents=True, exist_ok=True)
    # download with resume; retry until curl exits 0 (partial file resumes via -C -)
    for attempt in range(5):
        r = subprocess.run(
            ["curl", "-sSL", "-C", "-", "--connect-timeout", "20",
             "--max-time", "3600", "-o", str(pack), url],
            capture_output=True,
        )
        if r.returncode == 0 and pack.exists() and pack.stat().st_size > 0:
            break
        time.sleep(5 * (attempt + 1))
    if not pack.exists() or pack.stat().st_size == 0:
        return (name, "fail", "download empty")
    # test archive integrity; if truncated, remove and mark fail
    test = subprocess.run([SEVENZIP, "t", "-y", str(pack)], capture_output=True)
    if test.returncode != 0:
        size_mb = pack.stat().st_size / 1e6
        return (name, "fail", f"corrupt/incomplete ({size_mb:.0f}MB), kept for resume")
    ex = subprocess.run([SEVENZIP, "x", "-y", f"-o{voice_dir}", str(pack)],
                        capture_output=True)
    if ex.returncode != 0:
        return (name, "fail", f"extract error: {ex.stderr.decode(errors='ignore')[:200]}")
    n_wav = sum(1 for _ in voice_dir.rglob("*.wav"))
    pack.unlink(missing_ok=True)
    return (name, "ok", f"{n_wav} wavs")


def main():
    refresh_lists()
    tasks = []
    for game, (_, listfile) in GAMES.items():
        data = json.load(open(listfile))
        for item in data:
            tasks.append((game, item))
    log(f"total tasks: {len(tasks)}")
    done = fail = skip = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(process, g, it): (g, it["dataname"]) for g, it in tasks}
        for fut in as_completed(futs):
            g, n = futs[fut]
            try:
                name, status, msg = fut.result()
            except Exception as e:  # noqa: BLE001
                name, status, msg = n, "fail", repr(e)
            if status == "ok":
                done += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
            log(f"[{g}] {name}: {status} {msg} (ok={done} skip={skip} fail={fail})")
    log(f"FINISHED ok={done} skip={skip} fail={fail}")


if __name__ == "__main__":
    main()
