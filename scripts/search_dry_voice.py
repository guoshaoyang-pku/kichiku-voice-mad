from pathlib import Path
#!/usr/bin/env python3
"""Search bilibili for dry-voice (干声) material videos for Bocchi/MyGO/AveMujica.

Uses yt-dlp bilisearch for queries (avoids search API rate limit), then resolves
each aid's title via the view API. Outputs candidates JSON for review.
"""
import json
import re
import subprocess
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
      "Referer": "https://www.bilibili.com"}

KEYWORDS = [
    # bocchi
    "孤独摇滚 干声素材", "波奇 干声", "后藤一里 干声", "结束乐队 语音素材",
    "喜多郁代 干声", "伊地知虹夏 干声", "山田凉 干声",
    # mygo
    "MyGO 干声素材", "高松灯 干声", "千早爱音 干声", "要乐奈 干声",
    "长崎爽世 干声", "椎名立希 干声", "MyGO 语音素材",
    # ave mujica
    "Ave Mujica 干声", "丰川祥子 干声", "若叶睦 干声", "八幡海铃 干声",
    "祐天寺喵梦 干声", "三角初华 干声", "Ave Mujica 语音素材",
]
FILTER = re.compile(r"干声|语音|素材|台詞|台词|セリフ|voice", re.I)
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "materials/anime/candidates.json"


def bilisearch(kw, n=6):
    try:
        r = subprocess.run(
            ["yt-dlp", f"bilisearch{n}:{kw}", "--flat-playlist", "--print", "%(id)s"],
            capture_output=True, text=True, timeout=60)
        return [x.strip() for x in r.stdout.splitlines() if x.strip()]
    except Exception as e:
        print(f"  search err {e}")
        return []


def resolve(aid):
    url = "https://api.bilibili.com/x/web-interface/view?aid=" + str(aid)
    req = urllib.request.Request(url, headers=UA)
    try:
        d = json.load(urllib.request.urlopen(req, timeout=15))
        if d.get("code") != 0:
            return None
        v = d["data"]
        return {
            "aid": aid,
            "bvid": v.get("bvid"),
            "title": v.get("title"),
            "duration": v.get("duration"),
            "uploader": (v.get("owner") or {}).get("name"),
            "pages": len(v.get("pages", [])),
        }
    except Exception:
        return None


def main():
    seen = set()
    cands = []
    for kw in KEYWORDS:
        print(f"== {kw}", flush=True)
        ids = bilisearch(kw)
        for aid in ids:
            if aid in seen:
                continue
            seen.add(aid)
            time.sleep(1.2)
            info = resolve(aid)
            if not info:
                continue
            if FILTER.search(info["title"] or ""):
                info["keyword"] = kw
                cands.append(info)
                mins = info["duration"] // 60
                print(f"  + {info['bvid']} | {info['title'][:50]} | {mins}m{info['duration']%60}s | {info['uploader']}", flush=True)
        time.sleep(2)
    with open(OUT, "w") as f:
        json.dump(cands, f, ensure_ascii=False, indent=1)
    print(f"\nsaved {len(cands)} candidates -> {OUT}")


if __name__ == "__main__":
    main()
