#!/usr/bin/env python3
"""把 lib/*.json 里的绝对路径从一个机器根目录重写到另一个（换机/迁移用）。

用法：python3 engine/relocate_lib.py OLD_ROOT NEW_ROOT
例：python3 engine/relocate_lib.py /Users/guoshaoyang/Desktop/workdir/Ideas/kichiku-voice-mad /data2/guoshaoyang/kichiku-voice-mad
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
LIB = HERE.parent / "lib"

def main():
    old, new = sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/")
    for f in LIB.glob("library_*.json"):
        d = json.load(open(f))
        n = 0
        for c in d:
            for k in ("src", "path"):
                if isinstance(c.get(k), str) and c[k].startswith(old):
                    c[k] = new + c[k][len(old):]
                    n += 1
        if n:
            json.dump(d, open(f, "w"), ensure_ascii=False)
            print(f.name, n, "paths rewritten")
    p = LIB / "hires_map.json"
    if p.exists():
        m = json.load(open(p))
        m2 = {(new + k[len(old):] if k.startswith(old) else k): v for k, v in m.items()}
        json.dump(m2, open(p, "w"), ensure_ascii=False)
        print("hires_map keys:", len(m2))

if __name__ == "__main__":
    main()
