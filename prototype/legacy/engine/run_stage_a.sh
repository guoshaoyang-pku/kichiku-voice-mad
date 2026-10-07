#!/bin/bash
# Stage-A render matrix. Waits for freshly rebuilt libraries (mtime > script start),
# renders version matrix, encodes mp3, collects metrics.
set -u
cd "$(dirname "$0")"
LIB=../lib
OUT=../out
ENG=render.py
mkdir -p "$OUT"
ready() { # $1=json path, $2=min usable, $3=require named chars (anime repair check)
  python3 - "$1" "$2" "$3" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
usable = [s for s in d if s.get("f0_semi") and s.get("voiced_ratio", 0) > 0.3]
ok = len(usable) >= int(sys.argv[2])
if sys.argv[3] == "named":
    ok = ok and sum(1 for s in usable if s.get("char") not in ("未知", None)) > 100
sys.exit(0 if ok else 1)
PY
}

echo "[A] waiting for rebuilt library_genshin.json ..."
while ! ready "$LIB/library_genshin.json" 800 any; do sleep 20; done
echo "[A] genshin library ready"

GIORNO="--midi ../midis/il_vento_doro.mid --track 0 --transpose -12 --fold-range 52-83"

run() {
  name="$1"; shift
  echo "=== render $name ==="
  python3 "$ENG" "$@" --out "$OUT/$name.wav" --metrics "$OUT/$name.metrics.json" 2>&1 | tail -3
  ffmpeg -y -loglevel error -i "$OUT/$name.wav" -codec:a libmp3lame -qscale:a 4 "$OUT/$name.mp3"
}

run v1_giorno_genshin   $GIORNO --lib "$LIB/library_genshin.json" --palette genshin --version v1
run v2_giorno_genshin   $GIORNO --lib "$LIB/library_genshin.json" --palette genshin --version v2
run v2rs_giorno_genshin $GIORNO --lib "$LIB/library_genshin.json" --palette genshin --version v2 --shift-mode rs
run v3_giorno_genshin   $GIORNO --lib "$LIB/library_genshin.json" --palette genshin --version v3 --chords 2

echo "[A] waiting for rebuilt library_anime.json ..."
while ! ready "$LIB/library_anime.json" 1500 named; do sleep 30; done
echo "[A] anime library ready"

ROUND="--midi ../midis/roundabout.mid --track 0 --transpose 0 --fold-range 50-81"

run v3_giorno_anime     $GIORNO --lib "$LIB/library_anime.json" --palette anime --version v3 --chords 2
run v3_giorno_mix       $GIORNO --lib "$LIB/library_genshin.json" "$LIB/library_anime.json" --version v3 --chords 2
run v3_roundabout_anime $ROUND  --lib "$LIB/library_anime.json" --palette anime --version v3 --chords 2

echo "[A] ALL RENDERS DONE"
