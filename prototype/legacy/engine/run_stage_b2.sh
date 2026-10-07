#!/bin/bash
# Stage-B2: cores on capped libraries -> v4/v5 renders -> ALL DONE marker.
set -u
cd "$(dirname "$0")/../lib"
ENG=../engine/render.py
OUT=../out
LOG=stage_b.log

step() { echo "=== [B2] $* ===" | tee -a "$LOG"; }

step "make cores (capped libs: genshin 910 + anime 1310)"
python3 make_cores.py library_genshin.json library_anime.json 2>&1 | tee -a "$LOG" | tail -4

cd ../engine
GIORNO="--midi ../midis/il_vento_doro.mid --track 0 --transpose -12 --fold-range 52-83"
ROUND="--midi ../midis/roundabout.mid --track 0 --transpose 0 --fold-range 50-81"

run() {
  name="$1"; shift
  echo "=== render $name ===" | tee -a "$LOG"
  python3 "$ENG" "$@" --out "$OUT/$name.wav" --metrics "$OUT/$name.metrics.json" 2>&1 | tail -3 | tee -a "$LOG"
  ffmpeg -y -loglevel error -i "$OUT/$name.wav" -codec:a libmp3lame -qscale:a 4 "$OUT/$name.mp3"
}

run v4_giorno_genshin   $GIORNO --lib ../lib/library_genshin.json --palette genshin --version v4
run v5_giorno_genshin   $GIORNO --lib ../lib/library_genshin.json --palette genshin --version v5 --chords 2
run v4_giorno_anime     $GIORNO --lib ../lib/library_anime.json --palette anime --version v4
run v5_giorno_mix       $GIORNO --lib ../lib/library_genshin.json ../lib/library_anime.json --version v5 --chords 2
run v5_roundabout_anime $ROUND  --lib ../lib/library_anime.json --palette anime --version v5 --chords 2

echo "[B] ALL DONE" | tee -a "$LOG"
