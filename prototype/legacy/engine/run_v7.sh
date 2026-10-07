#!/bin/bash
# v7 full renders + ASR intelligibility eval + review page.
set -u
cd "$(dirname "$0")"
OUT=../out
LIB=../lib
log=v7.log
: > "$log"

run() {
  name="$1"; shift
  echo "=== $name ===" | tee -a "$log"
  python3 syllabic.py "$@" --asr-samples 30 --out "$OUT/$name" 2>&1 | rg -v "^\s+(rendered|[0-9]+/)" | tail -6 | tee -a "$log"
  for w in "$OUT/$name.wav" "$OUT/${name}_vocal.wav"; do
    [ -f "$w" ] && ffmpeg -y -loglevel error -i "$w" -codec:a libmp3lame -qscale:a 2 "${w%.wav}.mp3"
  done
}

run v7_roundabout_anime --midi ../midis/roundabout.mid --track 0 --lib $LIB/library_anime_full.json --palette anime --works mygo,ave_mujica --start 58 --pad 2
run v7_giorno_anime     --midi ../midis/il_vento_doro.mid --track 0 --lib $LIB/library_anime_full.json --palette anime --works mygo,ave_mujica --pad 2
run v7_giorno_genshin   --midi ../midis/il_vento_doro.mid --track 0 --lib $LIB/library_genshin_full.json --palette genshin --pad 2

echo "=== ASR eval ===" | tee -a "$log"
python3 eval_asr.py $OUT/v7_roundabout_anime_asr $OUT/v7_giorno_anime_asr $OUT/v7_giorno_genshin_asr 2>&1 | rg "_asr \{" | tee -a "$log"

cd .. && python3 make_review.py | tee -a engine/$log
echo "V7 ALL DONE" | tee -a engine/$log
