#!/usr/bin/env bash
# Download the public UrbanEV and CHARGED data from their official GitHub repositories into data/.
# No data are distributed with this repository.
set -u
ROOT=${EV_RFP_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${PY:-python}

# UrbanEV (https://github.com/IntelligentSystemsLab/UrbanEV): every file in data/
U=$ROOT/data/urbanev
mkdir -p "$U"
curl -s -m 60 "https://api.github.com/repos/IntelligentSystemsLab/UrbanEV/contents/data" |
  $PY -c 'import sys, json
for x in json.load(sys.stdin):
    if x["type"] == "file":
        print(x["name"], x["download_url"])' |
  while read -r name url; do
    [ -s "$U/$name" ] || curl -sL -m 900 --retry 3 -o "$U/$name" "$url"
    echo "UrbanEV $name"
  done

# CHARGED (https://github.com/IntelligentSystemsLab/CHARGED): *_remove_zero city folders
BASE=https://raw.githubusercontent.com/IntelligentSystemsLab/CHARGED/main/data
for c in AMS JHB LOA MEL SPO SZH; do
  mkdir -p "$ROOT/data/charged/$c"
  for f in info.csv sites.csv volume.csv distance.csv; do
    dst=$ROOT/data/charged/$c/$f
    [ -s "$dst" ] || curl -sL -m 1800 --retry 3 -o "$dst" "$BASE/${c}_remove_zero/$f"
    echo "CHARGED $c $f"
  done
done
