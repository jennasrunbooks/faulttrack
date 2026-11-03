#!/usr/bin/env bash
# Download RCAEval suites from Zenodo (CC BY 4.0, DOI 10.5281/zenodo.14590730) and extract METRICS ONLY.
#
#   ./download_rcaeval.sh                      # RE2-OB RE2-SS RE2-TT into data/raw
#   ./download_rcaeval.sh RE2-OB RE1-OB        # pick suites
#   DEST=somewhere ./download_rcaeval.sh       # other target dir
#   KEEP_ZIPS=0 ./download_rcaeval.sh          # delete zips after extracting
#
# The RE2 zips include multi-GB logs/traces (~4.2 GB total for RE2); only metrics + inject_time are extracted.
# Compare the printed md5 sums with the ones listed on https://zenodo.org/records/14590730 .
set -euo pipefail

RECORD="https://zenodo.org/records/14590730/files"
DEST="${DEST:-data/raw}"
KEEP_ZIPS="${KEEP_ZIPS:-1}"
SUITES=("$@")
[ ${#SUITES[@]} -eq 0 ] && SUITES=(RE2-OB RE2-SS RE2-TT)

command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }
command -v unzip >/dev/null || { echo "unzip is required" >&2; exit 1; }
MD5="md5sum"; command -v md5sum >/dev/null || MD5="md5 -r"     # macOS fallback

mkdir -p "$DEST"
for s in "${SUITES[@]}"; do
  case "$s" in RE1-OB|RE1-SS|RE1-TT|RE2-OB|RE2-SS|RE2-TT|RE3-OB|RE3-SS|RE3-TT) ;; *)
    echo "unknown suite: $s (expected e.g. RE2-OB)" >&2; exit 1;; esac
  zip="$DEST/$s.zip"
  if [ -s "$zip" ]; then
    echo "== $s: $zip already present, skipping download"
  else
    echo "== $s: downloading"
    curl -fL --retry 3 -o "$zip.part" "$RECORD/$s.zip?download=1"
    mv "$zip.part" "$zip"
  fi
  $MD5 "$zip"
  # metrics only: JSON or CSV metrics files + injection time. Logs/traces stay inside the zip.
  # each suite goes into its own folder, so the scripts can tell systems apart whatever the zip's root is
  unzip -o -q "$zip" '*inject_time.txt' '*metrics.json' '*data.csv' '*metrics.csv' -d "$DEST/$s" 2>/dev/null || true
  n=$(find "$DEST/$s" -name inject_time.txt | wc -l | tr -d ' ')
  if [ "$n" -eq 0 ]; then
    echo "!! $s: nothing extracted. Inspect the layout and adjust the patterns:" >&2
    unzip -l "$zip" | head -40 >&2
    exit 1
  fi
  echo "   extracted $n cases"
  [ "$KEEP_ZIPS" = "0" ] && rm -f "$zip"
done
du -sh "$DEST"
echo "Data: RCAEval (Pham et al., WWW 2025), CC BY 4.0, https://doi.org/10.5281/zenodo.14590730"
