#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
g++ -O3 -std=c++17 -fopenmp "$HERE/fixed_backend.cpp" \
  -I"$ROOT/tmp/upstreams/SPERR/include" -I"$ROOT/tmp/builds/SPERR-0.8.5" \
  -I"$ROOT/tmp/upstreams/Serf/src" "$ROOT/tmp/builds/SPERR-0.8.5/src/libSPERR.a" \
  "$ROOT/tmp/builds/Serf-b38450b/libserf.so" -lzstd \
  -Wl,-rpath,"$ROOT/tmp/builds/Serf-b38450b" -o "$HERE/fixed_backend"
python3 "$HERE/write_manifest.py"
