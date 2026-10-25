#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXTENSION_BUILD="$WORKSPACE/tmp/builds/qpet-correlation-guided-20261013-v1"
mkdir -p "$WORKSPACE/tmp/paper_budget_guided_20261013_v1"
exec > >(tee "$WORKSPACE/tmp/paper_budget_guided_20261013_v1/native_build.log") 2>&1
# Consume existing pinned official dependency builds without fetching, checking
# out, patching, reinstalling, or replacing any original source/binary.
CMAKE_BIN="$(command -v cmake || true)"
if [[ -z "$CMAKE_BIN" && -x "$HOME/.local/bin/cmake" ]]; then
  CMAKE_BIN="$HOME/.local/bin/cmake"
fi
if [[ -z "$CMAKE_BIN" ]]; then
  echo "cmake not available" >&2
  exit 1
fi
"$CMAKE_BIN" -S "$SCRIPT_DIR" -B "$EXTENSION_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$WORKSPACE/tmp/qpet-prefix" \
  -DQPET_SOURCE="$WORKSPACE/tmp/upstreams/QPET-szfamily-revision" \
  -DSPERR_SOURCE="$WORKSPACE/tmp/upstreams/SPERR" \
  -DSPERR_BUILD="$WORKSPACE/tmp/builds/SPERR-0.8.5" \
  -DSERF_SOURCE="$WORKSPACE/tmp/upstreams/Serf" \
  -DSERF_BUILD="$WORKSPACE/tmp/builds/Serf-b38450b"
if [[ "${1:-}" == "--clean" ]]; then
  "$CMAKE_BIN" --build "$EXTENSION_BUILD" --target clean
fi
"$CMAKE_BIN" --build "$EXTENSION_BUILD" --target qpet_correlation_batch test_correlation_qoi test_paper_budget --parallel 1
"$(dirname "$CMAKE_BIN")/ctest" --test-dir "$EXTENSION_BUILD" \
  -R '^(correlation_qoi_analytical_and_constraints|paper_uniform_budget_analytical_fixtures)$' --output-on-failure
mkdir -p "$SCRIPT_DIR/bin"
cp "$EXTENSION_BUILD/qpet_correlation_batch" "$SCRIPT_DIR/bin/qpet_correlation_batch"
echo "Isolated guided-budget binary: $SCRIPT_DIR/bin/qpet_correlation_batch"
