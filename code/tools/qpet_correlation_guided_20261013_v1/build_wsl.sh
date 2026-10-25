#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/../.." && pwd)"
UPSTREAMS="$WORKSPACE/tmp/upstreams"
BUILDS="$WORKSPACE/tmp/builds"
PREFIX="$WORKSPACE/tmp/qpet-prefix"
QPET_SOURCE="$UPSTREAMS/QPET-szfamily-revision"
QPET_BUILD="$BUILDS/QPET-szfamily-revision"
SYMENGINE_SOURCE="$UPSTREAMS/symengine-v0.14.0"
SYMENGINE_BUILD="$BUILDS/symengine-v0.14.0"
EXTENSION_BUILD="$BUILDS/qpet-correlation-refined-20261012-v2"
SPERR_SOURCE="$UPSTREAMS/SPERR"
SPERR_BUILD="$BUILDS/SPERR-0.8.5"
SERF_SOURCE="$UPSTREAMS/Serf"
SERF_BUILD="$BUILDS/Serf-b38450b"

QPET_COMMIT="5d17cb1647caf001bd7355c617856c72c4b6475e"
SYMENGINE_COMMIT="fac9314c78f2809570494017efc6603befeb4eda"
SPERR_COMMIT="9f930a3970f86c8e390e135c97162b79a40d165b"
SERF_COMMIT="b38450b56825eabc96be8e25d6880127dc688c95"

mkdir -p "$UPSTREAMS" "$BUILDS" "$PREFIX"

CMAKE_BIN="$(command -v cmake || true)"
if [[ -z "$CMAKE_BIN" && -x "$HOME/.local/bin/cmake" ]]; then
  CMAKE_BIN="$HOME/.local/bin/cmake"
fi
if [[ -z "$CMAKE_BIN" ]]; then
  echo "cmake is required (the official projects require CMake >= 3.20)" >&2
  exit 1
fi
CTEST_BIN="$(dirname "$CMAKE_BIN")/ctest"

git_at() {
  local source="$1"
  shift
  if [[ -f "$source/.git" ]]; then
    local git_dir
    git_dir="$(sed -n 's/^gitdir: //p' "$source/.git")"
    if [[ "$git_dir" =~ ^[A-Za-z]:/ ]]; then
      git_dir="$(wslpath -u "$git_dir")"
    elif [[ "$git_dir" != /* ]]; then
      git_dir="$source/$git_dir"
    fi
    (cd "$source" && git --git-dir="$git_dir" --work-tree="$source" "$@")
  else
    git -C "$source" "$@"
  fi
}

if ! git_at "$SYMENGINE_SOURCE" rev-parse --git-dir >/dev/null 2>&1; then
  git clone https://github.com/symengine/symengine.git "$SYMENGINE_SOURCE"
fi
git_at "$SYMENGINE_SOURCE" fetch --depth 1 origin "$SYMENGINE_COMMIT"
if [[ "$(git_at "$SYMENGINE_SOURCE" rev-parse HEAD)" != "$SYMENGINE_COMMIT" ]]; then
  git_at "$SYMENGINE_SOURCE" diff --quiet
  git_at "$SYMENGINE_SOURCE" checkout --detach "$SYMENGINE_COMMIT"
fi

"$CMAKE_BIN" -S "$SYMENGINE_SOURCE" -B "$SYMENGINE_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DBUILD_TESTS=ON \
  -DBUILD_BENCHMARKS=OFF
"$CMAKE_BIN" --build "$SYMENGINE_BUILD" --parallel
"$CTEST_BIN" --test-dir "$SYMENGINE_BUILD" --output-on-failure
"$CMAKE_BIN" --install "$SYMENGINE_BUILD"

if ! git_at "$SPERR_SOURCE" rev-parse --git-dir >/dev/null 2>&1; then
  git clone https://github.com/NCAR/SPERR.git "$SPERR_SOURCE"
fi
git_at "$SPERR_SOURCE" fetch --depth 1 origin "$SPERR_COMMIT"
if [[ "$(git_at "$SPERR_SOURCE" rev-parse HEAD)" != "$SPERR_COMMIT" ]]; then
  git_at "$SPERR_SOURCE" diff --quiet
  git_at "$SPERR_SOURCE" checkout --detach "$SPERR_COMMIT"
fi
"$CMAKE_BIN" -S "$SPERR_SOURCE" -B "$SPERR_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=17 \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_UNIT_TESTS=OFF \
  -DBUILD_CLI_UTILITIES=OFF
"$CMAKE_BIN" --build "$SPERR_BUILD" --parallel

if ! git_at "$SERF_SOURCE" rev-parse --git-dir >/dev/null 2>&1; then
  git clone https://github.com/Spatio-Temporal-Lab/Serf.git "$SERF_SOURCE"
fi
git_at "$SERF_SOURCE" fetch --depth 1 origin "$SERF_COMMIT"
if [[ "$(git_at "$SERF_SOURCE" rev-parse HEAD)" != "$SERF_COMMIT" ]]; then
  git_at "$SERF_SOURCE" diff --quiet
  git_at "$SERF_SOURCE" checkout --detach "$SERF_COMMIT"
fi
# Build the authors' core target directly. The repository-level project also
# builds all benchmark baselines, which are not part of this backend.
"$CMAKE_BIN" -S "$SERF_SOURCE/src" -B "$SERF_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release
"$CMAKE_BIN" --build "$SERF_BUILD" --parallel

if ! git_at "$QPET_SOURCE" rev-parse --git-dir >/dev/null 2>&1; then
  git clone --branch szfamily_qpet_revision \
    https://github.com/JLiu-1/QPET-Artifact.git "$QPET_SOURCE"
fi
git_at "$QPET_SOURCE" fetch --depth 1 origin "$QPET_COMMIT"
if [[ "$(git_at "$QPET_SOURCE" rev-parse HEAD)" != "$QPET_COMMIT" ]]; then
  git_at "$QPET_SOURCE" diff --quiet
  git_at "$QPET_SOURCE" checkout --detach "$QPET_COMMIT"
fi

QPET_BUFFER_PATCH="$SCRIPT_DIR/patches/0001-fix-tuning-huffman-buffer.patch"
if grep -Fq 'quant_inds.size() * quant_inds.size() / 8 + 65536' \
     "$QPET_SOURCE/include/QoZ/compressor/SZInterpolationCompressor.hpp" && \
   grep -Fq 'quant_inds.size() * quant_inds.size() / 8 + 65536' \
     "$QPET_SOURCE/include/QoZ/compressor/SZQoIInterpolationCompressor.hpp"; then
  echo "QPET Huffman tuning-buffer safety patch already applied"
else
  git_at "$QPET_SOURCE" apply --ignore-space-change --check "$QPET_BUFFER_PATCH"
  git_at "$QPET_SOURCE" apply --ignore-space-change "$QPET_BUFFER_PATCH"
fi

QPET_UB_PATCH="$SCRIPT_DIR/patches/0002-eliminate-sanitizer-undefined-behavior.patch"
if git_at "$QPET_SOURCE" apply --ignore-space-change --reverse --check \
     "$QPET_UB_PATCH" >/dev/null 2>&1; then
  echo "QPET sanitizer undefined-behavior patch already applied"
else
  # A partially applied or drifted checkout must fail here instead of silently
  # producing a binary that is not represented by the frozen patch artifact.
  git_at "$QPET_SOURCE" apply --ignore-space-change --check "$QPET_UB_PATCH"
  git_at "$QPET_SOURCE" apply --ignore-space-change "$QPET_UB_PATCH"
fi

QPET_FAIL_CLOSED_PATCH="$SCRIPT_DIR/patches/0003-fail-closed-huffman-and-symbol-count.patch"
if git_at "$QPET_SOURCE" apply --ignore-space-change --reverse --check \
     "$QPET_FAIL_CLOSED_PATCH" >/dev/null 2>&1; then
  echo "QPET fail-closed Huffman/symbol-count patch already applied"
else
  git_at "$QPET_SOURCE" apply --ignore-space-change --check \
    "$QPET_FAIL_CLOSED_PATCH"
  git_at "$QPET_SOURCE" apply --ignore-space-change \
    "$QPET_FAIL_CLOSED_PATCH"
fi

"$CMAKE_BIN" -S "$QPET_SOURCE" -B "$QPET_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_PREFIX_PATH="$PREFIX" \
  -DQoZ_DEBUG_TIMINGS=OFF \
  -DBUILD_TESTING=ON
"$CMAKE_BIN" --build "$QPET_BUILD" --parallel
# The pinned QPET revision registers no CTest cases; this command records that
# upstream fact. Its official hpez and qoi_val executables are nevertheless built.
"$CTEST_BIN" --test-dir "$QPET_BUILD" --output-on-failure
"$CMAKE_BIN" --install "$QPET_BUILD"

"$CMAKE_BIN" -S "$SCRIPT_DIR" -B "$EXTENSION_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PREFIX" \
  -DQPET_SOURCE="$QPET_SOURCE" \
  -DSPERR_SOURCE="$SPERR_SOURCE" \
  -DSPERR_BUILD="$SPERR_BUILD" \
  -DSERF_SOURCE="$SERF_SOURCE" \
  -DSERF_BUILD="$SERF_BUILD"
"$CMAKE_BIN" --build "$EXTENSION_BUILD" --parallel
"$CTEST_BIN" --test-dir "$EXTENSION_BUILD" --output-on-failure

mkdir -p "$SCRIPT_DIR/bin"
cp "$EXTENSION_BUILD/qpet_correlation_batch" "$SCRIPT_DIR/bin/qpet_correlation_batch"
echo "Built QPET correlation extension with QPET $QPET_COMMIT, SPERR $SPERR_COMMIT, Serf $SERF_COMMIT, and SymEngine $SYMENGINE_COMMIT"
