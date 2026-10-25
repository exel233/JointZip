#!/usr/bin/env bash
# Build untouched official CLI sources into ignored, project-local tmp/ paths.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PREFIX="$WORKSPACE/tmp/modern-codecs"
JOBS="${MODERN_BUILD_JOBS:-2}"
CMAKE_BIN="$(command -v cmake || true)"
if [[ -z "$CMAKE_BIN" && -x "$HOME/.local/bin/cmake" ]]; then
  CMAKE_BIN="$HOME/.local/bin/cmake"
fi
if [[ -z "$CMAKE_BIN" ]]; then
  echo 'cmake is required; install it before building the frozen CLIs' >&2
  exit 1
fi
mkdir -p "$PREFIX/bin" "$PREFIX/build" "$PREFIX/upstreams"

checkout_source() {
  local name="$1" url="$2" commit="$3"
  local source="$PREFIX/upstreams/$name"
  if [[ ! -d "$source/.git" ]]; then
    git init "$source"
    git -C "$source" remote add origin "$url"
  fi
  if [[ "$(git -C "$source" rev-parse HEAD 2>/dev/null || true)" != "$commit" ]]; then
    git -C "$source" diff --quiet
    git -C "$source" diff --cached --quiet
    git -C "$source" fetch --depth 1 origin "$commit"
    git -C "$source" checkout --detach "$commit"
  fi
  [[ "$(git -C "$source" rev-parse HEAD)" == "$commit" ]]
  git -C "$source" diff --exit-code
}

checkout_source SZ3 https://github.com/szcompressor/SZ3.git 0ebe6fa26b0aa2e274d13b78df81a81e2db1b268
checkout_source zfp https://github.com/LLNL/zfp.git f40868a6a1c190c802e7d8b5987064f044bf7812

# Disable optional hardware/library paths, use the Zstd 1.4.5 release selected
# by SZ3's untouched CMake file, and use serial execution for the CPU panel.
"$CMAKE_BIN" -S "$PREFIX/upstreams/SZ3" -B "$PREFIX/build/SZ3" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=ON \
  -DCMAKE_DISABLE_FIND_PACKAGE_GSL=ON -DSZ3_USE_BUNDLED_ZSTD=ON \
  -DBUILD_TESTING=OFF -DBUILD_SZ3_BINARY=ON -DBUILD_SHARED_LIBS=OFF
"$CMAKE_BIN" --build "$PREFIX/build/SZ3" --target sz3 --parallel "$JOBS"
"$CMAKE_BIN" -S "$PREFIX/upstreams/zfp" -B "$PREFIX/build/zfp" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DBUILD_UTILITIES=ON \
  -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DZFP_WITH_OPENMP=OFF -DZFP_WITH_CUDA=OFF
"$CMAKE_BIN" --build "$PREFIX/build/zfp" --target zfpcmd --parallel "$JOBS"
cp "$PREFIX/build/SZ3/tools/sz3/sz3" "$PREFIX/bin/sz3_official"
g++ -std=c++17 -O3 -DNDEBUG -DSZ3_USE_SKA_HASH=1 -DSZ3_DEBUG_TIMINGS=0 \
  -I"$PREFIX/upstreams/SZ3/include" -I"$PREFIX/build/SZ3/include" \
  -I"$PREFIX/build/SZ3/_deps/zstdfetched-src/lib" \
  "$SCRIPT_DIR/sz3_api_cli.cpp" -L"$PREFIX/build/SZ3/tools/zstd" \
  -Wl,-rpath,"$PREFIX/build/SZ3/tools/zstd" -lzstd -o "$PREFIX/bin/sz3"
cp "$PREFIX/build/zfp/bin/zfp" "$PREFIX/bin/zfp"

python3 - "$PREFIX" "$SCRIPT_DIR/build_modern_codecs.sh" "$CMAKE_BIN" <<'PY'
import hashlib, json, pathlib, platform, subprocess, sys
prefix = pathlib.Path(sys.argv[1])
def command(*args):
    return subprocess.check_output(args, text=True).strip()
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
manifest = {
    'schema': 'vldb.modern-codecs-build.v1',
    'system': platform.platform(),
    'compiler': command('c++', '--version').splitlines()[0],
    'cmake': command(sys.argv[3], '--version').splitlines()[0],
    'build_script_sha256': sha(pathlib.Path(sys.argv[2])),
    'configuration': 'Release; serial; ZFP static; SZ3 bundled shared Zstd 1.4.5; no GSL',
    'codecs': {},
}
for name, version in [('sz3', '3.3.2'), ('zfp', '1.0.1')]:
    source = prefix / 'upstreams' / ('SZ3' if name == 'sz3' else 'zfp')
    binary = prefix / 'bin' / name
    cache = prefix / 'build' / source.name / 'CMakeCache.txt'
    manifest['codecs'][name] = {
        'version': version, 'upstream_commit': command('git', '-C', str(source), 'rev-parse', 'HEAD'),
        'upstream_url': command('git', '-C', str(source), 'remote', 'get-url', 'origin'),
        'binary': str(binary), 'binary_sha256': sha(binary),
        'cmake_cache_sha256': sha(cache),
        'ldd': command('ldd', str(binary)),
    }
    if name == 'sz3':
        manifest['codecs'][name]['implementation'] = 'official SZ3 API with allocating overload; local thin CLI'
        manifest['codecs'][name]['wrapper_sha256'] = sha(pathlib.Path(sys.argv[2]).with_name('sz3_api_cli.cpp'))
        relative = 'build/SZ3/tools/zstd/libzstd.so'
        manifest['codecs'][name]['dependencies'] = [{
            'name': 'zstd', 'version': '1.4.5', 'relative_path': relative,
            'binary_sha256': sha(prefix / relative),
            'source_header_sha256': sha(prefix / 'build/SZ3/_deps/zstdfetched-src/lib/zstd.h'),
        }]
(prefix / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
PY
