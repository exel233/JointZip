#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_root="$script_dir/build"
source_dir="$build_root/sz-1.1"
binary_dir="$script_dir/bin"
repository="https://github.com/szcompressor/sz2.git"
commit="2123f4899b76e6dfe28861da0a70d63a6cd43f83"
patch_file="$script_dir/patches/0001-use-compress-bound-for-zlib-output.patch"
patch_sha256="63b4a1cc5758f5f71167f40c6b87b569fccda3f291b39e0891e7f44ffb45a0be"
build_kind="${1:-release}"

case "$build_kind" in
  release)
    compiler_flags=(-O3)
    output_binary="$binary_dir/sz2016_batch"
    ;;
  asan)
    compiler_flags=(-O1 -g -fsanitize=address -fno-omit-frame-pointer)
    output_binary="$binary_dir/sz2016_batch_asan"
    ;;
  *)
    echo "usage: $0 [release|asan]" >&2
    exit 2
    ;;
esac

case "$source_dir" in
  "$build_root"/*) ;;
  *)
    echo "refusing to clean an SZ source path outside $build_root" >&2
    exit 1
    ;;
esac

if [[ ! -f "$patch_file" ]]; then
  echo "missing required upstream patch: $patch_file" >&2
  exit 1
fi
actual_patch_sha256="$(sha256sum "$patch_file" | awk '{print $1}')"
if [[ "$actual_patch_sha256" != "$patch_sha256" ]]; then
  echo "SZ upstream patch SHA-256 mismatch: $actual_patch_sha256" >&2
  exit 1
fi

mkdir -p "$build_root" "$binary_dir"
if [[ ! -d "$source_dir/.git" ]]; then
  git clone "$repository" "$source_dir"
fi
git -C "$source_dir" fetch origin "$commit"
git -C "$source_dir" checkout --detach --force "$commit"
git -C "$source_dir" clean -ffd

if [[ "$(git -C "$source_dir" rev-parse HEAD)" != "$commit" ]]; then
  echo "SZ source checkout is not pinned to $commit" >&2
  exit 1
fi
if [[ -n "$(git -C "$source_dir" status --porcelain --untracked-files=all)" ]]; then
  echo "SZ source checkout is not clean before patching" >&2
  exit 1
fi

git -C "$source_dir" apply --check "$patch_file"
git -C "$source_dir" apply "$patch_file"
git -C "$source_dir" diff --check
if [[ "$(git -C "$source_dir" diff --name-only)" != "sz/src/test_zlib.c" ]]; then
  echo "SZ patch changed files outside sz/src/test_zlib.c" >&2
  exit 1
fi
git -C "$source_dir" apply --reverse --check "$patch_file"

# The historical checkout may acquire CRLF endings on a Windows worktree.
sed -i 's/\r$//' \
  "$source_dir/configure" \
  "$source_dir/install-sh" \
  "$source_dir/missing" \
  "$source_dir/depcomp" \
  "$source_dir/ar-lib" \
  "$source_dir/config.guess" \
  "$source_dir/config.sub" \
  "$source_dir/ltmain.sh"

(cd "$source_dir" && ./configure >/dev/null)

zlib_sources=(
  adler32.c compress.c crc32.c deflate.c gzclose.c gzlib.c gzread.c
  gzwrite.c infback.c inffast.c inflate.c inftrees.c trees.c uncompr.c zutil.c
)
zlib_paths=()
for source in "${zlib_sources[@]}"; do
  zlib_paths+=("$source_dir/zlib/$source")
done

gcc \
  -std=gnu89 \
  -fcommon \
  "${compiler_flags[@]}" \
  -DHAVE_CONFIG_H \
  -I"$source_dir" \
  -I"$source_dir/sz/include" \
  -I"$source_dir/zlib" \
  "$script_dir/sz2016_batch.c" \
  "$source_dir"/sz/src/*.c \
  "${zlib_paths[@]}" \
  -lm \
  -o "$output_binary"

chmod +x "$output_binary"
echo "Built $output_binary ($build_kind) from $repository@$commit + $patch_sha256"
