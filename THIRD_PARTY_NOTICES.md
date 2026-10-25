# Third-party source and dependency notices

The `code/` tree contains the local JointZip framework, adapters, repair helper,
tests and build recipes. Upstream library repositories and their compiled
artifacts are not vendored. Optional build scripts download the following
revisions, which remain subject to their upstream copyright and license terms.
This document does not grant a new license to third-party material.

| Dependency | Upstream | Pinned revision |
| --- | --- | --- |
| SZ3 3.3.2 | https://github.com/szcompressor/SZ3 | `0ebe6fa26b0aa2e274d13b78df81a81e2db1b268` |
| ZFP 1.0.1 | https://github.com/LLNL/zfp | `f40868a6a1c190c802e7d8b5987064f044bf7812` |
| SPERR 0.8.5 | https://github.com/NCAR/SPERR | `9f930a3970f86c8e390e135c97162b79a40d165b` |
| Serf | https://github.com/Spatio-Temporal-Lab/Serf | `b38450b56825eabc96be8e25d6880127dc688c95` |
| QPET-Artifact, szfamily revision | https://github.com/JLiu-1/QPET-Artifact | `5d17cb1647caf001bd7355c617856c72c4b6475e` |
| SymEngine 0.14.0 | https://github.com/symengine/symengine | `fac9314c78f2809570494017efc6603befeb4eda` |
| SZ 1.1 (SZ2016) | https://github.com/szcompressor/sz2 | `2123f4899b76e6dfe28861da0a70d63a6cd43f83` |

The shared native implementation uses QPET/QoZ headers and interfaces. Its
`patches/` directory contains three local safety patches against the pinned
QPET source: tuning Huffman buffer handling, sanitizer-detected undefined
behavior, and fail-closed Huffman/symbol-count validation. Patch context may
contain upstream source lines, which retain their upstream attribution and
terms. The QPET and Serf projects must be consulted for their applicable terms;
no permissive license is inferred merely from public repository access.

At the pinned Serf revision, the upstream README links CC BY-NC 4.0 and describes
the project as non-commercial; there is no separate top-level license file.
This dependency must not be treated as permissively licensed for commercial use.
The Serf sources and binaries are obtained separately from their authors.

The SZ2016 wrapper uses the official SZ 1.1 API. Its local patch increases the
zlib output allocation to the compression bound. The original SZ copyright and
license notice is preserved at `code/tools/sz2016/COPYRIGHT.txt`. The required
acknowledgment is:

> This product includes software produced by UChicago Argonne, LLC under Contract No. DE-AC02-06CH11357 with the Department of Energy.

The Python offline test depends on NumPy and python-zstandard at the versions in
`requirements.txt`; these packages are installed separately. Native builds also
use Zstd, GSL, GMP, OpenMP, CMake, Ninja and the system toolchain. The SZ3 recipe
uses the bundled Zstd 1.4.5 release selected by its upstream CMake configuration.
Each installed or fetched dependency retains its own notices and license.
