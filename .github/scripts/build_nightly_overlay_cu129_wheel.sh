#!/usr/bin/env bash
# Build an LMCache cu129 wheel by overlaying this checkout onto the official
# nightly cp312 wheel, keeping nightly's cuda_ops (and other GPU-native) .so
# files intact. Only lmcache_fs is rebuilt from this tree (needed for PR #4752
# C++ key-codec / salt changes). Abort #4136 is Python-only and is
# applied via the .py overlay.
#
# Env:
#   NIGHTLY_WHL_URL   override nightly wheel URL
#   WHEEL_VERSION     pretend / published version
#                     (default 0.5.5.dev118+abort4136.fs4752.overlay)
#   OUT_DIR           output directory for the wheel (default dist)
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

NIGHTLY_WHL_URL="${NIGHTLY_WHL_URL:-https://github.com/LMCache/LMCache/releases/download/nightly/lmcache-0.5.5.dev118-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl}"
WHEEL_VERSION="${WHEEL_VERSION:-0.5.5.dev118+abort4136.fs4752.overlay}"
OUT_DIR="${OUT_DIR:-dist}"

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/lmcache-overlay.XXXXXX")"
cleanup() { rm -rf "${WORKDIR}"; }
trap cleanup EXIT

echo "==> downloading nightly wheel"
curl -fsSL -o "${WORKDIR}/nightly.whl" "${NIGHTLY_WHL_URL}"
python3 - <<PY
import zipfile
from pathlib import Path
z = zipfile.ZipFile("${WORKDIR}/nightly.whl")
z.extractall("${WORKDIR}/pkg")
print("extracted", len(z.namelist()), "entries")
PY

PKG="${WORKDIR}/pkg"
LMCACHE_PKG="${PKG}/lmcache"

echo "==> overlaying Python sources from this checkout (leave nightly .so in place)"
# Copy every tracked .py under lmcache/ onto the unpacked wheel.
while IFS= read -r -d '' src; do
  rel="${src#${ROOT}/lmcache/}"
  dest="${LMCACHE_PKG}/${rel}"
  mkdir -p "$(dirname "${dest}")"
  cp -f "${src}" "${dest}"
done < <(find "${ROOT}/lmcache" -type f -name '*.py' -print0)

echo "==> building lmcache_fs only (NO_GPU_EXT=1 — no cuda_ops rebuild)"
# Common C++ extensions build without CUDA. We only publish lmcache_fs.so from
# this build; cuda_ops / lmcache_native / lmcache_redis stay from nightly.
export NO_GPU_EXT=1
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
python3 setup.py build_ext --inplace

FS_SO="$(ls -1 "${ROOT}/lmcache"/lmcache_fs.cpython-*-linux-gnu.so | head -1)"
test -n "${FS_SO}" && test -f "${FS_SO}"
echo "    built ${FS_SO}"
cp -f "${FS_SO}" "${LMCACHE_PKG}/$(basename "${FS_SO}")"

# Sanity: cuda_ops must still be the nightly binary (never overwritten above).
test -f "${LMCACHE_PKG}/cuda_ops.cpython-312-x86_64-linux-gnu.so"
python3 - <<PY
from pathlib import Path
cuda = Path("${LMCACHE_PKG}/cuda_ops.cpython-312-x86_64-linux-gnu.so")
fs = Path("${LMCACHE_PKG}") / Path("${FS_SO}").name
print(f"cuda_ops size={cuda.stat().st_size} mtime kept from nightly")
print(f"lmcache_fs size={fs.stat().st_size} from this checkout")
assert cuda.stat().st_size > 30_000_000, "cuda_ops looks too small; nightly binary missing?"
PY

echo "==> stamping version ${WHEEL_VERSION}"
python3 - <<PY
from pathlib import Path
import re, shutil

pkg = Path("${PKG}")
version = "${WHEEL_VERSION}"

# Rewrite _version.py
ver_py = pkg / "lmcache" / "_version.py"
text = ver_py.read_text()
text = re.sub(r"__version__ = version = '.*'", f"__version__ = version = '{version}'", text)
# Keep a simple tuple for setuptools-scm style consumers.
text = re.sub(
    r"__version_tuple__ = version_tuple = .*",
    "__version_tuple__ = version_tuple = (0, 5, 5, 'dev118', 'abort4136', 'fs4752', 'overlay')",
    text,
)
text = re.sub(r"__commit_id__ = commit_id = .*", "__commit_id__ = commit_id = None", text)
ver_py.write_text(text)

# Rename dist-info and rewrite METADATA Name/Version.
old_di = next(pkg.glob("*.dist-info"))
new_di_name = f"lmcache-{version}.dist-info"
new_di = pkg / new_di_name
if old_di.name != new_di_name:
    if new_di.exists():
        shutil.rmtree(new_di)
    old_di.rename(new_di)

meta = (new_di / "METADATA").read_text()
meta = re.sub(r"(?m)^Version:.*$", f"Version: {version}", meta)
(new_di / "METADATA").write_text(meta)

# Drop old RECORD; wheel pack regenerates it.
rec = new_di / "RECORD"
if rec.exists():
    rec.unlink()
print("dist-info ->", new_di.name)
PY

echo "==> packing wheel into ${OUT_DIR}"
mkdir -p "${OUT_DIR}"
rm -f "${OUT_DIR}"/lmcache-*.whl
python3 -m pip install -q wheel
python3 -m wheel pack "${PKG}" -d "${OUT_DIR}"
ls -lh "${OUT_DIR}"/lmcache-*.whl
test "$(find "${OUT_DIR}" -maxdepth 1 -name 'lmcache-*.whl' | wc -l)" -eq 1
echo "==> done"
