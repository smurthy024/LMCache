#!/usr/bin/env bash
# Build an LMCache ROCm wheel by overlaying this checkout onto the official
# nightly-rocm cp312 wheel, keeping nightly's hipified cuda_ops (and other
# GPU-native) .so files intact. Only lmcache_fs is rebuilt from this tree
# (needed for PR #4752 C++ key-codec / salt changes). Abort #4136 and IPC
# #4261 and the dead-producer wait (producer_wait.py) are Python-only and
# are applied via the .py overlay.
#
# Env:
#   NIGHTLY_WHL_URL   override nightly-rocm wheel URL (default: resolve
#                     latest cp312 asset from LMCache/LMCache nightly-rocm)
#   WHEEL_VERSION     pretend / published version
#                     (default 0.5.5.dev118+abort4136.fs4752.ipc4261.deadprod.overlay.rocm7.2)
#   OUT_DIR           output directory for the wheel (default dist)
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

WHEEL_VERSION="${WHEEL_VERSION:-0.5.5.dev118+abort4136.fs4752.ipc4261.deadprod.overlay.rocm7.2}"
OUT_DIR="${OUT_DIR:-dist}"

if [[ -z "${NIGHTLY_WHL_URL:-}" ]]; then
  echo "==> resolving latest cp312 asset from LMCache nightly-rocm"
  NIGHTLY_WHL_URL="$(python3 - <<'PY'
import json, urllib.request
url = "https://api.github.com/repos/LMCache/LMCache/releases/tags/nightly-rocm"
with urllib.request.urlopen(url) as resp:
    data = json.load(resp)
assets = [
    a for a in data.get("assets", [])
    if a["name"].endswith(".whl") and "cp312" in a["name"]
]
if not assets:
    raise SystemExit("no cp312 wheel on nightly-rocm release")
# Prefer the single cp312 asset; if multiple, take the first.
print(assets[0]["browser_download_url"])
PY
)"
fi
echo "    NIGHTLY_WHL_URL=${NIGHTLY_WHL_URL}"

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/lmcache-overlay-rocm.XXXXXX")"
cleanup() { rm -rf "${WORKDIR}"; }
trap cleanup EXIT

echo "==> downloading nightly-rocm wheel"
curl -fsSL -o "${WORKDIR}/nightly.whl" "${NIGHTLY_WHL_URL}"
python3 - <<PY
import zipfile
z = zipfile.ZipFile("${WORKDIR}/nightly.whl")
z.extractall("${WORKDIR}/pkg")
print("extracted", len(z.namelist()), "entries")
print("wheel files:", [n for n in z.namelist() if n.endswith(".so")])
PY

PKG="${WORKDIR}/pkg"
LMCACHE_PKG="${PKG}/lmcache"

echo "==> overlaying Python sources from this checkout (leave nightly .so in place)"
while IFS= read -r -d '' src; do
  rel="${src#${ROOT}/lmcache/}"
  dest="${LMCACHE_PKG}/${rel}"
  mkdir -p "$(dirname "${dest}")"
  cp -f "${src}" "${dest}"
done < <(find "${ROOT}/lmcache" -type f -name '*.py' -print0)

echo "==> building lmcache_fs only (NO_GPU_EXT=1 — no hipified cuda_ops rebuild)"
export NO_GPU_EXT=1
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
python3 setup.py build_ext --inplace

FS_SO="$(ls -1 "${ROOT}/lmcache"/lmcache_fs.cpython-*-linux-gnu.so | head -1)"
test -n "${FS_SO}" && test -f "${FS_SO}"
echo "    built ${FS_SO}"
cp -f "${FS_SO}" "${LMCACHE_PKG}/$(basename "${FS_SO}")"

# Sanity: hipified cuda_ops from nightly-rocm must still be present.
CUDA_OPS_SO="$(ls -1 "${LMCACHE_PKG}"/cuda_ops.cpython-*-linux-gnu.so | head -1)"
test -n "${CUDA_OPS_SO}" && test -f "${CUDA_OPS_SO}"
python3 - <<PY
from pathlib import Path
cuda = Path("${CUDA_OPS_SO}")
fs = Path("${LMCACHE_PKG}") / Path("${FS_SO}").name
print(f"cuda_ops (HIP) size={cuda.stat().st_size} kept from nightly-rocm")
print(f"lmcache_fs size={fs.stat().st_size} from this checkout")
assert cuda.stat().st_size > 5_000_000, "cuda_ops looks too small; nightly-rocm binary missing?"
PY

echo "==> stamping version ${WHEEL_VERSION}"
python3 - <<PY
from pathlib import Path
import re, shutil

pkg = Path("${PKG}")
version = "${WHEEL_VERSION}"
# setuptools-scm style tuple derived from the version string, e.g.
# 0.5.5.dev118+a.b -> (0, 5, 5, 'dev118', 'a', 'b')
version_tuple = tuple(int(p) if p.isdigit() else p for p in re.split(r"[.+]", version))

ver_py = pkg / "lmcache" / "_version.py"
text = ver_py.read_text()
text = re.sub(r"__version__ = version = '.*'", f"__version__ = version = '{version}'", text)
text = re.sub(
    r"__version_tuple__ = version_tuple = .*",
    f"__version_tuple__ = version_tuple = {version_tuple!r}",
    text,
)
text = re.sub(r"__commit_id__ = commit_id = .*", "__commit_id__ = commit_id = None", text)
ver_py.write_text(text)

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
