"""The ready-made strata-poly for Windows: strata-poly.exe and the image encoder strata-vision.exe, built for other
PCs (STRATA_PORTABLE: ggml's CPU code for AVX2, the MSVC runtime linked in) and for every card from RTX 20 to RTX 50
with CUDA 13.0, the version of the CUDA libraries setup.py installs from pip.  They are zipped with a BUILD.json under
the name setup.py asks the latest release for (poly_asset()), which carries the source_hash() of their source.

Run it on a clean checkout of the commit the release is made from, with setup's Python:

    .venv\\Scripts\\python.exe tools\\make_poly_release.py OUT_DIR

and upload OUT_DIR\\strata-poly-windows-x64-<source>.zip to that release.  setup.py checks the download against the
SHA-256 GitHub publishes for the asset.  The build folders stay in OUT_DIR, so a second run compiles what changed."""
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

ARCHS = [75, 80, 86, 89, 120]   # RTX 20, A100 and A-series, RTX 30, RTX 40, RTX 50


def main() -> int:
    if len(sys.argv) != 2 or not setup.WIN:
        sys.exit(__doc__)
    out = Path(sys.argv[1]).resolve()
    # the hash covers every file under these paths, so a file git does not have would name a source no clone has
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--ignored", "--",
                            *setup.ENGINE_SOURCES, *setup.VISION_SOURCES], capture_output=True, text=True).stdout
    if dirty.strip():
        sys.exit("the engine sources are not those of a commit:\n" + dirty)
    nvcc, cuda_v = setup.find_nvcc()
    if cuda_v != (13, 0):
        sys.exit(f"CUDA 13.0's nvcc is needed (pip's libraries are 13.0); found {cuda_v}")
    vcvars = setup.find_vcvars(cuda_v)
    llama = setup.get_llama_cpp()
    defs = ["-DCMAKE_CUDA_ARCHITECTURES=" + ";".join(f"{a}-real" for a in ARCHS), f"-DCMAKE_CUDA_COMPILER={nvcc}",
            "-DSTRATA_PORTABLE=ON"]
    poly, vis = out / "build-poly", out / "build-vision"   # GBs of CUDA objects for five cards: on OUT_DIR's disk
    for d in (poly, vis):
        d.mkdir(parents=True, exist_ok=True)
    setup.cmake_build(ROOT, poly, "strata-poly", ["-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_BUILD_TESTS=OFF",
                                                  f"-DSTRATA_GGML_DIR={llama}", *defs], vcvars, poly / "build.bat")
    setup.cmake_build(ROOT / "tools" / "vision", vis, "strata-vision", [f"-DLLAMA_DIR={llama}",
                                                                        "-DSTRATA_VISION_CUDA=ON", *defs],
                      vcvars, vis / "build.bat")
    src = setup.source_hash(setup.ENGINE_SOURCES)
    meta = {"source": "release", "version": setup.source_version(), "archs": ARCHS, "cuda": f"{cuda_v[0]}.{cuda_v[1]}",
            "src": src, "vision": "gpu", "vision_src": setup.source_hash(setup.VISION_SOURCES)}
    out.mkdir(parents=True, exist_ok=True)
    z = out / setup.poly_asset()
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
        f.write(poly / setup.POLY_EXE, setup.POLY_EXE)
        f.write(vis / "bin" / setup.VEXE, setup.VEXE)
        f.writestr("BUILD.json", json.dumps(meta, indent=1))
    print(f"{z}\n  {z.stat().st_size:,} bytes, sha256 {hashlib.sha256(z.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
