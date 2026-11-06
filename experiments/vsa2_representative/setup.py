"""Builds vsa2_rep._C, the native three-tier representative attention extension.

It compiles this package's csrc/rep sources together with the VSA2 kernels it reuses (pooling, PSP
router, exact global top-k, generic fine attention forward/backward, gate fusion), taken from a VSA2
/ EchoSparse-Attn source tree. Point VSA2_REP_SOURCE at that tree (it must contain csrc/vsa):

  RTX 5090: VSA2_REP_BUILD=1 VSA2_REP_SOURCE=/path/to/EchoSparse-Attn TORCH_CUDA_ARCH_LIST=12.0 \
            pip install -e . --no-build-isolation
  H100:     VSA2_REP_BUILD=1 VSA2_REP_SOURCE=/path/to/EchoSparse-Attn TORCH_CUDA_ARCH_LIST=9.0a \
            pip install -e . --no-build-isolation

VSA2_REP_TK=1 adds the ThunderKittens + TMA kernels (csrc/rep_tk) and VSA2's own TK fine kernels (the
native baseline and tier A's backward). It needs TORCH_CUDA_ARCH_LIST=9.0a and a ThunderKittens tree at
VSA2_REP_TK_ROOT (VSA2 pins 6c27e28c8115d1839d9eeeb530913c184a75fc87).

Without VSA2_REP_BUILD=1 the extension is skipped (no nvcc or torch needed), so the Python reference
and the pure-Python tests stay importable.
"""
import os

from setuptools import find_packages, setup

ROOT = os.path.dirname(os.path.abspath(__file__))
# VSA2 stages reused as they are; the three-tier kernels are this package's own.
VSA2_SOURCES = ("pool.cu", "router.cu", "topk.cu", "scores.cu")
VSA2_SOURCES_TOP = ("generic/attn_fwd.cu", "generic/attn_bwd.cu", "fuse.cu")
REP_SOURCES = ("rep_api.cpp", "rep_pool.cu", "rep_attn_fwd.cu", "rep_attn_bwd.cu")


def vsa2_root():
    p = os.environ.get("VSA2_REP_SOURCE")
    if not p:
        raise RuntimeError("set VSA2_REP_SOURCE to a VSA2 / EchoSparse-Attn source tree (containing csrc/vsa)")
    p = os.path.abspath(p)
    if not os.path.isfile(os.path.join(p, "csrc", "vsa", "vsa2_kernels.h")):
        raise RuntimeError(f"{p} does not look like a VSA2 source tree (no csrc/vsa/vsa2_kernels.h)")
    return p


def extension():
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    src = os.path.join(vsa2_root(), "csrc", "vsa")
    sources = [os.path.join(ROOT, "csrc", "rep", f) for f in REP_SOURCES]
    sources += [os.path.join(src, "router", f) for f in VSA2_SOURCES]
    sources += [os.path.join(src, f) for f in VSA2_SOURCES_TOP]
    nvcc = ["-O3", "-std=c++20", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-lineinfo",
            "-Xptxas=--warn-on-spills", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"]
    if os.environ.get("VSA2_REP_PTXAS_VERBOSE", "0") == "1":
        nvcc.append("-Xptxas=-v")
    include_dirs = [src, os.path.join(ROOT, "csrc", "rep")]
    cxx = ["-O3", "-std=c++17"]
    library_dirs, libraries = [], []
    if os.environ.get("VSA2_REP_TK", "0") == "1":
        tk = os.path.abspath(os.environ.get("VSA2_REP_TK_ROOT", ""))
        if not os.path.isfile(os.path.join(tk, "include", "kittens.cuh")):
            raise RuntimeError(f"VSA2_REP_TK=1 needs VSA2_REP_TK_ROOT pointing at ThunderKittens (got {tk!r})")
        if "9.0a" not in os.environ.get("TORCH_CUDA_ARCH_LIST", ""):
            raise RuntimeError("VSA2_REP_TK=1 needs TORCH_CUDA_ARCH_LIST to contain 9.0a (wgmma / TMA)")
        sources += [os.path.join(ROOT, "csrc", "rep_tk", "rep_tk.cu"), os.path.join(src, "sm90_tk", "fine_tk_h100.cu")]
        include_dirs += [os.path.join(tk, "include"), os.path.join(tk, "prototype")]
        cxx.append("-DVSA2_REP_TK")
        nvcc += ["-DVSA2_REP_TK", "-DKITTENS_HOPPER", "-DNDEBUG"]
        # ThunderKittens' TMA descriptors use the CUDA driver API (cuTensorMapEncodeTiled)
        cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        for d in ("lib64/stubs", "lib/stubs"):
            if os.path.exists(os.path.join(cuda_home, d, "libcuda.so")):
                library_dirs, libraries = [os.path.join(cuda_home, d)], ["cuda"]
                break
        else:
            raise RuntimeError(f"no libcuda.so link stub under {cuda_home}")
    ext = CUDAExtension(name="vsa2_rep._C", sources=sources, include_dirs=include_dirs,
                        library_dirs=library_dirs, libraries=libraries,
                        extra_compile_args={"cxx": cxx, "nvcc": nvcc})
    return [ext], {"build_ext": BuildExtension}


kw = {}
if os.environ.get("VSA2_REP_BUILD", "0") == "1":
    kw["ext_modules"], kw["cmdclass"] = extension()

setup(name="vsa2-rep", version="0.1.0", packages=find_packages(include=["vsa2_rep*"]),
      python_requires=">=3.9", **kw)
