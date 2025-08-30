# GFX906 Setup Guide

AMD has dropped support for GFX906 GPUs (MI50, etc.) for months.

Systems like Arch Linux with latest ML software stack updates will have problems running these GPUs.

This document will hopefully help you revive these cards.

## Host Environment

1. Arch Linux with latest packages installed. (no ROCM so far)
2. build tools installed. `gcc/clang/ninja`, etc.
3. package `downgrade` installed.
4. GFX906 hardwares installed.

## Caveats

### Softwares

1. Manually downgrade ROCm 6.4.x to 6.3.3
2. Recompile a lots of packages against ROCm 6.3.3
3. Many packages are not build-ready: need to edit sources. Mostly adding "gfx906" arch to sources.
4. Poor attention library support. Must manually build and install.
5. Poor quantization support. No `*_marlin`s. basically nothing except `fp16 or GGUF` in vLLM.


### Hardwares

1. No lower than `fp16`. no `bf*`.

## Software Stack for Text/Image/Video Generation

**for outdated packages, always use `downgrade` to install specific version.**

### ROCm 6.3.3

```sh
rocm-hip-libraries 6.3.3-1
rocm-hip-runtime 6.3.3-1
rocm-hip-sdk 6.3.3-1
rocm-opencl-runtime 6.3.3-1
rocm-opencl-sdk 6.3.3-1
hip-runtime-amd 6.3.3-1
comgr 6.3.3-2
composable-kernel 6.3.3-1
hipblas 6.3.3-1
hipblas-common 6.3.3-1
hipblaslt 6.3.3-1
hipcub 6.3.3-1
hipfft 6.3.3-1
hiprand 6.3.3-1
hipsolver 6.3.3-1
hipsparse 6.3.3-1
hsa-rocr 6.3.3-1
miopen-hip 6.3.3-1
rccl 6.3.3-1
rocalution 6.3.3-1
rocblas 6.3.3-1
rocfft 6.3.3-1
rocm-cmake 6.3.3-1
rocm-core 6.3.3-1
rocm-device-libs 6.3.3-2
rocminfo 6.3.3-1
rocm-llvm 6.3.3-2
rocm-smi-lib 6.3.3-1
rocprim 6.3.3-1
rocprofiler-register 6.3.3-1
rocrand 6.3.3-1
rocsolver 6.3.3-1
rocsparse 6.3.3-1
rocthrust 6.3.3-1
roctracer 6.3.3-1
amdsmi 6.3.3-1
rocprofiler 6.3.3-1
```

manually build and re-install rocblas [tag: 6.3.3-1]:

https://gitlab.archlinux.org/archlinux/packaging/packages/rocblas.git

```diff
diff --git a/PKGBUILD b/PKGBUILD
index 71b73da..5d77b98 100644
--- a/PKGBUILD
+++ b/PKGBUILD
@@ -67,6 +67,7 @@ build() {
     -D HIP_PLATFORM=amd
     -D BLAS_LIBRARY=cblas
     -D BUILD_WITH_TENSILE=ON
+    -D TENSILE_VERSION=4.43.0
     -D Tensile_LIBRARY_FORMAT=msgpack
     -D Tensile_TEST_LOCAL_PATH="$srcdir/$_tensile_dir"
     -D Tensile_COMPILER=hipcc
```

### Libraries

these are required by packages below.

```sh
protobuf 30.2-3
python-protobuf 30.2-3
abseil-cpp 20250127.1-2
grpc 1.72.0-1
python-grpcio 1.72.0-1
python-grpcio-tools 1.72.0-1
re2 1:20240702-4
arrow 19.0.1-1
python-pyarrow 19.0.1-1
apache-orc 2.0.3-6
```

if some packages fail:

- [tag 19.0.1-1] https://gitlab.archlinux.org/archlinux/packaging/packages/python-pyarrow.git 

### Torch

```sh
python-pytorch-opt-rocm 2.7.0-3
python-torchaudio-rocm 2.7.1-1
python-torchvision-rocm 0.22.1-1
torchvision-rocm 0.22.1-1
```

if some packages fail:

- [commit id 4a6d85c8fd526a484eec09f90c72b663490a3913] `git clone --branch python-torchaudio-rocm --single-branch https://github.com/archlinux/aur.git python-torchaudio-rocm`

### Recompile and install

#### bitsandbytes

[branch: rocm_enabled_multi_backend] https://github.com/ROCm/bitsandbytes.git


#### miopen-hip

see first patch

https://github.com/DKingAlpha/miopen-hip

#### orc

[tag: 0.4.41-1] https://gitlab.archlinux.org/archlinux/packaging/packages/orc.git

### triton

[branch: gfx906/main] https://github.com/nlzy/triton-gfx906.git

### vllm

- [latest] https://github.com/vllm-project/vllm.git

with patch for python 3.13 support:

```patch
diff --git a/CMakeLists.txt b/CMakeLists.txt
index e2cc0ccde..27b9599c5 100644
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -30,7 +30,7 @@ install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY TRUE)" ALL_COMPONENTS)
 # Supported python versions.  These versions will be searched in order, the
 # first match will be selected.  These should be kept in sync with setup.py.
 #
-set(PYTHON_SUPPORTED_VERSIONS "3.9" "3.10" "3.11" "3.12")
+set(PYTHON_SUPPORTED_VERSIONS "3.9" "3.10" "3.11" "3.12" "3.13")

 # Supported AMD GPU architectures.
 set(HIP_SUPPORTED_ARCHS "gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1200;gfx1201")
diff --git a/vllm/config.py b/vllm/config.py
index 44a8d871f..c05c2e58c 100644
--- a/vllm/config.py
+++ b/vllm/config.py
@@ -201,8 +201,18 @@ def get_attr_docs(cls: type[Any]) -> dict[str, str]:
         for b in iterator:
             yield a, b
             a = b
-
-    cls_node = ast.parse(textwrap.dedent(inspect.getsource(cls))).body[0]
+
+    try:
+        cls_node = ast.parse(textwrap.dedent(inspect.getsource(cls))).body[0]
+    except (OSError, KeyError, TypeError):
+        # getsource() fails in Python 3.13+ - use dataclass introspection
+        if is_dataclass(cls):
+            docs = {}
+            for field_obj in fields(cls):
+                docs[field_obj.name] = f"Configuration for {field_obj.name}."
+            return docs
+        else:
+            return {}

     if not isinstance(cls_node, ast.ClassDef):
         raise TypeError("Given object was not a class.")
```

### flash-attention (FA2)

[tag: v2.8.0.post2] https://github.com/Dao-AILab/flash-attention

```patch
diff --git a/setup.py b/setup.py
index cafc818f..919dd483 100644
--- a/setup.py
+++ b/setup.py
@@ -132,7 +132,7 @@ def rename_cpp_to_cu(cpp_files):

 def validate_and_update_archs(archs):
     # List of allowed architectures
-    allowed_archs = ["native", "gfx90a", "gfx950", "gfx942"]
+    allowed_archs = ["native", "gfx906", "gfx90a", "gfx950", "gfx942"]

     # Validate if each element in archs is in allowed_archs
     assert all(
```


### xformers

*Note: broken in comfy. `--use-flash-attention --disable-xformers` instead.

[tag: v0.0.32.post2] https://github.com/facebookresearch/xformers
[pretty much broken, dont use] https://github.com/ROCm/xformers.git

