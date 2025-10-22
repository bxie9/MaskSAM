import setuptools
import os
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# By default, we also build the SAM 2 CUDA extension.
# You may turn off CUDA build with `export SAM2_BUILD_CUDA=0`.
BUILD_CUDA = os.getenv("SAM2_BUILD_CUDA", "1") == "1"
# By default, we allow SAM 2 installation to proceed even with build errors.
# You may force stopping on errors with `export SAM2_BUILD_ALLOW_ERRORS=0`.
BUILD_ALLOW_ERRORS = os.getenv("SAM2_BUILD_ALLOW_ERRORS", "1") == "1"

# Catch and skip errors during extension building and print a warning message
# (note that this message only shows up under verbose build mode
# "pip install -v -e ." or "python setup.py build_ext -v")
CUDA_ERROR_MSG = (
    "{}\n\n"
    "Failed to build the SAM 2 CUDA extension due to the error above. "
    "You can still use SAM 2, but some post-processing functionality may be limited "
    "(see https://github.com/facebookresearch/segment-anything-2/blob/main/INSTALL.md).\n"
)


def get_extensions():
    if not BUILD_CUDA:
        return []

    try:
        srcs = ["nnunetv2/sam2/csrc/connected_components.cu"]
        compile_args = {
            "cxx": [],
            "nvcc": [
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ],
        }
        ext_modules = [CUDAExtension("sam2._C", srcs, extra_compile_args=compile_args)]
    except Exception as e:
        if BUILD_ALLOW_ERRORS:
            print(CUDA_ERROR_MSG.format(e))
            ext_modules = []
        else:
            raise e

    return ext_modules


class BuildExtensionIgnoreErrors(BuildExtension):

    def finalize_options(self):
        try:
            super().finalize_options()
        except Exception as e:
            print(CUDA_ERROR_MSG.format(e))
            self.extensions = []

    def build_extensions(self):
        try:
            super().build_extensions()
        except Exception as e:
            print(CUDA_ERROR_MSG.format(e))
            self.extensions = []

    def get_ext_filename(self, ext_name):
        try:
            return super().get_ext_filename(ext_name)
        except Exception as e:
            print(CUDA_ERROR_MSG.format(e))
            self.extensions = []
            return "_C.so"

if __name__ == "__main__":
    setuptools.setup(
        ext_modules=get_extensions(),
    )
