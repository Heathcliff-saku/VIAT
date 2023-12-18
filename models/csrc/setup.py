import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

# export LD_LIBRARY_PATH

os.environ.setdefault("CUDA_HOME", "/data/apps/cuda/11.3/")
os.environ.setdefault("LD_LIBRARY_PATH", "/data/apps/cuda/11.3/lib64")

ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]
# "helper_math.h" is copied from https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_math.h

sources = glob.glob('*.cpp')+glob.glob('*.cu')


setup(
    name='vren',
    version='2.0',
    author='kwea123',
    author_email='kwea123@gmail.com',
    description='cuda volume rendering library',
    long_description='cuda volume rendering library',
    ext_modules=[
        CUDAExtension(
            name='vren',
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={'cxx': ['-O2'],
                                'nvcc': ['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)