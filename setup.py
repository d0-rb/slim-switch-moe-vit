from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

CUDA_HELPER = os.environ.get('CUDA_HELPER', '/usr/local/cuda/samples/common/inc')
cxx_flags = [
        '-I{}'.format(CUDA_HELPER)
        ]
if os.environ.get('USE_NCCL', '0') == '1':
    cxx_flags.append('-DMOE_USE_NCCL')

setup(
    name='moe_cuda',
    ext_modules=[
        CUDAExtension(
            name='moe_cuda', 
            sources=[
                'cuda/moe.cpp',
                'cuda/cuda_stream_manager.cpp',
                'cuda/moe_compute_kernel.cu',
                'cuda/moe_comm_kernel.cu',
                'cuda/moe_fused_kernel.cu',
                ],
            extra_compile_args={
                'cxx': cxx_flags,
                'nvcc': cxx_flags
                }
            )
        ],
    cmdclass={
        'build_ext': BuildExtension
    })
