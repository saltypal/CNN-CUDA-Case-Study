"""Compile the small C ABI libraries without PyTorch C++ headers."""
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def build_native(cuda=True):
    build = ROOT / 'build'
    build.mkdir(exist_ok=True)
    source = Path(__file__).parent / 'native'
    compiler = shutil.which('g++-14') or shutil.which('g++')
    if compiler is None:
        raise RuntimeError('A Linux C++ compiler is required. On Windows use the supplied WSL command.')
    outputs = [build / 'libconv_cpu.so']
    if not outputs[0].exists() or outputs[0].stat().st_mtime < (source/'conv_cpu.cpp').stat().st_mtime:
        subprocess.run([compiler,'-O3','-std=c++17','-fPIC','-shared','-fopenmp',str(source/'conv_cpu.cpp'),'-o',str(outputs[0])],check=True)
    if cuda:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but no supported CUDA GPU is available.')
        major, minor = torch.cuda.get_device_capability()
        nvcc = shutil.which('nvcc')
        if nvcc is None:
            raise RuntimeError('nvcc is required for the custom CUDA backend.')
        gpu = build/'libconv_cuda.so'
        if not gpu.exists() or gpu.stat().st_mtime < (source/'conv_cuda.cu').stat().st_mtime:
            subprocess.run([nvcc,'-O3','-std=c++17','-ccbin',compiler,f'-arch=sm_{major}{minor}','-Xcompiler','-fPIC','-shared',str(source/'conv_cuda.cu'),'-o',str(gpu)],check=True)
        outputs.append(gpu)
    return outputs

if __name__ == '__main__':
    print('\n'.join(map(str,build_native())))
