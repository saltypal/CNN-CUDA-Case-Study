"""Interchangeable forward/backward convolution engines.

PyTorch owns tensors, pooling, the linear classifier, loss, and SGD in every
experiment. Custom C++/CUDA and direct cuDNN implement convolution only.
This controlled boundary is the central design decision of the case study.
"""
import atexit
import ctypes as ct
import sys
import threading
from pathlib import Path

import torch
import torch.nn.functional as F

from .build import ROOT, build_native

POINTER = ct.c_void_p
INT = ct.c_int
CPU = None
CUDA = None
CUDNN = {}
ENGINE_LOCK = threading.Lock()

def pointer(tensor):
    return POINTER(tensor.data_ptr()) if tensor is not None else POINTER()

def shape_arguments(x, weight):
    if x.dtype != torch.float32 or weight.dtype != torch.float32:
        raise TypeError('The teaching kernels support FP32 only.')
    if x.ndim != 4 or weight.ndim != 4 or weight.shape[1] != x.shape[1]:
        raise ValueError('Expected NCHW input and KCRS weights with matching channels.')
    n,c,h,width=x.shape
    k,_,r,s=weight.shape
    if r!=s or r>h or r>width:
        raise ValueError('Expected a square filter that fits the input.')
    return n,c,h,width,k,r

def native_library(gpu):
    global CPU, CUDA
    if gpu and CUDA is None:
        build_native(cuda=True)
        CUDA=ct.CDLL(str(ROOT/'build/libconv_cuda.so'))
        CUDA.conv_forward_cuda.argtypes=[POINTER]*4+[INT]*7+[POINTER]
        CUDA.conv_forward_cuda.restype=INT
        CUDA.conv_backward_cuda.argtypes=[POINTER]*6+[INT]*6+[POINTER]
        CUDA.conv_backward_cuda.restype=INT
        CUDA.launch_empty.argtypes=[POINTER]
        CUDA.launch_empty.restype=INT
    if not gpu and CPU is None:
        build_native(cuda=False)
        CPU=ct.CDLL(str(ROOT/'build/libconv_cpu.so'))
        CPU.conv_forward_cpu.argtypes=[POINTER]*4+[INT]*7
        CPU.conv_forward_cpu.restype=None
        CPU.conv_backward_cpu.argtypes=[POINTER]*6+[INT]*7
        CPU.conv_backward_cpu.restype=None
    return CUDA if gpu else CPU

def check_cuda(status):
    if status != 0:
        raise RuntimeError(f'Custom CUDA launch failed, runtime error code {status}')

class NativeConvolution(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, backend, threads):
        x=x.contiguous()
        weight=weight.contiguous()
        dimensions=shape_arguments(x,weight)
        n,c,h,width,k,r=dimensions
        y=torch.empty((n,k,h-r+1,width-r+1),device=x.device,dtype=x.dtype)
        lib=native_library(x.is_cuda)
        if x.is_cuda:
            with torch.cuda.device(x.device):
                stream=POINTER(torch.cuda.current_stream(x.device).cuda_stream)
                status=lib.conv_forward_cuda(pointer(x),pointer(weight),pointer(bias),pointer(y),*dimensions,int(backend=='cuda_shared'),stream)
                check_cuda(status)
        else:
            lib.conv_forward_cpu(pointer(x),pointer(weight),pointer(bias),pointer(y),*dimensions,threads)
        ctx.save_for_backward(x,weight)
        ctx.dimensions=dimensions
        ctx.threads=threads
        return y

    @staticmethod
    def backward(ctx, output_gradient):
        x,weight=ctx.saved_tensors
        output_gradient=output_gradient.contiguous()
        dx=torch.empty_like(x) if ctx.needs_input_grad[0] else None
        dw=torch.empty_like(weight)
        db=torch.empty(weight.shape[0],dtype=x.dtype,device=x.device)
        lib=native_library(x.is_cuda)
        arguments=[pointer(x),pointer(weight),pointer(output_gradient),pointer(dx),pointer(dw),pointer(db)]
        if x.is_cuda:
            with torch.cuda.device(x.device):
                status=lib.conv_backward_cuda(*arguments,*ctx.dimensions,POINTER(torch.cuda.current_stream(x.device).cuda_stream))
                check_cuda(status)
        else:
            lib.conv_backward_cpu(*arguments,*ctx.dimensions,ctx.threads)
        return dx,dw,db,None,None

class CudnnEngine:
    """Direct, fixed-function cuDNN API with cached NCHW descriptors.

The legacy API is deliberate here: it supports the laptop's Pascal GPU and
makes the operation boundary visible. This is distinct from nn.Conv2d.
"""
    def __init__(self,device):
        self.device=device
        self.lock=threading.RLock()
        candidates=[]
        for root in sys.path:
            candidates.extend(Path(root).glob('nvidia/cudnn/lib/libcudnn.so*'))
            candidates.extend(Path(root).glob('torch/lib/libcudnn.so*'))
        if not candidates:
            raise RuntimeError('The cuDNN library is missing from this Python environment.')
        self.lib=ct.CDLL(str(sorted(candidates)[0]))
        self.lib.cudnnGetErrorString.argtypes=[INT]
        self.lib.cudnnGetErrorString.restype=ct.c_char_p
        self.lib.cudnnGetVersion.restype=ct.c_size_t
        specifications={
            'cudnnCreate':[ct.POINTER(POINTER)],
            'cudnnDestroy':[POINTER],
            'cudnnSetStream':[POINTER,POINTER],
            'cudnnCreateTensorDescriptor':[ct.POINTER(POINTER)],
            'cudnnDestroyTensorDescriptor':[POINTER],
            'cudnnSetTensor4dDescriptor':[POINTER]+[INT]*6,
            'cudnnCreateFilterDescriptor':[ct.POINTER(POINTER)],
            'cudnnDestroyFilterDescriptor':[POINTER],
            'cudnnSetFilter4dDescriptor':[POINTER]+[INT]*6,
            'cudnnCreateConvolutionDescriptor':[ct.POINTER(POINTER)],
            'cudnnDestroyConvolutionDescriptor':[POINTER],
            'cudnnSetConvolution2dDescriptor':[POINTER]+[INT]*8,
            'cudnnSetConvolutionMathType':[POINTER,INT],
            'cudnnGetConvolutionForwardWorkspaceSize':[POINTER]*5+[INT,ct.POINTER(ct.c_size_t)],
            'cudnnGetConvolutionBackwardFilterWorkspaceSize':[POINTER]*5+[INT,ct.POINTER(ct.c_size_t)],
            'cudnnGetConvolutionBackwardDataWorkspaceSize':[POINTER]*5+[INT,ct.POINTER(ct.c_size_t)],
            'cudnnConvolutionForward':[POINTER]*7+[INT,POINTER,ct.c_size_t]+[POINTER]*3,
            'cudnnConvolutionBackwardFilter':[POINTER]*7+[INT,POINTER,ct.c_size_t]+[POINTER]*3,
            'cudnnConvolutionBackwardData':[POINTER]*7+[INT,POINTER,ct.c_size_t]+[POINTER]*3,
            'cudnnConvolutionBackwardBias':[POINTER]*7,
            'cudnnAddTensor':[POINTER]*7,
        }
        for name,arguments in specifications.items():
            fn=getattr(self.lib,name)
            fn.argtypes=arguments
            fn.restype=INT
        self.handle=POINTER()
        self.call('cudnnCreate',ct.byref(self.handle))
        self.cache={}
        self.one=ct.c_float(1)
        self.zero=ct.c_float(0)
        atexit.register(self.close)

    def call(self,name,*arguments):
        status=getattr(self.lib,name)(*arguments)
        if status:
            raise RuntimeError(f'{name}: {self.lib.cudnnGetErrorString(status).decode()}')

    def descriptor(self,kind):
        descriptor=POINTER()
        self.call(f'cudnnCreate{kind}Descriptor',ct.byref(descriptor))
        return descriptor

    def plan(self,x,weight):
        dimensions=shape_arguments(x,weight)
        if dimensions in self.cache:
            return self.cache[dimensions]
        n,c,h,width,k,r=dimensions
        p,q=h-r+1,width-r+1
        xd,yd,bd=[self.descriptor('Tensor') for _ in range(3)]
        wd=self.descriptor('Filter')
        cd=self.descriptor('Convolution')
        for desc,shape in [(xd,(n,c,h,width)),(yd,(n,k,p,q)),(bd,(1,k,1,1))]:
            self.call('cudnnSetTensor4dDescriptor',desc,0,0,*shape)
        self.call('cudnnSetFilter4dDescriptor',wd,0,0,k,c,r,r)
        self.call('cudnnSetConvolution2dDescriptor',cd,0,0,1,1,1,1,1,0)
        self.call('cudnnSetConvolutionMathType',cd,0)
        sizes=[]
        # Forward implicit GEMM, deterministic backward filter algorithm 1,
        # and backward-data algorithm 0. These choices remain fixed.
        for name,args,algo in [
            ('cudnnGetConvolutionForwardWorkspaceSize',(xd,wd,cd,yd),0),
            ('cudnnGetConvolutionBackwardFilterWorkspaceSize',(xd,yd,cd,wd),1),
            ('cudnnGetConvolutionBackwardDataWorkspaceSize',(wd,yd,cd,xd),0),
        ]:
            size=ct.c_size_t()
            self.call(name,self.handle,*args,algo,ct.byref(size))
            sizes.append(size.value)
        workspace=torch.empty(max(sizes),device=x.device,dtype=torch.uint8)
        plan=(xd,yd,bd,wd,cd,workspace)
        self.cache[dimensions]=plan
        return plan

    def prepare(self):
        self.call('cudnnSetStream',self.handle,POINTER(torch.cuda.current_stream().cuda_stream))

    def forward(self,x,weight,bias):
        with self.lock,torch.cuda.device(self.device):
            return self._forward(x,weight,bias)

    def _forward(self,x,weight,bias):
        xd,yd,bd,wd,cd,workspace=self.plan(x,weight)
        n,c,h,width,k,r=shape_arguments(x,weight)
        y=torch.empty((n,k,h-r+1,width-r+1),device=x.device,dtype=x.dtype)
        self.prepare()
        self.call('cudnnConvolutionForward',self.handle,ct.byref(self.one),xd,pointer(x),wd,pointer(weight),cd,0,pointer(workspace),workspace.numel(),ct.byref(self.zero),yd,pointer(y))
        self.call('cudnnAddTensor',self.handle,ct.byref(self.one),bd,pointer(bias),ct.byref(self.one),yd,pointer(y))
        return y

    def backward(self,x,weight,dy,need_dx):
        with self.lock,torch.cuda.device(self.device):
            return self._backward(x,weight,dy,need_dx)

    def _backward(self,x,weight,dy,need_dx):
        xd,yd,bd,wd,cd,workspace=self.plan(x,weight)
        dw=torch.empty_like(weight)
        db=torch.empty(weight.shape[0],device=x.device,dtype=x.dtype)
        dx=torch.empty_like(x) if need_dx else None
        self.prepare()
        self.call('cudnnConvolutionBackwardFilter',self.handle,ct.byref(self.one),xd,pointer(x),yd,pointer(dy),cd,1,pointer(workspace),workspace.numel(),ct.byref(self.zero),wd,pointer(dw))
        self.call('cudnnConvolutionBackwardBias',self.handle,ct.byref(self.one),yd,pointer(dy),ct.byref(self.zero),bd,pointer(db))
        if need_dx:
            self.call('cudnnConvolutionBackwardData',self.handle,ct.byref(self.one),wd,pointer(weight),yd,pointer(dy),cd,0,pointer(workspace),workspace.numel(),ct.byref(self.zero),xd,pointer(dx))
        return dx,dw,db

    def close(self):
        with torch.cuda.device(self.device):
            self._close()

    def _close(self):
        for xd,yd,bd,wd,cd,workspace in self.cache.values():
            for desc in [xd,yd,bd]:
                self.call('cudnnDestroyTensorDescriptor',desc)
            self.call('cudnnDestroyFilterDescriptor',wd)
            self.call('cudnnDestroyConvolutionDescriptor',cd)
        self.cache.clear()
        if self.handle:
            self.call('cudnnDestroy',self.handle)
            self.handle=POINTER()

def cudnn_engine(device=None):
    index=torch.cuda.current_device() if device is None else torch.device(device).index
    if index is None:
        index=torch.cuda.current_device()
    # DataParallel creates new worker threads on each forward call. Caching per
    # device avoids a new handle/workspace every iteration. Each handle is locked
    # while setting its stream and submitting operations.
    with ENGINE_LOCK:
        if index not in CUDNN:
            with torch.cuda.device(index):
                CUDNN[index]=CudnnEngine(index)
    return CUDNN[index]

class DirectCudnnConvolution(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight,bias):
        x,weight=x.contiguous(),weight.contiguous()
        ctx.save_for_backward(x,weight)
        return cudnn_engine(x.device).forward(x,weight,bias)

    @staticmethod
    def backward(ctx,dy):
        x,weight=ctx.saved_tensors
        return cudnn_engine(x.device).backward(x,weight,dy.contiguous(),ctx.needs_input_grad[0])

def convolution(x,weight,bias,backend,threads=4):
    if backend in ('pytorch_cpu','pytorch_cuda','pytorch_tpu'):
        return F.conv2d(x,weight,bias)
    if backend=='cudnn':
        if not x.is_cuda:
            raise ValueError('cuDNN requires CUDA tensors.')
        return DirectCudnnConvolution.apply(x,weight,bias)
    if backend not in ('cpu','cpu_openmp','cuda','cuda_shared'):
        raise ValueError(f'Unknown convolution backend: {backend}')
    if x.is_cuda != backend.startswith('cuda'):
        raise ValueError('Tensor device and selected convolution backend disagree.')
    return NativeConvolution.apply(x,weight,bias,backend,threads if backend=='cpu_openmp' else 1)
