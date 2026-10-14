import math
import torch
from torch import nn
from .backends import convolution
from .backends import native_library,cudnn_engine

PARALLEL_BACKENDS={'cuda_dataparallel':'cuda','cuda_shared_dataparallel':'cuda_shared',
    'cudnn_dataparallel':'cudnn','pytorch_dataparallel':'pytorch_cuda'}

def is_gpu_backend(backend):
    return backend in ('cuda','cuda_shared','cudnn','pytorch_cuda') or backend in PARALLEL_BACKENDS

def unwrap(model):
    return model.module if isinstance(model,nn.DataParallel) else model

def create_model(backend,channels=4,threads=4):
    base=PARALLEL_BACKENDS.get(backend,backend)
    return SmallCNN(base,channels,threads)

class ConvolutionProbe(nn.Module):
    """A convolution alone, including scatter/replication/gather when wrapped."""
    def __init__(self,weight,bias,backend,threads=4,trainable=True):
        super().__init__()
        self.weight=nn.Parameter(weight,requires_grad=trainable)
        self.bias=nn.Parameter(bias,requires_grad=trainable)
        self.backend=PARALLEL_BACKENDS.get(backend,backend)
        self.threads=threads

    def forward(self,x):
        return convolution(x,self.weight,self.bias,self.backend,self.threads)

def place_model(model,backend):
    device=torch.device('cuda:0' if is_gpu_backend(backend) else 'cpu')
    model=model.to(device)
    if backend in PARALLEL_BACKENDS:
        if torch.cuda.device_count()<2:
            raise RuntimeError(f'{backend} requires two GPUs. Select Kaggle GPU T4 x2.')
        if backend.startswith('cuda'):
            native_library(True)
        if backend=='cudnn_dataparallel':
            for index in (0,1):
                cudnn_engine(torch.device(f'cuda:{index}'))
        model=nn.DataParallel(model,device_ids=[0,1],output_device=0)
    return model,device

class SmallCNN(nn.Module):
    """1x28x28, convolution, ReLU, 2x2 pool, 676 features, 10 logits."""
    def __init__(self,backend,channels=4,threads=4):
        super().__init__()
        self.backend=backend
        self.threads=threads
        self.conv_weight=nn.Parameter(torch.empty(channels,1,3,3))
        self.conv_bias=nn.Parameter(torch.zeros(channels))
        self.classifier=nn.Linear(channels*13*13,10)
        nn.init.kaiming_uniform_(self.conv_weight,a=math.sqrt(5))
        nn.init.uniform_(self.conv_bias,-1/3,1/3)

    def forward(self,x):
        features=convolution(x,self.conv_weight,self.conv_bias,self.backend,self.threads)
        features=torch.relu(features)
        features=torch.nn.functional.max_pool2d(features,2)
        return self.classifier(features.flatten(1))
