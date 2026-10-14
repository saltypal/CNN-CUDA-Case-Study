import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime,timezone
from pathlib import Path
import torch
from .build import ROOT

def read_config(path=None):
    return json.loads(Path(path or ROOT/'config.json').read_text())

def configure(config):
    torch.set_num_threads(config['cpu_threads'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False

def synchronize(device):
    if torch.device(device).type=='cuda':
        # Include work on both devices for DataParallel wall-clock boundaries.
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)

def source_hash():
    digest=hashlib.sha256()
    for path in sorted((ROOT/'src').rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            digest.update(path.relative_to(ROOT).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()

def environment(include_gpu_context=True):
    result={
        'utc_time':datetime.now(timezone.utc).isoformat(),
        'python':platform.python_version(),
        'platform':platform.platform(),
        'torch':torch.__version__,
        'cuda_runtime':torch.version.cuda,
        'cudnn_version':torch.backends.cudnn.version(),
        'source_sha256':source_hash(),
        'git_commit':None,
        'cpu_logical_count':os.cpu_count(),
        'torch_cpu_threads':torch.get_num_threads(),
    }
    try:
        result['git_commit']=subprocess.run(['git','-C',str(ROOT),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
        result['source_dirty']=bool(subprocess.run(['git','-C',str(ROOT),'diff','--name-only','HEAD','--','src'],capture_output=True,text=True,check=True).stdout.strip())
    except (OSError,subprocess.CalledProcessError):
        result['source_dirty']=None
    try:
        cpu=Path('/proc/cpuinfo').read_text()
        result['cpu_name']=next(line.split(':',1)[1].strip() for line in cpu.splitlines() if line.startswith('model name'))
    except (OSError,StopIteration):
        result['cpu_name']=platform.processor()
    if include_gpu_context and torch.cuda.is_available():
        prop=torch.cuda.get_device_properties(0)
        result['gpu']={'name':prop.name,'compute_capability':[prop.major,prop.minor],
            'multiprocessors':prop.multi_processor_count,'total_memory_bytes':prop.total_memory}
        result['gpus']=[{'index':i,'name':torch.cuda.get_device_properties(i).name,
            'compute_capability':list(torch.cuda.get_device_capability(i)),
            'total_memory_bytes':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]
    for name,command in [('cuda_compiler',['nvcc','--version']),('cpu_compiler',['g++-14','--version'])]:
        try:
            result[name]=subprocess.run(command,capture_output=True,text=True,check=True).stdout.strip()
        except (OSError,subprocess.CalledProcessError):
            result[name]=None
    return result

def write_json(path,data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.partial')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)
