"""Read official IDX files and persist stratified, disjoint 10% splits."""
import gzip
import hashlib
import struct
import urllib.request
from pathlib import Path
import numpy as np
import torch

MNIST_URL='https://ossci-datasets.s3.amazonaws.com/mnist/'
FILES={
    'train-images-idx3-ubyte.gz':'f68b3c2dcbeaaa9fbdd348bbdeb94873',
    'train-labels-idx1-ubyte.gz':'d53e105ee54ea40749a09fcbcd1e9432',
    't10k-images-idx3-ubyte.gz':'9fb629c4189551a2d022fa330f9573f3',
    't10k-labels-idx1-ubyte.gz':'ec29112dd5afa0611ce80d1b7f02629c',
}

def download(root):
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    for filename,expected in FILES.items():
        target=root/filename
        if not target.exists():
            temporary=target.with_suffix('.partial')
            urllib.request.urlretrieve(MNIST_URL+filename,temporary)
            temporary.replace(target)
        actual=hashlib.md5(target.read_bytes()).hexdigest()
        if actual!=expected:
            raise ValueError(f'MNIST checksum mismatch: {filename}')

def read_idx(path):
    with gzip.open(path,'rb') as stream:
        header=stream.read(4)
        zero,dtype,dimensions=struct.unpack('>HBB',header)
        if zero!=0 or dtype!=8:
            raise ValueError('Expected unsigned-byte IDX data.')
        shape=struct.unpack('>'+'I'*dimensions,stream.read(4*dimensions))
        return np.frombuffer(stream.read(),dtype=np.uint8).reshape(shape).copy()

def balanced_indices(labels,count,rng):
    # Exactly 10 balanced classes, useful for a small controlled experiment.
    if count%10:
        raise ValueError('This experiment uses split sizes divisible by ten.')
    groups=[rng.permutation(np.flatnonzero(labels==digit))[:count//10] for digit in range(10)]
    return rng.permutation(np.concatenate(groups))

def prepare_data(root,config):
    root=Path(root)
    download(root)
    train_images=read_idx(root/'train-images-idx3-ubyte.gz')
    train_labels=read_idx(root/'train-labels-idx1-ubyte.gz')
    test_images=read_idx(root/'t10k-images-idx3-ubyte.gz')
    test_labels=read_idx(root/'t10k-labels-idx1-ubyte.gz')
    rng=np.random.default_rng(config['seed'])
    development=balanced_indices(train_labels,config['train_count']+config['validation_count'],rng)
    train_parts=[]
    validation_parts=[]
    for digit in range(10):
        group=development[train_labels[development]==digit]
        boundary=config['train_count']//10
        train_parts.append(group[:boundary])
        validation_parts.append(group[boundary:])
    train_indices=rng.permutation(np.concatenate(train_parts))
    validation_indices=rng.permutation(np.concatenate(validation_parts))
    test_indices=balanced_indices(test_labels,config['test_count'],rng)
    assert len(set(train_indices)&set(validation_indices))==0
    splits={'train':train_indices,'validation':validation_indices,'test':test_indices}
    np.savez(root/'split_indices.npz',**splits)
    data={}
    for name,indices in splits.items():
        source_x,source_y=(test_images,test_labels) if name=='test' else (train_images,train_labels)
        # Fixed normalization avoids estimating preprocessing from validation/test.
        x=source_x[indices].astype(np.float32)/255.0
        data[name]=(torch.from_numpy(x[:,None]),torch.from_numpy(source_y[indices].astype(np.int64)))
    return data,splits
