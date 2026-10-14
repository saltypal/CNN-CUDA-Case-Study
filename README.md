# CNN CUDA Case Study

**Inside a CNN: 2D convolution on CPU, one GPU and two GPUs.**

A small MNIST classifier with interchangeable convolution engines and a reproducible
Kaggle experiment. The goal is to connect the convolution equation to CPU loops,
CUDA thread mapping, cuDNN, and PyTorch DataParallel, then measure each path.

**Current status:** local native CPU/direct CUDA/shared CUDA forward and gradient
checks passed. Kaggle framework validation and 50-epoch results are pending. No
performance numbers are claimed before those runs complete.

## Training comparisons

| Path | Devices | Convolution |
|---|---:|---|
| PyTorch CPU | CPU, four framework threads | PyTorch CPU backend |
| Custom CUDA | One GPU | Handwritten CUDA C++ |
| Custom CUDA DataParallel | Two GPUs | Same CUDA kernel in each replica |
| Direct cuDNN | One GPU | Direct cuDNN C API |
| Direct cuDNN DataParallel | Two GPUs | A handle/workspace for each GPU |
| PyTorch CUDA | One GPU | Framework convolution backend |
| PyTorch DataParallel | Two GPUs | Framework convolution in each replica |

Native serial/OpenMP CPU and tiled CUDA convolution add operation benchmarks and
correctness controls. Only convolution differs in the custom paths. PyTorch owns
tensor storage and supplies ReLU, pooling, the dense classifier, loss and SGD.

## Model and data

| Stage | Per-image shape |
|---|---|
| Input | 1 x 28 x 28 |
| Four 3x3 filters, stride 1, valid padding | 4 x 26 x 26 |
| ReLU | 4 x 26 x 26 |
| 2x2 max pool | 4 x 13 x 13 |
| Flatten | 676 |
| Linear classifier | 10 logits |

The model has **6,810 parameters**. One image's convolution performs 24,336 MACs,
or 48,672 FLOPs excluding bias addition.

Exactly 6,000 images from MNIST's training collection are selected: 4,800 train
and 1,200 validation. Another 1,000 images come from its separate test collection.
Each split is balanced across ten digits, disjoint, and saved by exact indices.
Pixel scaling is x/255.

All training paths use 50 epochs, seed 42, the same initial weights and batch
orders, FP32, SGD learning rate 0.05 and momentum 0.9. The **global batch stays
64** for one and two GPUs. DataParallel assigns 32 images per GPU. Loss follows
gathering, preserving the global mean. Batch-one inference uses one replica.

## Run on Kaggle

Import [the notebook](notebooks/experiment.ipynb), select **GPU T4 x2**, enable
Internet and run all cells. Save a notebook version with outputs to retain results.

The first cell removes only a named temporary source copy, clones this GitHub
repository, checks out SOURCE_REF and records the exact commit. Set SOURCE_REF
to an earlier commit to reproduce that source. Results live separately under
/kaggle/working/cnn-cuda-case-study-results.

## Run locally under Linux or WSL

The two-GPU cases require two supported NVIDIA GPUs, a compatible PyTorch build,
cuDNN and nvcc. Native CPU convolution needs a Linux C++ compiler with OpenMP.

```bash
git clone https://github.com/saltypal/CNN-CUDA-Case-Study.git
cd CNN-CUDA-Case-Study
export PYTHONPATH="$PWD/src"
python -m hpc_cnn.run --run-root "$PWD/results/main"
```

Use --skip-completed to resume matching completed runs. Configuration is in
config.json. The portable notebook uses the same source modules.

## Metrics

- Single MNIST convolution, batch-64 convolution and larger synthetic scaling.
- Synchronized 50-epoch training time, validation time, setup and initial transfer.
- CNN inference latency at batch one and throughput at batch 64.
- One-versus-two GPU speedup and efficiency for each matching engine.
- Dispatch/completion overhead, transfers and a DataParallel identity diagnostic.
- Process RSS, GPU tensor allocation, calculated FLOPs/bytes and GPU-time cost.

Training time excludes warmup, validation and checkpoint I/O, which are recorded
separately. GPU wall timing waits for completed work. Separate CUDA-event samples
describe execution intervals and can include submission gaps. Monetary cost needs
an explicit per-GPU-hour price. Energy is not inferred from speed.

## Code map

| File | Responsibility |
|---|---|
| data.py | Official IDX download, checksums and saved balanced splits |
| native/conv_cpu.cpp | Serial/OpenMP forward and backward convolution |
| native/conv_cuda.cu | Direct/tiled kernels and gradient reductions |
| backends.py | Tensor pointers, current streams and direct cuDNN |
| model.py | Classifier and DataParallel wrapping |
| validate.py | Reference, numerical and multi-GPU correctness gates |
| train.py | Controlled training and checkpoints |
| benchmark.py | Latency, transfer, overhead and efficiency measurements |
| run.py | Independent backend processes |
| report.py | CSV, report and scientific plots |
| demo.py | Same held-out digits through final checkpoints |

All modules live in src/hpc_cnn. Important contracts appear in [DESIGN.md](DESIGN.md).

## Demo

```bash
PYTHONPATH=src python -m hpc_cnn.demo --run-root results/main
```

Each run records the Git commit, source hash, hardware, package/compiler versions,
config, split indices, epoch history and checkpoints. One seed and timing repeats
do not establish statistical robustness. DataParallel may lose on this small CNN.
Negative results remain part of the evidence.
