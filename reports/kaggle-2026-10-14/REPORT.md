# Inside a CNN: measured CPU/CUDA case study

Hardware: Intel(R) Xeon(R) CPU @ 2.00GHz, Tesla T4. The saved environment records the execution platform and available GPUs.
Source SHA-256: `8abc77c10910c0ff5f84fb7097a3304d2c651965b25ccdb6b238d4a4b1112056`. Experiment Git commit: `370e8a418f76c32129057a95eb7993f19095cd3f`.

## Model and dataset

One FP32 convolution with four 3x3 filters, stride 1, valid padding, ReLU, 2x2 max pooling, and a 676-to-10 linear classifier. Total: 6,810 trainable parameters.
Exactly 6,000 images from MNIST training data: 4,800 train and 1,200 validation. A separate 1,000-image subset of MNIST test data remains disjoint. Splits contain equal numbers of each digit. Input values use x/255.
Every trained backend starts with the same weights, shuffled indices, SGD learning rate 0.05, momentum 0.9, global batch size 64, seed 42 and 50 epochs. Two-GPU DataParallel assigns 32 examples to each GPU. Batch 1 uses only one GPU.

## Implementation boundary

Custom CPU and CUDA implement convolution forward, input gradient, weight gradient and bias gradient. The direct cuDNN backend invokes the cuDNN C API through ctypes. PyTorch supplies tensor storage, the remaining CNN layers, loss and SGD for all paths. Thus custom CUDA training is a hybrid CNN with a custom convolution, rather than a fully handwritten neural-network runtime.
The PyTorch CUDA baseline may use cuDNN internally. It tests framework dispatch and algorithm choice, rather than an independent hardware platform. CPU custom convolution uses one worker or four OpenMP workers. Shared classifier operations use four PyTorch CPU threads.

## Measured comparison

| Backend | One conv wall us | Batch 64 conv wall us | 50-epoch train s | Test accuracy | Single CNN inference us |
|---|---:|---:|---:|---:|---:|
| pytorch_cpu | 23.80 | 352.83 | 10.17 | 95.60% | 83.45 |
| cuda | 62.41 | 80.44 | 6.06 | 95.60% | 351.39 |
| cuda_dataparallel | 548.40 | 1637.85 | 15.99 | 95.60% | 913.92 |
| cudnn | 141.80 | 162.03 | 6.34 | 95.60% | 444.74 |
| cudnn_dataparallel | 633.95 | 1824.84 | 17.32 | 95.60% | 959.52 |
| pytorch_cuda | 41.68 | 72.58 | 5.42 | 95.60% | 379.61 |
| pytorch_dataparallel | 524.17 | 2539.44 | 18.39 | 95.60% | 867.08 |

## Timing meaning

Training time is the sum of synchronized epoch training intervals. It includes minibatch indexing, forward/backward computation, autograd integration and SGD. It excludes validation, checkpoint I/O, imports, downloads, compilation/warmup and the initial model/data transfer. Each of these separate times is retained in summary.json.
Single convolution is input 1x1x28x28, weights 4x1x3x3, output 1x4x26x26. It performs 24,336 multiply-accumulates, or 48,672 FLOPs excluding bias. Forward wall latency includes output allocation, dispatch and GPU completion synchronization. Inputs and weights are already resident. GPU event intervals use separate samples.
CNN inference measures a forward pass to 10 logits, without transfer or class selection. Batch 1 describes latency, while batch 64 describes throughput. Host-to-host convolution timing includes both transfers and allocation. Benchmarks retain warmup counts, repeated sample counts and p10/p90 latency spread.

## Parallel overhead, efficiency and cost

| CPU case | OpenMP speedup | Efficiency at 4 workers | 4*Tp - T1 ms |
|---|---:|---:|---:|
| mnist_single | 1.871x | 46.77% | 0.141225 |
| mnist_batch64 | 2.944x | 73.59% | 2.371089 |
| synthetic_larger | 2.255x | 56.38% | 45.766024 |

### One versus two GPUs

| Matching implementation | Training speedup | Efficiency on two GPUs | GPU-time cost ratio |
|---|---:|---:|---:|
| cuda_dataparallel | 0.379x | 18.96% | 5.275x |
| cudnn_dataparallel | 0.366x | 18.30% | 5.466x |
| pytorch_dataparallel | 0.295x | 14.74% | 6.782x |

Each GPU pair compares the same convolution implementation and global batch size. DataParallel replicates the model, scatters inputs, gathers outputs and reduces gradients. One image uses only one replica. Efficiency is T_one/(2*T_two) for the two identical GPUs. GPU-time cost ratio is 2*T_two/T_one and counts allocated GPU time, not measured energy.

CPU efficiency T1/(p*Tp) compares the same native convolution at one versus p OpenMP workers. GPU throughput is reported in images/s and calculated GFLOP/s. CUDA cores and CPU cores are heterogeneous, so dividing CPU-to-GPU speedup by a GPU core count does not produce a defensible parallel efficiency.
Aggregate host overhead is paired instrumented wall time minus the CUDA-event interval. It includes Python/C ABI calls, event management and synchronization. Separate empty-kernel and pageable H2D/D2H copy timings make overhead visible. These overhead measurements are diagnostic, and the event interval can include host submission gaps.
Computational cost is calculated MACs/FLOPs and logical tensor bytes. Memory cost is measured peak process RSS and the sum of per-device peak PyTorch GPU tensor allocation. Those peaks need not occur simultaneously. RSS includes the framework and context and is measured in a fresh backend process. GPU allocated bytes exclude driver/context memory and are not total board VRAM usage.
Energy was not measured. Monetary cost remains unspecified because no rental or electricity rate was provided. A configured per-GPU-hour rate can calculate a time-based charge without claiming that it measures electricity consumption or actual Kaggle quota accounting.

## Validation and limits

170 correctness checks passed. Forward and all convolution gradients match the PyTorch CPU reference within the declared tolerance. Checks cover multiple channels, batches, non-square inputs, partial CUDA tiles, a hand-computed case and an independent central-difference derivative.
This is one seed and one Kaggle runtime. Fifty epochs on a small balanced subset can overfit. Test accuracy measures this held-out 1,000-image subset, not the full MNIST test set. Timing samples are not independent repeated training trials. Runtime contention and host/framework overhead can affect rankings.
Training all paths for the same epoch count measures fixed-workload time. It does not establish equal time to a target accuracy. A slow custom CUDA result is valid evidence and must remain visible.

## Sources

- [PyTorch Conv2d](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Conv2d.html)
- [CUDA programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
- [CUDA timing semantics](https://docs.pytorch.org/docs/2.9/notes/cuda.html)
- [cuDNN 9.10.2 API](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.10.2/api/cudnn-cnn-library.html)
- [cuDNN support matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.10.2/reference/support-matrix.html)
- [MNIST reference loader and checksums](https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py)
