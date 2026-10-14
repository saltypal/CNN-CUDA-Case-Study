# Inside a CNN: measured CPU/CUDA case study

Hardware: Intel(R) Xeon(R) CPU @ 2.00GHz, Tesla T4. The saved environment records the execution platform and available GPUs.
Source SHA-256: `a58ae19c93ef14abf8722034c45f4edd63913cd9720e07af6c463923b9929414`. Experiment Git commit: `42624fca4ba9b278076097e076363057b2d32a27`.

## Model and dataset

One FP32 convolution with four 3x3 filters, stride 1, valid padding, ReLU, 2x2 max pooling, and a 676-to-10 linear classifier. Total: 6,810 trainable parameters.
Full MNIST: 48,000 training and 12,000 validation images cover the entire official training collection. All 10,000 official test images remain separate. Development splits preserve digit proportions. Input values use x/255.
Every trained backend starts with the same weights, shuffled indices, SGD learning rate 0.05, momentum 0.9, global batch size 64, seed 42 and 50 epochs. Two-GPU DataParallel assigns 32 examples to each GPU. Batch 1 uses only one GPU.

## Implementation boundary

The five CNN training configurations use PyTorch CPU, direct cuDNN on one/two GPUs and PyTorch GPU on one/two GPUs. GPU work uses library CUDA kernels; no handwritten CUDA convolution configuration is included. Native serial/OpenMP CPU convolution remains an additional operation benchmark. PyTorch supplies tensor storage, remaining layers, loss and SGD.
The PyTorch CUDA baseline may use cuDNN internally. It tests framework dispatch and algorithm choice, rather than an independent hardware platform. CPU custom convolution uses one worker or four OpenMP workers. Shared classifier operations use four PyTorch CPU threads.

## Measured comparison

| Backend | One conv wall us | Batch 64 conv wall us | 50-epoch train s | Test accuracy | Single CNN inference us |
|---|---:|---:|---:|---:|---:|
| pytorch_cpu | 22.65 | 367.62 | 94.59 | 97.13% | 88.96 |
| cudnn | 140.33 | 162.42 | 62.49 | 97.17% | 452.48 |
| cudnn_dataparallel | 632.60 | 1771.76 | 165.78 | 97.17% | 961.70 |
| pytorch_cuda | 43.05 | 75.35 | 50.80 | 97.17% | 347.21 |
| pytorch_dataparallel | 529.62 | 2554.28 | 181.97 | 97.15% | 851.90 |

## Timing meaning

Training time is the sum of synchronized epoch training intervals. It includes minibatch indexing, forward/backward computation, autograd integration and SGD. It excludes validation, checkpoint I/O, imports, downloads, compilation/warmup and the initial model/data transfer. Each of these separate times is retained in summary.json.
Single convolution is input 1x1x28x28, weights 4x1x3x3, output 1x4x26x26. It performs 24,336 multiply-accumulates, or 48,672 FLOPs excluding bias. Forward wall latency includes output allocation, dispatch and GPU completion synchronization. Inputs and weights are already resident. GPU event intervals use separate samples.
CNN inference measures a forward pass to 10 logits, without transfer or class selection. Batch 1 describes latency, while batch 64 describes throughput. Host-to-host convolution timing includes both transfers and allocation. Benchmarks retain warmup counts, repeated sample counts and p10/p90 latency spread.

## Parallel overhead, efficiency and cost

| CPU case | OpenMP speedup | Efficiency at 4 workers | 4*Tp - T1 ms |
|---|---:|---:|---:|
| mnist_single | 1.887x | 47.17% | 0.142295 |
| mnist_batch64 | 3.003x | 75.08% | 2.235184 |
| synthetic_larger | 2.363x | 59.08% | 40.731608 |

### One versus two GPUs

| Matching implementation | Training speedup | Efficiency on two GPUs | GPU-time cost ratio |
|---|---:|---:|---:|
| cudnn_dataparallel | 0.377x | 18.85% | 5.306x |
| pytorch_dataparallel | 0.279x | 13.96% | 7.164x |

Each GPU pair compares the same convolution implementation and global batch size. DataParallel replicates the model, scatters inputs, gathers outputs and reduces gradients. One image uses only one replica. Efficiency is T_one/(2*T_two) for the two identical GPUs. GPU-time cost ratio is 2*T_two/T_one and counts allocated GPU time, not measured energy.

CPU efficiency T1/(p*Tp) compares the same native convolution at one versus p OpenMP workers. GPU throughput is reported in images/s and calculated GFLOP/s. CUDA cores and CPU cores are heterogeneous, so dividing CPU-to-GPU speedup by a GPU core count does not produce a defensible parallel efficiency.
Aggregate host overhead is paired instrumented wall time minus the CUDA-event interval. It includes Python/C ABI calls, event management and synchronization. Pageable H2D/D2H copy timings make data-movement overhead visible. These measurements are diagnostic, and the event interval can include host submission gaps.
Computational cost is calculated MACs/FLOPs and logical tensor bytes. Memory cost is measured peak process RSS and the sum of per-device peak PyTorch GPU tensor allocation. Those peaks need not occur simultaneously. RSS includes the framework and context and is measured in a fresh backend process. GPU allocated bytes exclude driver/context memory and are not total board VRAM usage.
Energy was not measured. Monetary cost remains unspecified because no rental or electricity rate was provided. A configured per-GPU-hour rate can calculate a time-based charge without claiming that it measures electricity consumption or actual Kaggle quota accounting.

## Validation and limits

119 correctness checks passed. Forward and all convolution gradients match the PyTorch CPU reference within the declared tolerance. Checks cover multiple channels, batches, non-square inputs, partial CUDA tiles, a hand-computed case and an independent central-difference derivative.
This is one seed and one Kaggle runtime. Test accuracy uses all 10,000 official MNIST test images. Timing samples are not independent repeated training trials. Runtime contention and host/framework overhead can affect rankings.
Training all paths for the same epoch count measures fixed-workload time. It does not establish equal time to a target accuracy. Slower results remain visible in the measured comparison.

## Sources

- [PyTorch Conv2d](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Conv2d.html)
- [CUDA programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
- [CUDA timing semantics](https://docs.pytorch.org/docs/2.9/notes/cuda.html)
- [cuDNN 9.10.2 API](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.10.2/api/cudnn-cnn-library.html)
- [cuDNN support matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.10.2/reference/support-matrix.html)
- [MNIST reference loader and checksums](https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py)
