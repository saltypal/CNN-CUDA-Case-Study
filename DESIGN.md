# Experiment plan and acceptance gates

The objective is a simple MNIST classifier and measured convolution, training,
inference, overhead and resource-cost comparisons. Keep all implementation and
persistent results in this project directory. The source is versioned on GitHub.
The Kaggle notebook fetches the exact/latest commit and saves results separately.

## Current full-MNIST configuration

`config_full_mnist.json` uses 48,000 training, 12,000 validation and all 10,000 official test images. It preserves the four-filter CNN, 50 epochs, global batch 64, seed 42, FP32 and SGD settings.

The five training configurations are PyTorch CPU, direct cuDNN on one/two GPUs, and PyTorch GPU on one/two GPUs. Handwritten CUDA configurations are excluded. Native serial/OpenMP CPU convolution remains an operation benchmark. The full study needs the CUDA/cuDNN runtime and a C++ compiler for that CPU benchmark, without compiling a handwritten GPU convolution.

The implementation contracts below also describe code retained from the original 10% experiment. Its custom CUDA kernels remain historical source and are not part of the current full-dataset comparison. Current evidence is in `reports/full-mnist-2026-10-14`.

## Scope and contracts

All custom convolutions accept contiguous FP32 NCHW tensors, KCRS filters and a
per-output-channel bias. They implement valid, stride-one cross-correlation with
square filters. Their output is N x K x (H-R+1) x (W-R+1). Groups, dilation,
nonzero padding and mixed precision are outside this small case study.

PyTorch owns storage. Native libraries borrow tensor addresses and execute on the
current CUDA stream. They never free the caller's tensor memory. Each forward
output and each input-gradient element has one writer. CPU parameter gradients
use independent reductions. CUDA parameter gradients use a shared-memory block
reduction instead of concurrent unsynchronized writes.

The tiled CUDA kernel uses a 16x16 output tile and an (16+R-1)^2 input tile per
channel. Its halo supports the overlapping filter footprints. All threads,
including out-of-output-range threads, reach the loading barriers.

Direct cuDNN uses cached descriptors and a workspace, with fixed algorithms:
forward implicit GEMM (0), backward-filter algorithm 1, backward-data algorithm 0.
The API supports the laptop's Pascal GPU. PyTorch CUDA uses its own backend
selection, with cuDNN enabled, deterministic convolution and TF32 disabled.

## Gates

1. Detect the real GPU and verify native compilation and device execution.
2. Verify forward and every gradient against an independent framework reference.
   Cover multiple channels, batches, non-square inputs and partial blocks.
3. Verify a hand-calculated convolution and a numerical derivative.
4. Train every requested backend for exactly 50 epochs, with finite loss and the
   same initial weights, data and optimizer settings.
5. Preserve checkpoints, history, config, environment and split indices.
6. Measure latency only after warmup. Synchronize GPU completion. Keep transfers,
   allocation, initialization, validation and checkpoint time scopes explicit.
7. Generate the comparison report and presentation only from completed results.

## Risks and interpretation

Pascal requires a compatible CUDA/PyTorch/cuDNN combination. The laptop has only
2 GB of GPU memory. Small convolutions can be slower on a GPU because launch,
dispatch and transfer cost dominate. Shared memory can also lose to direct
convolution when its barriers and halo loads cost more than the reuse saves.

CPU speedup and efficiency compare one and four OpenMP workers on the same
compiled convolution. GPU and CPU cores cannot be counted as equal workers.
Report GPU throughput, memory and overhead without inventing a cross-platform
parallel-efficiency percentage.

One seed is enough for a reproducible classroom case study, but cannot establish
statistical robustness. Repeat latency samples characterize local timing spread.
They do not characterize variation across independent training runs.

The MX330 has no usable power telemetry in this setup. Energy cost is unavailable.
Monetary compute cost needs a configured price. Calculated FLOPs and logical bytes
must remain distinct from measured wall time, process RSS and tensor allocation.

## Two-GPU integration

DataParallel keeps global batch size 64 and scatters 32 examples per device.
cuDNN handles and workspaces belong to each device and survive worker-thread
lifetimes. Stream submission is protected by a per-handle lock. Native kernels
guard the tensor device before launching. Loss follows gathering, preserving
global normalization. Checkpoints save the unwrapped model.
