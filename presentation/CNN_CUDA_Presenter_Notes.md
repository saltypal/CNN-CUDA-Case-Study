# CNN CUDA Case Study — presenter notes

24 slides. Presenter 1: slides 1–8. Presenter 2: slides 9–16. Presenter 3: slides 17–24.

The expanded presentation teaches the algorithms first, then connects them to the completed Kaggle measurements.

## Slide 1: CNN CUDA Case Study

Presenter 1

The focus is how the same convolution becomes parallel work. Presenter 1 covers the equation, CPU memory access and OpenMP. Presenter 2 explains CUDA mapping, memory reuse, synchronization and gradient algorithms. Presenter 3 connects library algorithms and two-GPU execution to the measured experiment and live demo.

We keep the CNN small so the important work can be traced. The custom implementations replace convolution; PyTorch still supplies ReLU, pooling, the linear classifier, loss and SGD. The measurements are the completed Kaggle run, not a new experiment. The title photo comes from the retained Simple Dark Mode template.

References:


## Slide 2: The small CNN classifier

Presenter 1

A convolutional neural network uses local connections and shared filter weights. The spatial dimensions describe where a feature occurs. The channel dimension describes which feature response we are looking at. A feature map is a learned filter's response over the image.

Our batch input has shape N x 1 x 28 x 28 in NCHW order. One grayscale channel enters four 3x3 filters. Valid stride-one convolution produces N x 4 x 26 x 26. ReLU clamps negative responses to zero. A 2x2 maximum pool with stride two produces N x 4 x 13 x 13. Flattening produces 676 features per digit. A linear layer converts those features to ten logits. Cross entropy compares the logits with the integer label during training. Softmax is unnecessary before PyTorch cross_entropy, because the loss handles the required normalization internally.

The convolution has 4*1*3*3 = 36 weights and four biases. The classifier has 676*10 = 6,760 weights and ten biases. Total parameters: 6,810. A convolution is the local weighted sum. ReLU, pooling and the dense classifier are separate operations.

We scale pixels to [0,1] using x/255. The training collection contributes exactly 6,000 selected examples: 4,800 training and 1,200 validation. The separate official test collection contributes 1,000 examples. Equal numbers from each digit keep these small subsets balanced.

References:
https://docs.pytorch.org/docs/2.9/generated/torch.nn.Conv2d.html
https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py

Start with the data path. At every stage, name the independent dimension: images, output channels, pixels or features. These dimensions provide different opportunities for parallel execution.

## Slide 3: The 2D convolution operation

Presenter 1

The tiny example uses a 2x2 filter so every multiplication fits on the slide. It is also an exact correctness test in the implementation. With no filter reversal, each 2x2 patch gives -4. CNN libraries normally call this operation convolution, but the mathematical operation is cross-correlation.

For the implemented valid stride-one layer, y[n,k,p,q] = bias[k] + sum over c,r,s of x[n,c,p+r,q+s] * weight[k,c,r,s]. The filter combines all input channels. One output channel has a separate filter bank and one bias.

With general stride S, padding P and dilation D, output height is floor((H + 2P - D*(R-1) - 1)/S) + 1, and width follows the same rule. Our small kernels intentionally support S=1, P=0 and D=1.

For one MNIST image, N=1, C=1, K=4, H=W=28 and R=S=3. There are 4*26*26 = 2,704 output elements, with nine multiply-accumulates each. This gives 24,336 MACs. Counting one multiply and one addition as two floating-point operations gives 48,672 FLOPs excluding bias addition. These are calculated operation counts, not measured instruction counts or elapsed times.

During training, the gradient of each filter weight sums input values multiplied by the upstream output gradient. Input gradients gather all output contributions whose receptive fields cover that input pixel. Bias gradients sum the upstream gradients of the corresponding output channel.

References:
https://docs.pytorch.org/docs/2.9/generated/torch.nn.Conv2d.html

## Slide 4: Where convolution can run in parallel

Presenter 1

An output element is a weighted sum over its receptive field. It depends on input pixels and filter weights, but not on another output element. This is the independence that makes the outer output loop parallelizable.

For this classifier, each output has nine multiply-accumulates. Splitting such a small sum among workers would introduce communication and reduction overhead for little arithmetic. Instead, each worker computes a complete output. Independent workers may read overlapping input pixels safely because reading does not change them.

One image has 2,704 work items; a full batch of 64 has 173,056. Those are logical output work items, not the number of physical CPU cores or CUDA execution units. There are also independent sample and output-filter dimensions. Later we use sample parallelism across GPUs and output parallelism inside each GPU.

The same equation is used by serial C++, OpenMP, direct CUDA and the shared-memory CUDA kernel. The parallel version changes ownership and execution order, not the intended result.

References:
https://github.com/saltypal/CNN-CUDA-Case-Study/blob/370e8a418f76c32129057a95eb7993f19095cd3f/src/hpc_cnn/native/conv_cpu.cpp
https://github.com/saltypal/CNN-CUDA-Case-Study/blob/370e8a418f76c32129057a95eb7993f19095cd3f/src/hpc_cnn/native/conv_cuda.cu

## Slide 5: The serial CPU algorithm

Presenter 1

Trace a single output completely before discussing parallelism. Decode the flattened output index into sample, filter, row and column. Initialize the accumulator with the filter bias. Walk every input channel and spatial filter offset, multiply an input value by its corresponding weight, and add the product. Finally store the accumulator.

The handwritten baseline is compiled C++ with -O3; it is not a Python loop benchmark. The actual implementation uses explicit channel, filter-row and filter-column loops. Its runtime grows with the product of batch size, filter count, output dimensions, input channels and filter area.

The one-worker label refers to the native convolution operation. PyTorch CPU training is a separate baseline using its own convolution implementation. Avoid mixing their timings. A mature CPU library can apply blocking, vectorization and optimized kernels that this clear teaching implementation does not explicitly implement.

Useful demonstration: point to the accumulator and ask which worker owns it. In serial execution, one CPU thread owns each output in sequence. In OpenMP, different threads own different outputs while using the same inner sum.

References:
https://github.com/saltypal/CNN-CUDA-Case-Study/blob/370e8a418f76c32129057a95eb7993f19095cd3f/src/hpc_cnn/native/conv_cpu.cpp

## Slide 6: CPU memory access and locality

Presenter 1

A tensor shape describes logical coordinates; a contiguous tensor also has a physical layout. In NCHW order, width is the fastest-changing coordinate. The displayed address formula is exactly the indexing used by the native implementation.

Adjacent filter-column accesses are adjacent values. Nearby output windows revisit many pixels. CPU caches can reuse those values, so a source-code load is not automatically a new DRAM read. Locality reduces the cost of obtaining operands. Adding threads can increase arithmetic throughput, but the cores still compete for shared memory resources.

SIMD means one CPU instruction operates on several data lanes. OpenMP means several CPU threads execute work. These are compatible but different forms of parallelism. Our source has no explicit SIMD intrinsic or omp simd directive. Compilation uses optimization, but no disassembly or vectorization report was collected, so do not claim a measured SIMD width or SIMD speedup.

Keep output ownership contiguous where practical: this improves locality and limits cache-line sharing between workers. Blocking and explicit SIMD are possible extensions, not measured features of this implementation.

References:
https://github.com/saltypal/CNN-CUDA-Case-Study/blob/370e8a418f76c32129057a95eb7993f19095cd3f/src/hpc_cnn/native/conv_cpu.cpp

## Slide 7: OpenMP divides the output loop

Presenter 1

The illustration uses 16 toy outputs and four workers so the division is visible. It is schematic, not an exact trace of a particular OpenMP runtime. In the real layer the iteration space is N*K*P*Q.

The actual directive also has if(threads > 1), which bypasses a parallel team for the one-worker path. The parallel-for construct creates a team and assigns iterations. schedule(static) avoids repeatedly requesting new work; similarly sized outputs make that a sensible choice. Without an explicit chunk size, exact chunk sizes are runtime-dependent, so do not promise a particular boundary.

The accumulator and decoded coordinates are declared inside the loop body and are private to each iteration. Input, weights and bias point to shared read-only arrays. Output is shared storage with non-overlapping element ownership. There is no output data race and no atomic operation is needed. The implicit barrier completes before the function returns.

OpenMP does not discover the numerical algorithm for us. We choose the safe iteration space and data ownership; the runtime schedules that work on CPU threads.

References:
https://github.com/saltypal/CNN-CUDA-Case-Study/blob/370e8a418f76c32129057a95eb7993f19095cd3f/src/hpc_cnn/native/conv_cpu.cpp
https://www.openmp.org/spec-html/5.2/openmpse66.html

Illustration: generated explanatory schematic, checked against the implementation. Diagram labels show the conceptual work division; they are not a hardware execution trace.

## Slide 8: OpenMP correctness and scaling

Presenter 1

Both times compare the same handwritten C++ convolution with one and four workers. They are warmed median completed-call wall latencies. Batch-64 speedup is 6608.54/2244.91, approximately 2.94; efficiency against four workers is about 73.6%. The one-image gain is about 1.87. We measured four logical CPU processors, not four independently verified physical cores.

Forward needs no reduction between threads because each output is stored once. Backward is different: each weight gradient is a sum over all samples and output positions. Our CPU implementation assigns each weight sum to one output owner, then performs that owner's inner sum serially. Bias uses one owner per filter. Input gradients gather contributions to one input value. This avoids atomics and shared accumulators.

A common incorrect parallelization has every thread update the same sum variable. Unsynchronized read-modify-write loses contributions. Local partial sums plus a deliberate reduction repair that design.

Ideal fourfold scaling would require all cost to divide perfectly among four equal workers. Team overhead, barriers, data access and shared hardware prevent that ideal. These causes explain plausible limits; no hardware-counter analysis was performed to attribute an exact percentage to each cause.

References:
https://github.com/saltypal/CNN-CUDA-Case-Study/blob/370e8a418f76c32129057a95eb7993f19095cd3f/src/hpc_cnn/native/conv_cpu.cpp

## Slide 9: CPU threads and CUDA threads

Presenter 2

An OpenMP thread and a CUDA thread are different scheduling abstractions. The OpenMP team runs through many output iterations. The direct CUDA launch supplies enough logical threads for the output tensor, with a bounds check for the excess at the end.

Logical GPU threads are not all executing simultaneously. The GPU schedules groups onto a finite set of streaming multiprocessors. Several resident warps allow it to run available work while other warps wait. Registers and shared memory limit how much work can reside together.

The conceptual transformation is straightforward: move the outer output-loop iteration index into the CUDA thread index. Keep the inner weighted sum local to the thread. CUDA provides launch configuration, device addressing and synchronization primitives; it does not remove dependencies or memory costs.

The measured Kaggle accelerator consists of two Tesla T4 devices. This comparison describes execution models, not an equivalence between one CPU thread and one CUDA thread.

References:
https://github.com/saltypal/CNN-CUDA-Case-Study/blob/370e8a418f76c32129057a95eb7993f19095cd3f/src/hpc_cnn/native/conv_cuda.cu
https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html

## Slide 10: CUDA maps an output to a thread

Presenter 2

The CUDA source computes index = blockIdx.x*blockDim.x+threadIdx.x. With 256 threads per block, block 3 and local thread 232 give index 1000. Decode width first: 1000 mod 26 is 12. After division by 26, 38 mod 26 is 12. After dividing again, the filter index is 1 and the sample index is 0.

This thread reads a 3x3 patch starting at input row 12, column 12 for filter 1, sums nine products plus bias, and stores y[1000]. Neighboring threads work on neighboring output indices. One image has 2704 outputs, so ceil(2704/256)=11 blocks. The launch has 2816 threads; 112 take the bounds return. A batch of 64 has 173056 outputs, exactly 676 blocks.

This direct kernel has no block barrier, so an out-of-range thread may return safely. Later the tiled kernel requires participation in cooperative loading and barriers even when a thread has no valid output.

The pseudocode integer divisions use positive indices. It summarizes the same sequential decode used in the actual kernel.

References:
https://github.com/saltypal/CNN-CUDA-Case-Study/blob/370e8a418f76c32129057a95eb7993f19095cd3f/src/hpc_cnn/native/conv_cuda.cu

## Slide 11: Warps and neighboring memory accesses

Presenter 2

A warp groups 32 CUDA threads. The direct kernel uses 256-thread blocks, so each block consists of eight warps. All valid threads execute similar loops, with differing indices and data.

Output index i is contiguous, so neighboring threads store adjacent FP32 output values. For threads in the same output row and channel, a fixed filter offset also produces adjacent input addresses. Boundaries between rows, filters or samples alter those address relationships. Therefore do not claim that every input load is perfectly coalesced merely because output indices are consecutive.

Coalescing combines a warp's neighboring memory requests into efficient transactions. Reuse and caching address a different issue: multiple outputs revisit the same operands. Both affect performance, but no transaction counters were collected in this case study.

Boundary threads may follow a different branch. Their invalid outputs are skipped; the direct launch still pays scheduling overhead. Larger batches supply more blocks than the single-image case and expose more work to the scheduler.

References:
https://github.com/saltypal/CNN-CUDA-Case-Study/blob/370e8a418f76c32129057a95eb7993f19095cd3f/src/hpc_cnn/native/conv_cuda.cu
https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html

## Slide 12: The host launches; the GPU computes

Presenter 2

The host is the CPU-side process; the device is the GPU. The ctypes adapter passes addresses of contiguous PyTorch tensors to the native C ABI. The native code borrows this storage; it does not allocate or free the tensors. The caller retains their lifetime.

The wrapper obtains the current CUDA stream and guards the tensor's device. Operations in the same stream are ordered. Kernel submission is asynchronous relative to the host, so timing only the host call can measure enqueue cost while the device is still working. Completed-call wall timing waits for the relevant device work. The benchmark also records separate instrumented event intervals.

Resident convolution measurements use data already on the GPU. Training transfers prepared data before timed epochs. This removes a per-batch host upload from that training path. Transfer cost is retained separately rather than hidden in a kernel number.

One invocation still has tensor allocation, Python/ctypes overhead, launch and synchronization. For a tiny convolution those costs can dominate. This explains why a throughput accelerator can lose a single-image latency comparison.

References:
https://github.com/saltypal/CNN-CUDA-Case-Study/blob/370e8a418f76c32129057a95eb7993f19095cd3f/src/hpc_cnn/native/conv_cuda.cu
https://docs.pytorch.org/docs/2.9/notes/cuda.html

## Slide 13: Shared memory reuses overlapping pixels

Presenter 2

Explain the halo geometrically. Sixteen adjacent output positions with a three-wide filter cover 16+3-1=18 input positions. The two-dimensional footprint is 18x18=324 floats, or 1296 bytes, per channel. The illustrated input tile is a schematic; the dimension labels and code define the exact extent.

The direct code performs 256*9=2304 input references for a full 16x16 output region, per channel and filter. The tiled source cooperatively loads the 324 needed values, then performs its neighborhood reads from shared memory. Nominal source-level load reduction is about 7.11, not a measured 7.11-fold bandwidth reduction or speedup. Hardware caches already reuse some direct loads.

The actual tiled grid has a sample/filter pair in blockIdx.z. Each block iterates input channels, loading one channel's tile at a time. It does not reuse a tile across different blocks or output filters. Blocks compute partial edge tiles with bounds checks.

Shared memory makes reuse explicit but consumes per-block resources and adds loading and barrier instructions. The measured tiled kernel was slower than direct CUDA here, which we return to on the algorithm benchmark slide.

References:
https://github.com/saltypal/CNN-CUDA-Case-Study/blob/370e8a418f76c32129057a95eb7993f19095cd3f/src/hpc_cnn/native/conv_cuda.cu

Illustration: generated explanatory schematic, checked against the implementation. Diagram labels show the conceptual work division; they are not a hardware execution trace.

## Slide 14: Synchronization protects a shared tile

Presenter 2

Trace one input channel. Threads cooperatively fill the shared array. Because a thread may consume values loaded by another thread, everyone must complete loading before computation begins. The first barrier establishes that checkpoint.

After computation, some threads might finish sooner. Without the second barrier, they could overwrite shared memory with the next channel while another thread still reads the previous channel. The second barrier prevents that race.

The real load loop uses a flattened lane index and advances by 256. There are 324 tile values, so 68 threads load a second value for a full tile. Out-of-image input locations are filled with zero. An invalid output thread may still need to load a value used by a valid neighbor and must reach both barriers.

__syncthreads synchronizes a participating block, not the entire grid. No global block order is assumed. It cannot be used as a device-wide barrier. This algorithm keeps blocks independent and relies on stream ordering between separate kernels.

References:
https://github.com/saltypal/CNN-CUDA-Case-Study/blob/370e8a418f76c32129057a95eb7993f19095cd3f/src/hpc_cnn/native/conv_cuda.cu

## Slide 15: Training introduces gradient reductions

Presenter 2

Backpropagation receives dY, the derivative of loss with respect to each convolution output. For each weight, every output location using that weight contributes x*dY. Weight sharing in the forward algorithm therefore creates a reduction over batch and output positions in the backward algorithm.

For batch 64, each of the 36 weight reductions has 64*26*26=43264 terms. The handwritten CPU code assigns a parameter to one loop owner and sums those terms serially. CUDA assigns one block to a weight, so its 256 threads split the terms and then reduce their partial sums. Each bias also has one block.

Input gradients use a gather design: one writer owns an input pixel and sums all valid output/filter contributions covering it. This avoids multiple outputs atomically writing into the same dX location. The equations assume our stride-one, valid cross-correlation contract.

The classifier's raw input images do not require gradients, so timed training skips dX for its first layer. Validation requests and checks dX to verify the full implementation. ReLU, pooling, linear-layer backpropagation and SGD remain PyTorch operations in every path.

References:
https://github.com/saltypal/CNN-CUDA-Case-Study/blob/370e8a418f76c32129057a95eb7993f19095cd3f/src/hpc_cnn/native/conv_cpu.cpp
https://github.com/saltypal/CNN-CUDA-Case-Study/blob/370e8a418f76c32129057a95eb7993f19095cd3f/src/hpc_cnn/native/conv_cuda.cu

## Slide 16: CUDA reduces partial sums inside a block

Presenter 2

The weight-gradient kernel launches K*C*R*R blocks. For our layer, that is 36 blocks with 256 threads per block. Each thread walks a strided subset of the 43264 terms, accumulates locally, and stores one shared partial sum.

At stride 128, the first 128 threads add the upper-half partials. At stride 64, the first 64 threads combine again. The eight stages eventually produce partial[0]. Every thread reaches the barrier at every stage, including threads not adding in that stage. Thread zero stores the block's result once. The bias kernel uses the same reduction structure with four blocks.

A naive implementation that launches one writer per contribution would require atomic accumulation or another reduction stage. This one-block-per-parameter algorithm favors understandable ownership, but 36 weight blocks and four bias blocks expose limited block-level parallelism on a 40-SM T4. Threads still process many terms within each block; small output-parameter count limits the number of simultaneously scheduled independent blocks.

Floating-point sums may differ slightly when their order changes. We validate with tolerances instead of requiring exact bitwise equality of gradients. The initial model weights, however, were checked byte for byte across training paths.

References:
https://github.com/saltypal/CNN-CUDA-Case-Study/blob/370e8a418f76c32129057a95eb7993f19095cd3f/src/hpc_cnn/native/conv_cuda.cu

## Slide 17: cuDNN turns convolution into tuned work

Presenter 3

The GEMM formulation describes convolution as matrix multiplication. A conceptual patch matrix has one column per sample/output position and one row per channel/filter offset. A filter matrix has one row per output channel. Their product contains every output response before bias addition and a layout reshape.

For the implemented MNIST layer the dimensions are 4x9 times 9x(N*676). A batch of 64 therefore has 43264 conceptual patch columns. Explicit im2col materializes repeated input values, increasing storage and memory traffic. Implicit GEMM forms the necessary tile operands while computing rather than storing the entire expanded matrix.

Our direct cuDNN adapter uses forward algorithm 0, implicit GEMM, with fixed backward algorithm choices. It constructs tensor/filter/convolution descriptors and submits calls through ctypes. Handles and workspaces are cached by device. This is a different API and selection path from the PyTorch convolution interface; it is not a completely separate hardware engine. The saved PyTorch dispatch trace contains a cuDNN convolution.

No Tensor Core use was verified in this experiment. FP32 and disabled TF32 do not support a claim that a Tensor Core path ran. Do not attribute the measured ranking to Tensor Cores. Mature libraries offer several algorithms; this experiment does not establish that the selected fixed cuDNN algorithm is globally optimal.

References:
https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html
https://github.com/saltypal/CNN-CUDA-Case-Study/blob/370e8a418f76c32129057a95eb7993f19095cd3f/src/hpc_cnn/backends.py

## Slide 18: DataParallel splits images across GPUs

Presenter 3

The schematic represents our DataParallel path. Forward recreates replicas of the original module, scatters the batch, executes replica work in worker threads and gathers logits on GPU zero. Inside each device, the selected convolution engine still handles its own output-level parallelism.

A global batch of 64 gives 32 images to each replica. Both replicas use the same starting parameter values for that step. Cross entropy is computed once after gathering all logits. Its global mean supplies correctly scaled gradients for the full batch. Replica parameter gradients are summed into the original module, then one optimizer updates its parameters.

The native adapter guards the input device and uses its current stream. The direct cuDNN cache must be device-aware and protect mutable handle/stream use. This ensures GPU 1 does not accidentally use GPU 0's execution state.

A single-image call supplies only one nonempty chunk and uses one replica. Two configured GPUs do not mean two GPUs perform useful work for that case. This diagram illustrates execution, not measured communication volume.

References:
https://docs.pytorch.org/docs/main/generated/torch.nn.DataParallel.html

Illustration: generated explanatory schematic, checked against the implementation. Diagram labels show the conceptual work division; they are not a hardware execution trace.

## Slide 19: One parallel CNN training step

Presenter 3

The order is a dependency chain even though individual operations contain parallel work. The loss needs logits; convolution gradients need the upstream derivative; the optimizer needs completed gradients. Parallel execution within a layer does not remove these training dependencies.

PyTorch autograd records the forward graph. The custom convolution autograd function supplies its handwritten backward implementation. The shared classifier uses PyTorch backward for the other layers. DataParallel adds gradient paths from both replica outputs to the original parameters.

We compute mean cross entropy after gathering the full batch, rather than independently averaging unequal local batches. This also handles the last incomplete batch consistently. SGD uses learning rate 0.05 and momentum 0.9; all paths receive identical initial weights and epoch order. The final training batch has 4800 mod 64 = 0, so each epoch contains exactly 75 full batches.

Fifty epochs therefore contain 3750 optimizer steps per path. Forward and backward may launch several kernels per step. Multiplying one isolated convolution time by this count would not predict training time because the full training step also includes other layers, reductions, optimizer work and host orchestration.

References:
https://github.com/saltypal/CNN-CUDA-Case-Study/blob/370e8a418f76c32129057a95eb7993f19095cd3f/src/hpc_cnn/train.py

## Slide 20: Parallel work must exceed parallel overhead

Presenter 3

The approximate CPU equation is a teaching model, not a fitted measurement. If 10% of a workload remains serial and 90% splits perfectly over four equal workers, time becomes 0.1+0.9/4 = 0.325 of the original. Speedup is 1/0.325 = 3.08. This calculation omits overhead and is an illustrative upper bound under its assumptions.

GPU launch, Python submission and completion waiting add relatively fixed work. Two-GPU DataParallel adds model replication, worker dispatch, scatter/gather and parameter-gradient communication each step. When the model is tiny, halving the per-device batch may save less computation than these added costs.

Measured training efficiency compares a two-GPU run with its matching one-GPU engine: efficiency = T1/(2*T2). Custom CUDA is 18.96%, direct cuDNN is 18.30%, and PyTorch is 14.74%. GPU-time proxy = 2*T2/T1 is respectively 5.27, 5.47 and 6.78. These ratios are measured time-based resource proxies, not energy measurements, monetary charges or Kaggle quota accounting.

A larger CNN or batch may change the balance. This experiment does not measure a precise attribution of total overhead to its components, and it does not benchmark DistributedDataParallel.

References:


## Slide 21: Correctness and fair timing

Presenter 3

Performance is meaningful only after correctness. The validation suite compares forward results and the gradients with respect to input, filters and biases against PyTorch CPU. It uses multiple tensor shapes, including non-square inputs and edges that do not divide into a 16x16 tile. A hand-computed example independently checks the operator interpretation. A central-difference derivative checks one scalar weight gradient.

Every implementation uses contiguous FP32 tensors. We disable TF32 and automatic mixed precision. All runs start from the same weights and use seed 42, batch size 64, SGD learning rate 0.05 and momentum 0.9. Epoch shuffle indices derive from the same seed plus epoch number. Data preparation saves the exact indices so train, validation and test membership can be inspected.

Latency measurements discard warmup calls. Warmup covers just-in-time initialization, cuDNN descriptor/workspace setup and allocator effects. The ordinary completed-call wall samples are separate from the instrumented CUDA-event samples. This keeps the displayed CPU/GPU wall comparison free of extra event-recording calls. Event timing still provides a second execution interval view.

We retain the median and the 10th/90th percentile sample spread. These repetitions describe local timing variation. They do not substitute for multiple independent training trials. The small MNIST workload and a larger synthetic workload must remain labeled separately. The synthetic tensors test scaling and have no classification-accuracy interpretation.

CPU OpenMP efficiency compares the same compiled convolution at one and four workers. We do not divide CPU/GPU speedup by a CUDA core count, because those workers and cores have different capabilities and execution models.

References:
https://docs.pytorch.org/docs/2.9/notes/cuda.html

## Slide 22: What the algorithm benchmarks show

Presenter 3

This table isolates the convolution operation. All rows use the same mathematical tensor case within a column. It contrasts one native CPU worker, four native OpenMP workers, direct CUDA and the tiled CUDA kernel. It does not compare PyTorch CPU training with native C++ timings.

The batch-64 native CPU time drops from 6608.54 to 2244.91 microseconds with four workers. CUDA direct takes 80.44 microseconds for that resident batch. This is enough output-level parallel work to amortize submission overhead more effectively than one image.

The tiled CUDA implementation is slower than direct CUDA on all three saved cases: 63.51 versus 62.41 microseconds for one image; 93.82 versus 80.44 for batch 64; and 451.57 versus 307.44 for the larger synthetic case. Explicit reuse did not produce a wall-time win. Additional cooperative-load instructions, barriers and resource use are plausible contributors, but counters were not collected, so the table does not prove which dominates.

The synthetic case has 8 samples, 8 input channels, 64x64 input, and 16 output filters of size 3x3. Its output is 8x16x62x62. It tests operation scaling only and has no classification accuracy. A source-level optimization must be evaluated against hardware caching, occupancy and orchestration costs.

References:


## Slide 23: Measured CNN performance

Presenter 3

This table contains measured Kaggle results, not illustrative estimates. The hardware and package versions appear in the saved Kaggle environment record. The requested accelerator is two NVIDIA T4 GPUs. Every backend completes 50 epochs with the same model, seed, initial weights, split and optimizer settings.

The convolution column is one image, one input channel and four 3x3 filters. It includes bias and output allocation. The inference column measures the whole CNN forward pass to ten logits. Both use input already on the execution device. GPU wall latency waits for completion. Separate CUDA-event intervals remain in the detailed report.

The training column sums synchronized training epoch intervals and excludes validation, checkpoint I/O, compilation/warmup and initial transfer. The test column evaluates the final 50-epoch model on the disjoint 1,000-example subset. A validation-selected checkpoint also exists, but the displayed test result belongs to the final checkpoint so every row represents the same epoch count.

The direct cuDNN path explicitly calls the cuDNN C API for forward, weight gradient, bias gradient and input gradient where required. Its fixed algorithm choices are part of the implementation. PyTorch CUDA delegates convolution through framework dispatch and may choose a different cuDNN algorithm. The saved dispatch trace verifies whether this run actually enters a cuDNN operation. They are related software layers, not unrelated GPUs.

Small single-image workloads emphasize framework and launch overhead. Training and batched inference may rank differently. Slight accuracy differences can arise because floating-point reductions use different orders. This one run does not establish a general speed ranking for every CNN or GPU.

References:
https://docs.nvidia.com/deeplearning/cudnn/backend/v9.10.2/api/cudnn-cnn-library.html
https://docs.pytorch.org/docs/2.9/generated/torch.nn.Conv2d.html

Connect every result to the earlier mechanisms. The seven paths use the same classifier; their convolution ownership, API overhead and multi-device orchestration differ. The accuracy is 95.6% for every final checkpoint. No new training run was needed for this explanatory revision.

## Slide 24: Live demo

Presenter 3

The live demo runs in the saved Kaggle notebook after training. The equivalent command in that GPU runtime is:

cd /kaggle/working/cnn-cuda-case-study-source
PYTHONPATH=src python -m hpc_cnn.demo --run-root /kaggle/working/cnn-cuda-case-study-results

The completed Kaggle run already executed this demo successfully. A live rerun of all paths requires two GPUs. For an offline presentation, keep the saved result table, plots and test predictions ready.

The script prints the true labels of ten held-out test digits, predictions from each final checkpoint, the saved test accuracy and the measured 50-epoch training time. It does not retrain. Keep the experiment report and performance figure ready in case the live environment is unavailable.

First show that every path is a classifier using the same test examples. Then show the measured results and explain timing boundaries. The demo's cold initialization, native library loading and checkpoint reading are not steady-state performance measurements.

When discussing a fast result, identify the workload and scope. When discussing a slow result, identify the source of cost: small parallel work, dispatch, synchronization, transfer, or additional tile loading. Shared memory is an optimization to test, not a guaranteed win. cuDNN packages years of kernel and algorithm tuning. PyTorch provides a high-level model interface and can invoke cuDNN internally.

The case study demonstrates the mechanism of convolution and how CUDA expresses its parallelism. The experiment is one seed and one Kaggle runtime. It does not establish a universal GPU speedup or an energy-efficiency conclusion.

References:


Before running the demo, open conv_cpu.cpp and conv_cuda.cu. Trace y[0,1,12,12] and the matching direct CUDA index 1000. Point to the two tiled barriers and the weight-gradient reduction. Then run the saved checkpoint demo. End by relating the measured latency, training time and DataParallel slowdown to these mechanisms.
