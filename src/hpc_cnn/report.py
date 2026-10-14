"""Build a measured report and plots from completed, traceable runs."""
import argparse
import csv
import json
from pathlib import Path

def load(path):
    return json.loads(Path(path).read_text())

def create_report(run_root):
    root=Path(run_root)
    benchmark=load(root/'benchmarks.json')
    validation=load(root/'validation.json')
    backends=benchmark['config'].get('training_backends',benchmark['config']['backends'])
    summaries=[load(root/backend/'summary.json') for backend in backends]
    lookup={row['backend']:row for row in summaries}
    pairs={'cuda_dataparallel':'cuda','cudnn_dataparallel':'cudnn','pytorch_dataparallel':'pytorch_cuda'}
    parallel_results=[]
    for two,one in pairs.items():
        if two in lookup and one in lookup:
            t1,t2=lookup[one]['training_s'],lookup[two]['training_s']
            parallel_results.append({'single_backend':one,'parallel_backend':two,
                'training_speedup':t1/t2,'training_parallel_efficiency':t1/(2*t2),
                'gpu_time_cost_ratio':2*t2/t1,'gpu_time_overhead_s':2*t2-t1})
    if not validation['passed'] or any(row['epochs_completed']!=benchmark['config']['epochs'] for row in summaries):
        raise ValueError('The report requires passed validation and complete training runs.')
    report={'training':summaries,'benchmark':benchmark,'validation':validation,'two_gpu_comparison':parallel_results}
    (root/'all_results.json').write_text(json.dumps(report,indent=2))
    single={r['backend']:r for r in benchmark['convolutions'] if r['case']=='mnist_single'}
    batch={r['backend']:r for r in benchmark['convolutions'] if r['case']=='mnist_batch64'}
    headers=['backend','conv_single_wall_us','conv_single_device_us','conv_batch64_wall_us','training_50_epochs_s','test_accuracy_percent','inference_single_us','inference_batch64_ms','train_images_per_s','peak_rss_mib','peak_gpu_tensor_mib']
    rows=[]
    for summary in summaries:
        backend=summary['backend']
        inference=benchmark['costs'][backend]['inference']
        event=single[backend]['resident']['device_event_median_ms']
        peak=summary['peak_gpu_tensor_allocated_bytes']
        rows.append([backend,single[backend]['resident']['wall_median_ms']*1000,
            event*1000 if event is not None else None,batch[backend]['resident']['wall_median_ms']*1000,
            summary['training_s'],summary['test_accuracy']*100,
            inference[0]['resident']['wall_median_ms']*1000,inference[1]['resident']['wall_median_ms'],
            summary['training_images_per_s'],summary['peak_process_rss_bytes']/2**20,
            peak/2**20 if peak is not None else None])
    with (root/'comparison.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)
    hardware=benchmark['environment']
    text=['# Inside a CNN: measured CPU/CUDA case study','',
        f"Hardware: {hardware['cpu_name']}, {hardware.get('gpu',{}).get('name','CPU only')}. Execution under WSL2 where applicable.",
        f"Source SHA-256: `{hardware['source_sha256']}`. No hosted repository was requested.",
        '', '## Model and dataset','',
        'One FP32 convolution with four 3x3 filters, stride 1, valid padding, ReLU, 2x2 max pooling, and a 676-to-10 linear classifier. Total: 6,810 trainable parameters.',
        'Exactly 6,000 images from MNIST training data: 4,800 train and 1,200 validation. A separate 1,000-image subset of MNIST test data remains disjoint. Splits contain equal numbers of each digit. Input values use x/255.',
        'Every trained backend starts with the same weights, shuffled indices, SGD learning rate 0.05, momentum 0.9, global batch size 64, seed 42 and 50 epochs. Two-GPU DataParallel assigns 32 examples to each GPU. Batch 1 uses only one GPU.',
        '', '## Implementation boundary','',
        'Custom CPU and CUDA implement convolution forward, input gradient, weight gradient and bias gradient. The direct cuDNN backend invokes the cuDNN C API through ctypes. PyTorch supplies tensor storage, the remaining CNN layers, loss and SGD for all paths. Thus custom CUDA training is a hybrid CNN with a custom convolution, rather than a fully handwritten neural-network runtime.',
        'The PyTorch CUDA baseline may use cuDNN internally. It tests framework dispatch and algorithm choice, rather than an independent hardware platform. CPU custom convolution uses one worker or four OpenMP workers. Shared classifier operations use four PyTorch CPU threads.',
        '', '## Measured comparison','',
        '| Backend | One conv wall us | Batch 64 conv wall us | 50-epoch train s | Test accuracy | Single CNN inference us |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        text.append(f'| {row[0]} | {row[1]:.2f} | {row[3]:.2f} | {row[4]:.2f} | {row[5]:.2f}% | {row[6]:.2f} |')
    text+=['','## Timing meaning','',
        'Training time is the sum of synchronized epoch training intervals. It includes minibatch indexing, forward/backward computation, autograd integration and SGD. It excludes validation, checkpoint I/O, imports, downloads, compilation/warmup and the initial model/data transfer. Each of these separate times is retained in summary.json.',
        'Single convolution is input 1x1x28x28, weights 4x1x3x3, output 1x4x26x26. It performs 24,336 multiply-accumulates, or 48,672 FLOPs excluding bias. Forward wall latency includes output allocation, dispatch and GPU completion synchronization. Inputs and weights are already resident. GPU event intervals use separate samples.',
        'CNN inference measures a forward pass to 10 logits, without transfer or class selection. Batch 1 describes latency, while batch 64 describes throughput. Host-to-host convolution timing includes both transfers and allocation. Benchmarks retain warmup counts, repeated sample counts and p10/p90 latency spread.',
        '', '## Parallel overhead, efficiency and cost','',
        '| CPU case | OpenMP speedup | Efficiency at 4 workers | 4*Tp - T1 ms |',
        '|---|---:|---:|---:|']
    for r in benchmark['convolutions']:
        if r['backend']=='cpu_openmp':
            text.append(f"| {r['case']} | {r['speedup_vs_cpu_single_conv']:.3f}x | {r['parallel_efficiency']*100:.2f}% | {r['parallel_processor_time_overhead_ms']:.6f} |")
    text+=['','### One versus two GPUs','',
        '| Matching implementation | Training speedup | Efficiency on two GPUs | GPU-time cost ratio |',
        '|---|---:|---:|---:|']
    for r in parallel_results:
        text.append(f"| {r['parallel_backend']} | {r['training_speedup']:.3f}x | {r['training_parallel_efficiency']*100:.2f}% | {r['gpu_time_cost_ratio']:.3f}x |")
    text+=['','Each GPU pair compares the same convolution implementation and global batch size. DataParallel replicates the model, scatters inputs, gathers outputs and reduces gradients. One image uses only one replica. Efficiency is T_one/(2*T_two) for the two identical GPUs. GPU-time cost ratio is 2*T_two/T_one and counts allocated GPU time, not measured energy.']
    text+=['',
        'CPU efficiency T1/(p*Tp) compares the same native convolution at one versus p OpenMP workers. GPU throughput is reported in images/s and calculated GFLOP/s. CUDA cores and CPU cores are heterogeneous, so dividing CPU-to-GPU speedup by a GPU core count does not produce a defensible parallel efficiency.',
        'Aggregate host overhead is paired instrumented wall time minus the CUDA-event interval. It includes Python/C ABI calls, event management and synchronization. Separate empty-kernel and pageable H2D/D2H copy timings make overhead visible. These overhead measurements are diagnostic, and the event interval can include host submission gaps.',
        'Computational cost is calculated MACs/FLOPs and logical tensor bytes. Memory cost is measured peak process RSS and peak PyTorch GPU tensor allocation. RSS includes the framework and context and is measured in a fresh backend process. GPU allocated bytes exclude driver/context memory and are not total board VRAM usage.',
        'Energy remains unavailable because the MX330 does not expose usable power telemetry. Monetary cost remains unspecified because no rental or electricity rate was provided. A configured per-hour rate can calculate a time-based charge without claiming that it measures electricity consumption.',
        '', '## Validation and limits','',
        f"{len(validation['checks'])} correctness checks passed. Forward and all convolution gradients match the PyTorch CPU reference within the declared tolerance. Checks cover multiple channels, batches, non-square inputs, partial CUDA tiles, a hand-computed case and an independent central-difference derivative.",
        'This is one seed and one laptop run. Fifty epochs on a small balanced subset can overfit. Test accuracy measures this held-out 1,000-image subset, not the full MNIST test set. Timing samples are not independent repeated training trials. Thermals, CPU contention and WSL overhead can affect rankings.',
        'Training all paths for the same epoch count measures fixed-workload time. It does not establish equal time to a target accuracy. A slow custom CUDA result is valid evidence and must remain visible.',
        '', '## Sources','',
        '- [PyTorch Conv2d](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Conv2d.html)',
        '- [CUDA programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)',
        '- [CUDA timing semantics](https://docs.pytorch.org/docs/2.9/notes/cuda.html)',
        '- [cuDNN 9.10.2 API](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.10.2/api/cudnn-cnn-library.html)',
        '- [cuDNN support matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.10.2/reference/support-matrix.html)',
        '- [MNIST reference loader and checksums](https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py)',
        '']
    (root/'REPORT.md').write_text('\n'.join(text),encoding='utf-8')
    create_plots(root,backends,summaries,single,batch)
    print(f'Report saved: {root/"REPORT.md"}',flush=True)

def create_plots(root,backends,summaries,single,batch):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.style.use('dark_background')
    colors=['#cccccc','#ffffff','#89d9ff','#4ab5ff','#7de3bc','#ffcf7a','#efacff'][:len(backends)]
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    series=[([single[b]['resident']['wall_median_ms']*1000 for b in backends],'Single convolution (us)'),
        ([s['training_s'] for s in summaries],'50 epochs training (s)'),
        ([load(root/'benchmarks.json')['costs'][b]['inference'][0]['resident']['wall_median_ms']*1000 for b in backends],'Single CNN inference (us)')]
    for ax,(values,title) in zip(axes,series):
        ax.barh(backends,values,color=colors)
        ax.set_title(title)
        ax.set_xlabel('Lower is faster')
        ax.invert_yaxis()
        ax.grid(axis='x',alpha=0.15)
        for i,value in enumerate(values):
            ax.text(value,i,f' {value:.1f}',va='center',fontsize=9)
        ax.set_xlim(0,max(values)*1.25)
    fig.savefig(root/'performance.png',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for backend,color in zip(backends,colors):
        history=load(root/backend/'history.json')
        axes[0].plot([r['epoch'] for r in history],[r['train_loss'] for r in history],label=backend,color=color)
        axes[1].plot([r['epoch'] for r in history],[r['validation_accuracy']*100 for r in history],label=backend,color=color)
    axes[0].set_title('Training loss')
    axes[1].set_title('Validation accuracy (%)')
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.grid(alpha=0.15)
    axes[1].legend(fontsize=8)
    fig.savefig(root/'learning_curves.png',dpi=170)
    plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-root',required=True)
    args=parser.parse_args()
    create_report(args.run_root)
