"""Measure operation latency, transfers, host overhead and inference."""
import argparse
import ctypes as ct
import statistics
import time
from pathlib import Path
import torch

from .backends import convolution,native_library,check_cuda
from .build import ROOT
from .data import prepare_data
from .model import create_model,place_model,unwrap,is_gpu_backend,PARALLEL_BACKENDS,ConvolutionProbe
from .runtime import configure,environment,read_config,synchronize,write_json

@torch.no_grad()
def measure(function,device,repeats,warmup):
    for _ in range(warmup):
        output=function()
    synchronize(device)
    wall=[]
    event=[]
    instrumented_wall=[]
    enqueue=[]
    if device.type=='cuda':
        # Compare ordinary completed-call wall latency against the CPU first.
        # Event instrumentation uses a separate set of paired samples.
        for _ in range(repeats):
            synchronize(device)
            before=time.perf_counter_ns()
            output=function()
            submitted=time.perf_counter_ns()
            synchronize(device)
            wall.append((time.perf_counter_ns()-before)/1e6)
            enqueue.append((submitted-before)/1e6)
        start=torch.cuda.Event(enable_timing=True)
        end=torch.cuda.Event(enable_timing=True)
        for _ in range(repeats):
            synchronize(device)
            before=time.perf_counter_ns()
            start.record()
            output=function()
            end.record()
            end.synchronize()
            instrumented_wall.append((time.perf_counter_ns()-before)/1e6)
            event.append(start.elapsed_time(end))
    else:
        for _ in range(repeats):
            before=time.perf_counter_ns()
            output=function()
            wall.append((time.perf_counter_ns()-before)/1e6)
    result={'wall_median_ms':statistics.median(wall),
        'wall_p10_ms':sorted(wall)[int(0.1*(len(wall)-1))],
        'wall_p90_ms':sorted(wall)[int(0.9*(len(wall)-1))],
        'samples':repeats,'warmup':warmup,
        'device_event_median_ms':statistics.median(event) if event else None}
    if event:
        result['instrumented_wall_median_ms']=statistics.median(instrumented_wall)
        result['host_enqueue_median_ms']=statistics.median(enqueue)
        result['host_and_measurement_overhead_median_ms']=statistics.median([w-e for w,e in zip(instrumented_wall,event)])
    return result

def run_benchmarks(config,run_root):
    configure(config)
    data,_=prepare_data(ROOT/'data',config)
    records=[]
    costs={}
    cases=[('mnist_single',1,1,28,4),('mnist_batch64',64,1,28,4),('synthetic_larger',8,8,64,16)]
    for backend in config['backends']:
        device=torch.device('cuda:0' if is_gpu_backend(backend) else 'cpu')
        configure(config)
        model=create_model(backend,config['conv_channels'],config['cpu_threads'])
        model,device=place_model(model,backend)
        model.eval()
        initial=Path(run_root)/backend/'initial_weights.pt'
        if initial.exists():
            unwrap(model).load_state_dict(torch.load(initial,map_location=device,weights_only=True))
        for name,n,c,h,k in cases:
            generator=torch.Generator().manual_seed(123)
            if c==1:
                host_x=data['test'][0][:n].clone()
                host_w=unwrap(model).conv_weight.detach().cpu()
                host_b=unwrap(model).conv_bias.detach().cpu()
            else:
                host_x=torch.rand((n,c,h,h),generator=generator)
                host_w=torch.rand((k,c,3,3),generator=generator)*0.1
                host_b=torch.zeros(k)
            x,w,b=host_x.to(device),host_w.to(device),host_b.to(device)
            repeats=config['benchmark_repeats'] if name!='synthetic_larger' else min(30,config['benchmark_repeats'])
            if backend in PARALLEL_BACKENDS:
                probe=ConvolutionProbe(w,b,backend,config['cpu_threads'],trainable=False)
                probe,_=place_model(probe,backend)
                function=lambda:probe(x)
            else:
                function=lambda:convolution(x,w,b,backend,config['cpu_threads'])
            measurement=measure(function,device,repeats,config['benchmark_warmup'])
            p=h-2
            macs=n*k*p*p*c*9
            logical_bytes=4*(x.numel()+w.numel()+b.numel()+n*k*p*p)
            record={'backend':backend,'case':name,'input_shape':[n,c,h,h],'weight_shape':[k,c,3,3],
                'requested_gpu_count':2 if backend in PARALLEL_BACKENDS else int(device.type=='cuda'),
                'effective_gpu_count':min(n,2) if backend in PARALLEL_BACKENDS else int(device.type=='cuda'),
                'output_shape':[n,k,p,p],'multiply_accumulates':macs,
                'flops_excluding_bias':2*macs,'logical_tensor_bytes':logical_bytes,
                'resident':measurement,
                'gflops_per_s_wall':2*macs/(measurement['wall_median_ms']*1e6),
                'scope':'Forward convolution plus bias and output allocation. Input and weights already resident. GPU wall timing includes completion synchronization. Events use separate samples.'}
            if backend=='pytorch_cuda' and name=='mnist_single':
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                    function()
                    synchronize(device)
                names=sorted({entry.key for entry in profile.key_averages()})
                record['observed_dispatch_operators']=names
                record['cudnn_dispatch_observed']=any('cudnn' in name for name in names)
            if device.type=='cuda':
                def pipeline():
                    transferred=host_x.to(device)
                    result=probe(transferred) if backend in PARALLEL_BACKENDS else convolution(transferred,w,b,backend,config['cpu_threads'])
                    return result.cpu()
                record['host_to_host']=measure(pipeline,device,repeats,config['benchmark_warmup'])
                destination=torch.empty_like(x)
                output=function()
                host_destination=torch.empty_like(output,device='cpu')
                record['h2d_copy']=measure(lambda:destination.copy_(host_x),device,repeats,5)
                record['d2h_copy']=measure(lambda:host_destination.copy_(output),device,repeats,5)
            records.append(record)
            print(f'{backend} {name}: {measurement["wall_median_ms"]*1000:.2f} us wall',flush=True)
        inference=[]
        for batch in (1,64):
            images=data['test'][0][:batch].to(device)
            measurement=measure(lambda:model(images),device,config['benchmark_repeats'],config['benchmark_warmup'])
            inference.append({'batch_size':batch,'resident':measurement,
                'images_per_s_wall':batch*1000/measurement['wall_median_ms'],
                'scope':'CNN forward to 10 logits. Includes convolution, ReLU, pooling and dense layer. Excludes transfers and class selection.'})
        costs[backend]={'inference':inference}
        if backend in PARALLEL_BACKENDS:
            identity=torch.nn.DataParallel(torch.nn.Identity().to(device),device_ids=[0,1])
            images=data['test'][0][:64].to(device)
            costs[backend]['dataparallel_identity_diagnostic']=measure(lambda:identity(images),device,config['benchmark_repeats'],config['benchmark_warmup'])
            costs[backend]['dataparallel_identity_scope']='Scatter, worker dispatch and gather for an identity operation. No model parameters. Input and output are images, so this diagnostic is not an exact decomposition of CNN overhead.'
        for item in inference:
            item['effective_gpu_count']=min(item['batch_size'],2) if backend in PARALLEL_BACKENDS else int(device.type=='cuda')
        if device.type=='cuda' and backend.startswith('cuda'):
            lib=native_library(True)
            stream=ct.c_void_p(torch.cuda.current_stream().cuda_stream)
            def empty():
                check_cuda(lib.launch_empty(stream))
            costs[backend]['empty_kernel_launch']=measure(empty,device,config['benchmark_repeats'],config['benchmark_warmup'])
        write_json(Path(run_root)/'benchmark_partial.json',{'convolutions':records,'costs':costs})
    lookup={(r['backend'],r['case']):r for r in records}
    for record in records:
        t1=lookup[('cpu',record['case'])]['resident']['wall_median_ms']
        record['speedup_vs_cpu_single_conv']=t1/record['resident']['wall_median_ms']
        if record['backend']=='cpu_openmp':
            workers=config['cpu_threads']
            tp=record['resident']['wall_median_ms']
            record['parallel_efficiency']=t1/(workers*tp)
            record['parallel_processor_time_overhead_ms']=workers*tp-t1
            record['processor_time_cost_ratio']=workers*tp/t1
        if record['backend'] in PARALLEL_BACKENDS:
            one=lookup[(PARALLEL_BACKENDS[record['backend']],record['case'])]['resident']['wall_median_ms']
            speedup=one/record['resident']['wall_median_ms']
            record['gpu_speedup_vs_matching_single']=speedup
            record['gpu_parallel_efficiency']=speedup/2 if record['effective_gpu_count']==2 else None
            record['gpu_processor_time_overhead_ms']=2*record['resident']['wall_median_ms']-one if record['effective_gpu_count']==2 else None
    result={'environment':environment(),'config':config,'convolutions':records,'costs':costs,
        'methodology':{
            'precision':'FP32, TF32 disabled, fixed shapes, fixed random seed, no AMP',
            'gpu_interval':'CUDA-event interval. May include submission gaps. It is not an isolated instruction count or an occupancy measurement.',
            'overhead':'Paired wall minus CUDA-event time includes Python/C ABI dispatch, event instrumentation and synchronization. Reported as aggregate overhead, not a pure launch latency.',
            'cpu_efficiency':'T1/(p*Tp) for the same native CPU convolution at p OpenMP workers. It does not define efficiency for heterogeneous GPU cores.',
            'transfer':'Pageable host buffers and preallocated copy destinations. Host-to-host convolution includes allocations and both transfers.',
            'cost':'Logical tensor bytes and FLOPs are calculated. Process RSS and GPU tensor allocation are measured. Energy and monetary cost remain null without power telemetry and a configured price.',
            'statistics':'One local experiment, repeated latency samples. p10/p90 describe timing spread, not independent training-run uncertainty.'}}
    write_json(Path(run_root)/'benchmarks.json',result)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--run-root',default=str(ROOT/'results/main'))
    args=parser.parse_args()
    run_benchmarks(read_config(args.config),args.run_root)
