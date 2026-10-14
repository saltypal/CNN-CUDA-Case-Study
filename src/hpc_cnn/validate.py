"""Correctness gates independent of training accuracy."""
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from .backends import convolution
from .runtime import configure,read_config,write_json,environment
from .model import ConvolutionProbe,place_model,unwrap,is_gpu_backend,PARALLEL_BACKENDS

def run_validation(backends,output):
    config=read_config()
    configure(config)
    records=[]
    shapes=[(1,1,4,5,1,2),(2,3,7,9,4,3),(2,1,28,28,4,3),(2,2,33,35,3,3)]
    for backend in backends:
        device='cuda:0' if is_gpu_backend(backend) else 'cpu'
        base_backend=PARALLEL_BACKENDS.get(backend,backend)
        for n,c,h,width,k,r in shapes:
            generator=torch.Generator().manual_seed(7+n+c+h)
            inputs=torch.randn((n,c,h,width),generator=generator)
            weights=torch.randn((k,c,r,r),generator=generator)*0.1
            bias=torch.randn(k,generator=generator)*0.1
            gradient=torch.randn((n,k,h-r+1,width-r+1),generator=generator)*0.1
            x=inputs.to(device).requires_grad_()
            w=weights.to(device).requires_grad_()
            b=bias.to(device).requires_grad_()
            if backend in PARALLEL_BACKENDS:
                probe=ConvolutionProbe(w,b,backend,config['cpu_threads'])
                probe,_=place_model(probe,backend)
                w,b=unwrap(probe).weight,unwrap(probe).bias
                actual=probe(x)
            else:
                actual=convolution(x,w,b,backend,config['cpu_threads'])
            gradients=torch.autograd.grad(actual,(x,w,b),gradient.to(device))
            xr=inputs.clone().requires_grad_()
            wr=weights.clone().requires_grad_()
            br=bias.clone().requires_grad_()
            expected=F.conv2d(xr,wr,br)
            reference=torch.autograd.grad(expected,(xr,wr,br),gradient)
            pairs=[('forward',actual.cpu(),expected),('input_gradient',gradients[0].cpu(),reference[0]),('weight_gradient',gradients[1].cpu(),reference[1]),('bias_gradient',gradients[2].cpu(),reference[2])]
            for name,value,truth in pairs:
                torch.testing.assert_close(value,truth,rtol=3e-4,atol=3e-4)
                records.append({'backend':backend,'shape':[n,c,h,width,k,r],'check':name,'max_absolute_error':(value-truth).abs().max().item(),'passed':True})
        # Tiny exact case checks interpretation and avoids relying only on a library.
        x=torch.arange(1,10,dtype=torch.float32,device=device).reshape(1,1,3,3)
        w=torch.tensor([[[[1.,0.],[0.,-1.]]]],device=device)
        y=convolution(x,w,torch.zeros(1,device=device),base_backend,config['cpu_threads'])
        torch.testing.assert_close(y.cpu(),torch.full((1,1,2,2),-4.0),rtol=0,atol=0)
        # Central differences independently check one scalar weight derivative.
        x=torch.tensor([[[[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]]],device=device)
        w=torch.tensor([[[[0.2,-0.3],[0.1,0.4]]]],device=device,requires_grad=True)
        b=torch.zeros(1,device=device)
        loss=convolution(x,w,b,base_backend,config['cpu_threads']).square().sum()
        analytic=torch.autograd.grad(loss,w)[0][0,0,0,0].item()
        epsilon=0.001
        values=[]
        for direction in (-1,1):
            perturbed=w.detach().clone()
            perturbed[0,0,0,0]+=direction*epsilon
            values.append(convolution(x,perturbed,b,base_backend,config['cpu_threads']).square().sum().item())
        numerical=(values[1]-values[0])/(2*epsilon)
        if abs(analytic-numerical)>0.05:
            raise AssertionError(f'{backend}: finite-difference check failed: {analytic} vs {numerical}')
        records.append({'backend':backend,'check':'hand_computed_and_finite_difference','analytic':analytic,'numerical':numerical,'passed':True})
        print(f'{backend}: forward, backward, borders, channels and finite differences PASS',flush=True)
    report={'environment':environment(),'checks':records,'passed':True,'tolerance':{'rtol':0.0003,'atol':0.0003}}
    write_json(output,report)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--backends',nargs='+',default=read_config()['backends'])
    parser.add_argument('--output',default='results/validation.json')
    args=parser.parse_args()
    run_validation(args.backends,args.output)
