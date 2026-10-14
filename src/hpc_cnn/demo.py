"""Quick live demo using saved training checkpoints and measured timings."""
import argparse
import json
import time
from pathlib import Path
import torch
from .build import ROOT
from .data import prepare_data
from .model import create_model,place_model,unwrap,is_gpu_backend
from .runtime import configure,read_config,synchronize

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-root',default=str(ROOT/'results/main'))
    parser.add_argument('--count',type=int,default=10)
    parser.add_argument('--config',help='Use the same configuration as the saved training run.')
    args=parser.parse_args()
    config=read_config(args.config)
    configure(config)
    data,_=prepare_data(ROOT/'data',config)
    x,y=data['test']
    print('Inside a CNN: same held-out digits, different convolution engines')
    print('Truth:',y[:args.count].tolist())
    for backend in config.get('training_backends',config['backends']):
        model=create_model(backend,config['conv_channels'],config['cpu_threads'])
        model,device=place_model(model,backend)
        model.eval()
        checkpoint=torch.load(Path(args.run_root)/backend/'final.pt',map_location=device,weights_only=True)
        unwrap(model).load_state_dict(checkpoint['model'])
        with torch.no_grad():
            predicted=model(x[:args.count].to(device)).argmax(1).cpu()
        summary=json.loads((Path(args.run_root)/backend/'summary.json').read_text())
        print(f'{backend:14} predictions {predicted.tolist()}  test {summary["test_accuracy"]:.2%}  train {summary["training_s"]:.2f}s')
    print('\nFor the presentation, use benchmarks.json timings after warmup.')
    print('This demo includes checkpoint loading and setup, so its wall time is not a benchmark.')

if __name__=='__main__':
    main()
