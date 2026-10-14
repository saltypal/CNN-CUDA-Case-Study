"""One controlled 50-epoch run. Validation and I/O have separate stopwatches."""
import argparse
import csv
import hashlib
import math
import resource
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .build import ROOT
from .data import prepare_data
from .model import create_model,place_model,unwrap,is_gpu_backend,PARALLEL_BACKENDS
from .runtime import configure,environment,read_config,synchronize,write_json

@torch.no_grad()
def evaluate(model,x,y,batch_size):
    model.eval()
    losses=[]
    predictions=[]
    for start in range(0,len(x),batch_size):
        logits=model(x[start:start+batch_size])
        losses.append(F.cross_entropy(logits,y[start:start+batch_size],reduction='sum'))
        predictions.append(logits.argmax(1))
    predictions=torch.cat(predictions)
    loss=torch.stack(losses).sum().item()/len(x)
    accuracy=(predictions==y).float().mean().item()
    return loss,accuracy,predictions.cpu()

def train(backend,config,run_root):
    configure(config)
    device=torch.device('cuda:0' if is_gpu_backend(backend) else 'cpu')
    run=Path(run_root)/backend
    run.mkdir(parents=True,exist_ok=True)
    write_json(run/'config.json',config)
    write_json(run/'environment.json',environment(include_gpu_context=device.type=='cuda'))
    setup_start=time.perf_counter()
    data,splits=prepare_data(ROOT/'data',config)
    np.savez(run/'split_indices.npz',**splits)
    split_digest=hashlib.sha256((run/'split_indices.npz').read_bytes()).hexdigest()
    model=create_model(backend,config['conv_channels'],config['cpu_threads'])
    torch.save(model.state_dict(),run/'initial_weights.pt')
    transfer_start=time.perf_counter()
    model,device=place_model(model,backend)
    data={key:(x.to(device),y.to(device)) for key,(x,y) in data.items()}
    synchronize(device)
    data_transfer_s=time.perf_counter()-transfer_start
    setup_s=time.perf_counter()-setup_start
    optimizer=torch.optim.SGD(model.parameters(),lr=config['learning_rate'],momentum=config['momentum'])
    train_x,train_y=data['train']
    val_x,val_y=data['validation']
    warmup_start=time.perf_counter()
    # Compilation/library planning and gradient allocation stay outside timing.
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(train_x[:config['batch_size']]),train_y[:config['batch_size']]).backward()
    optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    warmup_s=time.perf_counter()-warmup_start
    if device.type=='cuda':
        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)
    history=[]
    best_validation=-1.0
    training_start=time.perf_counter()
    for epoch in range(1,config['epochs']+1):
        model.train()
        order=np.random.default_rng(config['seed']+epoch).permutation(len(train_x))
        order=torch.tensor(order,device=device)
        losses=[]
        synchronize(device)
        epoch_start=time.perf_counter()
        for start in range(0,len(train_x),config['batch_size']):
            indices=order[start:start+config['batch_size']]
            x,y=train_x[indices],train_y[indices]
            optimizer.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(x),y)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach()*len(indices))
        synchronize(device)
        training_s=time.perf_counter()-epoch_start
        training_loss=torch.stack(losses).sum().item()/len(train_x)
        if not math.isfinite(training_loss):
            raise RuntimeError(f'{backend}: non-finite training loss at epoch {epoch}')
        validation_start=time.perf_counter()
        val_loss,val_accuracy,_=evaluate(model,val_x,val_y,config['batch_size'])
        synchronize(device)
        validation_s=time.perf_counter()-validation_start
        row={'epoch':epoch,'train_loss':training_loss,'validation_loss':val_loss,'validation_accuracy':val_accuracy,'training_s':training_s,'validation_s':validation_s}
        history.append(row)
        write_json(run/'history.json',history)
        if val_accuracy>best_validation:
            best_validation=val_accuracy
            torch.save({'model':unwrap(model).state_dict(),'epoch':epoch,'validation_accuracy':val_accuracy},run/'best_validation.pt')
        if epoch%10==0 or epoch==1:
            print(f'{backend} epoch {epoch:02}/{config["epochs"]}: train loss {training_loss:.4f}, val accuracy {val_accuracy:.2%}, train {training_s:.3f}s',flush=True)
            torch.save({'model':unwrap(model).state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch},run/f'checkpoint_{epoch:02}.pt')
    training_and_validation_wall_s=time.perf_counter()-training_start
    torch.save({'model':unwrap(model).state_dict(),'epoch':config['epochs'],'backend':backend},run/'final.pt')
    test_start=time.perf_counter()
    test_loss,test_accuracy,predictions=evaluate(model,*data['test'],config['batch_size'])
    test_evaluation_s=time.perf_counter()-test_start
    test_truth=data['test'][1].cpu()
    with (run/'test_predictions.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['mnist_test_index','true_label','predicted_label'])
        writer.writerows(zip(splits['test'].tolist(),test_truth.tolist(),predictions.tolist()))
    total_training_s=sum(row['training_s'] for row in history)
    gpu_count=2 if backend in PARALLEL_BACKENDS else 1 if device.type=='cuda' else 0
    gpu_peaks=[torch.cuda.max_memory_allocated(i) for i in range(gpu_count)]
    peak_gpu=sum(gpu_peaks) if gpu_peaks else None
    result={
        'backend':backend,'epochs_completed':len(history),'training_s':total_training_s,
        'gpu_count':gpu_count,'global_batch_size':config['batch_size'],
        'gpu_training_allocation_seconds':gpu_count*total_training_s,
        'peak_gpu_tensor_allocated_bytes_per_device':gpu_peaks,
        'training_and_validation_wall_s':training_and_validation_wall_s,
        'validation_s':sum(row['validation_s'] for row in history),
        'mean_epoch_training_s':total_training_s/config['epochs'],
        'training_images_per_s':len(train_x)*config['epochs']/total_training_s,
        'test_loss':test_loss,'test_accuracy':test_accuracy,
        'final_validation_accuracy':history[-1]['validation_accuracy'],
        'best_validation_accuracy':best_validation,'setup_s':setup_s,
        'initial_data_and_model_transfer_s':data_transfer_s,'warmup_s':warmup_s,
        'test_evaluation_s':test_evaluation_s,
        'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'peak_gpu_tensor_allocated_bytes':peak_gpu,
        'split_sha256':split_digest,
        'timing_scope':'Sum of synchronized training epochs. Validation, checkpoint I/O, warmup and initial data/model transfer excluded. Mini-batch indexing, autograd and optimizer included.',
        'implementation_scope':'Only convolution differs. PyTorch supplies ReLU, max pooling, dense layer, cross entropy, autograd integration and SGD for every backend.',
        'monetary_training_cost':None,
        'energy_joules':None,
    }
    hourly=config.get('gpu_hour_price' if device.type=='cuda' else 'cpu_hour_price')
    if hourly is not None:
        result['monetary_training_cost']=total_training_s/3600*hourly*(gpu_count if gpu_count else 1)
    write_json(run/'summary.json',result)
    print(f'COMPLETE {backend}: {total_training_s:.2f}s training, test {test_accuracy:.2%}',flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend',required=True)
    parser.add_argument('--config')
    parser.add_argument('--run-root',default=str(ROOT/'results/main'))
    args=parser.parse_args()
    train(args.backend,read_config(args.config),args.run_root)
