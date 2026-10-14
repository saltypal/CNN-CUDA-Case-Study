"""Run independent backend processes and preserve each completed result."""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from .build import ROOT
from .runtime import read_config,write_json,source_hash

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default=str(ROOT/'config.json'))
    parser.add_argument('--run-root',default=str(ROOT/'results/main'))
    parser.add_argument('--skip-completed',action='store_true')
    args=parser.parse_args()
    config=read_config(args.config)
    run_root=Path(args.run_root)
    run_root.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env['PYTHONPATH']=str(ROOT/'src')
    env['OMP_NUM_THREADS']=str(config['cpu_threads'])
    env['MKL_NUM_THREADS']=str(config['cpu_threads'])
    env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    env['PYTHONUNBUFFERED']='1'
    validation=[sys.executable,'-m','hpc_cnn.validate','--backends',*config['backends'],'--output',str(run_root/'validation.json')]
    subprocess.run(validation,env=env,check=True)
    for backend in config.get('training_backends',config['backends']):
        summary=run_root/backend/'summary.json'
        if args.skip_completed and summary.exists():
            previous=read_config(summary)
            saved_config=read_config(run_root/backend/'config.json')
            saved_environment=read_config(run_root/backend/'environment.json')
            if previous['epochs_completed']==config['epochs'] and saved_config==config and saved_environment['source_sha256']==source_hash():
                print(f'Reusing completed {backend}',flush=True)
                continue
        command=[sys.executable,'-m','hpc_cnn.train','--backend',backend,'--config',args.config,'--run-root',str(run_root)]
        with (run_root/f'{backend}.log').open('w') as log:
            process=subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in process.stdout:
                print(line,end='',flush=True)
                log.write(line)
                log.flush()
            if process.wait()!=0:
                raise RuntimeError(f'{backend} failed. See {run_root/backend}.log')
    subprocess.run([sys.executable,'-m','hpc_cnn.benchmark','--config',args.config,'--run-root',str(run_root)],env=env,check=True)
    subprocess.run([sys.executable,'-m','hpc_cnn.report','--run-root',str(run_root)],env=env,check=True)

if __name__=='__main__':
    main()
