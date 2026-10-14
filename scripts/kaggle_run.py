"""Launch a pinned full-MNIST notebook and retrieve its persistent outputs."""
import argparse
import json
import subprocess
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

PROJECT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['launch','status','download'])
    parser.add_argument('--record',required=True)
    parser.add_argument('--launch-dir')
    parser.add_argument('--destination')
    args=parser.parse_args()
    record=Path(args.record)
    api=KaggleApi()
    api.authenticate()
    if args.action=='launch':
        if not args.launch_dir:
            parser.error('launch requires --launch-dir')
        folder=Path(args.launch_dir)
        folder.mkdir(parents=True,exist_ok=True)
        commit=subprocess.check_output(['git','-C',str(PROJECT),'rev-parse','HEAD'],text=True).strip()
        notebook=json.loads((PROJECT/'notebooks/experiment_full_mnist.ipynb').read_text(encoding='utf-8'))
        for cell in notebook['cells']:
            if cell['cell_type']=='code':
                text=''.join(cell['source']).replace('SOURCE_REF = "main"',f'SOURCE_REF = "{commit}"')
                cell['source']=text.splitlines(True)
        (folder/'experiment_full_mnist.ipynb').write_text(json.dumps(notebook,indent=2),encoding='utf-8')
        metadata={'id':'satyapaladugu/cnn-cuda-case-study-full-mnist',
            'title':'CNN CUDA Case Study Full MNIST','code_file':'experiment_full_mnist.ipynb',
            'language':'python','kernel_type':'notebook','is_private':True,
            'enable_gpu':True,'enable_internet':True,'dataset_sources':[],
            'competition_sources':[],'kernel_sources':[]}
        (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        response=api.kernels_push(str(folder.resolve()),timeout='7200',acc='NvidiaTeslaT4')
        payload=json.loads(str(response))
        if payload.get('error'):
            raise RuntimeError(payload['error'])
        url=payload['url']
        result={'kernel':url.split('/code/',1)[1],'url':url,'source_commit':commit,
            'version_number':payload['versionNumber'],'dataset_scope':'full',
            'epochs':50,'train_count':48000,'validation_count':12000,'test_count':10000}
        record.parent.mkdir(parents=True,exist_ok=True)
        record.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    else:
        result=json.loads(record.read_text(encoding='utf-8'))
        status=api.kernels_status(result['kernel'])
        status=status if isinstance(status,dict) else json.loads(str(status))
        result['status']=status
        if args.action=='download':
            if status['status']!='COMPLETE':
                raise RuntimeError(f"Run is not complete: {status}")
            if not args.destination:
                parser.error('download requires --destination')
            destination=Path(args.destination)
            destination.mkdir(parents=True,exist_ok=True)
            token=None
            count=0
            while True:
                files,token=api.kernels_output(result['kernel'],str(destination),force=True,
                    quiet=True,page_token=token,page_size=100)
                count+=len(files)
                if not token:
                    break
            result['downloaded_files']=count
            result['destination']=str(destination.resolve())
        record.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
