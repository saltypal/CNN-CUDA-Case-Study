"""Independently audit splits, predictions, initial weights and run provenance."""
import argparse
import csv
import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def audit(run_root, source_ref=None):
    root=Path(run_root)
    results=load(root/'all_results.json')
    config=results['benchmark']['config']
    checks=results['validation']['checks']
    assert results['validation']['passed'] and all(check['passed'] for check in checks)
    expected_backends=config['training_backends']
    assert [row['backend'] for row in results['training']]==expected_backends
    reference_splits=None
    initial_hashes={}
    source_hashes=set()
    commits=set()
    audited=[]
    for summary in results['training']:
        backend=summary['backend']
        folder=root/backend
        assert load(folder/'config.json')==config
        environment=load(folder/'environment.json')
        assert environment['source_dirty'] is False
        commits.add(environment['git_commit'])
        source_hashes.add(environment['source_sha256'])
        history=load(folder/'history.json')
        assert [row['epoch'] for row in history]==list(range(1,config['epochs']+1))
        assert summary['epochs_completed']==config['epochs']
        assert abs(sum(row['training_s'] for row in history)-summary['training_s'])<1e-8
        assert all(math.isfinite(row['train_loss']) and math.isfinite(row['validation_loss']) for row in history)
        splits=dict(np.load(folder/'split_indices.npz'))
        for name,indices in splits.items():
            assert len(indices)==config[name+'_count']
            assert len(np.unique(indices))==len(indices)
        assert not set(splits['train'])&set(splits['validation'])
        if config.get('dataset_scope')=='full':
            assert np.array_equal(np.sort(np.concatenate([splits['train'],splits['validation']])),np.arange(60000))
            assert np.array_equal(np.sort(splits['test']),np.arange(10000))
        if reference_splits is None:
            reference_splits=splits
        else:
            assert all(np.array_equal(splits[name],reference_splits[name]) for name in splits)
        digest=hashlib.sha256()
        with zipfile.ZipFile(folder/'initial_weights.pt') as archive:
            for name in sorted(name for name in archive.namelist() if '/data/' in name):
                digest.update(archive.read(name))
        initial_hashes[backend]=digest.hexdigest()
        with (folder/'test_predictions.csv').open(newline='') as stream:
            predictions=list(csv.DictReader(stream))
        assert len(predictions)==config['test_count']
        assert [int(row['mnist_test_index']) for row in predictions]==splits['test'].tolist()
        correct=sum(row['true_label']==row['predicted_label'] for row in predictions)
        assert abs(correct/config['test_count']-summary['test_accuracy'])<1e-6
        audited.append({'backend':backend,'epochs':len(history),'correct_test_predictions':correct,
            'test_count':len(predictions),'test_accuracy':correct/len(predictions),
            'initial_training_loss':history[0]['train_loss'],'final_training_loss':history[-1]['train_loss']})
    assert len(set(initial_hashes.values()))==1
    assert len(commits)==1 and len(source_hashes)==1
    if source_ref is not None:
        assert commits=={source_ref}
    dispatch=next(row for row in results['benchmark']['convolutions']
        if row['backend']=='pytorch_cuda' and row['case']=='mnist_single')
    assert dispatch['cudnn_dispatch_observed']
    result={'passed':True,'dataset_scope':config.get('dataset_scope','subset'),
        'identical_initial_weight_bytes':True,'identical_disjoint_splits':True,
        'full_official_dataset_coverage':config.get('dataset_scope')=='full',
        'epochs_every_backend':config['epochs'],'git_commit':next(iter(commits)),
        'source_sha256':next(iter(source_hashes)),'pytorch_cudnn_dispatch_observed':True,
        'validation_checks':len(checks),'training':audited}
    (root/'audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-root',required=True)
    parser.add_argument('--source-ref')
    args=parser.parse_args()
    audit(args.run_root,args.source_ref)
