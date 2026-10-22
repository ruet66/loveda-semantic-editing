"""Tiny real-data check, isolated from research runs. Does not train the final experiment."""
import argparse
from pathlib import Path
import torch
from common import ROOT,write,read,emit,sha,device_for
from train import run
from dataset import draw_batch
from model import Generator,Discriminator,condition,onehot

def check(data_root,out,device_name='auto'):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);device=device_for(device_name)
    config=dict(num_train_images=2,patch_size=256,patches_per_image=2,updates=3,batch_size=1,
        full_image_ratio=.5,balance_strength=.3,seed=20261022,lr=1e-4,save_every=1,amp=device.type=='cuda',cpu_threads=4)
    config_path=out/'smoke_config.json';write(config_path,config)
    before=sha(ROOT/'weights/semantic_04000.pt')
    run(config_path,data_root,out/'resumed',str(device),stop_after=1)
    if read(out/'resumed/status.json')['state']!='paused':raise AssertionError('Pause checkpoint missing')
    run(config_path,data_root,out/'resumed',str(device),resume=True)
    run(config_path,data_root,out/'uninterrupted',str(device))
    a=torch.load(out/'resumed/checkpoints/latest.pt',map_location='cpu',weights_only=False)
    b=torch.load(out/'uninterrupted/checkpoints/latest.pt',map_location='cpu',weights_only=False)
    differences={key:max(float((a[key][k]-b[key][k]).abs().max()) for k in a[key]) for key in ['generator','discriminator']}
    if device.type=='cpu' and max(differences.values())!=0:raise AssertionError(differences)
    if sha(ROOT/'weights/semantic_04000.pt')!=before:raise AssertionError('Original weights modified')
    plan=read(out/'resumed/plan.json');modes={r['mode'] for r in plan['rows']}
    if modes!={'partial','full_image'}:raise AssertionError('Did not exercise both masking modes')
    if a['step']!=3 or a['state']!='complete':raise AssertionError('Completion state invalid')
    if not all(torch.isfinite(v).all() for v in a['generator'].values()):raise AssertionError('Nonfinite parameters')
    initial=torch.load(ROOT/'weights/semantic_04000.pt',map_location='cpu',weights_only=True)
    changed=max(float((a['generator'][k]-initial['generator'][k]).abs().max()) for k in a['generator'])
    if changed==0:raise AssertionError('No learning update')
    result=dict(device=str(device),tested_updates=3,actual_updates=a['updates'],both_mask_modes=True,
        paused_and_resumed=True,resume_parameter_max_difference=differences,original_weights_unchanged=True,
        update_parameter_max_difference=changed,
        note='Only a tiny engineering smoke check. Not evidence of edit quality or convergence; cross-device bit identity is not promised.')
    write(out/'smoke_validation.json',result);emit('SMOKE_CHECK_OK',**result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='auto')
    a=p.parse_args();check(a.data_root,a.out,a.device)
