"""One configured experiment per invocation. Atomic saves and graceful stopping."""
import argparse
import collections
import contextlib
import os
import random
import shutil
import signal
import time
from pathlib import Path
import numpy as np
import torch
from common import ROOT,read,write,sha,canonical,emit,seed_all,device_for,environment,source_hashes,configuration
from dataset import make_plan,draw_batch,member_path
from model import Generator,Discriminator,FrozenFeatures,segmentation,condition,onehot,region_mean,generator_loss

def scalar(x):
    x=x.detach().float().cpu()
    return float(x) if x.ndim==0 else x.tolist()

def checkpoint(g,d,og,od,sg,sd,out,step,signature,exposure,counts,history,state):
    directory=out/'checkpoints';directory.mkdir(exist_ok=True)
    row=dict(generator=g.state_dict(),discriminator=d.state_dict(),optimizer_g=og.state_dict(),optimizer_d=od.state_dict(),
        scaler_g=sg.state_dict(),scaler_d=sd.state_dict(),step=step,signature=signature,source_exposure=dict(exposure),updates=counts,
        history=history,state=state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        numpy_rng=np.random.get_state(),python_rng=random.getstate(),decoder_semantic_bias=True,input_channels=11)
    tmp=directory/'latest.part'
    with tmp.open('wb') as f:torch.save(row,f);f.flush();os.fsync(f.fileno())
    current=directory/'latest.pt';metadata=directory/'latest.json'
    if current.exists() and metadata.exists():
        shutil.copyfile(current,directory/'previous.pt');shutil.copyfile(metadata,directory/'previous.json')
    os.replace(tmp,current)
    record=dict(step=step,state=state,sha256=sha(current),signature=signature,updates=counts)
    write(metadata,record)
    if step%1000==0 or state=='complete':
        snapshot=directory/f'generator_{step:06d}.pt'
        if not snapshot.exists():
            torch.save(dict(generator=g.state_dict(),step=step,signature=signature),snapshot)
            write(snapshot.with_suffix('.json'),dict(step=step,sha256=sha(snapshot),signature=signature))
    write(out/'status.json',record);emit('CHECKPOINT',step=step,state=state,updates=counts)

def resumable(out):
    for name in ['latest','previous']:
        path=out/'checkpoints'/(name+'.pt');meta=path.with_suffix('.json')
        if path.exists() and meta.exists() and sha(path)==read(meta)['sha256']:return path
    return None

def run(config_path,data_root,out,device_name='auto',resume=False,prepare_only=False,stop_after=None):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    config=configuration(config_path,out);torch.set_num_threads(config['cpu_threads']);seed_all(config['resolved_seed'])
    plan=make_plan(config,data_root,out)
    if prepare_only:
        emit('PLAN_READY',run=str(out),statistics=plan['statistics']);return
    device=device_for(device_name,training=True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if (out/'STOP').exists():raise RuntimeError('STOP file exists. Remove it deliberately before resuming.')
    initial_path=Path(config.get('initial_checkpoint','weights/semantic_04000.pt'))
    if not initial_path.is_absolute():initial_path=ROOT/initial_path
    inputs={}
    for row in plan['selected_sources']:
        for key in ['source_member','label_member']:inputs[row[key]]=sha(member_path(data_root,row[key]))
    signature=dict(config=canonical(config),plan=plan['rows_sha256'],code=source_hashes(),
        initial_weights=sha(initial_path),teacher=sha(ROOT/'weights/semantic_teacher.pt'),
        features=sha(ROOT/'weights/squeezenet_features.pt'),input_files=canonical(inputs))
    write(out/'input_files.json',inputs);write(out/'environment.json',environment())
    lock=out/'training.lock'
    with lock.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
    stop={'requested':False};old_handlers={}
    for sig in [signal.SIGINT,signal.SIGTERM]:
        old_handlers[sig]=signal.getsignal(sig);signal.signal(sig,lambda *_:stop.update(requested=True))
    start_step=0;history=[];exposure=collections.Counter();counts={'generator':0,'discriminator':0}
    try:
        g=Generator().to(device);d=Discriminator().to(device)
        initial=torch.load(initial_path,map_location='cpu',weights_only=False)
        if 'discriminator' not in initial:raise ValueError('Fine-tuning needs a checkpoint with both generator and discriminator; generator-only snapshots are inference-only')
        g.load_state_dict(initial['generator'],strict=True);d.load_state_dict(initial['discriminator'],strict=True);del initial
        og=torch.optim.Adam(g.parameters(),lr=config['lr'],betas=(.5,.999));od=torch.optim.Adam(d.parameters(),lr=config['lr'],betas=(.5,.999))
        amp=bool(config['amp'] and device.type=='cuda');sg=torch.amp.GradScaler('cuda',enabled=amp,init_scale=128);sd=torch.amp.GradScaler('cuda',enabled=amp,init_scale=128)
        teacher=segmentation(ROOT/'weights/semantic_teacher.pt',device);features=FrozenFeatures(ROOT/'weights/squeezenet_features.pt').to(device).eval()
        path=resumable(out)
        if path:
            if not resume:raise RuntimeError('Checkpoint exists. Use --resume or choose a new run name.')
            saved=torch.load(path,map_location='cpu',weights_only=False)
            if saved['signature']!=signature:raise ValueError('Code, configuration, data or weights changed since saved checkpoint')
            if saved['state']=='complete':emit('ALREADY_COMPLETE',run=str(out),step=saved['step']);return
            g.load_state_dict(saved['generator']);d.load_state_dict(saved['discriminator']);og.load_state_dict(saved['optimizer_g']);od.load_state_dict(saved['optimizer_d'])
            sg.load_state_dict(saved['scaler_g']);sd.load_state_dict(saved['scaler_d']);start_step=saved['step'];history=saved['history'];counts=saved['updates'];exposure.update(saved['source_exposure'])
            torch.set_rng_state(saved['torch_rng']);np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
            if device.type=='cuda':torch.cuda.set_rng_state_all(saved['cuda_rng'])
            del saved;emit('RESUMED',step=start_step,checkpoint=str(path))
        g.train();d.train();last=start_step;started=time.time()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        for step in range(start_step+1,config['updates']+1):
            rows=plan['rows'][(step-1)*config['batch_size']:step*config['batch_size']]
            image,label,mask=draw_batch(rows,data_root,device);cond=condition(image,onehot(label),mask)
            d.requires_grad_(True);od.zero_grad(set_to_none=True)
            with torch.no_grad(),torch.autocast(device.type,dtype=torch.float16,enabled=amp):raw=g(cond);fake=image*(1-mask)+raw*mask
            with torch.autocast(device.type,dtype=torch.float16,enabled=amp):
                support=mask*(label!=0)[:,None]
                dl=.5*(region_mean((d(torch.cat([cond,image],1)).float()-1).square(),support)+region_mean(d(torch.cat([cond,fake],1)).float().square(),support))
            if not torch.isfinite(dl):raise FloatingPointError('Nonfinite discriminator loss')
            sd.scale(dl).backward();sd.unscale_(od);dn=float(torch.nn.utils.clip_grad_norm_(d.parameters(),10))
            scale=sd.get_scale();sd.step(od);sd.update();counts['discriminator']+=int(sd.get_scale()>=scale)
            if sd.get_scale()<scale:emit('AMP_UPDATE_SKIPPED',role='discriminator',step=step,new_scale=sd.get_scale())
            d.requires_grad_(False);og.zero_grad(set_to_none=True)
            with torch.autocast(device.type,dtype=torch.float16,enabled=amp):gl,metrics=generator_loss(g,d,teacher,features,image,label,mask,step)
            if not torch.isfinite(gl):raise FloatingPointError('Nonfinite generator loss')
            sg.scale(gl).backward();sg.unscale_(og);gn=float(torch.nn.utils.clip_grad_norm_(g.parameters(),10))
            scale=sg.get_scale();sg.step(og);sg.update();counts['generator']+=int(sg.get_scale()>=scale)
            if sg.get_scale()<scale:emit('AMP_UPDATE_SKIPPED',role='generator',step=step,new_scale=sg.get_scale())
            if not amp and (not np.isfinite(gn) or not np.isfinite(dn)):raise FloatingPointError('Nonfinite unscaled gradient')
            for row in rows:exposure[row['source_member']]+=1
            last=step
            if step==1 or step%25==0 or step==config['updates']:
                row=dict(iteration=step,g_loss=float(gl.detach()),d_loss=float(dl.detach()),updates=dict(counts),
                    seconds=time.time()-started,unique_sources=len(exposure),full_masks=sum(r['mode']=='full_image' for r in rows),
                    peak_gpu_gib=torch.cuda.max_memory_allocated(device)/1024**3 if device.type=='cuda' else None,
                    **{k:scalar(v) for k,v in metrics.items()})
                history.append(row);write(out/'progress.json',row)
                with (out/'training.jsonl').open('a',encoding='utf-8') as f:
                    import json;f.write(json.dumps(row,allow_nan=False)+'\n')
                emit('TRAIN',**row)
            finished=step==config['updates'];paused=stop['requested'] or (out/'STOP').exists() or (stop_after is not None and step-start_step>=stop_after)
            if step%config['save_every']==0 or finished or paused:
                checkpoint(g,d,og,od,sg,sd,out,step,signature,exposure,dict(counts),history,'complete' if finished else 'paused' if paused else 'running')
            if paused:emit('STOPPED_SAFELY',step=step);break
        if last==config['updates']:emit('TRAINING_COMPLETE',iterations=last,actual_updates=counts)
    finally:
        for sig,handler in old_handlers.items():signal.signal(sig,handler)
        if lock.exists() and lock.read_text().strip()==str(os.getpid()):lock.unlink()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--data-root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='auto');p.add_argument('--resume',action='store_true');p.add_argument('--prepare-only',action='store_true');p.add_argument('--stop-after',type=int)
    a=p.parse_args();run(a.config,a.data_root,a.out,a.device,a.resume,a.prepare_only,a.stop_after)
