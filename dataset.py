"""Stored native crops and masks; soft balancing of actual masked labels by mode."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from common import ROOT,read,rgb_tensor,write,canonical,emit

def member_path(root,member):
    root=Path(root);p=root/member
    return p if p.is_file() else root/member.removeprefix('Train/')

@lru_cache(maxsize=16)
def source(root,member,label_member):
    image=np.asarray(Image.open(member_path(root,member)).convert('RGB')).copy()
    labels=np.asarray(Image.open(member_path(root,label_member))).copy()
    if image.shape!=(1024,1024,3) or labels.shape!=(1024,1024) or not np.isin(labels,np.arange(8)).all():
        raise ValueError('Invalid LoveDA image/label: '+member)
    return image,labels

def local_mask(labels,k,seed):
    rng=np.random.default_rng(seed);yy,xx=np.indices(labels.shape);ys,xs=np.where(labels==k)
    if not len(xs):raise ValueError('Anchor absent from crop')
    i=int(rng.integers(len(xs)));cy,cx=int(ys[i]),int(xs[i])
    ry=int(rng.integers(24,57));rx=int(rng.integers(24,57));box=(abs(yy-cy)<ry)&(abs(xx-cx)<rx)
    mask=ndimage.binary_dilation((labels==k)&box,iterations=int(rng.integers(2,9)))&box if rng.random()<.5 else box
    mask&=labels!=0
    if mask.sum()<256:mask=box&(labels!=0)
    if not mask.any():raise ValueError('Empty mask')
    return mask.astype('float32')

def select_sources(config):
    document=read(ROOT/'data/train_sources.json');rows=document['sources']
    excluded=set(document['excluded_historical_holdout'])
    if any(r['source_member'] in excluded for r in rows):raise ValueError('Holdout leakage')
    n=config['num_train_images']
    if n is None or n==len(rows):return rows
    if not isinstance(n,int) or not 2<=n<=len(rows):raise ValueError('num_train_images must be2..2490 or None')
    rng=np.random.default_rng(config['resolved_seed']);chosen=[]
    for domain,amount in [('Urban',n//2),('Rural',n-n//2)]:
        group=[r for r in rows if r['domain']==domain]
        chosen.extend(group[int(i)] for i in rng.permutation(len(group))[:amount])
    used={r['source_member'] for r in chosen};rest=[r for r in rows if r['source_member'] not in used]
    chosen.extend(rest[int(i)] for i in rng.permutation(len(rest))[:n-len(chosen)])
    return sorted(chosen,key=lambda r:r['source_member'])

def crop_candidate(root,row,size,rng,index):
    _,labels=source(str(root),row['source_member'],row['label_member'])
    eligible=[k for k in range(1,8) if (labels==k).sum()>=64]
    k=int(rng.choice(eligible));ys,xs=np.where(labels==k);best=None
    for _ in range(24):
        n=int(rng.integers(len(xs)));x=int(np.clip(xs[n]-rng.integers(size//4,3*size//4+1),0,1024-size))
        y=int(np.clip(ys[n]-rng.integers(size//4,3*size//4+1),0,1024-size));crop=labels[y:y+size,x:x+size]
        score=float((crop!=0).mean())
        if best is None or score>best[0]:best=(score,x,y)
        if score>=.95:break
    score,x,y=best
    if score<.95:raise ValueError('Cannot find >=95% valid crop: '+row['source_member'])
    crop=labels[y:y+size,x:x+size];mask_seed=int(rng.integers(2**31));mask=local_mask(crop,k,mask_seed)>0
    counts=lambda m:np.bincount(crop[m].ravel(),minlength=8)[1:].astype('int64').tolist()
    return dict(source_member=row['source_member'],label_member=row['label_member'],domain=row['domain'],
        crop_xywh=[x,y,size,size],anchor=k,mask_seed=mask_seed,candidate=index,
        partial_pixels=counts(mask),full_image_pixels=counts(crop!=0))

def summarize(rows):
    summary={}
    for mode in ['partial','full_image']:
        selected=[r for r in rows if r['mode']==mode];pixels=np.zeros(7,dtype='int64');events=np.zeros(7,dtype='int64');area=np.zeros(7)
        for row in selected:
            h=np.array(row['masked_pixels']);pixels+=h;events+=(h>=32);area+=h/max(h.sum(),1)
        summary[mode]=dict(draws=len(selected),masked_pixels=pixels.tolist(),events_at_least32pixels=events.tolist(),
            mean_per_draw_class_fraction=(area/max(len(selected),1)).tolist())
    return summary

def make_plan(config,data_root,out):
    out=Path(out);path=out/'plan.json'
    if path.exists():
        plan=read(path)
        if plan['config_sha256']!=canonical(config):raise ValueError('Existing plan belongs to another configuration')
        if canonical(plan['rows'])!=plan['rows_sha256']:raise ValueError('Saved schedule checksum changed')
        return plan
    rows=select_sources(config);n=len(rows);total=config['updates']*config['batch_size'];size=config['patch_size']
    if total<n:raise ValueError('Increase updates or reduce images: too few draws to visit selected sources')
    rng=np.random.default_rng(config['resolved_seed']);ratio=config['full_image_ratio']
    modes=np.array(['partial']*total,dtype='<U10');modes[:round(total*ratio)]='full_image';rng.shuffle(modes)
    planned=[];accum={m:np.zeros(7) for m in ['partial','full_image']};mode_n={m:0 for m in accum};candidate_id=0
    epoch_length=n*config['patches_per_image'];cycles=(total+epoch_length-1)//epoch_length
    target_log=[]
    for cycle in range(cycles):
        pool=[]
        for row in rows:
            for _ in range(config['patches_per_image']):
                pool.append(crop_candidate(data_root,row,size,rng,candidate_id));candidate_id+=1
        vectors={m:np.array([r[m+'_pixels'] for r in pool],float) for m in accum};targets={}
        for mode in accum:
            vectors[mode]/=vectors[mode].sum(1,keepdims=True).clip(1)
            natural=vectors[mode].mean(0);present=natural>0;uniform=present/max(present.sum(),1);a=config['balance_strength']
            targets[mode]=(1-a)*natural+a*uniform
        target_log.append({m:targets[m].tolist() for m in targets})
        used=np.zeros(len(pool));limit=min(epoch_length,total-len(planned));order=rng.permutation(n)
        for j in range(limit):
            mode=str(modes[len(planned)]);candidates=rng.choice(len(pool),min(24,len(pool)),replace=False)
            if cycle==0 and j<n:
                candidates=np.arange(int(order[j])*config['patches_per_image'],(int(order[j])+1)*config['patches_per_image'])
            if config['balance_strength']>0:
                deficit=(mode_n[mode]+1)*targets[mode]-accum[mode]
                scores=vectors[mode][candidates]@deficit-.02*used[candidates]
                index=int(candidates[np.argmax(scores)])
            else:index=int(rng.choice(candidates))
            row=dict(pool[index]);row.update(mode=mode,rotation=int(rng.integers(4)),horizontal_flip=bool(rng.integers(2)),
                vertical_flip=bool(rng.integers(2)),draw=len(planned),masked_pixels=pool[index][mode+'_pixels'])
            planned.append(row);used[index]+=1;accum[mode]+=vectors[mode][index];mode_n[mode]+=1
        emit('PLAN_CYCLE',cycle=cycle+1,cycles=cycles,draws=len(planned),total=total)
    if {r['source_member'] for r in planned}!={r['source_member'] for r in rows}:raise AssertionError('Source exposure mismatch')
    plan=dict(config_sha256=canonical(config),selected_sources=rows,rows=planned,statistics=summarize(planned),cycle_targets=target_log,
        note='Soft balance of per-mask normalized class areas within masking modes; not exact equal pixels or equal gradients. Each cycle regenerates candidate crops. Inspect source repetition and missing classes.')
    plan['rows_sha256']=canonical(planned);write(path,plan);write(out/'class_distribution.json',plan['statistics']);write(out/'selected_sources.json',rows)
    for mode,stats in plan['statistics'].items():
        missing=[k+1 for k,count in enumerate(stats['masked_pixels']) if count==0]
        if stats['draws'] and missing:emit('CLASS_NOT_EXPOSED',mode=mode,classes=missing,note='Inspect source count/crops; exact balancing is not feasible for missing classes.')
    return plan

def draw_batch(rows,root,device):
    images=[];labels=[];masks=[]
    for row in rows:
        image,label=source(str(root),row['source_member'],row['label_member']);x,y,w,h=row['crop_xywh']
        image=image[y:y+h,x:x+w].copy();label=label[y:y+h,x:x+w].copy()
        mask=np.ones((h,w),'float32') if row['mode']=='full_image' else local_mask(label,row['anchor'],row['mask_seed'])
        arrays=[np.rot90(a,row['rotation']) for a in [image,label,mask]]
        if row['horizontal_flip']:arrays=[a[:,::-1] for a in arrays]
        if row['vertical_flip']:arrays=[a[::-1] for a in arrays]
        image,label,mask=[np.ascontiguousarray(a) for a in arrays]
        if np.bincount(label[mask>0].ravel(),minlength=8)[1:].tolist()!=row['masked_pixels']:raise ValueError('Data/mask changed since planning')
        images.append(rgb_tensor(image));labels.append(torch.from_numpy(label.astype('int64')));masks.append(torch.from_numpy(mask)[None])
    return torch.stack(images).to(device),torch.stack(labels).to(device),torch.stack(masks).to(device)

def evaluation_cases():
    for row in read(ROOT/'data/evaluation_manifest.json')['cases']:
        d=ROOT/'data/eval'/row['id']
        yield {**row,'rgb':np.asarray(Image.open(d/'original.png').convert('RGB')).copy(),
            'labels':np.asarray(Image.open(d/'labels_original.png')).copy(),'edited':np.asarray(Image.open(d/'labels_target.png')).copy(),
            'mask':np.asarray(Image.open(d/'mask.png'))>0,'region':np.asarray(Image.open(d/'target_region.png'))>0}
