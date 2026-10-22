import hashlib
import json
import os
import platform
import random
import secrets
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.part')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)
def emit(event,**row):print(json.dumps(dict(event=event,**row),ensure_ascii=True,allow_nan=False),flush=True)
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def device_for(name,training=False):
    if name=='auto':
        if training and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable. Install the matching CUDA PyTorch build; CPU is only recommended for --smoke.')
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device=torch.device(name)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('Requested CUDA is unavailable.')
    return device
def environment():
    import torchvision
    return dict(python=platform.python_version(),platform=platform.platform(),torch=torch.__version__,
        torchvision=torchvision.__version__,cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
def rgb_tensor(image):return torch.from_numpy(image.copy()).permute(2,0,1).float()/127.5-1
def to_rgb(tensor):return ((tensor.detach().float().cpu().clamp(-1,1)+1)*127.5).round().byte().permute(1,2,0).numpy()
def source_hashes():return {p.name:sha(p) for p in sorted(ROOT.glob('*.py'))}

def configuration(path,out):
    value=read(path);out=Path(out)
    required=['num_train_images','patch_size','patches_per_image','updates','batch_size','full_image_ratio','balance_strength','seed','lr','save_every','amp','cpu_threads']
    if any(k not in value for k in required):raise ValueError('Missing configuration keys')
    if value['patch_size'] not in [256,512]:raise ValueError('patch_size must be256 or512')
    if not 0<=value['full_image_ratio']<=1 or not 0<=value['balance_strength']<=1:raise ValueError('Ratios must be in[0,1]')
    if any(not isinstance(value[k],int) or value[k]<1 for k in ['patches_per_image','updates','batch_size','save_every','cpu_threads']):raise ValueError('Counts must be positive integers')
    if not np.isfinite(value['lr']) or value['lr']<=0:raise ValueError('lr must be positive and finite')
    if value['seed'] is not None and (not isinstance(value['seed'],int) or not 0<=value['seed']<2**32):raise ValueError('seed must be None or a32-bit integer')
    existing=out/'config.json'
    if existing.exists():
        saved=read(existing);old=dict(saved);old.pop('resolved_seed')
        if old!=value:raise ValueError('Configuration differs. Use another run folder; do not overwrite an experiment.')
        return saved
    value['resolved_seed']=value['seed'] if value['seed'] is not None else secrets.randbits(32)
    write(existing,value);return value
