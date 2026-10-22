import argparse
from pathlib import Path
import torch
from PIL import Image,ImageDraw
from common import ROOT,read,write,sha,emit,environment,device_for,to_rgb
from model import Generator,Discriminator
from dataset import draw_batch

def verify():
    manifest=read(ROOT/'bundle_manifest.json')
    for name,digest in manifest['files'].items():
        if not (ROOT/name).is_file():
            raise FileNotFoundError('Missing package file: '+name+'. For weights, run: python setup_assets.py --archive /path/to/1022_weights.zip')
        if sha(ROOT/name)!=digest:raise ValueError('Bundle file mismatch: '+name)
    return len(manifest['files'])

def inspect(out,device_name='auto',run=None,data_root=None):
    n=verify();device=device_for(device_name);torch.set_num_threads(4)
    g=Generator().to(device).eval();d=Discriminator().to(device).eval()
    state=torch.load(ROOT/'weights/semantic_04000.pt',map_location='cpu',weights_only=True)
    g.load_state_dict(state['generator'],strict=True);d.load_state_dict(state['discriminator'],strict=True)
    with torch.no_grad():y=g(torch.zeros(1,11,256,256,device=device))
    if not torch.isfinite(y).all():raise ValueError('Nonfinite forward')
    result=dict(environment=environment(),verified_files=n,strict_G_D_load=True,forward_shape=list(y.shape),
        note='Forward check only. Run smoke_check.py for real-data backward and save/reload checks.')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);write(out/'environment_check.json',result)
    if run and data_root:
        rows=read(Path(run)/'plan.json')['rows'][:4];images,labels,masks=draw_batch(rows,data_root,'cpu')
        canvas=Image.new('RGB',(256*4,len(rows)*282),'white');draw=ImageDraw.Draw(canvas)
        palette=torch.tensor([[0,0,0],[255,255,255],[255,0,0],[255,255,0],[0,0,255],[159,129,183],[0,255,0],[255,195,128]],dtype=torch.uint8)
        for j,row in enumerate(rows):
            rgb=Image.fromarray(to_rgb(images[j])).resize((256,256));masked=Image.fromarray(to_rgb(images[j]*(1-masks[j]))).resize((256,256))
            lab=Image.fromarray(palette[labels[j]].numpy()).resize((256,256),Image.Resampling.NEAREST)
            mask=Image.fromarray((masks[j,0].numpy()*255).astype('uint8')).convert('RGB').resize((256,256),Image.Resampling.NEAREST)
            for i,(title,image) in enumerate([('real target',rgb),('model RGB input',masked),('original semantic map',lab),(row['mode']+' mask',mask)]):
                draw.text((i*256+3,j*282+3),title,fill='black');canvas.paste(image,(i*256,j*282+24))
        canvas.save(out/'training_input_preview.png')
    emit('PREFLIGHT_OK',**result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='checks');p.add_argument('--device',default='auto');p.add_argument('--run');p.add_argument('--data-root');p.add_argument('--require-cuda',action='store_true')
    a=p.parse_args()
    if a.require_cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is unavailable. Check GPU_INDEX and the server-matching torch/torchvision installation before downloading data.')
    inspect(a.out,a.device,a.run,a.data_root)
