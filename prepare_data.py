"""Fetch official Train ZIP only when requested; extract allowlisted members."""
import argparse
import hashlib
import os
import urllib.request
import zipfile
from pathlib import Path
from PIL import Image
from common import ROOT,read,write,emit
from dataset import member_path

TRAIN_URL='https://zenodo.org/records/5706578/files/Train.zip?download=1'
TRAIN_MD5='de2b196043ed9b4af1690b3f9a7d558f'

def prepare(data_root,archive=None,download=False):
    root=Path(data_root).resolve();root.mkdir(parents=True,exist_ok=True)
    rows=read(ROOT/'data/train_sources.json')['sources'];members={r[k] for r in rows for k in ['source_member','label_member']}
    missing=[m for m in members if not member_path(root,m).is_file()]
    if missing:
        archive=Path(archive) if archive else root/'Train.zip'
        if not archive.exists():
            if not download:raise FileNotFoundError('Need LoveDA Train.zip or --download; expected data root containing Train/Urban and Train/Rural')
            tmp=archive.with_suffix('.download');offset=tmp.stat().st_size if tmp.exists() else 0
            request=urllib.request.Request(TRAIN_URL,headers={'Range':f'bytes={offset}-'} if offset else {})
            with urllib.request.urlopen(request,timeout=120) as response:
                append=offset>0 and response.status==206
                with tmp.open('ab' if append else 'wb') as f:
                    total=offset if append else 0;reported=total
                    while True:
                        block=response.read(4*1024**2)
                        if not block:break
                        f.write(block);total+=len(block)
                        if total-reported>100*1024**2:emit('DOWNLOAD',bytes=total);reported=total
            os.replace(tmp,archive)
        h=hashlib.md5()
        with archive.open('rb') as f:
            for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
        if h.hexdigest()!=TRAIN_MD5:raise ValueError('Official Train ZIP MD5 mismatch; preserve file and inspect before retrying')
        with zipfile.ZipFile(archive) as z:
            for i,member in enumerate(sorted(members)):
                target=(root/member).resolve()
                if not target.is_relative_to(root):raise ValueError('Unsafe archive path')
                if not target.exists():
                    data=z.read(member);target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix('.part');tmp.write_bytes(data);os.replace(tmp,target)
                if i%200==0:emit('EXTRACT',members=i,total=len(members))
    for member in members:
        with Image.open(member_path(root,member)) as image:
            if image.size!=(1024,1024):raise ValueError('Invalid LoveDA shape: '+member)
    write(root/'prepared_data.json',dict(sources=len(rows),official_train_md5=TRAIN_MD5,
        source='official checksum verified ZIP' if missing else 'existing PNGs; original ZIP provenance not rechecked',
        excluded_holdout=32,layout='root/Train/{Urban,Rural}/{images_png,masks_png}'))
    emit('DATA_READY',sources=len(rows),root=str(root))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--train-zip');p.add_argument('--download',action='store_true')
    a=p.parse_args();prepare(a.data_root,a.train_zip,a.download)
