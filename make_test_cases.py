"""Optional reserved-Val test construction, fixed by labels before generation."""
import argparse
import io
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from common import ROOT,read,write,sha,emit,canonical

SCENARIOS=['building_add','building_remove','water_expand','water_shrink','forest_to_agriculture','agriculture_to_forest','road_remove']

def build(out,val_root=None,val_zip=None,count=28,seed=1729):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    config=dict(count=count,seed=seed,scenarios=SCENARIOS,selection='reserved sources, label-only geometry; one case per source')
    if not 1<=count<=len(read(ROOT/'data/final_test_sources.json')):raise ValueError('Invalid test count')
    selection=out/'selection_config.json'
    if selection.exists() and read(selection)!=config:raise ValueError('Test selection configuration changed; choose another folder')
    write(selection,config)
    if (out/'manifest.json').exists():
        if read(out/'manifest.json')['configuration']!=config:raise ValueError('Do not modify an existing final-test selection')
        emit('TEST_CASES_ALREADY_READY',out=str(out));return
    if not val_root and not val_zip:raise ValueError('Supply --val-root or --val-zip for the official LoveDA Val data')
    archive=zipfile.ZipFile(val_zip) if val_zip else None;allowed=read(ROOT/'data/final_test_sources.json')
    rng=np.random.default_rng(seed);sources=[allowed[int(i)] for i in rng.permutation(len(allowed))];cases=[];counts={s:0 for s in SCENARIOS}
    def load(member,rgb):
        if archive:stream=io.BytesIO(archive.read(member))
        else:
            path=Path(val_root)/member
            if not path.exists():path=Path(val_root)/member.removeprefix('Val/')
            stream=path
        with Image.open(stream) as image:return np.asarray(image.convert('RGB') if rgb else image).copy()
    try:
        for member in sources:
            label_member=member.replace('/images_png/','/masks_png/')
            try:labels=load(label_member,False)
            except (FileNotFoundError,KeyError):continue
            if labels.shape!=(1024,1024):raise ValueError('Unexpected Val shape')
            scenario=min(SCENARIOS,key=lambda s:counts[s]);selected=None
            for y in [0,256,512,768]:
                for x in [0,256,512,768]:
                    lab=labels[y:y+256,x:x+256]
                    if (lab!=0).mean()<.95:continue
                    center=np.zeros((256,256),bool);center[96:160,96:160]=True;support=np.zeros_like(center);support[64:192,64:192]=True
                    if scenario=='building_add':
                        region=np.zeros_like(center);region[114:142,110:146]=True;dest=2
                        if np.isin(lab[region],[5,7]).mean()<.98:continue
                    elif scenario=='building_remove':
                        cc,n=ndimage.label(lab==2);region=None;dest=6
                        for k in range(1,n+1):
                            component=cc==k
                            if 64<=component.sum()<=4096 and not (component&~center).any():region=component;break
                        if region is None:continue
                    elif scenario=='water_expand':
                        region=ndimage.binary_dilation(lab==4,iterations=8)&np.isin(lab,[5,7])&center;dest=4
                    elif scenario=='water_shrink':
                        region=(lab==4)&ndimage.binary_dilation(np.isin(lab,[1,3,5,6,7]),iterations=8)&center;dest=5
                    else:
                        source_class,dest={'forest_to_agriculture':(6,7),'agriculture_to_forest':(7,6),'road_remove':(3,5)}[scenario]
                        region=(lab==source_class)&center
                    if region.sum()<64:continue
                    selected=(x,y,lab,region,support,dest);break
                if selected:break
            if not selected:continue
            x,y,lab,region,support,dest=selected;image=load(member,True)[y:y+256,x:x+256];target=lab.copy();target[region]=dest
            case=f'test_{len(cases):03d}_{scenario}';d=out/case;d.mkdir(exist_ok=True);files={}
            for name,array in [('original',image),('labels_original',lab),('labels_target',target),('mask',support.astype('uint8')*255),('target_region',region.astype('uint8')*255)]:
                path=d/(name+'.png');Image.fromarray(array).save(path);files[path.name]=sha(path)
            cases.append(dict(id=case,source_member=member,label_member=label_member,scenario=scenario,split='reserved_final_test',destination_class=dest,crop_xywh=[x,y,256,256],files_sha256=files));counts[scenario]+=1
            emit('TEST_CASE_READY',case=case,total=len(cases),requested=count)
            if len(cases)>=count:break
    finally:
        if archive:archive.close()
    if len(cases)<count:raise RuntimeError(f'Only{len(cases)} valid cases found; source data/geometry insufficient. Inspect folder; no final manifest written.')
    write(out/'manifest.json',dict(configuration=config,cases=cases,scenario_counts=counts,
        note='Label geometry is fixed before model generation. Inspect source-label alignment; these synthetic target maps are not edited-RGB ground truth. Source separation does not guarantee geographic independence.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--val-root');p.add_argument('--val-zip');p.add_argument('--count',type=int,default=28);p.add_argument('--seed',type=int,default=1729)
    a=p.parse_args();build(a.out,a.val_root,a.val_zip,a.count,a.seed)
