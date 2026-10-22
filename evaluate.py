"""Reconstruction, edited-map response, independent audit and raw global probes."""
import argparse
import csv
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,PngImagePlugin
from common import ROOT,read,write,sha,emit,device_for,rgb_tensor,to_rgb
from dataset import evaluation_cases
from model import Generator,condition,onehot,segmentation,normalize

@torch.no_grad()
def evaluate(checkpoint,out,device_name='auto',audit=True,limit=None,global_probe=False,cases_dir=None,final_test=False):
    device=device_for(device_name);torch.set_num_threads(4);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    digest=sha(checkpoint)
    cases=list(evaluation_cases())
    if cases_dir:
        root=Path(cases_dir);manifest=read(root/'manifest.json');cases=[]
        allowed=set(read(ROOT/'data/final_test_sources.json'))
        for row in manifest['cases']:
            if final_test and row['source_member'] not in allowed:raise ValueError('Not in the reserved final source pool')
            d=root/row['id']
            for name,expected in row.get('files_sha256',{}).items():
                if sha(d/name)!=expected:raise ValueError('Test input checksum changed')
            cases.append({**row,'rgb':np.asarray(Image.open(d/'original.png').convert('RGB')).copy(),
                'labels':np.asarray(Image.open(d/'labels_original.png')).copy(),'edited':np.asarray(Image.open(d/'labels_target.png')).copy(),
                'mask':np.asarray(Image.open(d/'mask.png'))>0,'region':np.asarray(Image.open(d/'target_region.png'))>0})
    if final_test and not cases_dir:raise ValueError('Final test requires separately prepared reserved-source cases')
    if limit:cases=cases[:limit]
    lock=out/'evaluation_protocol.json'
    protocol=dict(checkpoint_sha256=digest,cases=[r['id'] for r in cases],audit=audit,global_probe=global_probe,final_test=final_test,
        cases_digest=sha(Path(cases_dir)/'manifest.json') if cases_dir else sha(ROOT/'data/evaluation_manifest.json'))
    protocol['case_input_sha256']={}
    for item in cases:
        h=hashlib.sha256()
        for key in ['rgb','labels','edited','mask','region']:
            value=np.ascontiguousarray(item[key])
            h.update(str((key,value.shape,str(value.dtype))).encode());h.update(value.tobytes())
        protocol['case_input_sha256'][item['id']]=h.hexdigest()
    if lock.exists() and read(lock)!=protocol:raise ValueError('Output belongs to a different evaluation; choose another directory')
    write(lock,protocol)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False);g=Generator().to(device).eval();g.load_state_dict(state['generator'],strict=True)
    step=state.get('step',state.get('total_steps',None));del state
    evaluator=segmentation(ROOT/'weights/semantic_audit.pt',device) if audit else None
    results=[];paired=[];sheets=[];palette=np.array([[0,0,0],[255,255,255],[255,0,0],[255,255,0],[0,0,255],[159,129,183],[0,255,0],[255,195,128]],np.uint8)
    def score(image,region,dest,reference):
        if evaluator is None:return {}
        probability=evaluator(normalize(rgb_tensor(image)[None].to(device)))['out'].float().softmax(1)[0]
        roi=torch.from_numpy(region).to(device);pred=probability.argmax(0)+1
        target=torch.from_numpy(reference.astype('int64')).to(device)
        return dict(target_pixel_fraction=float((pred[roi]==dest).float().mean()),mean_target_probability=float(probability[dest-1][roi].mean()),
            map_label_agreement_in_roi=float((pred[roi]==target[roi]).float().mean()))
    for index,item in enumerate(cases):
        image=rgb_tensor(item['rgb'])[None].to(device);m=torch.from_numpy(item['mask'].astype('float32'))[None,None].to(device)
        if not item['region'].any() or (item['region']&~item['mask']).any():raise ValueError('Invalid target/support')
        if not np.array_equal(item['labels'][~item['region']],item['edited'][~item['region']]):raise ValueError('Unexpected label edits')
        generated={};real_score=score(item['rgb'],item['region'],item['destination_class'],item['labels'])
        for name,label in [('original_map',item['labels']),('edited_map',item['edited']),('no_map',item['labels'])]:
            semantic=onehot(torch.from_numpy(label.astype('int64'))[None].to(device))
            if name=='no_map':semantic.zero_()
            raw=to_rgb(g(condition(image,semantic,m))[0]);final=item['rgb'].copy();final[item['mask']]=raw[item['mask']]
            error=int(np.abs(final.astype('int16')-item['rgb'].astype('int16'))[~item['mask']].max()) if (~item['mask']).any() else 0
            if error:raise AssertionError('Outside RGB changed')
            generated[name]=final;folder=out/'images'/item['id'];folder.mkdir(parents=True,exist_ok=True)
            info=PngImagePlugin.PngInfo();info.add_text('SYNTHETIC','true');info.add_text('configuration',json.dumps(dict(case=item['id'],condition=name,checkpoint_sha256=digest,step=step,outside_preserved=True)))
            Image.fromarray(final).save(folder/(name+'.png'),pnginfo=info)
            results.append(dict(case=item['id'],split=item['split'],scenario=item['scenario'],condition=name,outside_max_error=error,
                **score(final,item['region'],item['destination_class'],item['labels'] if name=='original_map' else item['edited'])))
            if global_probe:
                full=to_rgb(g(condition(image,semantic,torch.ones_like(m)))[0])
                info2=PngImagePlugin.PngInfo();info2.add_text('SYNTHETIC','true');info2.add_text('scope','whole-frame diagnostic, no original RGB preserved')
                Image.fromarray(full).save(folder/(name+'_global.png'),pnginfo=info2)
        roi=item['region'];support=item['mask']
        paired.append(dict(case=item['id'],split=item['split'],scenario=item['scenario'],real_original_audit=real_score,
            map_change_rgb_mae=float(np.abs(generated['edited_map'].astype(float)-generated['original_map'].astype(float))[roi].mean()),
            original_map_reconstruction_mae=float(np.abs(generated['original_map'].astype(float)-item['rgb'].astype(float))[support].mean())))
        canvas=Image.new('RGB',(256*5,288),'white');draw=ImageDraw.Draw(canvas)
        panels=[('real',item['rgb']),('target map',palette[item['edited']]),('original map',generated['original_map']),('edited map',generated['edited_map']),('no map (ablation)',generated['no_map'])]
        for col,(title,rgb) in enumerate(panels):
            draw.text((col*256+3,2),item['id'],fill='black');draw.text((col*256+3,16),title,fill='black');canvas.paste(Image.fromarray(rgb),(col*256,32))
        sheet=out/'comparison'/f'{item["id"]}.png';sheet.parent.mkdir(exist_ok=True);canvas.save(sheet)
        yy,xx=np.where(roi);side=min(256,max(96,2*max(int(np.ptp(xx))+1,int(np.ptp(yy))+1)))
        x=int(np.clip((xx.min()+xx.max()+1)//2-side//2,0,256-side));y=int(np.clip((yy.min()+yy.max()+1)//2-side//2,0,256-side))
        focus=Image.new('RGB',canvas.size,'white');draw=ImageDraw.Draw(focus)
        for col,(title,rgb) in enumerate(panels):
            draw.text((col*256+3,2),item['id'],fill='black');draw.text((col*256+3,16),title+' (same crop)',fill='black')
            focus.paste(Image.fromarray(rgb).crop((x,y,x+side,y+side)).resize((256,256),Image.Resampling.NEAREST),(col*256,32))
        focus_path=out/'focus'/sheet.name;focus_path.parent.mkdir(exist_ok=True);focus.save(focus_path);sheets.append(focus_path.relative_to(out).as_posix())
        emit('EVAL_CASE',case=item['id'],index=index+1,total=len(cases),step=step)
    warning='Target-pixel fraction is not edit success or realism. No-map is an untrained ablation in this GAN. Original-map MAE measures reconstruction only. GAN generation is deterministic; generation seed does not create diversity.'
    write(out/'metrics.json',dict(checkpoint_sha256=digest,step=step,rows=results,paired=paired,warning=warning))
    if not (out/'human_review.csv').exists():
        with (out/'human_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f);writer.writerow(['case','scenario','target_edit_pass_partial_fail','texture_boundary_pass_partial_fail','notes'])
            writer.writerows([r['case'],r['scenario'],'','',''] for r in paired)
    content='<meta charset="utf-8"><title>GAN comparison</title><h1>Saved GAN evaluation</h1><p>'+warning+'</p>'
    content+=''.join('<p><a href="'+p+'"><img style="max-width:100%" src="'+p+'"></a></p>' for p in sheets)
    (out/'index.html').write_text(content,encoding='utf-8');emit('EVALUATION_COMPLETE',cases=len(cases),out=str(out))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='auto')
    p.add_argument('--no-audit',action='store_true');p.add_argument('--limit',type=int);p.add_argument('--global-probe',action='store_true');p.add_argument('--cases-dir');p.add_argument('--final-test',action='store_true')
    a=p.parse_args();evaluate(a.checkpoint,a.out,a.device,not a.no_audit,a.limit,a.global_probe,a.cases_dir,a.final_test)
