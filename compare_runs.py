"""Compare existing evaluations only; never starts training."""
import argparse
import csv
import html
import os
from pathlib import Path
from common import read,write

def compare(evaluations,out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    runs=[(Path(p).resolve(),read(Path(p)/'metrics.json')) for p in evaluations]
    if len(runs)<2:raise ValueError('Choose at least two evaluation folders')
    maps=[{r['case']:r for r in data['paired']} for _,data in runs];common=set.intersection(*(set(m) for m in maps))
    if not common:raise ValueError('No common cases')
    protocols=[read(folder/'evaluation_protocol.json') for folder,_ in runs]
    if any('case_input_sha256' not in p for p in protocols):
        raise ValueError('Evaluation lacks input fingerprints. Evaluate with the current code into a new output folder.')
    for case in common:
        if len({p['case_input_sha256'].get(case) for p in protocols})!=1 or not protocols[0]['case_input_sha256'].get(case):
            raise ValueError('Different image/map/mask inputs for case: '+case)
    if len({p['audit'] for p in protocols})!=1 or len({p['final_test'] for p in protocols})!=1:
        raise ValueError('Evaluation audit/final-test settings differ.')
    rows=[];web=['<meta charset="utf-8"><h1>Saved model comparisons</h1><p>Map response is not edit accuracy. Review target shapes and texture.</p>']
    for case in sorted(common):
        web.append('<h2>'+html.escape(case)+'</h2>')
        for (folder,data),mapping in zip(runs,maps):
            row=dict(run=str(folder),case=case,**{k:v for k,v in mapping[case].items() if k in ['map_change_rgb_mae','original_map_reconstruction_mae']})
            rows.append(row);image=folder/'focus'/f'{case}.png'
            if not image.exists():image=folder/'comparison'/f'{case}.png'
            relative=os.path.relpath(image,out).replace('\\','/')
            web.append('<p>'+html.escape(str(folder))+'</p><img style="max-width:100%" src="'+html.escape(relative,quote=True)+'">')
    with (out/'comparison_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'index.html').write_text(''.join(web),encoding='utf-8')
    write(out/'comparison_scope.json',dict(common_cases=len(common),case_counts=[len(m) for m in maps],
        input_fingerprints_verified=True,
        note='Common cases have identical RGB/map/mask fingerprints. Training histories may still differ; this is not proof of a controlled experiment.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evaluations',nargs='+',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();compare(a.evaluations,a.out)
