"""Reconstruct the frozen paired cohort using the original algorithm kernels."""
from __future__ import annotations
import argparse, csv, json, os, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
METHODS=('pca','adaptive_poisson','fixed_poisson','ball_pivoting')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=METHODS+('all',),default='pca')
    parser.add_argument('--limit',type=int,default=0,help='0 means the full cohort')
    parser.add_argument('--leaf',help='Run one numeric leaf ID')
    parser.add_argument('--cohort',choices=('benchmark','all'),default='benchmark')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'runs/reconstruction')
    args=parser.parse_args()
    if args.limit<0:parser.error('--limit must be nonnegative')
    os.chdir(ROOT)
    with (ROOT/'data/leaf_manifest.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    if args.cohort=='benchmark':rows=[r for r in rows if r['in_benchmark'].lower()=='true']
    if args.leaf:rows=[r for r in rows if r['leaf']==args.leaf.zfill(4)]
    if args.limit:rows=rows[:args.limit]
    if not rows:raise ValueError('No matching leaves')
    output=args.output_dir.resolve()
    if output==ROOT/'data' or ROOT/'data' in output.parents:
        raise ValueError('Reconstruction output must not overwrite the frozen data directory')
    output.mkdir(parents=True,exist_ok=True)
    selected=METHODS if args.method=='all' else (args.method,)
    records=[]
    for method in selected:
        folder=output/method;folder.mkdir(exist_ok=True)
        if method=='pca':
            import pca_reconstruct as core
        else:
            import baseline_core as core
            core._init_worker()
        for i,row in enumerate(rows,1):
            src=ROOT/row['txt_path'];scale=float(row['scale_factor'])
            if method=='pca':
                # Match the source evaluation's ReconstructionConfig defaults.
                rec=core.process_file(src,folder,core.ReconstructionConfig())
                rec['leaf']=row['leaf'];rec['method']=method
                if rec.get('success'):
                    from recalculate_areas import surface_area
                    rec['area']=surface_area(folder/(row['leaf']+'_pca_mesh.obj'))
            else:
                rec=core.process_one(str(src),str(folder),method,'reduced')
            rec.update(genotype=row['genotype'],storage_folder=row['storage_folder'],scale_factor=scale)
            if rec.get('success'):rec['area_cm2']=float(rec['area'])*scale*scale
            for key in ('input','obj','ply'):
                if rec.get(key):
                    try:rec[key]=Path(rec[key]).resolve().relative_to(ROOT).as_posix()
                    except ValueError:rec[key]=Path(rec[key]).name
            records.append(rec)
            print(f"{method}: {i}/{len(rows)} leaf={row['leaf']} success={rec.get('success')}",flush=True)
    with (output/'reconstruction_summary.json').open('w',encoding='utf-8') as f:
        json.dump(records,f,ensure_ascii=False,indent=2)
    return 0 if all(r.get('success') for r in records) else 1

if __name__=='__main__':raise SystemExit(main())
