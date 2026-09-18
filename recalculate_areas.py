"""Recalculate triangle surface area without replacing reference measurements."""
from __future__ import annotations
import argparse,csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
METHODS={'fixed_poisson':'_fixed_poisson.obj','adaptive_poisson':'_adaptive_poisson.obj',
         'ball_pivoting':'_ball_pivoting.obj','pca':'_pca_mesh.obj'}

def surface_area(path):
    vertices=[];triangles=[]
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.startswith('v '):vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('f '):
                face=[int(t.split('/')[0]) for t in line.split()[1:]]
                face=[i-1 if i>0 else len(vertices)+i for i in face]
                triangles.extend((face[0],face[i],face[i+1]) for i in range(1,len(face)-1))
    v=np.asarray(vertices,dtype=float);t=np.asarray(triangles,dtype=int)
    if not len(v) or not len(t):raise ValueError(f'Empty mesh: {path}')
    if t.min()<0 or t.max()>=len(v):raise ValueError(f'Invalid face index: {path}')
    return float(np.linalg.norm(np.cross(v[t[:,1]]-v[t[:,0]],v[t[:,2]]-v[t[:,0]]),axis=1).sum()/2)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh-root',type=Path,default=ROOT/'data/benchmark')
    p.add_argument('--output',type=Path,default=ROOT/'runs/recalculated_areas.csv')
    args=p.parse_args()
    with (ROOT/'data/benchmark/sampled_four_method_leaf_areas.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    results=[]
    for row in rows:
        scale=float(row['scale_factor'])
        if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid scale factor')
        for method,suffix in METHODS.items():
            raw=surface_area(args.mesh_root/method/(row['leaf']+suffix))
            area=raw*scale*scale;archived=float(row[method+'_area_cm2'])
            results.append({'leaf':row['leaf'],'storage_folder':row['storage_folder'],'genotype':row['genotype'],
                'method':method,'scale_factor':scale,'mesh_area_raw':raw,'area_cm2':area,
                'archived_area_cm2':archived,'difference_cm2':area-archived,
                'reference_area_cm2':row['measured_area_cm2']})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    maxdiff=max(abs(r['difference_cm2']) for r in results)
    print(f'{len(results)} mesh areas recalculated. Maximum absolute difference from archived table: {maxdiff:.9g} cm2')
    return 0

if __name__=='__main__':raise SystemExit(main())
