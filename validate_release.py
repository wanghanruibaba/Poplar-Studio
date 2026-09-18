"""Check leaf pairing, 40 balanced storage folders, metadata and optional SHA-256."""
import argparse, ast, csv, hashlib, json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hashes',action='store_true');args=parser.parse_args()
    leaves=read(ROOT/'data/leaf_manifest.csv');bench=read(ROOT/'data/benchmark/sampled_four_method_leaf_areas.csv')
    storage=read(ROOT/'data/storage_manifest.csv')
    byid={r['leaf']:r for r in leaves};benchmark_ids={r['leaf'] for r in bench}
    assert len(leaves)==len(byid)==839 and len(bench)==len(benchmark_ids)==240
    assert {r['leaf'] for r in leaves if r['in_benchmark']=='True'}==benchmark_ids
    folders={f'{i:04d}' for i in range(1,41)}
    assert len(storage)==40 and {r['storage_folder'] for r in storage}==folders
    assert {p.name for p in (ROOT/'data/raw').iterdir() if p.is_dir()}==folders
    expected=set()
    for r in leaves:
        assert len(r['leaf'])==4 and r['leaf'].isdigit()
        for k in ('txt_path','pca_obj_path'):
            p=ROOT/r[k]
            assert not Path(r[k]).is_absolute() and ROOT in p.resolve().parents
            assert p.is_file() and p.parent.name==r['storage_folder']
            expected.add(r[k])
    assert {p.relative_to(ROOT).as_posix() for p in (ROOT/'data/raw').rglob('*') if p.is_file()}==expected
    totals=[]
    for folder in storage:
        members=[r for r in leaves if r['storage_folder']==folder['storage_folder']]
        totals.append(len(members))
        assert len(members)==int(folder['leaf_count'])
        assert sum(r['leaf'] in benchmark_ids for r in members)==int(folder['benchmark_leaf_count'])==6
        for g in ('01','02','03','04'):
            assert sum(r['genotype']==g for r in members)==int(folder[f'group_{g}_leaf_count'])
    assert sorted(totals)==[20]+[21]*39
    assert Counter(r['genotype'] for r in bench)=={'01':60,'02':60,'03':60,'04':60}
    methods={'pca':'_pca_mesh.obj','fixed_poisson':'_fixed_poisson.obj',
             'adaptive_poisson':'_adaptive_poisson.obj','ball_pivoting':'_ball_pivoting.obj'}
    for r in bench:
        for k in ('genotype','scale_factor','storage_folder','txt_path','pca_obj_path'):
            assert r[k]==byid[r['leaf']][k]
    for method,suffix in methods.items():
        assert {p.name[:-len(suffix)] for p in (ROOT/'data/benchmark'/method).glob('*'+suffix)}==benchmark_ids
    for path in (ROOT/'scripts').glob('*.py'):ast.parse(path.read_text(encoding='utf8'))
    if args.hashes:
        hashes=read(ROOT/'data/file_checksums.csv')
        actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()
                and not any(x in p.relative_to(ROOT).parts for x in ('.git','runs','__pycache__','.venv'))
                and p!=ROOT/'data/file_checksums.csv'}
        assert actual=={r['path'] for r in hashes}
        for r in hashes:
            p=ROOT/r['path']
            assert p.stat().st_size==int(r['bytes']) and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],r['path']
    print(json.dumps({'status':'PASS','leaves':839,'storage_folders':40,'leaves_per_folder':'20-21',
          'benchmark_leaves_per_folder':6,'hashes_checked':args.hashes}))
if __name__=='__main__':main()
