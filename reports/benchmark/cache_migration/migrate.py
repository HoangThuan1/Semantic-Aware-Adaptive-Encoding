"""One-time verified cache-key migration after policy-only oracle fix."""
import ast
import inspect
import json
from pathlib import Path

import pandas as pd
from vcu_surrogate.benchmark import VERSION, COUNT_COLUMNS, measure_grid, _sample_video, _save_table
from vcu_surrogate.common import load_config, sha256, stable_hash, write_json

root = Path('/workspace')
old_path = root/'reports/benchmark/cache_migration/benchmark_before_oracle_fix.py'
new_path = root/'src/vcu_surrogate/benchmark.py'
old_ast, new_ast = ast.parse(old_path.read_text()), ast.parse(new_path.read_text())
def function_body(tree, name):
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.body = [n for n in node.body if not (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'source_hash' for t in n.targets))]
    return ast.dump(node, include_attributes=False)
for function in ('measure_grid', '_sample_video', '_save_table'):
    assert function_body(old_ast, function) == function_body(new_ast, function), function
c = load_config()
shared = dict(metrics_implementation=sha256(root/'src/vcu_surrogate/benchmark_metrics.py'),
              detections_sha256=sha256(root/'artifacts/metadata/detections.parquet'),
              config=c['_config_sha256'], frames_per_gop=3, version=VERSION)
old_source = stable_hash(dict(implementation=sha256(old_path), **shared))
new_source = stable_hash(dict(implementation=stable_hash([inspect.getsource(measure_grid), inspect.getsource(_sample_video), inspect.getsource(_save_table), COUNT_COLUMNS]),
                             inference_implementation=sha256(root/'src/vcu_surrogate/inference.py'), **shared))
refs = pd.read_parquet(root/'artifacts/reference/manifest.parquet').set_index('video_id')
encodes = pd.read_parquet(root/'artifacts/encodes/manifest.parquet')
records=[]
for e in encodes.itertuples(index=False):
    p=root/'reports/benchmark/cache'/e.video_id/f'action_{e.action_id:02d}.parquet'
    cl=p.with_name(p.stem+'.classes.parquet')
    meta=p.with_suffix('.json')
    if not all(v.is_file() for v in (p,cl,meta)):
        continue
    m=json.loads(meta.read_text())
    inputs=dict(reference=refs.loc[e.video_id].reference_sha256, encode=e.encoded_sha256)
    if m['job_hash'] != stable_hash(dict(source=old_source, **inputs)):
        continue
    assert sha256(p)==m['output_sha256'] and sha256(cl)==m['classes_sha256']
    f=pd.read_parquet(p)
    assert len(f)==int(refs.loc[e.video_id].gop_count)*3
    assert set(f.action_id)=={e.action_id}
    old_job=m['job_hash']
    m['job_hash']=stable_hash(dict(source=new_source, **inputs))
    m['cache_key_migration']={'old_job_hash':old_job, 'old_module_sha256':sha256(old_path),
                              'reason':'Measurement function AST unchanged excluding cache-key assignment; policy oracle fix only.'}
    write_json(meta,m)
    records.append(dict(video_id=e.video_id,action_id=e.action_id,old_job_hash=old_job,new_job_hash=m['job_hash']))
write_json(root/'reports/benchmark/cache_migration/manifest.json', {'measurement_ast_verified_identical':True,
             'old_module_sha256':sha256(old_path),'new_module_sha256':sha256(new_path),'shards':records})
print('Verified and migrated',len(records),'measurement shards; no metric values modified.')
