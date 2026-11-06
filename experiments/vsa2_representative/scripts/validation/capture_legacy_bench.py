#!/usr/bin/env python3
"""Keep legacy timing boundaries; capture measured F+B before its printed B=FB-F estimate."""
from pathlib import Path
import importlib,json,math,sys
out=Path(sys.argv[1]);sys.argv=['bench_tk.py',*sys.argv[2:]]
b=importlib.import_module('bench_tk');original=b.timeit;records=[]
def capture(fn,iters,warmup):
 ms=original(fn,iters,warmup)
 assert math.isfinite(ms) and ms>0
 cells=dict(zip(fn.__code__.co_freevars,[x.cell_contents for x in (fn.__closure__ or ())]))
 if cells.get('be') in ('generic','tk') and cells.get('ph') in (1,7):
  records.append(dict(backend=cells['be'],phases=cells['ph'],scope='fwd_bwd' if fn.__name__=='fb' else 'fwd',ms=ms,iters=iters,warmup=warmup))
 return ms
b.timeit=capture;b.main()
assert len(records)==8 and len({(r['backend'],r['phases'],r['scope']) for r in records})==8,records
abc={}
for be in ['generic','tk']:
 row={r['scope']:r['ms'] for r in records if r['backend']==be and r['phases']==7};row['bwd_estimate']=row['fwd_bwd']-row['fwd'];abc[be]=row
assert all(r['bwd_estimate']>0 for r in abc.values())
r=dict(measurement='one warmed CUDA-event loop average, not independent samples',scope='rep_fine_attention ABC, frozen selection; excludes routing, coarse branch and gate',native_A_only_is_semantic_baseline=False,backward_scope='estimate F+B minus separately timed F',records=records,abc=abc)
(out/'legacy_bench.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
