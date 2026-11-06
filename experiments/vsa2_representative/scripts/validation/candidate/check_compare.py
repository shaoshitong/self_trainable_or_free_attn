#!/usr/bin/env python3
"""Strict CPU acceptance of matched benchmark outputs; no Torch/device access."""
from pathlib import Path
import json,math,statistics,sys

def validate(profile,path):
 r=json.loads(path.read_text());s=r['settings'];out=path.parent
 assert r['profile']==profile and r['status']=='ok' and not r['identity_differences']
 assert r['run_order']==['baseline','candidate','candidate','baseline'] and len(r['children'])==4
 expected=json.loads((out/'import.json').read_text())['extension_sha256']
 manifest=json.loads((out/'work/SOURCE_MANIFEST.yaml').read_text())
 ids={'kernel.native_tk.fwd.A','kernel.tk.fwd.A'}
 if profile=='representative':
  ids|={'kernel.tk.fwd.A|B','kernel.tk.fwd.A|B|C','kernel.tk.fwd.B|C','kernel.generic.fwd.A|B|C','kernel.tk.tables','kernel.tk.bwd.A','kernel.tk.bwd.B','kernel.tk.bwd.C'}
  ids|={p+'.'+v for p in ['fine_op.tk.A|B|C','fine_op.tk.A','fine_op.generic.A|B|C','full_op.tk.routed'] for v in ['f','f_plus_b','b_direct']}
  assert r['claims']['output_validated'] and r['claims']['may_claim_op_level_change']
 else:assert not r['claims']['output_validated'] and not r['claims']['may_claim_op_level_change']
 samples={}
 for c,arm in zip(r['children'],r['run_order']):
  assert c['arm']==arm and c['config']==r['configs'][arm]
  i=c['identity'];assert i['build']['extension_sha256']==expected and i['source']['rep_tk_sha256']==manifest['rep_tk_sha256']
  assert 'H100' in i['hardware']['name'] and i['hardware']['capability']==[9,0]
  assert i['shape']==dict(B=1,H=16,L=s['len'],D=128,NB=s['len']//64,RM=4,dtype='bf16',blk_cnt='full')
  assert len(c['rows'])==len(ids) and {x['id'] for x in c['rows']}==ids
  for x in c['rows']:
   n=s['batches']*s['iters'] if x['meta']['stage']=='b_direct' else s['batches']
   assert len(x['ms'])==n and all(isinstance(t,(int,float)) and math.isfinite(t) and t>0 for t in x['ms'])
   samples.setdefault((x['id'],arm),[]).extend(x['ms'])
  if profile=='representative':
   checks=c['output_check'];assert len(checks)==6 and all(x['ok'] is True for x in checks)
   for x in checks:
    if 'max_err' in x:
     err,scale=x['max_err'],x['scale'];assert math.isfinite(err) and math.isfinite(scale) and err>=0 and scale>=0
     tol=1e-4 if x['name'].startswith('lse') else 1e-2 if x['name'].startswith('o ') else 2e-2
     assert x['tol_rel']==tol and err<=tol*max(scale,1)+1e-12
    else:assert x['name']=='router reproduces the frozen selection'
 measured={x['id']:x for x in r['summary'] if x.get('measured')}
 assert set(measured)==ids
 for rid,x in measured.items():
  for arm in ['baseline','candidate']:
   v=samples[rid,arm];assert x[arm]['n']==len(v) and abs(x[arm]['median_ms']-statistics.median(v))<=1e-12
  assert abs(x['candidate_over_baseline']-x['candidate']['median_ms']/x['baseline']['median_ms'])<=1e-12
 for x in r['summary']:
  if not x.get('measured'):
   assert x['stage']=='b_estimate' and x['complete_call'] is False
   base=x['id'].removesuffix('.b_estimate')
   for arm in ['baseline','candidate']:
    v=statistics.median(samples[base+'.f_plus_b',arm])-statistics.median(samples[base+'.f',arm])
    assert math.isfinite(x['b_estimate_ms'][arm]) and abs(x['b_estimate_ms'][arm]-v)<=1e-12
 return dict(passed=True,profile=profile,children=4,measured_rows=len(ids),matched_build_source_inputs_selection_gpu=True,output_validated=profile=='representative',timing_only=profile!='representative',summary=r['summary'])

def main():
 mode,a,b=sys.argv[1:4];a=Path(a);b=Path(b)
 if mode=='decide':
  def ratio(p):return next(x['candidate_over_baseline'] for x in json.loads(p.read_text())['summary'] if x['id']=='full_op.tk.routed.f_plus_b')
  aa,main=ratio(a),ratio(b);noise=abs(1-aa);gain=1-main
  r=dict(run_second_length=gain>max(.01,noise),aa_ratio=aa,main_ratio=main,gain=gain,threshold=max(.01,noise),rule='measured full-op F+B gain exceeds both A/A displacement and a1percent minimum; experiment routing only, not significance')
  b=b.parent/'second_length_decision.json'
 else:r=validate(mode,a)
 b.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps(r,allow_nan=False))
if __name__=='__main__':main()
