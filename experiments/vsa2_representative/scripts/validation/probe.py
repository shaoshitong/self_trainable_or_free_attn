#!/usr/bin/env python3
"""Worker-local identity checks. Raw output is private; share only the separate sanitized feedback."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys
import torch
out=Path(sys.argv[2]);mode=sys.argv[1];work=out/'work'
if mode=='prepare':
 m=json.loads((work/'SOURCE_MANIFEST.yaml').read_text())
 bad=[r['path'] for r in m['files'] if hashlib.sha256((work/r['path']).read_bytes()).hexdigest()!=r['sha256']]
 assert not bad,bad
 tk=Path(os.environ['REP_TK_ROOT']);tm=json.loads((work/'_control/TK_SOURCE_MANIFEST.yaml').read_text())
 bad=[r['path'] for r in tm['files'] if not (tk/r['path']).is_file() or hashlib.sha256((tk/r['path']).read_bytes()).hexdigest()!=r['sha256']]
 assert not bad,bad
 assert torch.cuda.device_count()==1 and torch.cuda.get_device_capability()==(9,0)
 prop=torch.cuda.get_device_properties(0);assert 'H100' in prop.name,prop.name
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),gpu=prop.name,uuid=str(prop.uuid),capability=[prop.major,prop.minor],multiprocessors=prop.multi_processor_count,torch=torch.__version__,cuda=torch.version.cuda,source_commit=os.environ['REP_SOURCE_COMMIT'],source_sha256=os.environ['REP_SOURCE_SHA'],source_files=len(m['files']),all_source_sha_verified=True,tk_pin=tm['git_commit'],tk_files=len(tm['files']),all_tk_sha_verified=True,nvcc=subprocess.check_output(['nvcc','--version'],text=True),compiler=subprocess.check_output([os.environ['CXX'],'--version'],text=True).splitlines()[0])
 (out/'hardware.json').write_text(json.dumps(r,indent=2)+'\n')
else:
 from vsa2_rep.ops import ext
 C=ext();assert C.has_tk()
 p=Path(C.__file__);assert p.is_relative_to(work)
 elf=subprocess.check_output(['readelf','-d',str(p)],text=True)
 assert '[libcuda.so.1]' in elf and 'RPATH' not in elf and 'RUNPATH' not in elf
 r=dict(module=str(p),extension_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,has_tk=True,readelf=elf,config=1)
 (out/'import.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))
