#!/usr/bin/env python3
from pathlib import Path
import json,re,sys
name=sys.argv[1];out=Path(sys.argv[2]);text=(out/(name+'.log')).read_text()
expected={'cpu':('all CPU reference tests passed',6,0),'generic':('all GPU tests passed',1,7),'tk':('all TK GPU tests passed',1,5)}
marker,minimum,families=expected[name]
assert marker in text and not re.search(r'(^FAIL|Traceback|^FAILED)',text,re.M)
checks=len(re.findall(r'^ok\s',text,re.M));observed=len(re.findall(r'^--- test_',text,re.M))
assert checks>=minimum and observed==families,(name,checks,observed)
r=dict(suite=name,passed=True,check_lines=checks,test_families=observed,allskip=False,terminal_marker=marker)
p=out/'suite_checks.json';all=json.loads(p.read_text()) if p.exists() else {};all[name]=r;p.write_text(json.dumps(all,indent=2)+'\n');print(json.dumps(r))
