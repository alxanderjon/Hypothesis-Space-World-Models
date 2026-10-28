#!/usr/bin/env python3
import hashlib,json,os,sys
from pathlib import Path
def sha(p):
 h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()
root=Path(sys.argv[1]); plan=json.load(open(sys.argv[2])); adm=json.load(open(root/'admission_decision.json'))
err={'status':'ORIGINAL_COMPLETION_ERROR_PRESERVED','error_type':'KeyError','missing_path':'plan[scope][qualification]','actual_path':'plan[qualification]','scientific_work_repeated':False,'original_counts':adm['counts']}
(root/'terminal_error.json').write_text(json.dumps(err,indent=2,sort_keys=True))
res={'classification':'NOT_SUPPORTED_AT_ACQUISITION_ADMISSION','admissions':adm['admissions'],'counts':adm['counts'],'qualification':plan['qualification'],'recovery_qualification':'Administrative reconstruction from the atomic admission decision only; zero simulator, model, kernel, solve, score, or training repetition.'}
tmp=root/'result.json.tmp'; tmp.write_text(json.dumps(res,indent=2,sort_keys=True)); os.replace(tmp,root/'result.json')
comp={'status':'COMPLETE_AFTER_ADDITIVE_METADATA_RECOVERY','classification':res['classification'],'result_sha256':sha(root/'result.json'),'original_error_sha256':sha(root/'terminal_error.json'),'counts':adm['counts'],'repeated_scientific_work':0}
(root/'completion.json').write_text(json.dumps(comp,indent=2,sort_keys=True))
