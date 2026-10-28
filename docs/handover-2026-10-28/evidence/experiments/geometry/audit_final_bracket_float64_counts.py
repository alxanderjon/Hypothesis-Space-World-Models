"""Count-only safety gate for the requested discrete minimizer audit.

No candidate enumeration, score evaluation, model execution, or root search.
"""
import csv
import hashlib
import json
import struct
import time
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path('outputs/sho/stage1B')
SOURCE=ROOT/'defer_aware_stationary_locator_audit'
OUT=ROOT/'stationary_float64_minimizer_audit'
LIMIT=10000


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run():
    start=time.monotonic();OUT.mkdir(parents=True,exist_ok=False)
    manifest_path=SOURCE/'defer_aware_stationary_locator_validated.json'
    source_path=SOURCE/'defer_aware_root_recovery.csv'
    manifest=json.loads(manifest_path.read_text());prior=json.loads((SOURCE/'results.json').read_text())
    assert manifest['status']=='VALIDATED_UNDER_TEACHER_EQUIVALENCE'
    assert sha(manifest_path)==prior['assembly_sha256']
    protected={str(p):sha(p) for p in ROOT.rglob('*.pt')}
    protected.update({str(p):sha(p) for p in ROOT.rglob('*manifest.json')})
    protected[str(manifest_path)]=sha(manifest_path);protected[str(source_path)]=sha(source_path)
    records=list(csv.DictReader(source_path.open()));assert len(records)==7093
    rows=[]
    for record in records:
        L=float(record['final_L']);R=float(record['final_R']);assert 0<L<R<1
        # Positive finite binary64 values are ordered by their unsigned bit codes.
        left_bits=struct.unpack('>Q',struct.pack('>d',L))[0]
        right_bits=struct.unpack('>Q',struct.pack('>d',R))[0]
        count=right_bits-left_bits+1  # include BOTH endpoints
        assert count==int(record['final_width_in_float_steps'])+1
        rows.append(dict(index=int(record['index']),scope=record['scope'],case=record['case'],
            final_L=L,final_L_hex=L.hex(),final_R=R,final_R_hex=R.hex(),
            L_uint64=left_bits,R_uint64=right_bits,inclusive_float64_count=count,
            bracket_width=R-L,exceeds_10000=count>LIMIT,
            left_state=record['left_state'],right_state=record['right_state'],
            c_teacher_decimal=record['c_teacher_decimal'],previous_termination=record['termination']))
    counts=np.array([r['inclusive_float64_count'] for r in rows],dtype=np.int64)
    oversized=sorted([r for r in rows if r['exceeds_10000']],key=lambda r:r['inclusive_float64_count'],reverse=True)
    histogram=[]
    for lo,hi in [(1,1),(2,2),(3,4),(5,8),(9,16),(17,32),(33,64),(65,128),(129,256),(257,1024),(1025,10000),(10001,int(counts.max()))]:
        histogram.append(dict(min_count=lo,max_count=hi,brackets=int(np.sum((counts>=lo)&(counts<=hi)))))
    exact=[dict(inclusive_float64_count=int(c),brackets=int(n)) for c,n in sorted(Counter(counts).items())]
    write_csv(OUT/'final_bracket_float64_counts.csv',rows)
    write_csv(OUT/'final_bracket_count_histogram.csv',histogram)
    write_csv(OUT/'final_bracket_exact_count_distribution.csv',exact)
    if oversized:write_csv(OUT/'oversized_final_brackets.csv',oversized)
    result=dict(status='STOP_SIZE_LIMIT' if oversized else 'COUNT_GATE_PASSED',brackets=len(rows),
        inclusive_candidate_limit=LIMIT,minimum_count=int(counts.min()),median_count=float(np.median(counts)),
        p99_count=float(np.quantile(counts,.99)),p99_method='NumPy linear interpolation',maximum_count=int(counts.max()),
        oversized_brackets=len(oversized),within_limit_brackets=int(np.sum(counts<=LIMIT)),
        sum_inclusive_counts=int(counts.sum()),
        counting_rule='For positive finite float64 cuts: uint64_bits(R)-uint64_bits(L)+1; includes both endpoints.',
        enumeration_performed=False,teacher_scores_evaluated=0,learned_scores_evaluated=0,
        downstream_minimum_comparisons_performed=False,checkpoint_or_model_created=False,
        action='Stopped after counting all brackets; did not enumerate even the smaller brackets because the global size gate failed.' if oversized else 'Count-only implementation completed; enumeration not implemented in this module.',
        source_brackets=str(source_path),source_brackets_sha256=sha(source_path),source_locator_manifest=str(manifest_path),
        source_locator_sha256=sha(manifest_path),implementation_sha256=sha(__file__),elapsed_seconds=time.monotonic()-start)
    assert all(sha(p)==h for p,h in protected.items());result['all_protected_hashes_unchanged']=True
    (OUT/'results.json').write_text(json.dumps(result,indent=2));(OUT/'protected_hashes.json').write_text(json.dumps(protected,indent=2))
    lines=['FINAL-BRACKET DISCRETE MINIMIZER AUDIT — STOPPED AT COUNT GATE','',
        f"Counted all {len(rows):,} saved final trusted brackets, including both endpoints.",
        f"Candidate count minimum / median / p99 / maximum: {counts.min()} / {np.median(counts):g} / {np.quantile(counts,.99):.12g} / {counts.max()}.",
        'P99 uses linear interpolation; individual bracket counts are exact integers.',
        f"{len(oversized)} brackets exceed the 10,000-float limit; {np.sum(counts<=LIMIT):,} are within it.",'',
        'Count range | brackets']
    lines += [f"{h['min_count']}–{h['max_count']} | {h['brackets']}" for h in histogram]
    lines += ['', 'OVERSIZED BRACKETS:']
    for r in oversized:
        lines.append(f"index {r['index']}; {r['scope']} / {r['case']}; {r['inclusive_float64_count']:,} floats; L={r['final_L']:.17g} ({r['final_L_hex']}); R={r['final_R']:.17g} ({r['final_R_hex']}); width={r['bracket_width']:.17g}.")
    lines += ['',result['action'],
        'No score evaluations, minimizer selections, root searches, interval/parent minimum comparisons, training, recursion, global search, or checkpoint edits.',
        'stationary_float64_minimizer was NOT promoted. The existing locator remains unchanged.',
        'Outputs: final_bracket_float64_counts.csv; final_bracket_count_histogram.csv; final_bracket_exact_count_distribution.csv; oversized_final_brackets.csv; results.json; protected_hashes.json.']
    (OUT/'final_bracket_float64_minimizer_summary.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2));print(json.dumps(oversized,indent=2))


if __name__=='__main__':run()
