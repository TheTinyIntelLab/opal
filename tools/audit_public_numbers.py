"""Ground public numerical literals in compact saved records, never NPZ banks.

Value agreement is distinct from checking a sentence's scientific interpretation.
All possible matching sources and unresolved literals remain explicit.
"""
import argparse
import bisect
import csv
import json
import math
import heapq
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from opal.replicated.contracts import sha,write_json

NUMBER=re.compile(r'(?<![\w.])[-+]?\d+\.\d+(?:[eE][-+]?\d+|\\times10\^\{[-+]?\d+\})?(?:\\?%)?(?!\w|\.\d)')
FORBIDDEN={'responses','doubles','predictions','raw_counts','counts','expression','singles','gene_names'}


def numbers(value,pointer=''):
    if isinstance(value,dict):
        for key,v in value.items():
            if key in FORBIDDEN or ':' in key or key.startswith('\\\\'):continue
            yield from numbers(v,pointer+'/'+key.replace('~','~0').replace('/','~1'))
    elif isinstance(value,list):
        if len(value)>200:return
        for i,v in enumerate(value):yield from numbers(v,pointer+'/'+str(i))
    elif isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value):
        yield pointer,float(value)


def source_index(root):
    records=[]
    for directory in ('results','configs','images'):
        for path in sorted((root/directory).rglob('*')):
            rel=path.relative_to(root).as_posix()
            if path.suffix not in {'.json','.csv'} or path.stat().st_size>20_000_000:continue
            if any(s in rel for s in ('source_snapshots','release-closeout','replicated_fixed_state_v1','fitted','checkpoint')):continue
            try:
                if path.suffix=='.json':items=list(numbers(json.loads(path.read_text(encoding='utf-8'))))
                else:
                    items=[]
                    with path.open(encoding='utf-8',newline='') as stream:
                        for i,row in enumerate(csv.DictReader(stream),2):
                            for key,v in row.items():
                                try:number=float(v)
                                except (ValueError,TypeError):continue
                                if math.isfinite(number):items.append((f'row:{i}/{key}',number))
            except (ValueError,OSError):continue
            digest=sha(path)
            zeros=0
            for pointer,value in items:
                if value==0:
                    zeros+=1
                    if zeros>3:continue
                records.append({'source_path':rel,'source_sha256':digest,'pointer':pointer,'raw_value':value})
    return records


def literal_value(token):
    percent=token.endswith('%');plain=token.removesuffix('%').removesuffix('\\')
    scientific=re.fullmatch(r'([-+]?\d+\.\d+)\\times10\^\{([-+]?\d+)\}',plain)
    if scientific:mantissa,exponent=scientific.groups();exponent=int(exponent)
    else:
        parts=re.split('[eE]',plain);mantissa=parts[0];exponent=int(parts[1]) if len(parts)>1 else 0
    digits=len(mantissa.split('.')[1])
    return float(mantissa)*10**exponent,.500001*10**(exponent-digits),percent


def audit(root=ROOT):
    records=source_index(root)
    transformed=[]
    for record in records:
        value=record['raw_value']
        transformations=[(1,'identity'),(-1,'sign_reversal')]
        pointer=record['pointer'].lower()
        if any(k in pointer for k in ('gain','skill','coverage','probability','relative','reduction','concentration')):
            transformations.extend([(100,'fraction_to_percent'),(-100,'loss_fraction_to_percent')])
        if 'seconds' in pointer:transformations.append((1000,'seconds_to_milliseconds'))
        if 'bytes' in pointer:
            transformations.extend([(1e-6,'bytes_to_decimal_MB'),(1/1024**2,'bytes_to_MiB'),
                                    (1e-9,'bytes_to_decimal_GB'),(1/1024**3,'bytes_to_GiB')])
        for multiplier,label in transformations:
            transformed.append((value*multiplier,label,record))
    transformed.sort(key=lambda x:x[0]);values=[r[0] for r in transformed]
    claims=[]
    surfaces=[root/'README.md',root/'docs/models.md',root/'paper/main.tex',*sorted((root/'paper/evidence').glob('*.tex'))]
    for path in surfaces:
        lines=path.read_text(encoding='utf-8').splitlines()
        for line_number,line in enumerate(lines,1):
            if line.lstrip().startswith('%'):continue
            # Layout and version strings are not empirical outcome claims.
            clean=re.sub(r'\\(?:url|href)\{[^}]*\}|\\includegraphics\[[^]]*\]|\\(?:hspace|vspace)\{[^}]*\}|\b\d+\.\d+(?:\.\d+)?(?:\.dev\d+)?\b(?=\s*(?:through|environment|/CUDA))','',line)
            clean=re.sub(r'10\.\d{4,}/[a-zA-Z0-9_-]+','',clean)
            for match in NUMBER.finditer(clean):
                token=match.group();before=clean[max(0,match.start()-25):match.start()]
                if re.search(r'(?:version\s*=\s*\{|\bv|\bPython\s+|\bBY\s+|Opal\s*\*?\*?|0\.)$',before):continue
                if token.count('.')>1:continue
                value,tolerance,percent=literal_value(token)
                range_endpoint=token.startswith('-') and match.start()>0 and clean[match.start()-1]=='-'
                if range_endpoint:value=abs(value)
                lo=bisect.bisect_left(values,value-tolerance);hi=bisect.bisect_right(values,value+tolerance)
                candidates=[]
                context=' '.join(lines[max(0,line_number-4):min(len(lines),line_number+3)])
                terms=set(re.findall(r'[a-z][a-z0-9_-]{3,}',context.lower()))
                nearest=heapq.nsmallest(256,range(lo,hi),key=lambda i:abs(values[i]-value))
                for index in nearest:
                    displayed,transform,record=transformed[index]
                    if percent and transform not in {'identity','fraction_to_percent','loss_fraction_to_percent'}:continue
                    if not percent and transform in {'fraction_to_percent','loss_fraction_to_percent'} and not (re.search(r'gain|skill|coverage|risk|\\%',context,re.I) or path.name in {'ceiling_risk.tex','stage1_results.tex','canonical_claims.tex'}):continue
                    if transform.startswith('bytes_to') and not re.search(r'\b(?:MB|GB|MiB|GiB)\b',context):continue
                    if transform=='seconds_to_milliseconds' and not re.search('millisecond|\\bms\\b',context):continue
                    score=sum(1 for term in terms if term in (record['source_path']+' '+record['pointer']).lower())
                    candidates.append((score,{**record,'transform':transform,'displayed_value':displayed}))
                candidates.sort(key=lambda c:(-c[0],c[1]['source_path'],c[1]['pointer']))
                claims.append({'surface':path.relative_to(root).as_posix(),'line':line_number,'literal':token,
                    'value':value,'rounding_tolerance':tolerance,'context':line,
                    'tex_range_endpoint':range_endpoint,
                    'status':'value_grounded_requires_context_review' if candidates else 'unresolved_literal',
                    'matching_sources':[c[1] for c in candidates[:8]],'nearest_sources_considered':len(nearest),
                    'numerically_possible_source_count':hi-lo,'matching_source_count':len(candidates)})
    return {'schema':'opal-public-number-audit-v1','claims':claims,'literals':len(claims),
            'grounded':sum(bool(c['matching_sources']) for c in claims),
            'unresolved':sum(not c['matching_sources'] for c in claims),
            'source_numeric_fields':len(records),'expression_banks_opened':False,
            'scope':'decimal empirical literals and percentages across README, model guide, manuscript and evidence tables',
            'limitations':['Integer sample counts and external/version/layout facts need separate review',
                'Value matches do not certify candidate/comparator semantics or scientific interpretation',
                'Candidate lookup is bounded to the nearest 256 numerical fields per literal',
                'Ambiguous and unresolved claims remain visible; this is not an automatic whole-paper pass']}


def check(record,root=ROOT):
    cache={}
    for claim in record['claims']:
        lines=(root/claim['surface']).read_text(encoding='utf-8').splitlines()
        if claim['context']!=lines[claim['line']-1]:raise ValueError('Changed audited public sentence')
        for source in claim['matching_sources']:
            path=root/source['source_path']
            if source['source_path'] not in cache:cache[source['source_path']]=sha(path)
            if cache[source['source_path']]!=source['source_sha256']:raise ValueError('Changed audited numerical source')
            if abs(source['displayed_value']-claim['value'])>claim['rounding_tolerance']:
                raise ValueError('Numerical disagreement exceeds declared rounding')
    return {'checked_literals':record['literals'],'source_hashes_checked':len(cache),'whole_paper_semantics_certified':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--check',action='store_true');args=parser.parse_args()
    if args.check:print(check(json.loads(args.output.read_text())));return
    if args.output.exists():raise FileExistsError('Preserve prior numerical audits')
    record=audit();write_json(args.output,record)
    print({k:v for k,v in record.items() if k not in {'claims','limitations'}})


if __name__=='__main__':main()
