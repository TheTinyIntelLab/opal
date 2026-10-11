"""Reproduce author metadata filtering, without reading expression or graph assets."""
from collections import Counter
from pathlib import Path
import re
import sys
import urllib.request
import h5py
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from opal.artifacts import read_json,write_json,sha256
from prepare_information import OUT,LOCAL,resource_check
from prepare_pertspectra_series import column

def run():
 resource_check(100000)
 raw=ROOT/'data/raw/pertspectra-model-series/inhouse.h5ad'
 with h5py.File(raw,'r') as f:
  targets=column(f['obs/target_gene_name']);treatments=column(f['obs/Treatment'])
 conditions=[]
 for target in targets:
  condition=re.sub(r'\bINTERGENIC\w*\b','intergenic',str(target).replace('|','+'))
  condition=re.sub(r'\bCTRL00\w*\b','ctrl',condition)
  if condition=='ctrl+ctrl':condition='ctrl'
  elif 'ctrl' in condition and condition.count('+')==1:
   pair=condition.split('+');condition=pair[1] if pair[0]=='ctrl' else pair[0]
  conditions.append(condition)
 counts=Counter(c+'+'+str(t) for c,t in zip(conditions,treatments))
 retained=sum(n for n in counts.values() if n>=50)
 headers=[]
 for asset in ['StringDB.HQ.txt','Homo_sapiens.gene_info']:
  url='https://pert-spectra.s3.amazonaws.com/references/'+asset
  try:
   with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=20) as response:
    record={'asset':asset,'url':url,'status':response.status,'content_length':response.headers.get('Content-Length'),'etag':response.headers.get('ETag')}
  except Exception as error:record={'asset':asset,'url':url,'error':str(error)}
  headers.append(record)
 write_json(OUT/'comparators/PertSpectra-metadata-filter.json',{
  'deposited_rows':len(targets),'author_50_cell_retained_rows':retained,'excluded_rows':len(targets)-retained,
  'condition_treatment_groups':len(counts),'retained_groups':sum(n>=50 for n in counts.values()),
  'reported_filtered_rows':65648,'matches_reported_filtered_rows':retained==65648,
  'recipe':'Pinned inhouse_preprocess ctrl/intergenic replacements, ctrl singleton collapse, then condition+Treatment >=50 across SampleIndex',
  'utils_sha256':sha256(LOCAL/'source-contract-snapshots/utils.txt'),
  'train_script_sha256':sha256(Path('D:/opal-model-series-20261010-v2/native/PertSpectra/PertSpectra_training_scripts/train_PertSpectra_inhouse.py')),
  'graph_asset_head_requests':headers,'expression_read':False,'large_assets_downloaded':False,
  'population_changed_for_frozen_study':False,'fully_eligible':False,
  'remaining':'Exact graph asset content hashes/terms/order and author gene/normalization/input-task parity; no native score produced.'})
 print(retained,'retained author-filter rows; source metadata only')
if __name__=='__main__':run()
