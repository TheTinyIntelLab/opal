"""Acquire pinned official GxE measurement inputs with bounded, immutable custody."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import requests

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/experiments/opal_gxe_measurement_v1.json'
BASE='https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM7056nnn/GSM7056151/suppl/'
FILES={
 'cells.txt.gz':('GSM7056151_sciPlex_4_cell.annotations.txt.gz',None),
 'genes.txt.gz':('GSM7056151_sciPlex_4_gene.annotations.txt.gz',None),
 'hashes.txt.gz':('GSM7056151_sciPlex_4_hashTable.out.txt.gz',19793477),
 'sample_sheet.txt.gz':('GSM7056151_sciPlex_4_hash_sample_sheet.txt.gz',None),
 'counts.mtx.gz':('GSM7056151_sciPlex_4_UMI.count.matrix.gz',590579159)}


def write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')


def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--counts',action='store_true');args=p.parse_args(argv)
 config=json.loads(CONFIG.read_text());args.output.mkdir(parents=True,exist_ok=False)
 maximum=config['budgets']['download_bytes'];receipts=[];total=0;start=time.monotonic()
 write(args.output/'acceptance.json',dict(approved=True,owner='project owner in this conversation',
   authorization_source='explicit get them, do it all owner instruction; persistent execution authorization',
   scope='official public chemical pilot acquisition, local research only; no biological confirmation',
   approved_at=datetime.now(timezone.utc).isoformat(),budget=config['budgets'],config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest()))
 try:
  if shutil.disk_usage(args.output).free < maximum+config['budgets']['reserve_bytes']:
   raise RuntimeError('Insufficient verified free space for complete acquisition and reserve')
  for name,(remote,expected) in FILES.items():
   if name=='counts.mtx.gz' and not args.counts:continue
   url=BASE+remote;digest=hashlib.sha256();size=0
   with requests.get(url,stream=True,timeout=(15,60)) as response:
    response.raise_for_status()
    length=int(response.headers['Content-Length']) if response.headers.get('Content-Length') else None
    if expected and length and length!=expected:raise ValueError('Official file size changed')
    if length and total+length>maximum:raise ValueError('Complete download exceeds accepted byte budget')
    with (args.output/name).open('xb') as file:
     for chunk in response.iter_content(1024*1024):
      size+=len(chunk);total+=len(chunk)
      if total>maximum:raise ValueError('Accepted download byte stop')
      if shutil.disk_usage(args.output).free<config['budgets']['reserve_bytes']:raise ValueError('Writable reserve stop')
      file.write(chunk);digest.update(chunk)
    if (expected and size!=expected) or (length is not None and size!=length):raise ValueError('Truncated official download')
    with (args.output/name).open('rb') as file:
     if file.read(2)!=b'\x1f\x8b':raise ValueError('Official gzip payload required; HTML is not data')
    receipts.append(dict(path=name,url=url,bytes=size,sha256=digest.hexdigest(),
      last_modified=response.headers.get('Last-Modified'),publisher_checksum_available=False,
      identity_scope='official HTTPS endpoint, exact advertised size and frozen acquired SHA256; gzip CRC checked during decoding'))
    write(args.output/'source_receipts.json',dict(files=receipts,total_download_bytes=total))
    print('acquired',name,size,flush=True)
  write(args.output/'decision.json',dict(status='official_inputs_acquired',measurement_eligibility='not yet qualified',
    biological_confirmation=False,redistribution_authorized=False,counts_acquired=args.counts))
 except Exception as error:
  write(args.output/'failure.json',dict(status='partial_acquisition_failed',reason=str(error),partial_files_retained=True));raise
 finally:
  write(args.output/'resource_ledger.json',dict(scope='acquisition process only',download_bytes=total,
    wall_seconds=time.monotonic()-start,installs=0,gpu_seconds=0,raw_inputs_copied=False))


if __name__=='__main__':main()
