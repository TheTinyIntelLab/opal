"""Pin two pure-Python parser wheels and install them in an isolated tmp tree."""
import argparse
from email.parser import BytesParser
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.continuation_colo1_validator_v2 import NoRedirect, sha, unique_object

OUT = ROOT / 'results/research-continuation'
LOCAL = ROOT / 'tmp/colo1-rds-parser-v1'


def write(path, value):
    with path.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def gates():
    freeze = json.loads((OUT/'colo1-schema-freeze-v1.json').read_bytes())
    binding = json.loads((OUT/'colo1-parser-setup-code-binding-v1.json').read_bytes())
    if sha(OUT/'colo1-schema-freeze-v1.json') != binding['freeze_sha256']:
        raise ValueError('Parser setup freeze drift')
    for path, expected in {**freeze['repo_sha256'], **freeze['local_input_sha256'], **binding['code_sha256']}.items():
        if sha(ROOT/path) != expected:
            raise ValueError('Parser setup input drift: '+path)
    if shutil.disk_usage(ROOT).free < 2**30:
        raise RuntimeError('Unchanged disk reserve')


def fetch(url, cap, mime):
    request = urllib.request.Request(url, headers={'User-Agent': 'Opal-pinned-metadata-reader/1.0', 'Accept-Encoding': 'identity'})
    with urllib.request.build_opener(NoRedirect()).open(request, timeout=30) as stream:
        if stream.status != 200 or stream.headers.get_content_type() not in mime:
            raise ValueError('Package metadata/wheel envelope refused')
        raw = stream.read(cap+1)
    if len(raw) > cap:
        raise ValueError('Package byte cap')
    return raw


def catalog():
    from packaging.specifiers import SpecifierSet
    gates()
    output = OUT/'colo1-parser-package-pins-v1.json'
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for name in ('rdata', 'xarray'):
        url = 'https://pypi.org/pypi/'+name+'/json'
        raw = fetch(url, 4*1024**2, {'application/json'})
        value = json.loads(raw, object_pairs_hook=unique_object)
        info = value['info']
        candidates = [f for f in value['urls'] if f['filename'].endswith('py3-none-any.whl') and not f.get('yanked')]
        if len(candidates) != 1:
            raise ValueError('Unique stable pure-Python wheel required')
        wheel = candidates[0]
        if not SpecifierSet(wheel.get('requires_python') or info.get('requires_python') or '').contains('.'.join(map(str, sys.version_info[:3]))):
            raise ValueError('Parser wheel Python version incompatible')
        parsed = urllib.parse.urlsplit(wheel['url'])
        if parsed.scheme != 'https' or parsed.hostname != 'files.pythonhosted.org' or parsed.query or wheel['size'] > 16*1024**2:
            raise ValueError('Unexpected wheel origin/size')
        rows.append(dict(name=name, version=info['version'], filename=wheel['filename'], url=wheel['url'],
                         bytes=wheel['size'], sha256=wheel['digests']['sha256'], requires_dist=info.get('requires_dist') or [],
                         requires_python=wheel.get('requires_python'), license_expression=info.get('license_expression'),
                         catalog_url=url, catalog_raw_sha256=hashlib.sha256(raw).hexdigest()))
    write(output, dict(schema='opal-colo1-parser-package-pins-v1', packages=rows, scientific_data_requests=0))
    gates()
    print(json.dumps([dict(name=r['name'], version=r['version'], bytes=r['bytes']) for r in rows]))


def install():
    from packaging.requirements import Requirement
    gates()
    pins_path = OUT/'colo1-parser-package-pins-v1.json'
    binding = json.loads((OUT/'colo1-parser-package-binding-v1.json').read_bytes())
    if sha(pins_path) != binding['pins_sha256'] or sha(__file__) != binding['setup_code_sha256']:
        raise ValueError('Package pins drift')
    pins = json.loads(pins_path.read_bytes())['packages']
    selected = {p['name']:p['version'] for p in pins}
    reused = {}
    LOCAL.mkdir(exist_ok=False)
    wheels = LOCAL/'wheels'
    wheels.mkdir()
    paths = []
    for pin in pins:
        raw = fetch(pin['url'], 16*1024**2, {'application/octet-stream', 'application/zip', 'binary/octet-stream'})
        if len(raw) != pin['bytes'] or hashlib.sha256(raw).hexdigest() != pin['sha256']:
            raise ValueError('Pinned wheel identity mismatch')
        path = wheels/pin['filename']
        with path.open('xb') as stream:
            stream.write(raw)
        paths.append(str(path))
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.endswith('.dist-info/METADATA')]
            if len(names) != 1:
                raise ValueError('Wheel metadata identity')
            metadata = BytesParser().parsebytes(z.read(names[0]))
            if metadata['Name'].lower() != pin['name'] or metadata['Version'] != pin['version']:
                raise ValueError('Wheel package identity mismatch')
            for raw_req in metadata.get_all('Requires-Dist', []):
                req = Requirement(raw_req)
                if req.marker and not req.marker.evaluate({'extra':''}):
                    continue
                try:
                    have = selected.get(req.name.lower()) or version(req.name)
                except PackageNotFoundError as error:
                    raise ValueError('Unregistered parser dependency: '+req.name) from error
                if not req.specifier.contains(have):
                    raise ValueError('Incompatible parser dependency: '+str(req))
                if req.name.lower() not in selected:
                    reused[req.name] = have
    target = LOCAL/'installed'
    subprocess.run([sys.executable, '-B', '-m', 'pip', 'install', '--no-deps', '--no-compile', '--target', str(target), *paths], check=True)
    files = {str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted(target.rglob('*')) if p.is_file()}
    write(OUT/'colo1-parser-install-manifest-v1.json', dict(schema='opal-colo1-parser-install-v1', pins_sha256=sha(pins_path),
          target=str(target), packages=selected, reused_host_dependencies=reused, installed_sha256=files, venv_modified=False))
    gates()
    print(json.dumps(dict(installed=selected, host_dependencies=reused, hashed_files=len(files))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['catalog', 'install'])
    args = parser.parse_args()
    catalog() if args.stage == 'catalog' else install()
