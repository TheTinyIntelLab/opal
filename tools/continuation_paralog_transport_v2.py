"""One diagnostic GET, transport headers only, with no body or redirects."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/research-continuation'
URL = 'https://figshare.com/articles/dataset/A_compendium_of_synthetic_lethal_gene_pairs_defined_by_extensive_combinatorial_pan-cancer_CRISPR_screening/25954027/4'
FIELDS = ('Content-Type', 'Content-Length', 'Server', 'Retry-After', 'X-Cache',
          'X-Amzn-Waf-Action', 'Cf-Mitigated')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def envelope(response):
    """Inspect only whitelisted HTTP transport fields; never consume a body."""
    fields = {}
    for name in FIELDS:
        value = response.headers.get(name)
        if value is not None:
            fields[name] = dict(value=value[:256], truncated=len(value) > 256)
    raw_mime = response.headers.get('Content-Type')
    mime = raw_mime.split(';', 1)[0].strip().lower() if raw_mime else None
    status = int(response.status)
    location = response.headers.get('Location')
    parsed = urllib.parse.urlsplit(location) if location else None
    return dict(status=status, mime=mime, transport_fields=fields,
                location=dict(host=parsed.hostname, path=parsed.path,
                              query_redacted=bool(parsed.query)) if parsed else None,
                matches_v1_transport_gate=status == 200 and mime == 'text/html',
                body_bytes_read=0, redirects_followed=0)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def main():
    freeze_path = OUT / 'paralog-transport-freeze-v2.json'
    binding_path = OUT / 'paralog-transport-code-binding-v2.json'
    freeze = json.loads(freeze_path.read_text(encoding='utf8'))
    binding = json.loads(binding_path.read_text(encoding='utf8'))
    if binding['freeze_sha256'] != digest(freeze_path):
        raise ValueError('Transport freeze drift')
    for name, expected in binding['code_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Bound diagnostic code drift')
    for name, expected in freeze['sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Bound diagnostic input drift')
    if shutil.disk_usage(ROOT).free < 2**30:
        raise RuntimeError('Unchanged disk reserve')
    output = OUT / 'paralog-transport-receipt-v2.json'
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    result = dict(url=URL, method='GET', observed_utc=datetime.now(timezone.utc).isoformat(),
                  freeze_sha256=digest(freeze_path), tool_sha256=digest(__file__),
                  current_observation_not_reconstruction_of_v1=True,
                  source_qualified=False, source_assets_acquired=0,
                  response_tables_or_headers_opened=False, fits=0, scores=0)
    response = None
    try:
        request = urllib.request.Request(URL, method='GET',
            headers={'User-Agent': 'Opal-response-free-metadata-audit/1.0'})
        try:
            response = urllib.request.build_opener(NoRedirect).open(request, timeout=30)
        except urllib.error.HTTPError as error:
            response = error
        result.update(envelope(response))
    except (urllib.error.URLError, TimeoutError, ValueError) as error:
        result.update(error=type(error).__name__ + ': ' + str(error), body_bytes_read=0)
    finally:
        if response is not None:
            response.close()
    result['elapsed_seconds'] = time.perf_counter() - started
    with output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('status', 'mime', 'transport_fields',
                     'body_bytes_read', 'matches_v1_transport_gate', 'error') if k in result}))


if __name__ == '__main__':
    main()
