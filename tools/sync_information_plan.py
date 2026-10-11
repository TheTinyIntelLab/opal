"""Update active statuses, leaving the historical appendix byte-for-byte intact."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from opal.artifacts import read_json,write_json,sha256


def run():
    root=ROOT/'PLAN.md';mirror=ROOT/'docs/next_cycle/model_series_v2_plan.md'
    if root.read_bytes()!=mirror.read_bytes():raise ValueError('Resolve existing user plan/mirror divergence before editing')
    text=root.read_text();marker='## Historical appendix: previous model-series plan, unchanged content'
    active,appendix=text.split(marker,1)
    status=read_json(ROOT/'results/opal_information_v1/task_status.json')
    delivery_authorized=(ROOT/'results/opal_information_v1/verification/delivery_authorization.json').exists()
    for task in status['tasks']:
        if task['status'].startswith('complete_'):
            active=active.replace('- [ ] '+task['id']+':','- [x] '+task['id']+':')
    heading='## Current implementation evidence: opal_information_v1'
    if heading in active:active=active.split(heading)[0].rstrip()+'\n\n'
    delivery=' The subsequent owner instruction authorizes builds, small lowercase commits and push; see `verification/delivery_authorization.json`. Existing on-push CI must be inspected on the delivered revision. Raw assets, unpublished weights, registration and scientific publication remain outside that authorization.' if delivery_authorized else ''
    lines=[heading,'','The owner invoked the implementation handoff on 11 October 2026. The following statuses refer to actual local execution; scientific eligibility and blocked external work remain separate. The historical appendix below is unchanged. No commits, uploads or remote dispatch were authorized during the initial bounded experiment.'+delivery,'',
        '| Task | Status | Evidence and next action |','| --- | --- | --- |']
    for task in status['tasks']:
        location=task['outputs']
        if not location.startswith(('src/','docs/')):location='results/opal_information_v1/'+location
        lines.append('| '+task['id']+' | `'+task['status']+'` | '+task['scientific_verdict']+'. Evidence: `'+location+'`. Next: '+task['next_action']+'. |')
    lines+=['','Evidence repair register: E01/E02/E03/E04 remain `blocked_external` for native contracts/assets/environment; E05 has new measured adaptation profiles but historical native RAM stays `historically_unknown`; E06 has a verified descriptive efficiency table with native eligibility still blocked; E07 remains `historically_unknown`, and current measured CPU is explicitly a lower bound; E08 remains `blocked_external` pending authorized exact-revision remote CI. E09/E10/E11/E12 remain `blocked_external` for independent cultures, matched provenance, custody/registration and redistribution terms. E13/E14 remain `conditional_not_started`. E15 is `complete_verified` for this bounded local cycle, while larger assets/environments need additional approved storage.','',
        'The frozen 96-fold information study fails its administrative gate. TNF-positive partner variance MSE 0.08256839 versus Base 0.08275970 is a 0.231163% gain, but matched mean/capacity controls are better. TNF-negative partner nested support is unavailable and every head uses Base fallback, leaving that information question inconclusive. No further branch, external confirmation or variance deployment refit was executed. The full local Python3.14 suite passes 282 tests; new training/replay, installed verification and manuscript evidence are in the cycle verification/receipt records.','',
        'See `docs/next_cycle/opal_information_report.md` and `results/opal_information_v1/task_status.json`, `resource_ledger.json`, `report/claims.json`, `report/efficiency.json`, and `report/dependencies.json` for exact counts, measurements, roles, unknowns and remaining inputs. The follow-up `verification/completion_audit.json` verifies all96 prediction hashes and unchanged scientific code/config/protocol, with no pending_local task inside the freeze. Broader duplicate/DEG/pathway diagnostics were not executed or retrospectively added; complete new CPU certification remains unavailable. The previous PDF is preserved under the cycle local artifact root; the current manuscript has a separately verified development addendum.','']
    if delivery_authorized:
        lines=[line.replace('E08 remains `blocked_external` pending authorized exact-revision remote CI','E08 remains `blocked_external` until the owner-authorized delivery revision passes every remote CI cell') for line in lines]
    data=(active+'\n'.join(lines)+'\n'+marker+appendix).encode('utf-8')
    root.write_bytes(data);mirror.write_bytes(data)
    write_json(ROOT/'results/opal_information_v1/verification/plan_sync.json',{'byte_identical':root.read_bytes()==mirror.read_bytes(),'sha256':sha256(root),
        'historical_appendix_unchanged':appendix==text.split(marker,1)[1],'scope':'Active statuses only, no historical task reactivation'})
    print('Root/mirror synchronized; historical appendix unchanged')


if __name__=='__main__':run()
