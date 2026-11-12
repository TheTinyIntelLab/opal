"""Portable archive policy shared by packager, validator and membership tests."""
from pathlib import PurePosixPath

PRIVATE_NAMES={'plan.md','outreach.md','outreach_register.json','context.md','roadmap.md','opal_next_master_plan.md','build_evidence_demo.py',
               'release_cleanup_plan.md','release_deletions.json','opal_repository_audit.md'}
DENIED_PARTS={'web','vendor','external','__pycache__','raw','prepared','sealed','custodian','source_snapshots'}
COMPACT_CSV_NAMES={'comparisons.csv','coverage.csv','summary.csv','scores.csv'}


def safe_archive_name(name):
    if not isinstance(name,str) or not name or '\\' in name or ':' in name or '\x00' in name:
        return False
    path=PurePosixPath(name)
    return not path.is_absolute() and not {'..','.'}&set(name.split('/'))


def public_name(name):
    if not safe_archive_name(name):return False
    path=PurePosixPath(name)
    return path.name.casefold() not in PRIVATE_NAMES and not DENIED_PARTS & {p.casefold() for p in path.parts}


def allowed_source(name):
    if not public_name(name):return False
    if name == 'docs/archive/pre-v1-documentation.zip':return True
    path=PurePosixPath(name)
    if tuple(p.casefold() for p in path.parts[:3]) == ('docs', 'frontier', 'historical'):
        return False
    if len(path.parts)==1:
        return path.suffix in {'.md','.toml','.cff','.in','.py'} or name in {'LICENSE','.gitignore','.gitattributes','pytest.ini'} or (path.name.startswith('requirements') and path.suffix=='.txt')
    return path.parts[0] in {'src','tools','tests','configs','docs','paper','images','.github'} and path.suffix in {
        '.py','.json','.yaml','.yml','.md','.tex','.bib','.sty','.pdf','.txt','.png','.svg'}


def allowed_evidence(name):
    if not public_name(name):return False
    if name == 'docs/archive/pre-v1-documentation.zip':return True
    path=PurePosixPath(name)
    return path.parts[0] in {'results','data','paper'} and (
        path.suffix in {'.json','.md','.tex','.pdf'} or path.name in COMPACT_CSV_NAMES) and not any(
        p.casefold().startswith(('demo-','local-release-','replicated-demo')) for p in path.parts)
