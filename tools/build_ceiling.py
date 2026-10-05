"""Reproduce the joint-noise and data-efficiency research extension."""
from pathlib import Path
import argparse,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from opal.artifacts import write_json,sha256


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--skip-paper',action='store_true');p.add_argument('--skip-tests',action='store_true');args=p.parse_args(argv)
    status={'status':'running','stages':[],'scope':'exploratory development extension; no evaluator access'}
    path=ROOT/'results/architecture/ceiling_pipeline_status.json'
    write_json(path,status)
    def run(label,*command):
        subprocess.run([sys.executable,*command],cwd=ROOT,check=True)
        status['stages'].append(label);write_json(path,status)
    try:
        run('joint technical bootstrap resources','tools/prepare_joint_uncertainty.py')
        for v,seed in [(1,20261008),(2,20261009),(3,20261010)]:
            run('covariance seed '+str(seed),'tools/study_joint_uncertainty.py','--seed',str(seed),'--output',f'results/architecture/joint-uncertainty-v{v}')
        run('subset-retuned curves and isolated radii','tools/study_retuned_curves.py')
        if not args.skip_tests:run('tests','-m','pytest','tests','-q','-p','no:cacheprovider','--basetemp','.pytest-joint-tmp')
        run('verified evidence','tools/report_ceiling.py')
        if not args.skip_paper:run('paper','tools/build_paper.py')
        status.update(status='complete',report_sha256=sha256(ROOT/'results/architecture/ceiling_report.json'),tests_skipped=args.skip_tests,paper_skipped=args.skip_paper)
        write_json(path,status)
    except Exception as error:
        status.update(status='failed',error=str(error));write_json(path,status);raise


if __name__=='__main__':main()
