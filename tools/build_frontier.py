"""One-command architecture research build with immutable completed evidence."""
from pathlib import Path
import argparse,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from opal.artifacts import write_json,read_json,sha256


def run(*args):
    subprocess.run([sys.executable,*args],cwd=ROOT,check=True)


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--skip-tabular',action='store_true')
    parser.add_argument('--skip-transfer',action='store_true')
    parser.add_argument('--skip-paper',action='store_true')
    args=parser.parse_args(argv)
    status={'scope':'expanded exploratory architecture research; preserves historical tests', 'stages':[], 'status':'running'}
    path=ROOT/'results/architecture/pipeline_status.json'
    write_json(path,status)
    def stage(label,artifact,*command):
        artifact=ROOT/artifact
        if not artifact.exists():run(*command)
        if not artifact.exists():raise ValueError('Stage did not create expected artifact: '+label)
        status['stages'].append({'stage':label,'artifact':str(artifact.relative_to(ROOT)),'sha256':sha256(artifact)})
        write_json(path,status)
    try:
        stage('public pathways','data/architecture_resources/hallmark.txt','tools/download_architecture_resources.py')
        run('tools/prepare_architecture.py','--expanded','--protected','--datasets','norman-log2','wessels-log2','wessels-expanded-log2','wessels-expanded-protected-log2')
        stage('nested architecture ablations','results/architecture/v1/summary.json','main.py','architecture')
        stage('protected architecture ablations','results/architecture/protected-v1/summary.json','tools/study_architecture.py','--datasets','wessels-expanded-protected-log2','--output','results/architecture/protected-v1')
        stage('GP and retrieval','results/architecture/experts-v1/summary.json','tools/study_experts.py')
        if not args.skip_tabular:
            stage('frozen tabular comparator','results/architecture/tabular-v1/summary.json','tools/study_tabular.py')
        run('tools/download_architecture_resources.py','--presage',*(['--replogle'] if not args.skip_transfer else []))
        stage('verified external archive','data/architecture_resources/presage_cache.tar.gz','tools/download_architecture_resources.py','--presage')
        if not args.skip_transfer:
            stage('verified independent screen','data/architecture_resources/ReplogleWeissman2022_K562_essential.h5ad','tools/download_architecture_resources.py','--replogle')
            stage('independent transfer preparation','data/architecture_resources/enriched-wessels-expanded-log2/manifest.json','tools/prepare_transfer.py','--datasets','norman-log2','wessels-log2','wessels-expanded-log2','wessels-expanded-protected-log2')
            stage('transfer and linear-loss controls','results/architecture/transfer-v3/summary.json','tools/study_experts.py','--transfer','--output','results/architecture/transfer-v3','--datasets','norman-log2','wessels-log2','wessels-expanded-log2','wessels-expanded-protected-log2')
        if not args.skip_transfer:
            stage('bounded linear extrapolation control','results/architecture/linear-bounded-v1/summary.json','tools/study_experts.py','--transfer','--bounded-linear','--output','results/architecture/linear-bounded-v1','--datasets','norman-log2','wessels-log2','wessels-expanded-log2','wessels-expanded-protected-log2')
        stage('semantic prior preparation','data/architecture_resources/semantic-only-wessels-expanded-protected-log2/manifest.json','tools/prepare_transfer.py','--semantic-only','--datasets','norman-log2','wessels-log2','wessels-expanded-log2','wessels-expanded-protected-log2')
        stage('matched prior controls','results/architecture/prior-kernels-v1/summary.json','tools/study_prior_kernels.py','--prefix','semantic-only-','--datasets','norman-log2','wessels-log2','wessels-expanded-log2','wessels-expanded-protected-log2')
        for version,seed in [(1,20261005),(2,20261006),(3,20261007)]:
            stage('measurement weighting seed '+str(seed),f'results/architecture/noise-kernels-v{version}/summary.json','tools/study_noise_kernels.py','--seed',str(seed),'--output',f'results/architecture/noise-kernels-v{version}')
        stage('common-condition coverage','results/architecture/common-coverage-held-v1/results.json','tools/study_coverage.py','--partner-policy','held','--output','results/architecture/common-coverage-held-v1')
        for version,seed in [(2,20261006),(3,20261007)]:
            stage('common-condition coverage seed '+str(seed),f'results/architecture/common-coverage-held-v{version}/results.json','tools/study_coverage.py','--seed',str(seed),'--partner-policy','held','--output',f'results/architecture/common-coverage-held-v{version}')
        run('main.py','ceiling','--skip-paper','--skip-tests')
        run('-m','pytest','tests','-q','-p','no:cacheprovider','--basetemp','.pytest-final-architecture-tmp')
        run('tools/report_architecture.py',*(['--allow-partial'] if args.skip_transfer or args.skip_tabular else []))
        if not args.skip_paper:run('tools/build_paper.py')
        status['status']='partial' if args.skip_transfer or args.skip_tabular else 'complete'
        status['interpretation']='Completed registered exploratory build; no frontier or independent-confirmation claim implied'
        write_json(path,status)
    except Exception as error:
        status['status']='failed';status['error']=str(error);write_json(path,status);raise


if __name__=='__main__':main()
