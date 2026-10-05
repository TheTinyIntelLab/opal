"""Preserve development-only shared-reference bootstrap draws and covariance."""
from pathlib import Path
import argparse,sys
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from opal.artifacts import load_npz,read_json,write_json,sha256
from tools.prepare_architecture import joint_reference_draws


def prepare(name,draws=16,seed=90317):
    folder=ROOT/'data/architecture_resources'/('joint-'+name)
    if (folder/'manifest.json').exists():
        manifest=read_json(folder/'manifest.json')
        if manifest['seed']!=seed or manifest['draws']!=draws:raise ValueError('Joint sampling protocol differs; preserve the resource and use a new name')
        if manifest['profiles_sha256']!=sha256(ROOT/'data/prepared'/name/'profiles.npz') or manifest['draws_sha256']!=sha256(folder/'draws.npz'):raise ValueError('Joint resource changed')
        return
    if folder.exists():raise FileExistsError('Incomplete joint resource; inspect before resuming')
    folder.mkdir()
    data=load_npz(ROOT/'data/prepared'/name/'profiles.npz')
    singles,doubles,residual=joint_reference_draws(name,data,draws,seed,True)
    centered=residual.astype(np.float64)-residual.mean(0,dtype=np.float64)
    covariance=np.einsum('dpg,dqg->pq',centered,centered,optimize=True)/((draws-1)*residual.shape[2])
    np.testing.assert_allclose(np.diag(covariance),residual.var(0,ddof=1).mean(1),rtol=2e-5)
    base=load_npz(ROOT/'data/architecture_resources'/name/'resources.npz')
    if draws==16 and seed==90317:np.testing.assert_allclose(residual.var(0,ddof=1),base['residual_variance'],rtol=2e-5,atol=1e-7)
    np.savez_compressed(folder/'draws.npz',single_draws=singles,double_draws=doubles,residual_covariance=covariance,pair_indices=data['pair_indices'],genes=data['genes'])
    correlation=covariance/np.sqrt(np.maximum(np.outer(np.diag(covariance),np.diag(covariance)),1e-20))
    write_json(folder/'manifest.json',{'dataset':name,'draws':draws,'seed':seed,'draws_sha256':sha256(folder/'draws.npz'),
        'profiles_sha256':sha256(ROOT/'data/prepared'/name/'profiles.npz'),'covariance':'sample covariance across bootstrap draws, averaged over frozen output genes',
        'evaluator_access':False,'scope':'technical sampling only; frozen QC, gene universe and normalizer; shared single/control draws',
        'minimum_covariance_eigenvalue':float(np.linalg.eigvalsh(covariance).min()),'maximum_absolute_correlation':float(np.max(np.abs(correlation-np.eye(len(covariance)))))})
    print(name,'joint draws ready',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['norman-log2','wessels-log2','wessels-expanded-protected-log2']);args=p.parse_args()
    with threadpool_limits(limits=1):
        for name in args.datasets:prepare(name)


if __name__=='__main__':main()
