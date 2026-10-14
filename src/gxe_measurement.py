"""Literal chemical assignments and bounded MatrixMarket measurement streaming."""
import gzip
import re
import numpy as np


def assignment(label):
    parts=label.split('_')
    if len(parts)!=5 or not parts[0].startswith('plate') or parts[4] not in ('rep1','rep2'):
        raise ValueError('Malformed literal hash assignment')
    t,d=float(parts[1]),float(parts[3])
    if not np.isfinite([t,d]).all() or min(t,d)<0:
        raise ValueError('Physical nonnegative doses required')
    return dict(plate=parts[0],trametinib_dose=t,drug=parts[2],drug_dose=d,replicate=parts[4])


def rt_context(cell):
    match=re.search(r'_RT_BC_(\d+)_',cell)
    if not match or not 1<=int(match[1])<=96:raise ValueError('Unqualified RT cell-origin barcode')
    return ('A172','T98G','U87MG')[((int(match[1])-1)%12)//4]


def matrix_blocks(path,check,chunk_lines=250000,expected_shape=None):
    """Read exact coordinate counts; never materialize the full count matrix."""
    with gzip.open(path,'rt') as stream:
        header=stream.readline().strip()
        pending=None
        if header in ('%%MatrixMarket matrix coordinate integer general','%%MatrixMarket matrix coordinate real general'):
            line=stream.readline()
            while line.startswith('%'):line=stream.readline()
            rows,cols,entries=map(int,line.split())
        elif expected_shape is not None:
            rows,cols=expected_shape;entries=None;pending=header+'\n'
        else:raise ValueError('Headerless coordinates require verified annotation axes')
        yield (rows,cols,entries)
        seen=0
        while True:
            lines=[pending] if pending is not None else [];pending=None
            for _ in range(chunk_lines):
                line=stream.readline()
                if not line:break
                lines.append(line)
            if not lines:break
            block=np.fromstring(''.join(lines),sep=' ')
            if block.size!=len(lines)*3:raise ValueError('Malformed coordinate payload')
            block=block.reshape(-1,3)
            if not np.isfinite(block).all() or np.any(block!=np.floor(block)) or np.any(block[:,2]<0):
                raise ValueError('Nonnegative integer UMI counts required')
            row,col=block[:,0].astype(np.int64)-1,block[:,1].astype(np.int64)-1
            if np.any(row<0) or np.any(row>=rows) or np.any(col<0) or np.any(col>=cols):
                raise ValueError('Count coordinate bounds mismatch')
            seen+=len(block);check();yield row,col,block[:,2]
        if entries is not None and seen!=entries:raise ValueError('Declared coordinate population mismatch')


def matched_keys(context,record):
    prefix=(context,record['plate'],record['replicate'])
    # Condition grouping is literal source assignment, not culture replication.
    return [prefix+('vehicle',0.,0.),prefix+(record['drug'],record['drug_dose'],0.),
            prefix+('vehicle',0.,record['trametinib_dose'])]


def key(context,record):
    return (context,record['plate'],record['replicate'],record['drug'],record['drug_dose'],record['trametinib_dose'])


def reference_variance(identity_strata):
    """Equal single-identity weights after equally averaging matched strata."""
    if len(identity_strata)<2:raise ValueError('Reference-only panel variance unsupported')
    values=np.stack([np.mean(identity_strata[i],axis=0) for i in sorted(identity_strata)])
    return np.var(values,axis=0)
