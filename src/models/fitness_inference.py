"""Safe named-gene inference for the validation-selected growth references."""
from pathlib import Path
import json
import numpy as np
from .fitness import FitnessCompletion, design
from ..artifacts import sha256


class FitnessPredictor:
    def __init__(self,path):
        self.path=Path(path)
        self.manifest=json.loads((self.path/'manifest.json').read_text(encoding='utf-8'))
        for name,digest in self.manifest['files'].items():
            if sha256(self.path/name)!=digest:raise ValueError('Model asset hash mismatch')
        with np.load(self.path/'reference.npz',allow_pickle=False) as z:
            self.arrays={k:z[k].copy() for k in z.files}
        self.genes=[str(g) for g in self.arrays['genes']]
        self.index={g:i for i,g in enumerate(self.genes)}
        self.contexts=self.manifest['contexts']
        self.models={c:FitnessCompletion.load(self.path/entry['checkpoint']) for c,entry in self.contexts.items()}

    def predict(self,pairs,context):
        if context not in self.contexts:raise ValueError('Unsupported growth context')
        try:ids=np.asarray([[self.index[str(a)],self.index[str(b)]] for a,b in pairs],dtype=int)
        except (KeyError,TypeError,ValueError) as exc:raise ValueError('Unknown or malformed gene pair') from exc
        entry=self.contexts[context];c=entry['context_index'];kind=entry['reference_kind']
        # Completion validates gene support, bounds and distinct identities.
        residual=self.models[context].predict(ids)[c]
        singles=self.arrays['singles'][c]
        if kind=='zero':base=np.zeros(len(ids))
        elif kind=='additive':base=singles[ids].sum(1)
        elif kind=='minimum':base=singles[ids].min(1)
        else:
            prefix=f'c{c}_'
            x=design(singles,ids,kind=='node')
            base=singles[ids].sum(1)+self.arrays[prefix+'mean']
            base=base+(x-self.arrays[prefix+'center'])/self.arrays[prefix+'scale']@self.arrays[prefix+'coefficient']
        return base+residual
