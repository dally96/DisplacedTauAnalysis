import awkward as ak
import uproot

import os, argparse, importlib, pdb

from itertools import islice
def take(n, iterable):
    """Return the first n items of the iterable as a list."""
    return list(islice(iterable, n))

def process_n_files(nfiles, fileset):
## restrict to n files
    if nfiles != -1:
        for k in fileset.keys():
            if nfiles < len(fileset[k]['files']):
                fileset[k]['files'] = take(nfiles, fileset[k]['files'])
                
    return fileset


def is_rootcompat(a):
    """Is it a flat or 1-d jagged array?"""
    t = ak.type(a)
    if isinstance(t, ak.types.ArrayType):
        if isinstance(t.content, ak.types.NumpyType):
            return True
        if isinstance(t.content, ak.types.ListType) and isinstance(t.content.content, ak.types.NumpyType):
            return True
    return False

## save specific HLT branches
def is_good_hlt(name, good_hlts):

    if not name.startswith("HLT"):
        return False

    parts = name.split(".")
    if len(parts) == 2:
        name = parts[1]
        return name in good_hlts
    return False
   

def is_included(name, include_prefixes):
        return any(name.startswith(prefix) for prefix in include_prefixes)


## exclude not used
exclude_prefixes = ['Flag', 'JetSVs', 'GenJetAK8_', 'SubJet', 
                    'Photon', 'TrigObj', 'TrkMET', 'HLT',
                    'Puppi', 'OtherPV', 'GenJetCands',
                    'FsrPhoton', ''
                    ## tmp
                    'diele', 'LHE', 'dimuon', 'Rho', 'JetPFCands', 'GenJet', 'GenCands', 
                    'Electron'
                    ]


