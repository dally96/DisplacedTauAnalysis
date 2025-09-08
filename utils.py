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
