import awkward as ak
import uproot
import os, argparse, importlib, pdb
from itertools import islice


def take(n, iterable):
    """Return the first n items of the iterable as a list."""
    return list(islice(iterable, n))

## restrict to n files
def process_n_files(nfiles, fileset):
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



### columns that uproot can write out
def uproot_writeable_selected(events, include_all, include_prefixes, include_postfixes):
    '''
      - keep all branches starting with any prefix in include_all.
      - for branches starting with include_prefixes, keep only fields in include_postfixes.
    '''
    out = {}

    for bname in events.fields:
        keep_branch = any(bname.startswith(p) for p in include_all)
        reduced_branch = any(bname.startswith(p) for p in include_prefixes)

        # handle structured branches (records with subfields)
        if events[bname].fields:
            fields = {}
            for n in events[bname].fields:
                if is_rootcompat(events[bname][n]):
#                     print (bname, n)
                    if keep_branch or (reduced_branch and n in include_postfixes):
                        fields[n] = ak.to_packed(ak.without_parameters(events[bname][n]))

            if fields:  # avoid ak.zip({})
                out[bname] = ak.zip(fields)

        # handle flat branches
        else:
            if keep_branch or reduced_branch:
                out[bname] = ak.to_packed(ak.without_parameters(events[bname]))

    return out


## exclude not used
exclude_prefixes = ['Flag', 'JetSVs', 'GenJetAK8_', 'SubJet', 
                    'Photon', 'TrigObj', 'TrkMET', 'HLT',
                    'Puppi', 'OtherPV', 'GenJetCands',
                    'FsrPhoton', ''
                    ## tmp
                    'diele', 'LHE', 'dimuon', 'Rho', 'JetPFCands', 'GenJet', 'GenCands', 
                    'Electron'
                    ]


