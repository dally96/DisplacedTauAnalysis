import os, sys, pdb
import argparse, importlib
import pickle

from coffea.dataset_tools import (
    apply_to_fileset,
    max_chunks,
    preprocess,
)
from dask import config as cfg
cfg.set({'distributed.scheduler.worker-ttl': None}) # Check if this solves some dask issues
from uproot.exceptions import KeyInFileError

import time, logging
# from distributed import Client
from dask.distributed import Client, wait, progress, LocalCluster
from itertools import islice
# from dask_lxplus import CernCluster


parser = argparse.ArgumentParser(description="")
parser.add_argument(
	"--sample",
	choices=['QCD','DY', 'signal', 'Wto2Q', 'WtoLNu', 'TT'],
	required=True,
	help='Specify the sample you want to process')
parser.add_argument(
	"--subsample",
	nargs='*',
	default='all',
	required=False,
	help='Specify the exact sample you want to process')
parser.add_argument(
	"--nfiles",
	default='-1',
	required=False,
	help='Specify the number of input files to process')
parser.add_argument(
	"--skim",
	default='',
	required=False,
	help='Specify, if working on the skimmed samples, the skim name (name of the folder inside samples)')
args = parser.parse_args()


def take(n, iterable):
    """Return the first n items of the iterable as a list."""
    return list(islice(iterable, n))


outdir_p = ''
outdir_s = ''
if args.skim != '':
    outdir_p = args.skim + '.'
    outdir_s = args.skim + '/'
  
samples = {
    "Wto2Q"  : f"samples.{outdir_p}fileset_WTo2Q",
    "WtoLNu" : f"samples.{outdir_p}fileset_WToLNu",
    "QCD"    : f"samples.{outdir_p}fileset_QCD",
    "DY"     : f"samples.{outdir_p}fileset_DY",
    "signal" : f"samples.{outdir_p}fileset_signal",
    "TT"     : f"samples.{outdir_p}fileset_TT",
}

module = importlib.import_module(samples[args.sample])
all_fileset = module.fileset  #['Stau_100_0p1mm'] 


if args.subsample == 'all':
    fileset = all_fileset
else:  
    fileset = {k: all_fileset[k] for k in args.subsample}

nfiles = int(args.nfiles)
if nfiles != -1:
    for k in fileset.keys():
        if nfiles < len(fileset[k]['files']):
            fileset[k]['files'] = take(nfiles, fileset[k]['files'])

print("Will process {} files from the following samples:".format(nfiles), fileset.keys())

## first element of value is step_size, second is files_per_barch
pars_per_sample = {
    "Wto2Q"  : [20_000, 100], 
    "WtoLNu" : [20_000, 100],  
    "QCD"    : [20_000, 1],  
    "DY"     : [10_000, 1000],  
    "signal" : [20_000, 1],  
    "TT"     : [20_000, 1000],  
    "data"   : [20_000, 100] 
}



if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)    
    tic = time.time()

#     n_port = 8786
#     import socket
#     cluster = CernCluster(
#             cores=1,
#             memory='3000MB',
#             disk='1000MB',
#             death_timeout = '60',
#             lcg = True,
#             nanny = False,
#             container_runtime = "none",
#             log_directory = "/eos/user/f/fiorendi/condor/log",
#             scheduler_options={
#                 'port': n_port,
#                 'host': socket.gethostname(),
#                 },
#             job_extra={
#                 '+JobFlavour': '"longlunch"',
#                 },
#             extra = ['--worker-port 10000:10100']
#             )
#     
#     # minimum > 0: https://github.com/CoffeaTeam/coffea/issues/465
#     cluster.adapt(minimum=1, maximum=1000)
 
    cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    client = Client(cluster)

    dataset_runnable, dataset_updated = preprocess(
       fileset,
       align_clusters=False,
       step_size=pars_per_sample[args.sample][0],
       files_per_batch=pars_per_sample[args.sample][1],
       skip_bad_files=True,
       save_form=False,
       file_exceptions=(OSError, KeyInFileError),
       allow_empty_datasets=False,
    )
    with open(f"samples/{outdir_s}{args.sample}_preprocessed.pkl", "wb") as f:
       pickle.dump(dataset_runnable, f)

    elapsed = time.time() - tic 
    print(f"Preprocessing datasets finished in {elapsed:.1f}s") 

    client.shutdown()
    cluster.close()
