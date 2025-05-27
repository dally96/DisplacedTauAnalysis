import sys
import os
import fsspec_xrootd
from  fsspec_xrootd import XRootDFileSystem
import uproot
import uproot.exceptions
from uproot.exceptions import KeyInFileError
import pickle

import argparse
import numpy as np
from coffea import processor
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema, NanoAODSchema
from coffea.dataset_tools import (
    apply_to_fileset,
    max_chunks,
    preprocess,
)
from coffea.lumi_tools import LumiData, LumiList, LumiMask
import awkward as ak
import dask
from dask import config as cfg
cfg.set({'distributed.scheduler.worker-ttl': None}) # Check if this solves some dask issues
import dask_awkward as dak
from dask_jobqueue import HTCondorCluster
from dask.distributed import Client, wait, progress, LocalCluster
import fsspec_xrootd
from  fsspec_xrootd import XRootDFileSystem

import time
from distributed import Client
from lpcjobqueue import LPCCondorCluster

import warnings
warnings.filterwarnings("ignore", module="coffea") # Suppress annoying deprecation warnings for coffea vector, c.f. https://github.com/CoffeaTeam/coffea/blob/master/src/coffea/nanoevents/methods/candidate.py
import logging

from fileset import *
from skimmed_fileset import *
from lower_lifetime_fileset import *
from W_fileset import *
from data_fileset import *


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    #XRootDFileSystem(hostid = "root://cmsxrootd.fnal.gov/", filehandle_cache_size = 250)
    tic = time.time()
    cluster = LPCCondorCluster(ship_env=True, transfer_input_files='/uscms/home/dally/x509up_u57864')
    # minimum > 0: https://github.com/CoffeaTeam/coffea/issues/465
    cluster.adapt(minimum=1, maximum=1000)
    client = Client(cluster)

    #dataset_runnable, dataset_updated = preprocess(
    #    fileset,
    #    align_clusters=False,
    #    step_size=100_000,
    #    files_per_batch=1000,
    #    skip_bad_files=True,
    #    save_form=False,
    #    file_exceptions=(OSError, KeyInFileError),
    #    allow_empty_datasets=False,
    #)

    #with open("preprocessed_fileset.pkl", "wb") as f:
    #    pickle.dump(dataset_runnable, f)

    #skimmed_dataset_runnable, skimmed_dataset_updated = preprocess(
    #    skimmed_fileset,
    #    align_clusters=False,
    #    step_size=10_000,
    #    files_per_batch=100,
    #    skip_bad_files=True,
    #    save_form=False,
    #    file_exceptions=(OSError, KeyInFileError),
    #    allow_empty_datasets=False,
    #)
    #
    #with open("skimmed_preprocessed_fileset.pkl", "wb") as f:
    #    pickle.dump(skimmed_dataset_runnable, f)

    #lower_lifetime_dataset_runnable, lower_lifetime_dataset_updated = preprocess(
    #    lower_lifetime_fileset,
    #    align_clusters=False,
    #    step_size=10_000,
    #    files_per_batch=100,
    #    skip_bad_files=True,
    #    save_form=False,
    #    file_exceptions=(OSError, KeyInFileError),
    #    allow_empty_datasets=False,
    #)

    #with open("lower_lifetime_preprocessed_fileset.pkl", "wb") as f:
    #    pickle.dump(lower_lifetime_dataset_runnable, f)

    #W_dataset_runnable, W_dataset_updated = preprocess(
    #    W_fileset,
    #    align_clusters=False,
    #    step_size=10_000,
    #    files_per_batch=100,
    #    skip_bad_files=True,
    #    save_form=False,
    #    file_exceptions=(OSError, KeyInFileError),
    #    allow_empty_datasets=False,
    #)

    #with open("W_preprocessed_fileset.pkl", "wb") as f:
    #    pickle.dump(W_dataset_runnable, f)

    data_dataset_runnable, data_dataset_updated = preprocess(
        data_fileset,
        align_clusters=False,
        step_size=10_000,
        files_per_batch=100,
        skip_bad_files=True,
        save_form=False,
        file_exceptions=(OSError, KeyInFileError),
        allow_empty_datasets=False,
    )
    
    with open("data_preprocessed_fileset.pkl", "wb") as f:
        pickle.dump(data_dataset_runnable, f)

    elapsed = time.time() - tic 
    print(f"Preproccessing datasets finished in {elapsed:.1f}s") 

    client.shutdown()
    cluster.close()
