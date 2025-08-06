import sys
import os
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

import time
from distributed import Client
from dask_lxplus import CernCluster

import warnings
warnings.filterwarnings("ignore", module="coffea") # Suppress annoying deprecation warnings for coffea vector, c.f. https://github.com/CoffeaTeam/coffea/blob/master/src/coffea/nanoevents/methods/candidate.py
import logging

from fileset import fileset
from skimmed_fileset import skimmed_fileset
from lower_lifetime_fileset import lower_lifetime_fileset
from DY_fileset import DY_fileset
from W_fileset import W_fileset
from TT_fileset import TT_fileset
from data_fileset import data_fileset
# from merged_fileset import merged_fileset
# from jet_dxy_fileset import jet_dxy_fileset


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    #XRootDFileSystem(hostid = "root://cmsxrootd.fnal.gov/", filehandle_cache_size = 250)
    tic = time.time()
    n_port = 8786
    import socket
    cluster = CernCluster(
            cores=1,
            memory='3000MB',
            disk='1000MB',
            death_timeout = '60',
            lcg = True,
            nanny = False,
            container_runtime = "none",
            log_directory = "/eos/user/f/fiorendi/condor/log",
            scheduler_options={
                'port': n_port,
                'host': socket.gethostname(),
                },
            job_extra={
                '+JobFlavour': '"longlunch"',
                },
            extra = ['--worker-port 10000:10100']
            )
    
    # minimum > 0: https://github.com/CoffeaTeam/coffea/issues/465
    cluster.adapt(minimum=1, maximum=1000)
    client = Client(cluster)

    dataset_runnable, dataset_updated = preprocess(
       fileset,
       align_clusters=False,
       step_size=20_000,
       files_per_batch=1,
       skip_bad_files=True,
       save_form=False,
       file_exceptions=(OSError, KeyInFileError),
       allow_empty_datasets=False,
    )

    with open("signal_test_preprocessed_fileset.pkl", "wb") as f:
       pickle.dump(dataset_runnable, f)
#
    #TT_dataset_runnable, TT_dataset_updated = preprocess(
    #    TT_fileset,
    #    align_clusters=False,
    #    step_size=20_000,
    #    files_per_batch=1000,
    #    skip_bad_files=True,
    #    save_form=False,
    #    file_exceptions=(OSError, KeyInFileError),
    #    allow_empty_datasets=False,
    #)

    #with open("TT_preprocessed_fileset.pkl", "wb") as f:
    #    pickle.dump(TT_dataset_runnable, f)

    #DY_dataset_runnable, DY_dataset_updated = preprocess(
    #    DY_fileset,
    #    align_clusters=False,
    #    step_size=10_000,
    #    files_per_batch=1000,
    #    skip_bad_files=True,
    #    save_form=False,
    #    file_exceptions=(OSError, KeyInFileError),
    #    allow_empty_datasets=False,
    #)

    #with open("DY_preprocessed_fileset.pkl", "wb") as f:
    #    pickle.dump(DY_dataset_runnable, f)

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

#    W_dataset_runnable, W_dataset_updated = preprocess(
#        W_fileset,
#        align_clusters=False,
#        step_size=20_000,
#        files_per_batch=100,
#        skip_bad_files=True,
#        save_form=False,
#        file_exceptions=(OSError, KeyInFileError),
#        allow_empty_datasets=False,
#    )
#
#    with open("W_preprocessed_fileset.pkl", "wb") as f:
#        pickle.dump(W_dataset_runnable, f)
#
#    data_dataset_runnable, data_dataset_updated = preprocess(
#        data_fileset,
#        align_clusters=False,
#        step_size=20_000,
#        files_per_batch=100,
#        skip_bad_files=True,
#        save_form=False,
#        file_exceptions=(OSError, KeyInFileError),
#        allow_empty_datasets=False,
#    )
#     
#    with open("data_preprocessed_fileset.pkl", "wb") as f:
#        pickle.dump(data_dataset_runnable, f)

#     merged_dataset_runnable, merged_dataset_updated = preprocess(
#         merged_fileset,
#         align_clusters=False,
#         step_size=10_000,
#         files_per_batch=100,
#         skip_bad_files=True,
#         save_form=False,
#         file_exceptions=(OSError, KeyInFileError),
#         allow_empty_datasets=False,
#     )
     
#     with open("merged_preprocessed_fileset.pkl", "wb") as f:
#         pickle.dump(merged_dataset_runnable, f)

    elapsed = time.time() - tic 
    print(f"Preproccessing datasets finished in {elapsed:.1f}s") 

    client.shutdown()
    cluster.close()
