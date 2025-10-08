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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from filesets.fileset import fileset
from filesets.skimmed_fileset import skimmed_fileset
from filesets.lower_lifetime_fileset import lower_lifetime_fileset
from filesets.DY_fileset import DY_fileset
from filesets.W_fileset import W_fileset
from filesets.TT_fileset import TT_fileset
from filesets.data_fileset import data_fileset
from filesets.merged_fileset import merged_fileset
from filesets.new_merged_fileset import new_merged_fileset
from filesets.second_skim_fileset import second_skim_fileset
from filesets.jet_dxy_fileset import jet_dxy_fileset
from filesets.merged_lower_lifetime_fileset import merged_lower_lifetime_fileset
from filesets.local_fileset import local_fileset

if __name__ == "__main__":

    samp = input("Which sample to preprocess? ") 

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    #XRootDFileSystem(hostid = "root://cmsxrootd.fnal.gov/", filehandle_cache_size = 250)
    tic = time.time()
    cluster = LPCCondorCluster(ship_env=True, transfer_input_files='/uscms/home/dally/x509up_u57864')
    # minimum > 0: https://github.com/CoffeaTeam/coffea/issues/465
    cluster.adapt(minimum=1, maximum=1000)
    client = Client(cluster)

    if "QCD" or "Stau" in samp: 
        dataset_runnable, dataset_updated = preprocess(
            fileset,
            align_clusters=False,
            step_size=20_000,
            files_per_batch=1000,
            skip_bad_files=True,
            save_form=False,
            file_exceptions=(OSError, KeyInFileError),
            allow_empty_datasets=False,
        )

        with open("processed_filesets/preprocessed_fileset.pkl", "wb") as f:
            pickle.dump(dataset_runnable, f)

    if "TT" in samp: 
        TT_dataset_runnable, TT_dataset_updated = preprocess(
            TT_fileset,
            align_clusters=False,
            step_size=20_000,
            files_per_batch=1000,
            skip_bad_files=True,
            save_form=False,
            file_exceptions=(OSError, KeyInFileError),
            allow_empty_datasets=False,
        )

        with open("processed_filesets/TT_preprocessed_fileset.pkl", "wb") as f:
            pickle.dump(TT_dataset_runnable, f)

    elif "DY" in samp:
        DY_dataset_runnable, DY_dataset_updated = preprocess(
            DY_fileset,
            align_clusters=False,
            step_size=20_000,
            files_per_batch=1000,
            skip_bad_files=True,
            save_form=False,
            file_exceptions=(OSError, KeyInFileError),
            allow_empty_datasets=False,
        )

        with open("processed_filesets/DY_preprocessed_fileset.pkl", "wb") as f:
            pickle.dump(DY_dataset_runnable, f)

    elif "Lower Lifetime" in samp:
        lower_lifetime_dataset_runnable, lower_lifetime_dataset_updated = preprocess(
            lower_lifetime_fileset,
            align_clusters=False,
            step_size=10_000,
            files_per_batch=100,
            skip_bad_files=True,
            save_form=False,
            file_exceptions=(OSError, KeyInFileError),
            allow_empty_datasets=False,
        )
        with open("processed_filesets/lower_lifetime_preprocessed_fileset.pkl", "wb") as f:
            pickle.dump(lower_lifetime_dataset_runnable, f)

    elif "W" in samp:    
        W_dataset_runnable, W_dataset_updated = preprocess(
            W_fileset,
            align_clusters=False,
            step_size=20_000,
            files_per_batch=100,
            skip_bad_files=True,
            save_form=False,
            file_exceptions=(OSError, KeyInFileError),
            allow_empty_datasets=False,
        )

        with open("processed_filesets/W_preprocessed_fileset.pkl", "wb") as f:
            pickle.dump(W_dataset_runnable, f)

    elif "data" in samp or "JetMET" in samp: 
        data_dataset_runnable, data_dataset_updated = preprocess(
            data_fileset,
            align_clusters=False,
            step_size=20_000,
            files_per_batch=100,
            skip_bad_files=True,
            save_form=False,
            file_exceptions=(OSError, KeyInFileError),
            allow_empty_datasets=False,
        )
         
        with open("processed_filesets/data_preprocessed_fileset.pkl", "wb") as f:
            pickle.dump(data_dataset_runnable, f)

    elif "merged" in samp:
        merged_dataset_runnable, merged_dataset_updated = preprocess(
            new_merged_fileset,
            align_clusters=False,
            step_size=10_000,
            files_per_batch=100,
            skip_bad_files=True,
            save_form=False,
            file_exceptions=(OSError, KeyInFileError),
            allow_empty_datasets=False,
        )
         
        with open("processed_filesets/merged_W_preprocessed_fileset.pkl", "wb") as f:
            pickle.dump(merged_dataset_runnable, f)

    elif "second skim" in samp:
        second_skim_dataset_runnable, second_skim_dataset_updated = preprocess(
            second_skim_fileset,
            align_clusters=False,
            step_size=10_000,
            files_per_batch=100,
            skip_bad_files=True,
            save_form=False,
            file_exceptions=(OSError, KeyInFileError),
            allow_empty_datasets=False,
        )
         
        with open("processed_filesets/second_skim_preprocessed_fileset.pkl", "wb") as f:
            pickle.dump(second_skim_dataset_runnable, f)

    elif "second lower lifetime" in samp:
        print("Making merged lower lifetime")
        merged_lower_lifetime_dataset_runnable, merged_lower_lifetime_dataset_updated = preprocess(
            merged_lower_lifetime_fileset,
            align_clusters=False,
            step_size=10_000,
            files_per_batch=100,
            skip_bad_files=True,
            save_form=False,
            file_exceptions=(OSError, KeyInFileError),
            allow_empty_datasets=False,
        )
         
        with open("processed_filesets/merged_lower_lifetime_preprocessed_fileset.pkl", "wb") as f:
            pickle.dump(merged_lower_lifetime_dataset_runnable, f)

    elif "local" in samp:
        local_dataset_runnable, local_dataset_updated = preprocess(
            local_fileset,
            align_clusters=False,
            step_size=10_000,
            files_per_batch=100,
            skip_bad_files=True,
            save_form=False,
            file_exceptions=(OSError, KeyInFileError),
            allow_empty_datasets=False,
        )
         
        with open("processed_filesets/local_preprocessed_fileset.pkl", "wb") as f:
            pickle.dump(local_dataset_runnable, f)

    else:
        print("That isn't an option. Options are: 'TT', 'DY', 'W', 'Lower Lifetime', 'data', 'merged', 'second skim', 'merged lower lifetime', 'local', or 'skimmed'")

    elapsed = time.time() - tic 
    print(f"Preproccessing datasets finished in {elapsed:.1f}s") 

    #client.shutdown()
    #cluster.close()
