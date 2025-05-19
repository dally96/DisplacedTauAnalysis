import sys
import os
import fsspec_xrootd
from  fsspec_xrootd import XRootDFileSystem
import uproot
import uproot.exceptions
from uproot.exceptions import KeyInFileError
import dill as pickle

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

from skimmed_fileset import *

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-m"    , "--muon"    , dest = "muon"   , help = "Leading muon variable"    , default = "pt")
parser.add_argument("-j"    , "--jet"     , dest = "jet"    , help = "Leading jet variable"     , default = "pt")         

leading_var = parser.parse_args()

selections = {
              "muon_pt":                    30, ##GeV
              "muon_ID":                    "muon_tightId",
              "muon_dxy_prompt_max":        50E-4, ##cm
              "muon_dxy_prompt_min":        0E-4, ##cm
              "muon_dxy_displaced_min":     0.1, ##cm
              "muon_dxy_displaced_max":     10, ##cm
              "muon_iso_max":               0.19,

              "jet_score":                  0.9, 
              "jet_pt":                     32, ##GeV

              "MET_pt":                     105, ##GeV
             }

class skimProcessor(processor.ProcessorABC):
    def __init__(self, leading_muon_var, leading_jet_var):
        self.leading_muon_var = leading_var.muon
        self.leading_jet_var  = leading_var.jet

        self._accumulator = {}
        for samp in skimmed_fileset:
            self._accumulator[samp] = dak.from_awkward(ak.Array([]), npartitions = 1)

    def process(self, events):
        logger.info(f"Start process for {events.metadata['dataset']}")

        leading_muon_var = self.leading_muon_var
        leading_jet_var = self.leading_jet_var
        
        out_dict = {}

        events["Stau"] = events.GenPart[(abs(events.GenPart.pdgId) == 1000015) &\
                                        (events.GenPart.hasFlags("isLastCopy"))]

        logger.info(f"Stau branch created")        

        events["StauTau"] = events.Stau.distinctChildren[(abs(events.Stau.distinctChildren.pdgId) == 15) &\
                                                         (events.Stau.distinctChildren.hasFlags("isLastCopy"))]

        events["StauTau"] = ak.firsts(events.StauTau[ak.argsort(events.StauTau.pt, ascending=False)], axis = 2) 
        
        logger.info(f"StauTau branch  created")

        #events["DisMuon"] = events.DisMuon[ak.argsort(events["DisMuon"][leading_muon_var], ascending=False, axis = 1)]
        #events["Jet"] = events.Jet[ak.argsort(events["Jet"][leading_jet_var], ascending=False, axis = 1)]

        #events["DisMuon"] = ak.singletons(ak.firsts(events.DisMuon))
        #events["Jet"] = ak.singletons(ak.firsts(events.Jet))

        logger.info(f"Chose leading muon and jet")

        good_muons  = dak.flatten((events.DisMuon.pt > selections["muon_pt"])           &\
                       (abs(events.DisMuon.eta) < 2.4)                                  &\
                       (events.DisMuon.tightId == 1)                                    &\
                       (abs(events.DisMuon.dxy) > selections["muon_dxy_prompt_min"]) &\
                       (abs(events.DisMuon.dxy) < selections["muon_dxy_prompt_max"]) &\
                       (events.DisMuon.pfRelIso03_all < selections["muon_iso_max"])
                      )

        good_jets   = dak.flatten((events.Jet.disTauTag_score1 < selections["jet_score"])   &\
                       (events.Jet.pt > selections["jet_pt"])                               #&\
                       #(abs(events.Jet.dxy) > selections["muon_dxy_prompt_min"])            &\
                       #(abs(events.Jet.dxy) < selections["muon_dxy_prompt_max"])
                      )

        good_events = (events.PFMET.pt > selections["MET_pt"])
            
        ### ONLY FOR Z PEAK CALCULATION WHERE WE NEED AT LEAST 2 MUONS ###
        events['DisMuon'] = ak.drop_none(events.DisMuon[good_muons])
        logger.info("Applied mask to DisMuon")
        num_good_muons = ak.count_nonzero(good_muons, axis=-1)
        num_muon_mask = (num_good_muons > 0)

        events['Jet'] = ak.drop_none(events.Jet[good_jets])
        logger.info("Applied mask to Jet")
        num_good_jets = ak.count_nonzero(good_jets, axis=-1)
        num_jet_mask = (num_good_jets > 0)

        events = events[num_muon_mask & num_jet_mask & good_events]
        #################

        #events = events[good_muons & good_jets & good_events]

        logger.info(f"Filtered events")        

        muon_vars =  [ 
                     "pt",
                     "eta",
                     "phi",
                     "charge",
                     "mass",
        #             "dxy",
        #             "dxyErr",
        #             "dz",
        #             "dzErr",
        #             "looseId",
        #             "mediumId",
        #             "tightId",
        #             "pfRelIso03_all",
        #             "pfRelIso03_chg",
        #             "pfRelIso04_all",
        #             "tkRelIso",
                     "genPartIdx",
                     ]

        #jet_vars =   [
        #             "pt",
        #             "eta",
        #             "phi",
        #             "disTauTag_score1",
        #             "dxy",
        #             ]

        #gpart_vars = [
        #             "genPartIdxMother", 
        #             "statusFlags", 
        #             "pdgId",
        #             "status", 
        #             "eta", 
        #             "mass", 
        #             "phi", 
        #             "pt", 
        #             "vertexR", 
        #             "vertexRho", 
        #             "vx", 
        #             "vy", 
        #             "vz",
        #             ]

        #gvist_vars = [
        #             "genPartIdxMother", 
        #             "charge",
        #             "status", 
        #             "eta", 
        #             "mass", 
        #             "phi", 
        #             "pt", 
        #             ]

        #tau_vars   = events.Tau.fields                        
        tau_vars    = [
                     "pt",
                     "eta",
                     "phi",
                     "charge",
                     "mass",
        #             "dxy",
        #             "dxyErr",
        #             "dz",
        #             "dzErr",
        #             "looseId",
        #             "mediumId",
        #             "tightId",
        #             "pfRelIso03_all",
        #             "pfRelIso03_chg",
        #             "pfRelIso04_all",
        #             "tkRelIso",
                     "genPartIdx",
                     ]
                        

        #MET_vars   = events.PFMET.fields  

        meta = ak.Array([0], backend = "typetracer")
        event_counts = events.map_partitions(lambda part: ak.num(part, axis = 0), meta = meta)
        partition_counts = event_counts.compute()
        logger.info(f"Computing the number of events in each partition {type(partition_counts)}")
        non_empty_partitions = []
        if str(type(partition_counts)) == "<class 'awkward.highlevel.Array'>":
            non_empty_partitions = [events.partitions[i] for i in range(len(partition_counts)) if partition_counts[i] > 0]
        else:
            if partition_counts > 0:
                non_empty_partitions = [events.partitions[0]]
        logger.info(f"Printing the type of non_empty_partitons {type(non_empty_partitions)}")
        logger.info(f"Printing the non_empty_partitons {non_empty_partitions}")
                               
        if non_empty_partitions:
            
            events = dak.concatenate(non_empty_partitions)
            logger.info(f"Concatenated non-empty partitions")

            for branch in muon_vars:
                out_dict["DisMuon_"   + branch]  = ak.drop_none(events["DisMuon"][branch])
            #for branch in jet_vars:
            #    out_dict["Jet_"       + branch]  = ak.drop_none(events["Jet"][branch])
            #for branch in gpart_vars:
            #    out_dict["GenPart_"   + branch]  = ak.drop_none(events["GenPart"][branch])
            #for branch in gpart_vars:
            #    out_dict["Stau_"      + branch]  = ak.drop_none(events["Stau"][branch])
            #for branch in gpart_vars:
            #    out_dict["StauTau_"   + branch]  = ak.drop_none(events["StauTau"][branch])
            #for branch in gvist_vars:
            #    out_dict["GenVisTau_" + branch]  = ak.drop_none(events["GenVisTau"][branch])
            for branch in tau_vars:
                out_dict["Tau_"       + branch]  = ak.drop_none(events["Tau"][branch])
            #for branch in MET_vars: 
            #    out_dict["PFMET_"     + branch]  = ak.drop_none(events["PFMET"][branch])    

            out_dict["event"]           = ak.drop_none(events.event)
            out_dict["run"]             = ak.drop_none(events.run)
            out_dict["luminosityBlock"] = ak.drop_none(events.luminosityBlock)
            out_dict["nDisMuon"]        = dak.num(ak.drop_none(events.DisMuon))
            #out_dict["nJet"]            = dak.num(ak.drop_none(events.Jet))
            #out_dict["nGenPart"]        = dak.num(ak.drop_none(events.GenPart))
            #out_dict["nGenVisTau"]      = dak.num(ak.drop_none(events.GenVisTau))
            out_dict["nTau"]            = dak.num(ak.drop_none(events.Tau))
            
        else:
            logger.info("No events  after cuts")

            for branch in muon_vars:
                out_dict["DisMuon_"   + branch]  = dak.from_awkward(ak.Array([]), npartitions = 1)
            #for branch in jet_vars:
            #    out_dict["Jet_"       + branch]  = dak.from_awkward(ak.Array([]), npartitions = 1)
            #for branch in gpart_vars:
            #    out_dict["GenPart_"   + branch]  = dak.from_awkward(ak.Array([]), npartitions = 1)
            #for branch in gpart_vars:
            #    out_dict["Stau_"      + branch]  = dak.from_awkward(ak.Array([]), npartitions = 1)
            #for branch in gpart_vars:
            #    out_dict["StauTau_"   + branch]  = dak.from_awkward(ak.Array([]), npartitions = 1)
            #for branch in gvist_vars:
            #    out_dict["GenVisTau_" + branch]  = dak.from_awkward(ak.Array([]), npartitions = 1)
            for branch in tau_vars:
                out_dict["Tau_"       + branch]  = dak.from_awkward(ak.Array([]), npartitions = 1)
            #for branch in MET_vars: 
            #    out_dict["PFMET_"     + branch]  = dak.from_awkward(ak.Array([]), npartitions = 1)

            out_dict["event"]           = dak.from_awkward(ak.Array([]), npartitions = 1)
            out_dict["run"]             = dak.from_awkward(ak.Array([]), npartitions = 1)
            out_dict["luminosityBlock"] = dak.from_awkward(ak.Array([]), npartitions = 1)
            out_dict["nDisMuon"]        = dak.from_awkward(ak.Array([]), npartitions = 1)
            #out_dict["nJet"]            = dak.from_awkward(ak.Array([]), npartitions = 1)
            #out_dict["nGenPart"]        = dak.from_awkward(ak.Array([]), npartitions = 1)
            #out_dict["nGenVisTau"]      = dak.from_awkward(ak.Array([]), npartitions = 1)
            out_dict["nTau"]            = dak.from_awkward(ak.Array([]), npartitions = 1)
            
        try:
            out_dict = dak.zip(out_dict, depth_limit = 1)

            return out_dict

        except Exception as e:
            logger.info(f"Error processing {events.metadata['dataset']}: {e}")        
            


    def postprocess(self, accumulator):
        return accumulator
    
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    XRootDFileSystem(hostid = "root://cmsxrootd.fnal.gov/", filehandle_cache_size = 250)
    tic = time.time()
    cluster = LPCCondorCluster(ship_env=True, transfer_input_files='/uscms/home/dally/x509up_u57864')
    # minimum > 0: https://github.com/CoffeaTeam/coffea/issues/465
    cluster.adapt(minimum=1, maximum=10000)
    client = Client(cluster)

    with open("sub_skimmed_preprocessed_fileset.pkl", "rb") as  f:
        dataset_runnable = pickle.load(f)    

    #to_compute = apply_to_fileset(
    #             skimProcessor(leading_var.muon, leading_var.jet),
    #             max_chunks(dataset_runnable, 10000),
    #             schemaclass=PFNanoAODSchema
    #)

    #for samp in skimmed_fileset: 
    for samp in dataset_runnable.keys():
        if "TT" in samp:
            print(samp)
            samp_runnable = {}
            samp_runnable[samp] = dataset_runnable[samp]
            to_compute = apply_to_fileset(
                         skimProcessor(leading_var.muon, leading_var.jet),
                         max_chunks(samp_runnable, 10000),
                         schemaclass=PFNanoAODSchema
            )

            try:
                outfile = uproot.dask_write(to_compute[samp], "root://cmseos.fnal.gov//store/user/dally/second_skim_muon_root/prompt_score_muon_tau_only" + samp, compute=False, tree_name='Events')
                dask.compute(outfile)
            
            except Exception as e:
                logger.info(f"Error writing out files: {e}")

    elapsed = time.time() - tic
    print(f"Finished in {elapsed:.1f}s")
    client.shutdown()
    cluster.close()
