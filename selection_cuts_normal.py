import sys, argparse, os
import uproot
import uproot.exceptions
from uproot.exceptions import KeyInFileError
import dill as pickle
import json, gzip, correctionlib
import fsspec_xrootd
from  fsspec_xrootd import XRootDFileSystem

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

import time
from distributed import Client
from lpcjobqueue import LPCCondorCluster

import warnings
warnings.filterwarnings("ignore", module="coffea") # Suppress annoying deprecation warnings for coffea vector, c.f. https://github.com/CoffeaTeam/coffea/blob/master/src/coffea/nanoevents/methods/candidate.py
import logging

from skimmed_fileset import *
from xsec import *
from selection_function import SR_selections, loose_SR_selections, loose_noIso_SR_selections, event_selection, Zpeak_selection

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-m"    , "--muon"    , dest = "muon"   , help = "Leading muon variable"    , default = "pt")
parser.add_argument("-j"    , "--jet"     , dest = "jet"    , help = "Leading jet variable"     , default = "pt")         

leading_var = parser.parse_args()

selections = {
              "muon_pt":                    30., ##GeV
              "muon_ID":                    "muon_tightId",
              "muon_dxy_prompt_max":        50E-4, ##cm
              "muon_dxy_prompt_min":        0E-4, ##cm
              "muon_dxy_displaced_min":     0.1, ##cm
              "muon_dxy_displaced_max":     10.,  ##cm
              "muon_iso_max":               0.18,

              "jet_score":                  0.9, 
              "jet_pt":                     32.,  ##GeV
              "jet_dxy_displaced_min":      0.02, ##cm

              "MET_pt":                     105., ##GeV
             }

class skimProcessor(processor.ProcessorABC):
    def __init__(self, leading_muon_var, leading_jet_var):
        self.leading_muon_var = leading_var.muon
        self.leading_jet_var  = leading_var.jet

        self._accumulator = {}
        for samp in skimmed_fileset:
            self._accumulator[samp] = dak.from_awkward(ak.Array([]), npartitions = 1)

        # Load pileup weights evaluators 
        jsonpog = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration"
        pileup_file = jsonpog + "/POG/LUM/2022_Summer22EE/puWeights.json.gz"

        with gzip.open(pileup_file, 'rt') as f:
            self.pileup_set = correctionlib.CorrectionSet.from_string(f.read().strip())

    def get_pileup_weights(self, events, also_syst=False):
        # Apply pileup weights
        evaluator = self.pileup_set["Collisions2022_359022_362760_eraEFG_GoldenJson"]
        sf = evaluator.evaluate(events.Pileup.nTrueInt, "nominal")
#         if also_syst:
#             sf_up = evaluator.evaluate(events.Pileup.nTrueInt, "up")
#             sf_down = evaluator.evaluate(events.Pileup.nTrueInt, "down")
#         return {'nominal': sf, 'up': sf_up, 'down': sf_down}
        return {'nominal': sf}


    def process_weight_corrs_and_systs(self, events, weights):
        pileup_weights = self.get_pileup_weights(events)
        # Compute nominal weight and systematic variations by multiplying relevant factors
        # For pileup, do not multiply nominal correction factor, as it is already included in the up/down variations
        # To see this, one can reproduce the ratio in
        # https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/blob/master/misc/LUM/2018_UL/puWeights.png?ref_type=heads
        # from the plain correctionset
        weight_dict = {
            'weight': weights * pileup_weights['nominal'] #* muon_weights['muon_trigger_SF'],
#             'weight_pileup_up': weights * pileup_weights['up'] * muon_weights['muon_trigger_SF'],
#             'weight_pileup_down': weights * pileup_weights['down'] * muon_weights['muon_trigger_SF'],
#             'weight_muon_trigger_up': weights * pileup_weights['nominal'] * (muon_weights['muon_trigger_SF'] + muon_weights['muon_trigger_SF_syst']),
#             'weight_muon_trigger_down': weights * pileup_weights['nominal'] * (muon_weights['muon_trigger_SF'] - muon_weights['muon_trigger_SF_syst']),
        }

        return weight_dict

    def process(self, events):
        logger.info(f"Start process for {events.metadata['dataset']}")

        leading_muon_var = self.leading_muon_var
        leading_jet_var = self.leading_jet_var
        out_dict = {}


        # Determine if dataset is MC or Data
        is_MC = True if hasattr(events, "GenPart") else False
        if is_MC: 
            sumWeights = num_events[events.metadata["dataset"]]
        

            events["Stau"] = events.GenPart[(abs(events.GenPart.pdgId) == 1000015) &\
                                            (events.GenPart.hasFlags("isLastCopy"))]

            logger.info(f"Stau branch created")        

            events["StauTau"] = events.Stau.distinctChildren[(abs(events.Stau.distinctChildren.pdgId) == 15) &\
                                                             (events.Stau.distinctChildren.hasFlags("isLastCopy"))]

            events["StauTau"] = ak.firsts(events.StauTau[ak.argsort(events.StauTau.pt, ascending=False)], axis = 2) 
            
            logger.info(f"StauTau branch  created")

        events["DisMuon"] = events.DisMuon[ak.argsort(events["DisMuon"][leading_muon_var], ascending=False, axis = 1)]
        events["Jet"] = events.Jet[ak.argsort(events["Jet"][leading_jet_var], ascending=False, axis = 1)]

        events["DisMuon"] = ak.singletons(ak.firsts(events.DisMuon))
        events["Jet"] = ak.singletons(ak.firsts(events.Jet))

        logger.info(f"Chose leading muon and jet")

        good_muons  = dak.flatten((events.DisMuon.pt > selections["muon_pt"])           &\
                       (events.DisMuon.mediumId == 1)                                    &\
                       (abs(events.DisMuon.dxy) > selections["muon_dxy_displaced_min"]) &\
                       (abs(events.DisMuon.dxy) < selections["muon_dxy_displaced_max"]) &\
                       (events.DisMuon.pfRelIso03_all < selections["muon_iso_max"])
                      )

        good_jets   = dak.flatten((events.Jet.disTauTag_score1 > selections["jet_score"])   &\
                       (events.Jet.pt > selections["jet_pt"])                               &\
                       (abs(events.Jet.dxy) > selections["jet_dxy_displaced_min"])          #&\
                       #(abs(events.Jet.dxy) < selections["muon_dxy_prompt_max"])
                      )

        good_events = (events.PFMET.pt > selections["MET_pt"])
            
        events = events[good_muons & good_jets & good_events]
        #events = event_selection(events, SR_selections, "tight_TT_CR")

        ### ONLY FOR Z PEAK CALCULATION WHERE WE NEED AT LEAST 2 MUONS ###
        #### Make sure to comment out leading jet and leading muon selection ####
        #good_muons  = (events.DisMuon.pt > selections["muon_pt"])           &\
        #               (abs(events.DisMuon.eta) < 2.4)                                  &\
        #               (events.DisMuon.tightId == 1)                                    &\
        #               (abs(events.DisMuon.dxy) > selections["muon_dxy_prompt_min"]) &\
        #               (abs(events.DisMuon.dxy) < selections["muon_dxy_prompt_max"]) &\
        #               (events.DisMuon.pfRelIso03_all < selections["muon_iso_max"])
        #              

        #good_jets   = (events.Jet.disTauTag_score1 < selections["jet_score"])   &\
        #               (events.Jet.pt > selections["jet_pt"])                               #&\
        #               #(abs(events.Jet.dxy) > selections["muon_dxy_prompt_min"])            &\
        #               #(abs(events.Jet.dxy) < selections["muon_dxy_prompt_max"])
        #              
        #events['DisMuon'] = ak.drop_none(events.DisMuon[good_muons])
        #logger.info("Applied mask to DisMuon")
        #num_good_muons = ak.count_nonzero(good_muons, axis=1)
        #num_muon_mask = (num_good_muons > 0)

        #events['Jet'] = ak.drop_none(events.Jet[good_jets])
        #logger.info("Applied mask to Jet")
        #num_good_jets = ak.count_nonzero(good_jets, axis=1)
        #num_jet_mask = (num_good_jets > 0)


        #events = events[num_muon_mask & num_jet_mask & good_events]
        #################

        logger.info(f"Filtered events")        

        muon_vars = events.DisMuon.fields

        jet_vars =   [   
                     "pt",
                     "eta",
                     "phi",
                     "disTauTag_score1",
                     "dxy",
                     ]   


        tau_vars   = events.Tau.fields                        
        #tau_vars    = [
        #             "pt",
        #             "eta",
        #             "phi",
        #             "charge",
        #             "mass",
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
        #             "genPartIdx",
        #             ]

        MET_vars   = events.PFMET.fields

        if is_MC: 
            weights = events.event/events.event
            weights = weights / sumWeights 
        else: 
            weights = events.event/events.event

        logger.info("mc weights")

        # scale factor correction
        # Handle systematics and weights
        if is_MC:
            weight_branches = self.process_weight_corrs_and_systs(events, weights)
        else:
            weight_branches = {'weight': weights}
        logger.info("all weights")
        
        events = ak.with_field(events, weight_branches["weight"], "weight")

        for branch in muon_vars:
            out_dict["DisMuon_"   + branch]  = ak.drop_none(events["DisMuon"][branch])
        for branch in jet_vars:
            out_dict["Jet_"       + branch]  = ak.drop_none(events["Jet"][branch])
        for branch in tau_vars:
            out_dict["Tau_"       + branch]  = ak.drop_none(events["Tau"][branch])
        for branch in MET_vars: 
            out_dict["PFMET_"     + branch]  = ak.drop_none(events["PFMET"][branch])    
        if is_MC:
            gpart_vars =   ['eta', 
                            'genPartIdxMother', 
                            'mass', 
                            'pdgId', 
                            'phi', 
                            'pt', 
                            'status', 
                            'statusFlags', 
                            #'vertexR', 
                            #'vertexRho', 
                            #'vx', 
                            #'vy', 
                            #'vz',
                            ]

            gvist_vars = events.GenVisTau.fields   
            gvtx_vars = events.GenVtx.fields

            for branch in gpart_vars:
                out_dict["GenPart_"   + branch]  = ak.drop_none(events["GenPart"][branch])
            for branch in gpart_vars:
                out_dict["Stau_"      + branch]  = ak.drop_none(events["Stau"][branch])
            for branch in gpart_vars:
                out_dict["StauTau_"   + branch]  = ak.drop_none(events["StauTau"][branch])
            for branch in gvist_vars:
                out_dict["GenVisTau_" + branch]  = ak.drop_none(events["GenVisTau"][branch])
            for branch in gvtx_vars:
                out_dict["GenVtx_" + branch]     = ak.drop_none(events["GenVtx"][branch])

        out_dict["event"]           = ak.drop_none(events.event)
        out_dict["weight"]          = dak.drop_none(events.weight)
        out_dict["run"]             = ak.drop_none(events.run)
        out_dict["luminosityBlock"] = ak.drop_none(events.luminosityBlock)
        out_dict["nDisMuon"]        = dak.num(ak.drop_none(events.DisMuon))
        out_dict["nJet"]            = dak.num(ak.drop_none(events.Jet))
        out_dict["nTau"]            = dak.num(ak.drop_none(events.Tau))
        if is_MC:
            out_dict["nGenPart"]        = dak.num(ak.drop_none(events.GenPart))
            out_dict["nGenVisTau"]      = dak.num(ak.drop_none(events.GenVisTau))
            
            
        try:
            out_dict = dak.zip(out_dict, depth_limit = 1)
            logger.info(f"Dictionary zipped: {events.metadata['dataset']}: {out_dict}")
            outfile = uproot.dask_write(out_dict, "root://cmseos.fnal.gov//store/user/dally/second_skim/test_SR_medium_" + samp, compute=False, tree_name='Events')

            return outfile

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
     #minimum > 0: https://github.com/CoffeaTeam/coffea/issues/465
    cluster.adapt(minimum=1, maximum=10000)
    client = Client(cluster)

    with open("merged_preprocessed_fileset.pkl", "rb") as  f:
        dataset_runnable = pickle.load(f)    
    print(f"Keys in dataset_runnable {dataset_runnable.keys()}")
    del dataset_runnable["QCD50_80"]["files"]["root://cmsxrootd.fnal.gov:1094//store/user/dally/first_skim/merged/merged_QCD50_80/merged_QCD50_80_2.root"]
    #to_compute = apply_to_fileset(
    #             skimProcessor(leading_var.muon, leading_var.jet),
    #             max_chunks(dataset_runnable, 10000),
    #             schemaclass=PFNanoAODSchema
    #)

    #for samp in skimmed_fileset: 
    for samp in dataset_runnable.keys():
        print(samp)
        if "DY" in samp or "MET" in samp or "TT" in samp or "W" in samp or "Stau" in samp: continue
        samp_runnable = {}
        samp_runnable[samp] = dataset_runnable[samp]
        to_compute = apply_to_fileset(
                     skimProcessor(leading_var.muon, leading_var.jet),
                     max_chunks(samp_runnable, 1000000),
                     schemaclass=PFNanoAODSchema,
                     uproot_options={"coalesce_config": uproot.source.coalesce.CoalesceConfig(max_request_ranges=10, max_request_bytes=1024*1024)}
        )

        try:
            dask.compute(to_compute)
        
        except Exception as e:
            logger.info(f"Error writing out files: {e}")

        elapsed = time.time() - tic 
        print(f"Finished in {elapsed:.1f}s")
        client.shutdown()
        cluster.close()
