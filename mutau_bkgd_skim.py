import sys, argparse, os
import fsspec_xrootd
from  fsspec_xrootd import XRootDFileSystem
import uproot
import uproot.exceptions
from uproot.exceptions import KeyInFileError
import pickle
import json, gzip, correctionlib

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
from fileset import *

from selections.lumi_selections import select_lumis

import time
from datetime import datetime
from distributed import Client
from lpcjobqueue import LPCCondorCluster

import warnings
warnings.filterwarnings("ignore", module="coffea") # Suppress annoying deprecation warnings for coffea vector, c.f. https://github.com/CoffeaTeam/coffea/blob/master/src/coffea/nanoevents/methods/candidate.py
import logging 


# Can also be put in a utils file later
def delta_r_mask(first: ak.highlevel.Array, second: ak.highlevel.Array, threshold: float) -> ak.highlevel.Array:
            mval = first.metric_table(second)
            return ak.all(mval > threshold, axis=-1)

PFNanoAODSchema.mixins["DisMuon"] = "Muon"

class MyProcessor(processor.ProcessorABC):
    def __init__(self):
        self._accumulator = {} 
        for samp in fileset:
            self._accumulator[samp] = dak.from_awkward(ak.Array([]), npartitions = 1)

    def process(self, events):
        
        if events is None: 
            return output

        # Determine if dataset is MC or Data
        is_MC = True if hasattr(events, "GenPart") else False
        if not is_MC:
            try:
                lumimask = select_lumis('2022', events)
                events = events[lumimask]
            except:
                print (f"[ lumimask ] Skip now! Unable to find year info of {dataset_name}")

        logger.info("Starting process")

        charged_sel = events.Jet.constituents.pf.charge != 0            

        events["Jet_dxy"] = ak.flatten(ak.drop_none(events.Jet.constituents.pf[ak.argmax(events.Jet.constituents.pf[charged_sel].pt, axis=2, keepdims=True)].d0), axis=-1)

        # Define the "good muon" condition for each muon per event
        good_muon_mask = (
            #((events.DisMuon.isGlobal == 1) | (events.DisMuon.isTracker == 1)) & # Equivalent to loose ID cut without isPFcand requirement. Reduce background from non-prompt muons
             (events.DisMuon.pt > 20)
            & (abs(events.DisMuon.eta) < 2.4) # Acceptance of the CMS muon system
        )
        logger.info(f"Defined good muons")
        events['DisMuon'] = events.DisMuon[good_muon_mask]
        logger.info(f"Applied mask to DisMuon")
        num_evts = ak.num(events, axis=0)
        logger.info("Counted the number of original events")
        num_good_muons = ak.count_nonzero(good_muon_mask, axis=1)
        logger.info("Counted the number of events with good muons")
        events = events[num_good_muons >= 1]
        logger.info("Counted the number of events with one or more good muons")
        logger.info(f"Cut muons")

        logger.info(f"Before  overlap removal")
        # Perform the overlap removal with respect to muons, electrons and photons, dR=0.4
        events['Jet'] = events.Jet[delta_r_mask(events.Jet, events.Photon, 0.4)]
        events['Jet'] = events.Jet[delta_r_mask(events.Jet, events.Electron, 0.4)]
        events['Jet'] = events.Jet[delta_r_mask(events.Jet, events.Muon, 0.4)]
        events['Jet'] = events.Jet[delta_r_mask(events.Jet, events.DisMuon, 0.4)]
        logger.info(f"Performed overlap removal")

        good_jet_mask = (
            (events.Jet.pt > 20)
            & (abs(events.Jet.eta) < 2.4) 
        )
        logger.info("Defined good jets")
        
        events['Jet'] = events.Jet[good_jet_mask]
        num_good_jets = ak.count_nonzero(good_jet_mask, axis=1)
        events = events[num_good_jets >= 1]
        logger.info(f"Cut jets")

        #Noise filter
        noise_mask = (
                     (events.Flag.goodVertices == 1) 
                     & (events.Flag.globalSuperTightHalo2016Filter == 1)
                     & (events.Flag.EcalDeadCellTriggerPrimitiveFilter == 1)
                     & (events.Flag.BadPFMuonFilter == 1)
                     & (events.Flag.BadPFMuonDzFilter == 1)
                     & (events.Flag.hfNoisyHitsFilter == 1)
                     & (events.Flag.eeBadScFilter == 1)
                     & (events.Flag.ecalBadCalibFilter == 1)
                         )

        events = events[noise_mask] 
        logger.info(f"Filtered noise")

        #Trigger Selection
        trigger_mask = (
                        events.HLT.PFMET120_PFMHT120_IDTight                                    |\
                        events.HLT.PFMET130_PFMHT130_IDTight                                    |\
                        events.HLT.PFMET140_PFMHT140_IDTight                                    |\
                        events.HLT.PFMETNoMu120_PFMHTNoMu120_IDTight                            |\
                        events.HLT.PFMETNoMu130_PFMHTNoMu130_IDTight                            |\
                        events.HLT.PFMETNoMu140_PFMHTNoMu140_IDTight                            |\
                        events.HLT.PFMET120_PFMHT120_IDTight_PFHT60                             |\
                        #events.HLT.MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight        |\ #Trigger not included in current Nanos
                        events.HLT.PFMETTypeOne140_PFMHT140_IDTight                             |\
                        events.HLT.MET105_IsoTrk50                                              |\
                        events.HLT.PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF                   |\
                        events.HLT.MET120_IsoTrk50                                              |\
                        events.HLT.IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1   |\
                        events.HLT.IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1   |\
                        events.HLT.Ele30_WPTight_Gsf                                            |\
                        events.HLT.DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1                    |\
                        events.HLT.DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1        |\
                        events.HLT.DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1                 
        )
        
        events = events[trigger_mask]
        logger.info(f"Applied trigger mask")

        out_dict = {}

        muon_vars  = events.DisMuon.fields 
        jet_vars   = events.Jet.fields 
        tau_vars   = events.Tau.fields                        
        MET_vars   = events.PFMET.fields  

        if is_MC:
            gpart_vars  = events.GenPart.fields 
            gvist_vars  = events.GenVisTau.fields 
            gvtx_vars   = events.GenVtx.fields
            pileup_vars = events.Pileup.fields 
 
        for branch in muon_vars:
            if branch[-1] == "G": continue
            out_dict["DisMuon_"   + branch]  = dak.drop_none(events["DisMuon"][branch])
        for branch in jet_vars:
            if branch[-1] == "G": continue
            out_dict["Jet_"       + branch]  = dak.drop_none(events["Jet"][branch])
        for branch in tau_vars:
            if branch[-1] == "G": continue
            out_dict["Tau_"       + branch]  = dak.drop_none(events["Tau"][branch])
        for branch in MET_vars: 
            if branch[-1] == "G": continue
            out_dict["PFMET_"     + branch]  = dak.drop_none(events["PFMET"][branch])    
        if is_MC:
            for branch in gpart_vars:          
                if branch[-1] == "G": continue
                out_dict["GenPart_"   + branch]  = dak.drop_none(events["GenPart"][branch])
            for branch in gvist_vars:
                if branch[-1] == "G": continue
                out_dict["GenVisTau_" + branch]  = dak.drop_none(events["GenVisTau"][branch])
            for branch in gvtx_vars:
                if branch[-1] == "G": continue
                out_dict["GenVtx_"    + branch]  = dak.drop_none(events["GenVtx"][branch])
            for branch in pileup_vars:
                out_dict["Pileup_"    + branch]  = dak.drop_none(events["Pileup"][branch])

        out_dict["event"]           = dak.drop_none(events.event)
        out_dict["run"]             = dak.drop_none(events.run)
        out_dict["luminosityBlock"] = dak.drop_none(events.luminosityBlock)
        out_dict["Jet_dxy"]         = dak.drop_none(events["Jet_dxy"])
        out_dict["nDisMuon"]        = dak.num(dak.drop_none(events.DisMuon))
        out_dict["nTau"]            = dak.num(dak.drop_none(events.Tau))
        out_dict["nJet"]            = dak.num(dak.drop_none(events.Jet))

        if is_MC:
            out_dict["nGenPart"]        = dak.num(dak.drop_none(events.GenPart))
            out_dict["nGenVisTau"]      = dak.num(dak.drop_none(events.GenVisTau))


        logger.info(f"Filled dictionary")
        
        out_dict = dak.zip(out_dict, depth_limit = 1)

        logger.info(f"Dictionary zipped: {events.metadata['dataset']}: {out_dict}")
        out_file = uproot.dask_write(out_dict, "root://cmseos.fnal.gov//store/user/dally/first_skim_muon_root_pileup_genvtx/"+events.metadata['dataset'], compute=False, tree_name='Events')
        return out_file

    def postprocess(self, accumulator):
        return accumulator
        

if __name__ == "__main__":

    print("Time started:", datetime.now().strftime("%H:%M:%S"))
    #XRootDFileSystem(hostid = "root://cmseos.fnal.gov/", filehandle_cache_size = 2500)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    tic = time.time()
    cluster = LPCCondorCluster(ship_env=True, transfer_input_files='/uscms/home/dally/x509up_u57864')
    # minimum > 0: https://github.com/CoffeaTeam/coffea/issues/465
    cluster.adapt(minimum=1, maximum=10000)
    client = Client(cluster)

    with open("preprocessed_fileset.pkl", "rb") as  f:
        Stau_QCD_DY_dataset_runnable = pickle.load(f)    
    with open("TT_preprocessed_fileset.pkl", "rb") as  f:
        TT_dataset_runnable = pickle.load(f)    
    with open("W_preprocessed_fileset.pkl", "rb") as  f:
        W_dataset_runnable = pickle.load(f)    
    dataset_runnable = TT_dataset_runnable | Stau_QCD_DY_dataset_runnable
#    with open("lower_lifetime_preprocessed_fileset.pkl", "rb") as  f:
#        lower_lifetime_dataset_runnable = pickle.load(f)    
#    with open("W_preprocessed_fileset.pkl", "rb") as  f:
#        W_dataset_runnable = pickle.load(f)    
#    
#    dataset_runnable = lower_lifetime_dataset_runnable | W_dataset_runnable
#    print(lower_lifetime_dataset_runnable.keys())
#    print(W_dataset_runnable.keys())
#    print(dataset_runnable.keys())
#
#    to_compute = apply_to_fileset(
#                 MyProcessor(),
#                 max_chunks(dataset_runnable, 1000000),
#                 schemaclass=PFNanoAODSchema
#    )

    for samp in W_dataset_runnable.keys(): 
        #if ("2022" not in samp) and ("100_" not in samp) and ("200_" not in samp) and ("500_" not in samp):
        if  ("W" in samp):
            samp_runnable = {}
            samp_runnable[samp] = W_dataset_runnable[samp]
            print("Time before comupute:", datetime.now().strftime("%H:%M:%S")) 
            to_compute = apply_to_fileset(
                     MyProcessor(),
                     max_chunks(samp_runnable, 100000),
                     schemaclass=PFNanoAODSchema,
                     uproot_options={"coalesce_config": uproot.source.coalesce.CoalesceConfig(max_request_ranges=10, max_request_bytes=1024*1024),
                                     "allow_read_errors_with_report": True}
            )
            #failed_chunks = get_failed_steps_for_fileset(
            print(to_compute)
            #outfile = uproot.dask_write(to_compute[0][samp], "root://cmseos.fnal.gov//store/user/dally/first_skim_muon_root_test/"+samp, compute=False, tree_name='Events')
            dask.compute(to_compute)
    
    client.shutdown()
    cluster.close()
    
    elapsed = time.time() - tic
    print(f"Finished in {elapsed:.1f}s")
    
