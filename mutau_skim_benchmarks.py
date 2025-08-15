import awkward as ak
from coffea import processor
from coffea.nanoevents.methods import candidate

import uproot
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema

# import sys, argparse, os
import fsspec_xrootd
from  fsspec_xrootd import XRootDFileSystem

import time
from datetime import datetime

PFNanoAODSchema.warn_missing_crossrefs = False


# import uproot
# import uproot.exceptions
# from uproot.exceptions import KeyInFileError
# import pickle
# import json, gzip, correctionlib
# 
# import numpy as np
# from coffea import processor
# from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema, NanoAODSchema
# from coffea.dataset_tools import (
#     apply_to_fileset,
#     max_chunks,
#     preprocess,
# )
# from coffea.lumi_tools import LumiData, LumiList, LumiMask
# import awkward as ak
# import dask
# from dask import config as cfg
# cfg.set({'distributed.scheduler.worker-ttl': None}) # Check if this solves some dask issues
# import dask_awkward as dak
# from dask_jobqueue import HTCondorCluster
# from dask.distributed import Client, wait, progress, LocalCluster
# from fileset import *
# from dask_lxplus import CernCluster
# 
# from selections.lumi_selections import select_lumis
# 
# from distributed import Client
# # from lpcjobqueue import LPCCondorCluster
# 
# import warnings
# warnings.filterwarnings("ignore", module="coffea") # Suppress annoying deprecation warnings for coffea vector, c.f. https://github.com/CoffeaTeam/coffea/blob/master/src/coffea/nanoevents/methods/candidate.py
# import logging 


PFNanoAODSchema.mixins["DisMuon"] = "Muon"

exclude_prefixes = ['Flag', 'JetSVs', 'GenJetAK8_', 'SubJet', 
                    'Photon', 'TrigObj', 'TrkMET', 'HLT',
                    'Puppi', 'OtherPV', 'GenJetCands',
                    'FsrPhoton', ''
                    ## tmp
                    'diele', 'LHE', 'dimuon', 'Rho', 'JetPFCands', 'GenJet', 'GenCands', 
                    'Electron'
                    ]
                    
include_prefixes = ['DisMuon', 'Jet', 'Tau', 'PFMET', 'SV', 'PV', 'GenPart', 'GenVisTau', 'GenVtx', 'Pileup',
                    'event', 'run', 'luminosityBlock', 'nDisMuon', 'nTau', 'nJet', 'nGenPart', 'nGenVisTau'
                   ]                    

good_hlts = [
  "HLT_PFMET120_PFMHT120_IDTight",                  
  "HLT_PFMET130_PFMHT130_IDTight",
  "HLT_PFMET140_PFMHT140_IDTight",
  "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight",
  "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight",
  "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight",
  "HLT_PFMET120_PFMHT120_IDTight_PFHT60",
  "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF",
  "HLT_PFMETTypeOne140_PFMHT140_IDTight",
  "HLT_MET105_IsoTrk50",
  "HLT_MET120_IsoTrk50",
  "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1",
  "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1",
  "HLT_Ele30_WPTight_Gsf",                                         
  "HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1",                 
  "HLT_DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1",     
  "HLT_DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1"
]

def is_included(name):
        return any(name.startswith(prefix) for prefix in include_prefixes)

def is_good_hlt(name):
        return (name in good_hlts)
   


## for old coffea
def is_rootcompat_old_coffea(a):
    """Check if the data is a flat or 1-d jagged array for compatibility with uproot."""
    t = ak.type(a)
    if isinstance(t, ak.types.NumpyType):
        return True
    if isinstance(t, ak.types.ListType) and isinstance(t.content, ak.types.NumpyType):
        return True
    return False

### for old coffea
def uproot_writeable_old_coffea(events):
    """Restrict to columns that uproot can write compactly."""
    out_event = events[
        [x for x in events.fields if (is_included(x)) or (is_good_hlt(x))]
    ]    
    for bname in events.fields:
        if events[bname].fields and ((is_included(bname) or is_good_hlt(bname))):
            print (bname)
            out_event[bname] = ak.zip(
                {
                    n: ak.without_parameters(events[bname][n])
                    for n in events[bname].fields
                    if is_rootcompat(events[bname][n])
                }
            )
    return out_event


def is_rootcompat(a):
    """Is it a flat or 1-d jagged array?"""
    t = ak.type(a)
    if isinstance(t, ak.types.ArrayType):
        if isinstance(t.content, ak.types.NumpyType):
            return True
        if isinstance(t.content, ak.types.ListType) and isinstance(t.content.content, ak.types.NumpyType):
            return True
    return False

## for new coffea and uproot.recreate    
def uproot_writeable(events):
    """Restrict to columns that uproot can write compactly"""
    out = {}
    for bname in events.fields:
        if events[bname].fields and ((is_included(bname) or is_good_hlt(bname))):
            out[bname] = ak.zip(
                {
                    n: ak.to_packed(ak.without_parameters(events[bname][n])) 
                    for n in events[bname].fields 
                    if is_rootcompat(events[bname][n])
                }
            )
    return out    


class MyProcessor(processor.ProcessorABC):
    def __init__(self):
        pass
#         self._accumulator = {} 
#         for samp in fileset:
#             self._accumulator[samp] = dak.from_awkward(ak.Array([]), npartitions = 1)

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

#         logger.info("Starting process")


        # Define the "good muon" condition for each muon per event
        good_muon_mask = (
            #((events.DisMuon.isGlobal == 1) | (events.DisMuon.isTracker == 1)) & # Equivalent to loose ID cut without isPFcand requirement. Reduce background from non-prompt muons
             (events.DisMuon.pt > 20)
            & (abs(events.DisMuon.eta) < 2.4) # Acceptance of the CMS muon system
        )
#         logger.info(f"Defined good muons")
        events['DisMuon'] = events.DisMuon[good_muon_mask]
#         logger.info(f"Applied mask to DisMuon")
        num_evts = ak.num(events, axis=0)
#         logger.info("Counted the number of original events")
        num_good_muons = ak.count_nonzero(good_muon_mask, axis=1)
#         logger.info("Counted the number of events with good muons")
        events = events[num_good_muons >= 1]
#         logger.info("Counted the number of events with one or more good muons")
#         logger.info(f"Cut muons")

        good_jet_mask = (
            (events.Jet.pt > 20)
            & (abs(events.Jet.eta) < 2.4)
            & ~(ak.all(events.Jet.constituents.pf.charge == 0, axis = -1)) 
        )
#         logger.info("Defined good jets")
        
        events['Jet'] = events.Jet[good_jet_mask]
        num_good_jets = ak.count_nonzero(good_jet_mask, axis=1)
        events = events[num_good_jets >= 1]
#         logger.info(f"Cut jets")

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
#         logger.info(f"Filtered noise")

        n_jets = ak.num(events.Jet.pt)
        charged_sel = events.Jet.constituents.pf.charge != 0
        dxy = ak.where(ak.all(events.Jet.constituents.pf.charge == 0, axis = -1), -999, ak.flatten(events.Jet.constituents.pf[ak.argmax(events.Jet.constituents.pf[charged_sel].pt, axis=2, keepdims=True)].d0, axis = -1))
        events["Jet"] = ak.with_field(events.Jet, dxy, where = "dxy")

        out_dict = {}

        muon_vars  = events.DisMuon.fields 
        jet_vars   = events.Jet.fields 
        tau_vars   = events.Tau.fields                        
        MET_vars   = events.PFMET.fields  
        SV_vars    = events.SV.fields
        PV_vars    = events.PV.fields
        HLT_vars   = [
                        "PFMET120_PFMHT120_IDTight",                  
                        "PFMET130_PFMHT130_IDTight",
                        "PFMET140_PFMHT140_IDTight",
                        "PFMETNoMu120_PFMHTNoMu120_IDTight",
                        "PFMETNoMu130_PFMHTNoMu130_IDTight",
                        "PFMETNoMu140_PFMHTNoMu140_IDTight",
                        "PFMET120_PFMHT120_IDTight_PFHT60",
                        "PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF",
                        "PFMETTypeOne140_PFMHT140_IDTight",
                        "MET105_IsoTrk50",
                        "MET120_IsoTrk50",
                        "IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1",
                        "IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1",
                        "Ele30_WPTight_Gsf",                                         
                        "DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1",                 
                        "DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1",     
                        "DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1",              
                        #"MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight",      
                    ]

        for branch in muon_vars:
            ## coffea specific: in order to use their functions, they create fields in each collection 
            ## that end in G for a global index so they can cross reference between them.
            ## These aren't able to be copied so this is to stop the code from erroring
            if branch[-1] == "G": continue
#             out_dict["DisMuon_"   + branch]  = dak.drop_none(events["DisMuon"][branch])
            out_dict["DisMuon_"   + branch]  = ak.drop_none(events["DisMuon"][branch])
        for branch in jet_vars:
            if branch[-1] == "G": continue
            out_dict["Jet_"       + branch]  = ak.drop_none(events["Jet"][branch])
        for branch in tau_vars:
            if branch[-1] == "G": continue
            out_dict["Tau_"       + branch]  = ak.drop_none(events["Tau"][branch])
        for branch in MET_vars: 
            if branch[-1] == "G": continue
            out_dict["PFMET_"     + branch]  = ak.drop_none(events["PFMET"][branch])    
        for branch in SV_vars:
            if branch[-1] == "G": continue
            out_dict["SV_"        + branch]  = ak.drop_none(events["SV"][branch])    
        for branch in PV_vars:
            if branch[-1] == "G": continue
            out_dict["PV_"        + branch]  = ak.drop_none(events["PV"][branch])    
        for branch in HLT_vars:
            if branch[-1] == "G": continue
            if "EphemeralPhysics_TestMasking" in branch:
                out_dict["HLT_"       + branch] = ak.from_awkward(ak.Array([]), npartitions = 1)
            out_dict["HLT_"       + branch]  = ak.drop_none(events["HLT"][branch])    
            
        if is_MC:
            gpart_vars  = events.GenPart.fields 
            gvist_vars  = events.GenVisTau.fields 
            gvtx_vars   = events.GenVtx.fields
            pileup_vars = events.Pileup.fields 
            for branch in gpart_vars:          
                if branch[-1] == "G": continue
                out_dict["GenPart_"   + branch]  = ak.drop_none(events["GenPart"][branch])
            for branch in gvist_vars:
                if branch[-1] == "G": continue
                out_dict["GenVisTau_" + branch]  = ak.drop_none(events["GenVisTau"][branch])
            for branch in gvtx_vars:
                if branch[-1] == "G": continue
                out_dict["GenVtx_"    + branch]  = ak.drop_none(events["GenVtx"][branch])
            for branch in pileup_vars:
                out_dict["Pileup_"    + branch]  = ak.drop_none(events["Pileup"][branch])

        out_dict["event"]           = ak.drop_none(events.event)
        out_dict["run"]             = ak.drop_none(events.run)
        out_dict["luminosityBlock"] = ak.drop_none(events.luminosityBlock)
        out_dict["nDisMuon"]        = ak.num(ak.drop_none(events.DisMuon))
        out_dict["nTau"]            = ak.num(ak.drop_none(events.Tau))
        out_dict["nJet"]            = ak.num(ak.drop_none(events.Jet))
#         out_dict["event"]           = dak.drop_none(events.event)
#         out_dict["run"]             = dak.drop_none(events.run)
#         out_dict["luminosityBlock"] = dak.drop_none(events.luminosityBlock)
#         out_dict["nDisMuon"]        = dak.num(dak.drop_none(events.DisMuon))
#         out_dict["nTau"]            = dak.num(dak.drop_none(events.Tau))
#         out_dict["nJet"]            = dak.num(dak.drop_none(events.Jet))

        if is_MC:
            out_dict["nGenPart"]        = ak.num(ak.drop_none(events.GenPart))
            out_dict["nGenVisTau"]      = ak.num(ak.drop_none(events.GenVisTau))


#         logger.info(f"Filled dictionary")
        
#         out_dict = dak.zip(out_dict, depth_limit = 1)
        out_dict = ak.zip(out_dict, depth_limit = 1)
        
        ## sara
#         out_dict_sara = events
#         conv = uproot_writeable(out_dict_sara)
#         rows = 50000 
#         out_file = uproot.dask_write(
#             conv.repartition(rows_per_partition=rows),
#             compute=True,
#             destination= "./"+events.metadata['dataset'],
# #             prefix=f"{dataset}_{era}_skim{file_index-1}",
#             tree_name="Events"
#         )

#         logger.info(f"Dictionary zipped: {events.metadata['dataset']}: {out_dict}")
        ## daniel
        out_file = uproot.dask_write(out_dict, "./daniel_"+events.metadata['dataset'], compute=True, tree_name='Events')
        return out_file

    def postprocess(self, accumulator):
        pass
#         return accumulator
  
  


class MyProcessorSara(processor.ProcessorABC):
    def __init__(self):
        pass
#         self._accumulator = {} 
#         for samp in fileset:
#             self._accumulator[samp] = dak.from_awkward(ak.Array([]), npartitions = 1)

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

#         logger.info("Starting process")


        # Define the "good muon" condition for each muon per event
        good_muon_mask = (
            #((events.DisMuon.isGlobal == 1) | (events.DisMuon.isTracker == 1)) & # Equivalent to loose ID cut without isPFcand requirement. Reduce background from non-prompt muons
             (events.DisMuon.pt > 20)
            & (abs(events.DisMuon.eta) < 2.4) # Acceptance of the CMS muon system
        )
#         logger.info(f"Defined good muons")
        events['DisMuon'] = events.DisMuon[good_muon_mask]
#         logger.info(f"Applied mask to DisMuon")
        num_evts = ak.num(events, axis=0)
#         logger.info("Counted the number of original events")
        num_good_muons = ak.count_nonzero(good_muon_mask, axis=1)
#         logger.info("Counted the number of events with good muons")
        events = events[num_good_muons >= 1]
#         logger.info("Counted the number of events with one or more good muons")
#         logger.info(f"Cut muons")

        good_jet_mask = (
            (events.Jet.pt > 20)
            & (abs(events.Jet.eta) < 2.4)
            & ~(ak.all(events.Jet.constituents.pf.charge == 0, axis = -1)) 
        )
#         logger.info("Defined good jets")
        
        events['Jet'] = events.Jet[good_jet_mask]
        num_good_jets = ak.count_nonzero(good_jet_mask, axis=1)
        events = events[num_good_jets >= 1]
#         logger.info(f"Cut jets")

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
#         logger.info(f"Filtered noise")

        n_jets = ak.num(events.Jet.pt)
        charged_sel = events.Jet.constituents.pf.charge != 0
        dxy = ak.where(ak.all(events.Jet.constituents.pf.charge == 0, axis = -1), -999, ak.flatten(events.Jet.constituents.pf[ak.argmax(events.Jet.constituents.pf[charged_sel].pt, axis=2, keepdims=True)].d0, axis = -1))
        events["Jet"] = ak.with_field(events.Jet, dxy, where = "dxy")
#         logger.info(f"Filled dictionary")
        
        out_dict_sara = events
        events_to_write = uproot_writeable(events)

### old
#         out_file = uproot.dask_write(
#             conv,
# #             events_to_write.repartition(rows_per_partition=rows),
#             compute=True,
#             destination= "./"+events.metadata['dataset'],
# #             prefix=f"{dataset}_{era}_skim{file_index-1}",
#             tree_name="Events"
#         )
#         uproot.dask_write(conv, "out.root", "Events")
        with uproot.recreate("out.root") as f:
          f["Events"] = events_to_write  # regular Awkward array

#         logger.info(f"Dictionary zipped: {events.metadata['dataset']}: {out_dict}")
#         return out_file

    def postprocess(self, accumulator):
        pass
#         return accumulator
  
  
  
fileset = {
    'signal': [
        'root://cms-xrd-global.cern.ch///store/user/fiorendi/displacedTaus/inputFiles_fortesting/nano_10_0.root',
    ]
}


if __name__ == "__main__":
## https://github.com/CoffeaTeam/coffea-hats/blob/master/04-processor.ipynb

    print("Time started:", datetime.now().strftime("%H:%M:%S"))
    tic = time.time()

    filename = 'nano_10_0.root'
#     filename = '/eos/cms/store/user/fiorendi/displacedTaus/inputFiles_fortesting/nano_10_0.root'
#     filename = 'root://cms-xrd-global.cern.ch///store/user/fiorendi/displacedTaus/inputFiles_fortesting/nano_10_0.root'

## simple processing
#     file = uproot.open({filename:"Events"})
    events = NanoEventsFactory.from_root(
      {filename: "Events"}, 
      schemaclass=PFNanoAODSchema, 
      mode="virtual",
      metadata={"dataset": "Stau"},
    ).events()

## old versions
#     file = uproot.open({filename:"Events"})
#     events = NanoEventsFactory.from_root(
#         file,
# #         entry_stop=10000,
#         metadata={"dataset": "Stau"},
#         schemaclass=PFNanoAODSchema,
#     ).events()

    p = MyProcessorSara()
#     p = MyProcessor()
    out = p.process(events)
## end simple processing       


#     iterative_run = processor.Runner(
#         executor = processor.IterativeExecutor(compression=None),
#         schema=PFNanoAODSchema,
#         maxchunks=4,
#     )
#     
#     out = iterative_run(
#         fileset,
#         treename="Events",
#         processor_instance=MyProcessorSara(),
#     )
    elapsed = time.time() - tic
    print(f"Finished in {elapsed:.1f}s")
#     out


# if __name__ == "__main__":
# 
#     print("Time started:", datetime.now().strftime("%H:%M:%S"))
#     #XRootDFileSystem(hostid = "root://cmseos.fnal.gov/", filehandle_cache_size = 2500)
# 
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)
#     
#     tic = time.time()
#     import socket
#     n_port = 8786
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
#     cluster.adapt(minimum=1, maximum=5000)
#     client = Client(cluster)
# 
#     with open("signal_test_preprocessed_fileset.pkl", "rb") as  f:
#         Stau_dataset_runnable = pickle.load(f)    
#     dataset_runnable = Stau_dataset_runnable  
# #    print(dataset_runnable.keys())
# 
#     for samp in dataset_runnable.keys(): 
#         if samp not in os.listdir("/eos/cms/store/user/fiorendi/displacedTaus/skim/mutau/v3/"):
# #             if "TT" in samp or "DY" in samp or "Stau" in samp or "QCD" in samp or "MET" in samp: continue
#             
#             samp_runnable = {}
#             samp_runnable[samp] = dataset_runnable[samp]
#             print("Time before comupute:", datetime.now().strftime("%H:%M:%S")) 
#             to_compute = apply_to_fileset(
#                      MyProcessor(),
#                      max_chunks(samp_runnable, 100000),
#                      schemaclass=PFNanoAODSchema,
#                      uproot_options={"coalesce_config": uproot.source.coalesce.CoalesceConfig(max_request_ranges=10, max_request_bytes=1024*1024),
#                                      "allow_read_errors_with_report": True}
#             )
# #             dak.necessary_columns(to_compute)
# 
#             #failed_chunks = get_failed_steps_for_fileset(
#             print(to_compute)
#             #outfile = uproot.dask_write(to_compute[0][samp], "root://cmseos.fnal.gov//store/user/dally/first_skim_muon_root_test/"+samp, compute=False, tree_name='Events')
#             dask.compute(to_compute)
#     
#             elapsed = time.time() - tic
#             print(f"Finished in {elapsed:.1f}s")
#         else:    
#             print(f"Folder already exists in /eos/cms/store/user/fiorendi/displacedTaus/skim/mutau/v3/")
#         
# 
#     client.shutdown()
#     cluster.close()
#     
#     
