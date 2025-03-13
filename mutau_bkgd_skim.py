import sys
import uproot
import uproot.exceptions
from uproot.exceptions import KeyInFileError
import argparse
import json
import pyarrow as pa
import pyarrow.parquet as pq
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
# from dask_jobqueue import HTCondorCluster
from dask.distributed import Client, wait, progress, LocalCluster
from fileset import *

import ROOT
import warnings
warnings.filterwarnings("ignore", module="coffea") # Suppress annoying deprecation warnings for coffea vector, c.f. https://github.com/CoffeaTeam/coffea/blob/master/src/coffea/nanoevents/methods/candidate.py
import logging 

# Can also be put in a utils file later
def delta_r_mask(first: ak.highlevel.Array, second: ak.highlevel.Array, threshold: float) -> ak.highlevel.Array:
            mval = first.metric_table(second)
            return ak.all(mval > threshold, axis=-1)

PFNanoAODSchema.mixins["DisMuon"] = "Muon"


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    cluster = LocalCluster()
    client = Client(cluster)

    cluster.scale(10)

    print(f'Dask dashboard available at: {client.dashboard_link}')

    class MyProcessor(processor.ProcessorABC):
        def __init__(self):
            pass
    
        def process(self, events):
            # Determine if dataset is MC or Data
            is_MC = True if hasattr(events, "GenPart") else False
            if is_MC: sumWeights = ak.sum(events.genWeight)
            logger.info("Starting process")
    
            lumi = 0
            if not is_MC:
                lumi_mask = LumiMask("./tools/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt")
                lumi_data = LumiData("./tools/LumiData_2018_20200401.csv", is_inst_lumi=True)
                events = events[lumi_mask(events.run, events.luminosityBlock)]
                lumi_list = LumiList(*dask.compute(events.run, events.luminosityBlock))
                lumi = lumi_data.get_lumi(lumi_list)/10**9 # convert from inverse microbarn to inverse femtobarn

            charged_sel = events.Jet.constituents.pf.charge != 0        
    
            events["Jet_dxy"] = dak.flatten(ak.drop_none(events.Jet.constituents.pf[ak.argmax(events.Jet.constituents.pf[charged_sel].pt, axis=2, keepdims=True)].d0), axis=-1)    


            # Define the "good muon" condition for each muon per event
            good_muon_mask = (
                #((events.DisMuon.isGlobal == 1) | (events.DisMuon.isTracker == 1)) & # Equivalent to loose ID cut without isPFcand requirement. Reduce background from non-prompt muons
                 (events.DisMuon.pt > 20)
                & (abs(events.DisMuon.eta) < 2.4) # Acceptance of the CMS muon system
            )
            logger.info("Defined good muons")
    
            events['DisMuon'] = ak.drop_none(events.DisMuon[good_muon_mask])
            logger.info("Applied mask to DisMuon")
            num_evts = ak.num(events, axis=0)
            logger.info("Counted the number of original events")
            num_good_muons = ak.count_nonzero(good_muon_mask, axis=1)
            logger.info("Counted the number of events with good muons")
            events = events[num_good_muons >= 1]
            logger.info("Counted the number of events with one or more good muons")
            logger.info("Cut muons")
    
            # Perform the overlap removal with respect to muons, electrons and photons, dR=0.4
            events['Jet'] = events.Jet[delta_r_mask(events.Jet, events.Photon, 0.4)]
            events['Jet'] = events.Jet[delta_r_mask(events.Jet, events.Electron, 0.4)]
            events['Jet'] = events.Jet[delta_r_mask(events.Jet, events.Muon, 0.4)]
            events['Jet'] = events.Jet[delta_r_mask(events.Jet, events.DisMuon, 0.4)]
            logger.info("Performed overlap removal")
    
            good_jet_mask = (
                (events.Jet.pt > 20)
                & (abs(events.Jet.eta) < 2.4) 
            )
            logger.info("Defined good jets")
            
            events['Jet'] = events.Jet[good_jet_mask]
            num_good_jets = ak.count_nonzero(good_jet_mask, axis=1)
            events = events[num_good_jets >= 1]
            logger.info("Cut jets")

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

            #Trigger Selection
            trigger_mask = (
                            events.HLT.PFMET120_PFMHT120_IDTight                                    |\
                            events.HLT.PFMET130_PFMHT130_IDTight                                    |\
                            events.HLT.PFMET140_PFMHT140_IDTight                                    |\
                            events.HLT.PFMETNoMu120_PFMHTNoMu120_IDTight                            |\
                            events.HLT.PFMETNoMu130_PFMHTNoMu130_IDTight                            |\
                            events.HLT.PFMETNoMu140_PFMHTNoMu140_IDTight                            |\
                            events.HLT.PFMET120_PFMHT120_IDTight_PFHT60                             |\
                            #events.HLT.MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight        |\
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

            meta = ak.Array([0], backend = "typetracer")
            event_counts = events.map_partitions(lambda part: ak.num(part, axis = 0), meta = meta)
            partition_counts = event_counts.compute()
            non_empty_partitions = [
                                    events.partitions[i] for i in range(len(partition_counts)) if partition_counts[i] > 0
                                   ]
            if non_empty_partitions:
                events = dak.concatenate(non_empty_partitions) 


            if is_MC: weights = events.genWeight / sumWeights 
            else: weights = ak.ones_like(events.event) # Classic move to create a 1d-array of ones, the appropriate weight for data
            logger.info("Defined weights")
       
             
            out_dict = {}
            muon_vars =  events.DisMuon.fields#[ 
                         #"pt",
                         #"eta",
                         #"phi",
                         #"charge",
                         #"dxy",
                         #"dxyErr",
                         #"dz",
                         #"dzErr",
                         #"looseId",
                         #"mediumId",
                         #"tightId",
                         #"pfRelIso03_all",
                         #"pfRelIso03_chg",
                         #"pfRelIso04_all",
                         #"tkRelIso",
                         #]
            jet_vars =   events.Jet.fields#[
                         #"pt",
                         #"eta",
                         #"phi",
                         #"disTauTag_score1",
                         #]

            gpart_vars = events.GenPart.fields#[
                         #"genPartIdxMother", 
                         #"statusFlags", 
                         #"pdgId",
                         #"status", 
                         #"eta", 
                         #"mass", 
                         #"phi", 
                         #"pt", 
                         #"vertexR", 
                         #"vertexRho", 
                         #"vx", 
                         #"vy", 
                         #"vz",
                         #]

            gvist_vars = events.GenVisTau.fields#[
                         #"genPartIdxMother", 
                         #"charge",
                         #"status", 
                         #"eta", 
                         #"mass", 
                         #"phi", 
                         #"pt", 
                         #]

            tau_vars   = events.Tau.fields                        
            MET_vars   = events.PFMET.fields  
     
            for branch in muon_vars:
                if branch[-1] == 'G': continue
                out_dict["DisMuon_"   + branch]  = dak.drop_none(events["DisMuon"][branch])
            for branch in jet_vars:
                if branch[-1] == 'G': continue
                out_dict["Jet_"       + branch]  = dak.drop_none(events["Jet"][branch])
            for branch in gpart_vars:
                if branch[-1] == 'G': continue
                out_dict["GenPart_"   + branch]  = dak.drop_none(events["GenPart"][branch])
            for branch in gvist_vars:
                if branch[-1] == 'G': continue
                out_dict["GenVisTau_" + branch]  = dak.drop_none(events["GenVisTau"][branch])
            for branch in tau_vars:
                if branch[-1] == 'G': continue
                out_dict["Tau_"       + branch]  = dak.drop_none(events["Tau"][branch])
            for branch in MET_vars: 
                if branch[-1] == 'G': continue
                out_dict["PFMET_"     + branch]  = dak.drop_none(events["PFMET"][branch])    

            out_dict["event"]           = dak.drop_none(events.event)
            out_dict["run"]             = dak.drop_none(events.run)
            out_dict["luminosityBlock"] = dak.drop_none(events.luminosityBlock)
            out_dict["Jet_dxy"]         = dak.drop_none(events["Jet_dxy"])
            out_dict["nDisMuon"]        = dak.num(events.DisMuon)
            out_dict["nJet"]            = dak.num(events.Jet)
            out_dict["nGenPart"]        = dak.num(events.GenPart)
            out_dict["nGenVisTau"]      = dak.num(events.GenVisTau)
            out_dict["nTau"]            = dak.num(events.Tau)


            logger.info(f"Filled dictionary")

            out_dict = dak.zip(out_dict, depth_limit = 1)
            logger.info(f"Dictionary zipped: {out_dict}")
            return uproot.dask_write(out_dict, "/eos/uscms/store/user/dally/first_skim_muon_root/" + events.metadata['dataset'], tree_name="Events")
            logger.info("Dictionary written to root files")


        def postprocess(self, accumulator):
            pass

    dataset_runnable, dataset_updated = preprocess(
        fileset,
        align_clusters=False,
        step_size=100_00,
        files_per_batch=1,
        skip_bad_files=True,
        save_form=False,
        file_exceptions=(OSError, KeyInFileError),
        allow_empty_datasets=False,
    )
    to_compute = apply_to_fileset(
                 MyProcessor(),
                 max_chunks(dataset_runnable, 1000000000000),
                 schemaclass=PFNanoAODSchema
    )
    (out,) = dask.compute(to_compute)
    print(out)

if __name__ == "__main__":
    main()
