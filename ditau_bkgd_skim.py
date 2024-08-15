import sys
import argparse
import json
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
from distributed import Client
from fileset import *
import ROOT
import warnings
import logging
warnings.filterwarnings("ignore", module="coffea") # Suppress annoying deprecation warnings for coffea vector, c.f. https://github.com/CoffeaTeam/coffea/blob/master/src/coffea/nanoevents/methods/candidate.py

# Can also be put in a utils file later
def delta_r_mask(first: ak.highlevel.Array, second: ak.highlevel.Array, threshold: float) -> ak.highlevel.Array:
            mval = first.metric_table(second)
            return ak.all(mval > threshold, axis=-1)

def main(): 
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    client = Client()
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
    
            # Trigger requirement
            # events = events[events.HLT.IsoMu24 == True]
    
    
            good_jet_mask = (
                (events.Jet.pt > 20)
                & (abs(events.Jet.eta) < 2.4) 
            )
            logger.info("Defined good jets")

            events['Jet'] = ak.drop_none(events.Jet[good_jet_mask])
            num_good_jets = ak.count_nonzero(good_jet_mask, axis=1)
            events = events[num_good_jets >= 2]
            logger.info("Cut jets")
            #charged_sel = events.Jet.constituents.pf.charge != 0

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
            
            # Perform the overlap removal with respect to muons, electrons and photons, dR=0.4
            events['Jet'] = events.Jet[delta_r_mask(events.Jet, events.Photon, 0.4)]
            events['Jet'] = events.Jet[delta_r_mask(events.Jet, events.Electron, 0.4)]
            events['Jet'] = events.Jet[delta_r_mask(events.Jet, events.Muon, 0.4)]
            logger.info("Performed overlap removal") 
 
            if is_MC: weights = events.genWeight / sumWeights 
            else: weights = ak.ones_like(events.event) # Classic move to create a 1d-array of ones, the appropriate weight for data
    
    
            out = ak.zip({
                "jet_pt": events.Jet.pt, 
                "jet_eta": events.Jet.eta,
                "jet_phi": events.Jet.phi,
                #"jet_d0" : ak.flatten(events.Jet.constituents.pf[ak.argmax(events.Jet.constituents.pf[charged_sel].pt, axis=2, keepdims=True)].d0, axis=-1),
                "jet_score": events.Jet.disTauTag_score1,
    
                "MET_pT": events.MET.pt,
                "NJets": ak.num(events.Jet),
                "weight": weights, #*valsf,
                "intLumi": ak.ones_like(weights)*lumi,
            }, depth_limit = 1)
            logger.info("Send only some tuples out")  
            logger.info("Write tuple out") 
            skim = dak.to_parquet(out, 'my_skim_ditau_' + events.metadata['dataset'], compute=False)
            return skim
            logger.info("Tuple written out to parquet") 
        
        def postprocess(self, accumulator):
            pass


    dataset_runnable, dataset_updated = preprocess(
        fileset,
        align_clusters=False,
        step_size=100_000_000,
        # step_size=100_000,
        files_per_batch=1,
        skip_bad_files=True,
        save_form=False,
    )
    to_compute = apply_to_fileset(
                    MyProcessor(),
                    max_chunks(dataset_runnable, 1000000),
                    schemaclass=NanoAODSchema
                )
    (out,) = dask.compute(to_compute)
    print(out)

if __name__ == "__main__":
    main()
