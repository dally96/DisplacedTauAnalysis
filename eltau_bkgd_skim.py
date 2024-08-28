import sys
import argparse
import json
import numpy as np
from coffea import processor

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, PFNanoAODSchema 
 
from coffea.dataset_tools import (
    apply_to_fileset,
    max_chunks,
    preprocess,
)
from coffea.lumi_tools import LumiData, LumiList, LumiMask
import awkward as ak
import dask
from dask import array as da
from dask import config as cfg
cfg.set({'distributed.scheduler.worker-ttl': None}) # Check if this solves some dask issues
import dask_awkward as dak
# from dask_jobqueue import HTCondorCluster
from dask.distributed import Client, wait, progress, LocalCluster
from fileset import *

import ROOT
import warnings
import logging
warnings.filterwarnings("ignore", module="coffea") # Suppress annoying deprecation warnings for coffea vector, c.f. https://github.com/CoffeaTeam/coffea/blob/master/src/coffea/nanoevents/methods/candidate.py

# Can also be put in a utils file later
def delta_r_mask(first: ak.highlevel.Array, second: ak.highlevel.Array, threshold: float) -> ak.highlevel.Array:
            mval = first.metric_table(second)
            return ak.all(mval > threshold, axis=-1)

PFNanoAODSchema.mixins["LowPtElectron"] = "Electron"
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

            
            # Define the "good muon" condition for each muon per event
            good_electron_mask = (
                (events.Electron.pt > 20)
                & (abs(events.Electron.eta) < 2.4) 
            )
            logger.info("Defined good electrons")
    
            events['Electron'] = ak.drop_none(events.Electron[good_electron_mask])
            logger.info("Applied mask to Electron")
            num_evts = ak.num(events, axis=0)
            logger.info("Counted the number of original events")
            num_good_electrons = ak.count_nonzero(good_electron_mask, axis=1)
            logger.info("Counted the number of events with good electrons")
            events = events[num_good_electrons >= 1]
            logger.info("Counted the number of events with one or more good electrons")
            logger.info("Cut electrons")
    
            good_jet_mask = (
                (events.Jet.pt > 20)
                & (abs(events.Jet.eta) < 2.4) 
            )
            logger.info("Defined good jets")
    
            events['Jet'] = ak.drop_none(events.Jet[good_jet_mask])
            num_good_jets = ak.count_nonzero(good_jet_mask, axis=1)
            events = events[num_good_jets >= 1]
            logger.info("Cut jets")
            #print("Number of remaining events:", ak.num(events.Jet.pt, axis=0).compute())
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
    
            events['Electron'] = events.Electron[ak.argsort(events.Electron.pt, ascending=False)]
            events['Jet'] = events.Jet[ak.argsort(events.Jet.pt, ascending=False)] ## why not order by tagger?
            leadingel = ak.firsts(events.Electron)
            leadingjet = ak.firsts(events.Jet)
            logger.info("Defined leading electrons and jets")
    
   
            if is_MC: weights = events.genWeight / sumWeights 
            ele: weights = ak.ones_like(events.event) # Classic move to create a 1d-array of ones, the appropriate weight for data
            logger.info("Defined weights")
            out = ak.zip({
                "electron_pt": events.Electron.pt,
                "electron_eta": events.Electron.eta,
                "electron_phi": events.Electron.phi,
                "electron_charge": events.Electron.charge,
                "electron_dxy": events.Electron.dxy,
                "electron_dxyErr": events.Electron.dxyErr,
                "electron_dz": events.Electron.dz,
                "electron_dzErr": events.Electron.dzErr,
    
                "jet_pt": events.Jet.pt, 
                "jet_eta": events.Jet.eta,
                "jet_phi": events.Jet.phi,
                #"jet_d0" : ak.flatten(events.Jet.constituents.pf[ak.argmax(events.Jet.constituents.pf[charged_sel].pt, axis=2, keepdims=True)].d0, axis=-1).compute(),
                "jet_score": events.Jet.disTauTag_score1,
                "jet_partonFlavor": events.Jet.partonFlavour,

                "leadingjet_pt": leadingjet.pt,
                "leadingjet_eta": leadingjet.eta,
                "leadingjet_phi": leadingjet.phi,
                "leadingjet_score": leadingjet.disTauTag_score1,
                "leadingjet_partonFlavor": leadingjet.partonFlavour,

                "leadingelectron_pt": leadingel.pt,
                "leadingelectron_eta": leadingel.eta,
                "leadingelectron_phi": leadingel.phi,
                "leadingelectron_charge": leadingel.charge,
                "leadingelectron_dxy": leadingel.dxy,
                "leadingelectron_dxyErr": leadingel.dxyErr,
                "leadingelectron_dz": leadingel.dz,
                "leadingelectron_dzErr": leadingel.dzErr,
    
                "deta": leadingel.eta - leadingjet.eta,
                "dphi": leadingel.delta_phi(leadingjet),
                "dR" : leadingel.delta_r(leadingjet),
                
                "MET_pT": events.MET.pt,
                "NJets": ak.num(events.Jet),
                "NGoodElectrons": ak.num(events.Electron),
                "weight": weights, #*valsf,
                "intLumi": ak.ones_like(weights)*lumi,
            }, depth_limit = 1)
            logger.info("Send only some tuples out")
    
           # out = dak.from_awkward(out, npartitions = 3)
            logger.info("Prepare tuple to be written")
            logger.info("Write tuple out")
            skim = dak.to_parquet(out, 'my_skim_electron_' + events.metadata['dataset'], compute=False)
            return skim
            logger.info("Tuple written outg to parquet")
        
        def postprocess(self, accumulator):
            pass
   
    
    dataset_runnable, dataset_updated = preprocess(
        fileset,
        align_clusters=False,
        step_size=1_000_000_000,
        files_per_batch=1,
        skip_bad_files=True,
        save_form=False,
    )
    to_compute = apply_to_fileset(
                    MyProcessor(),
                    max_chunks(dataset_runnable, 10000000),
                    schemaclass=PFNanoAODSchema
                )
    (out,) = dask.compute(to_compute)
    print(out)

if __name__ == "__main__":
    main()

