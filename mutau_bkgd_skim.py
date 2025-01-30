import sys
import uproot
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
            
    
            events['DisMuon'] = events.DisMuon[ak.argsort(events.DisMuon.pt, ascending=False)]
            events['Jet'] = events.Jet[ak.argsort(events.Jet.pt, ascending=False)] ## why not order by tagger?
            leadingmu = ak.firsts(events.DisMuon)
            leadingjet = ak.firsts(events.Jet)
            logger.info("Defined leading muons and jets")
            events['nDisMuon'] = dak.num(events.DisMuon)
            events['nJet'] = dak.num(events.Jet)
    
            if is_MC: weights = events.genWeight / sumWeights 
            else: weights = ak.ones_like(events.event) # Classic move to create a 1d-array of ones, the appropriate weight for data
            logger.info("Defined weights")
            out_dict = {"DisMuon": events.DisMuon}
            #muon_vars = [ 
            #            "pt",
            #            "eta",
            #            "phi",
            #            "charge",
            #            "dxy",
            #            "dxyErr",
            #            "dz",
            #            "dzErr",
            #            "looseId",
            #            "mediumId",
            #            "tightId",
            #            "pfRelIso03_all",
            #            "pfRelIso03_chg",
            #            "pfRelIso04_all",
            #            ]
            #jet_vars = [
            #            "pt",
            #            "eta",
            #            "phi",
            #            "disTauTag_score1",
            #            ]
            #for branch in muon_vars:
            #    out_dict["DisMuon"][branch] = events["DisMuon"][branch]
            #for branch in jet_vars:
            #    out_dict["Jet"][branch] = events["Jet"][branch]
                
    
            out = ak.zip({
                "DisMuon_pt": events.DisMuon.pt,
                "DisMuon_eta": events.DisMuon.eta,
                "DisMuon_phi": events.DisMuon.phi,
                "DisMuon_charge": events.DisMuon.charge,
                "DisMuon_dxy": events.DisMuon.dxy,
                "DisMuon_dxyErr": events.DisMuon.dxyErr,
                "DisMuon_dz": events.DisMuon.dz,
                "DisMuon_dzErr": events.DisMuon.dzErr,
                "DisMuon_ntrk": events.DisMuon.nTrackerLayers,
                "DisMuon_looseId": events.DisMuon.looseId,
                "DisMuon_mediumId": events.DisMuon.mediumId,
                "DisMuon_tightId": events.DisMuon.tightId,
                "DisMuon_pfRelIso03_all": events.DisMuon.pfRelIso03_all,
                "DisMuon_pfRelIso03_chg": events.DisMuon.pfRelIso03_chg,
                "DisMuon_pfRelIso04_all": events.DisMuon.pfRelIso04_all,
    
                "Jet_pt": events.Jet.pt, 
                "Jet_eta": events.Jet.eta,
                "Jet_phi": events.Jet.phi,
                #"jet_d0": ak.flatten(events.Jet.constituents.pf[ak.argmax(events.Jet.constituents.pf[charged_sel].pt, axis=2, keepdims=True)].d0, axis=-1), 
                "Jet_score": events.Jet.disTauTag_score1, 
                "Jet_partonFlavor": events.Jet.partonFlavour,

                "event": events.event,
                "run": events.run,
                "luminosityBlock": events.luminosityBlock,
                
                "nDisMuon": events.nDisMuon,
                "nJet": events.nJet,
    
            ##    "LeadingJet_pt": leadingjet.pt,
            #    "LeadingJet_eta": leadingjet.eta,
            #    "LeadingJet_phi": leadingjet.phi,
            #    "LeadingJet_score": leadingjet.disTauTag_score1,
            #    "LeadingJet_partonFlavor": leadingjet.partonFlavour,

            #    "LeadingMuon_pt": leadingmu.pt,
            #    "LeadingMuon_eta": leadingmu.eta,
            #    "LeadingMuon_phi": leadingmu.phi,
            #    "LeadingMuon_charge": leadingmu.charge,
            #    "LeadingMuon_dxy": leadingmu.dxy,
            #    "LeadingMuon_dxyErr": leadingmu.dxyErr,
            #    "LeadingMuon_dz": leadingmu.dz,
            #    "LeadingMuon_dzErr": leadingmu.dzErr,
            #    "LeadingMuon_ntrk": leadingmu.nTrackerLayers,

            #    "deta": leadingmu.eta - leadingjet.eta,
            #    "dphi": leadingmu.delta_phi(leadingjet),
            #    "dR" : leadingmu.delta_r(leadingjet),
    
            #    "MET_pT": events.MET.pt,
            #    "NJets": ak.num(events.Jet),
            #    "NGoodMuons": ak.num(events.DisMuon),
            #    "weight": weights, #*valsf,
            #    "intLumi": ak.ones_like(weights)*lumi,
            #    "generator_scalePDF": events.Generator.scalePDF,
            }, depth_limit = 1)
            
            #out = dak.Record(out_dict, "out")
            #out_dict = dak.Array(out_dict)
            #out = dak.from_awkward(out, npartitions = 1)
            #out_dict["DisMuon"] = ak.zip(out_dict["DisMuon"], depth_limit = 1)
            #out_dict["Jet"] = ak.zip(out_dict["Jet"], depth_limit = 1)
            out_dict["event"] = events.event
            out_dict["run"] = events.run
            out_dict["luminosityBlock"] = events.luminosityBlock
            out_dict["nDisMuon"] = dak.num(events.DisMuon)
            out_dict = ak.zip(out_dict, depth_limit = 1)
            
            #out_array = ak.Array([events.DisMuon, events.nDisMuon, events.event, events.run, events.luminosityBlock])

            logger.info("Prepare tuple to be written")
            logger.info("Write tuple out")
            #print("out is ", out.compute())
            #skim = pq.write_table(out_table, "output.parquet")
            skim = dak.to_parquet(events, 'my_skim_muon_' + events.metadata['dataset'], compute=False)
            return skim
            logger.info("Tuple written outg to parquet")
   
        def postprocess(self, accumulator):
            pass

    dataset_runnable, dataset_updated = preprocess(
        fileset,
        align_clusters=False,
        step_size=100_000_000,
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
