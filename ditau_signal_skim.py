import sys
import argparse
import json
import numpy as np
from coffea import processor

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
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
from dask.distributed import Client, wait, progress
from distributed import Client

import ROOT
import warnings
warnings.filterwarnings("ignore", module="coffea") # Suppress annoying deprecation warnings for coffea vector, c.f. https://github.com/CoffeaTeam/coffea/blob/master/src/coffea/nanoevents/methods/candidate.py

# Can also be put in a utils file later
def delta_r_mask(first: ak.highlevel.Array, second: ak.highlevel.Array, threshold: float) -> ak.highlevel.Array:
            mval = first.metric_table(second)
            return ak.all(mval > threshold, axis=-1)

class MyProcessor(processor.ProcessorABC):
    def __init__(self):
        pass

    def process(self, events):
        # Determine if dataset is MC or Data
        is_MC = True if hasattr(events, "GenPart") else False
        if is_MC: sumWeights = ak.sum(events.genWeight)

        lumi = 0
        if not is_MC:
            lumi_mask = LumiMask("./tools/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt")
            lumi_data = LumiData("./tools/LumiData_2018_20200401.csv", is_inst_lumi=True)
            events = events[lumi_mask(events.run, events.luminosityBlock)]
            lumi_list = LumiList(*dask.compute(events.run, events.luminosityBlock))
            lumi = lumi_data.get_lumi(lumi_list)/10**9 # convert from inverse microbarn to inverse femtobarn


        events['Jet'] = events.GenVisTau.nearest(events.Jet, threshold = 0.3)

        good_jet_mask = (
            (events.Jet.pt > 20)
            & (abs(events.Jet.eta) < 2.4) 
            & (events.Jet.disTauTag_score1 > 0.9)
            & (events.Jet.genJetIdx > -1)
            & (events.Jet.genJetIdx < ak.num(events.GenJet.pt))
        )
        
        events['Jet'] = ak.drop_none(events.Jet[good_jet_mask])
        num_good_jets = ak.count_nonzero(good_jet_mask, axis=1)
        events = events[num_good_jets >= 2]
        print ("Number of ", events.metadata['dataset'], " events after jet cut:", ak.num(events, axis=0).compute())

        
        events['Jet'] = events.Jet[ak.argsort(events.Jet.pt, ascending=False)] ## why not order by tagger?
        leadingjet = ak.firsts(events.Jet)
        
        
        if is_MC: weights = events.genWeight / sumWeights 
        else: weights = ak.ones_like(events.event) # Classic move to create a 1d-array of ones, the appropriate weight for data

        out = ak.zip({
            "jet_pt": events.Jet.pt.compute(), 
            "jet_eta": events.Jet.eta.compute(),
            "jet_phi": events.Jet.phi.compute(),
            "jet_disTauTag_score1": events.Jet.disTauTag_score1.compute(),

            "deta": events.Jet.eta[:0].compute() - events.Jet.eta[:1].compute(),
            "dphi": events.Jet.phi[:0].delta_phi(events.Jet.phi[:1]).compute(),
            "dR" : events.Jet[:0].delta_r(events.Jet[:1]).compute(),
            

            "MET_pT": events.MET.pt.compute(),
            "NJets": ak.num(events.Jet).compute(),

            "weight": weights.compute(), #*valsf,
            "intLumi": ak.ones_like(weights).compute()*lumi,
        }, depth_limit = 1)

        out = dak.from_awkward(out, npartitions = 1)

        
        skim = dak.to_parquet(out, 'my_skim_electron_' + events.metadata['dataset'], compute=False)
        return skim, #muons_out, jets_out

    
    def postprocess(self, accumulator):
        pass

fileset = {
    'Stau_100_100mm': {
        "files": {
            'Staus_M_100_100mm_13p6TeV_Run3Summer22_lpcdisptau_NanoAOD_ExtraDisMuonBranches.root': "Events",
        }
    },
}

dataset_runnable, dataset_updated = preprocess(
    fileset,
    align_clusters=False,
    step_size=100_000,
    # step_size=100_000,
    files_per_batch=1,
    skip_bad_files=True,
    save_form=False,
)
to_compute = apply_to_fileset(
                MyProcessor(),
                max_chunks(dataset_runnable, 100),
                schemaclass=NanoAODSchema
            )
(out,) = dask.compute(to_compute)
print(out)
