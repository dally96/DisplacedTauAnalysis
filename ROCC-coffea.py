import coffea
import ROOT
import uproot
import scipy
import numpy
import awkward as ak
from coffea import processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, PFNanoAODSchema
from coffea.analysis_tools import PackedSelection
from coffea.dataset_tools import (
    apply_to_fileset,
    max_chunks,
    preprocess,
)
import dask 
import dask_awkward as dak

from fileset import *
# Silence obnoxious warning
NanoAODSchema.warn_missing_crossrefs = False
PFNanoAODSchema.mixins["DisMuon"] = "Muon"

SAMP = [ 
        'Stau_100_100mm',
        'QCD50_80',
        'QCD80_120',
        'QCD120_170',
        'QCD170_300',
        'QCD300_470',
        'QCD470_600',
        'QCD600_800',
        'QCD800_1000',
        'QCD1000_1400',
        'QCD1400_1800',
        'QCD1800_2400',
        'QCD2400_3200',
        'QCD3200',
      ]

selections = { 
              "muon_pt":                    30, ##GeV
              "muon_eta":                   1.5,
              "muon_ID":                    "muon_tightId",
              "muon_dxy_prompt_max":        50E-4, ##cm
              "muon_dxy_prompt_min":        0E-4, ##cm
              "muon_dxy_displaced_min":     100E-4, ##cm
              "muon_dxy_displaced_max":     10, ##cm
 
              "jet_score":                  0.9, 
              "jet_pt":                     32, ##GeV
 
              "MET_pT":                     105, ##GeV
             }



# Import dataset
fname = "SMS-TStauStau_MStau-100_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_NanoAOD.root"

# Can also be put in a utils file later
def delta_r_mask(first: ak.highlevel.Array, second: ak.highlevel.Array, threshold: float) -> ak.highlevel.Array:
            mval = first.metric_table(second)
            return ak.all(mval > threshold, axis=-1)

class IsoROCC(processor.ProcessorABC): 
    def __init__(self): 
        pass 

    def initialize_eff(self):
        eff = {}
        eff["pfRelIso03_all"] = ROOT.TEfficiency("pfRelIso03_all_eff", ";pfRelIso03_all;Efficiency", 100, 0, 10)
        eff["pfRelIso03_chg"] = ROOT.TEfficiency("pfRelIso03_chg_eff", ";pfRelIso03_chg;Efficiency", 100, 0, 10) 
        eff["pfRelIso04_all"] = ROOT.TEfficiency("pfRelIso04_all_eff", ";pfRelIso04_all;Efficiency", 100, 0, 10)
        return eff

    def process(self, events):
     # Define the "good muon" condition for each muon per event
        good_muon_mask = (
             (events.DisMuon.pt > 20)
            & (abs(events.DisMuon.eta) < 2.4) # Acceptance of the CMS muon system
        )      
        events['DisMuon'] = ak.drop_none(events.DisMuon[good_muon_mask])
        num_evts = ak.num(events, axis=0)
        num_good_muons = ak.count_nonzero(good_muon_mask, axis=1)
        events = events[num_good_muons >= 1]
        print("Processed muon preselection")
              
        # Perform the overlap removal with respect to muons, electrons and photons, dR=0.4
        events['Jet'] = events.Jet[delta_r_mask(events.Jet, events.Photon, 0.4)]
        events['Jet'] = events.Jet[delta_r_mask(events.Jet, events.Electron, 0.4)]
        events['Jet'] = events.Jet[delta_r_mask(events.Jet, events.DisMuon, 0.4)]
              
        good_jet_mask = (
            (events.Jet.pt > 20)
            & (abs(events.Jet.eta) < 2.4) 
        )     
              
        events['Jet'] = events.Jet[good_jet_mask]
        num_good_jets = ak.count_nonzero(good_jet_mask, axis=1)
        events = events[num_good_jets >= 1]
        print("Processed jet preselection")
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
        print("Processed noise mask")
        better_muons = (
                         (events.DisMuon.pt > selections["muon_pt"])
                       & (events.DisMuon.tightId == 1)
                       & (abs(events.DisMuon.dxy) > selections["muon_dxy_displaced_min"])
                       & (abs(events.DisMuon.dxy) < selections["muon_dxy_displaced_max"])
                       )

        better_jets =  ( 
                         (events.Jet.disTauTag_score1 > selections["jet_score"])
                       & (events.Jet.pt > selections["jet_pt"])
                       )

        better_events = (events.MET.pt > selections["MET_pT"])

        num_better_muons = ak.num(events.DisMuon[better_muons])
        num_better_jets  = ak.num(events.Jet[better_jets])
        
        better_muon_event_mask = num_better_muons > 0 
        better_jet_event_mask  = num_better_jets  > 0

        events = events[better_muon_event_mask & better_jet_event_mask & better_events]
        print("Applied current event selections")
        RecoMuons = events.DisMuon[(abs(events.GenPart[events.DisMuon.genPartIdx].pdgId) == 13)
                  & (abs(events.GenPart[events.DisMuon.genPartIdx].distinctParent.distinctParent.pdgId) == 1000015)
                  ]
        
        if "QCD" in str(dataset_runnable): 
            RecoMuons = events.DisMuon

        tot_muons = ak.num(dak.flatten(RecoMuons.pt, axis = None), axis = 0)
        return tot_muons
#        eff = self.intialize_eff()
    def postprocess(self):
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
             IsoROCC(),
             max_chunks(dataset_runnable, 10000000),
             schemaclass=PFNanoAODSchema,
)
(out,) = dask.compute(to_compute)
print(out)



 
