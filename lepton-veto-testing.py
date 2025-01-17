import coffea
import uproot
import scipy
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

NanoAODSchema.mixins["DisMuon"] = "Muon"

# Silence obnoxious warning
NanoAODSchema.warn_missing_crossrefs = False

# Import dataset
signal_fname = "/eos/user/d/dally/DisplacedTauAnalysis/SMS-TStauStau_MStau-100_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_NanoAOD.root" # signal

# Pass dataset info to coffea objects
signal_events = NanoEventsFactory.from_root(
    {signal_fname: "Events"},
    schemaclass=NanoAODSchema,
    metadata={"dataset": "signal"},
    delayed = False).events()

min_pT = 20
max_eta = 2.4
max_dr = 0.3
max_lep_dr = 0.4

### --- debug --- ###
print(ak.sum(ak.num(signal_events.Jet)), "original jets, cut to ")

def delta_r_mask(first: ak.highlevel.Array, second: ak.highlevel.Array, threshold: float) -> ak.highlevel.Array:
            mval = first.metric_table(second)
            return ak.all(mval > threshold, axis=-1)

def apply_cuts(collection):
    pt_mask = collection.pt >= min_pT
    eta_mask = abs(collection.eta) < max_eta
    valid_mask = collection.genJetIdx > 0
    
    collection_length = ak.sum(ak.num(collection.partonFlavour))
    inclusion_mask = collection.genJetIdx < collection_length
    
    cut_collection = collection[
        pt_mask & eta_mask & valid_mask & inclusion_mask ]
    
    return cut_collection

# Signal processing
taus = signal_events.GenPart[signal_events.GenVisTau.genPartIdxMother] # hadronically-decaying taus
stau_taus = taus[abs(taus.distinctParent.pdgId) == 1000015] # h-decay taus with stau parents
# --- Lepton veto --- #
signal_events['Jet'] = signal_events.Jet[delta_r_mask(signal_events.Jet, signal_events.DisMuon, max_lep_dr)]
signal_events['Jet'] = signal_events.Jet[delta_r_mask(signal_events.Jet, signal_events.Photon, max_lep_dr)]
signal_events['Jet'] = signal_events.Jet[delta_r_mask(signal_events.Jet, signal_events.Electron, max_lep_dr)]
# --- Basic cuts --- #
cut_signal_jets = apply_cuts(signal_events.Jet)
true_tau_jets = stau_taus.nearest(cut_signal_jets, threshold = max_dr) # jets dr-matched to stau_taus
matched_signal_scores = true_tau_jets.disTauTag_score1

print(ak.sum(ak.num(signal_events.Jet)), "jets after the lepton veto and")
print(ak.sum(ak.num(cut_signal_jets)), "jets after the basic cuts")
