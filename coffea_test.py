import uproot
import scipy
import awkward as ak
import numpy as np
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import matplotlib.pyplot as plt
import vector

NanoAODSchema.warn_missing_crossrefs = False

fname = "/eos/user/d/dally/DisplacedTauAnalysis/SMS-TStauStau_MStau-100_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_NanoAOD.root" # signal
#fname = "/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_1-1.root" # background

events = NanoEventsFactory.from_root(
    {fname: "Events"},
    schemaclass=NanoAODSchema,
    metadata={"dataset": "signal"},
    delayed = False).events()

dr_max = 0.4 

taus = events.GenPart[events.GenVisTau.genPartIdxMother] # hadronically-decaying taus
stau_taus = taus[abs(taus.distinctParent.pdgId) == 1000015] # h-decay taus with stau parents

jets = events.Jet
scores = jets.disTauTag_score1
tau_jets = stau_taus.nearest(events.Jet, threshold = dr_max) # jets dr-matched to stau_taus
#matched_scores = tau_jets.disTauTag_score1 # scores of matched jets

true_tau_jet_mask = ak.any(jets.genJetIdxG[:, None] == tau_jets.genJetIdxG, axis=-1)
true_tau_jets = jets[true_tau_jet_mask]
false_tau_jets = jets[~true_tau_jet_mask]

print("events.GenVisTau.genPartIdxMother")
print(events.GenVisTau.genPartIdxMother)
