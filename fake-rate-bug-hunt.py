import coffea
import uproot
import scipy
import numpy
import awkward as ak
import matplotlib.pyplot as plt
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

# Silence obnoxious warning
NanoAODSchema.warn_missing_crossrefs = False

# Import dataset
fname = "/eos/user/d/dally/DisplacedTauAnalysis/SMS-TStauStau_MStau-100_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_NanoAOD.root" # signal

# Pass dataset info to coffea object
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
matched_scores = tau_jets.disTauTag_score1 # scores of matched jets

true_tau_jet_mask = ak.any(jets.genJetIdxG[:, None] == tau_jets.genJetIdxG, axis=-1)
true_tau_jets = jets[true_tau_jet_mask]
false_tau_jets = jets[~true_tau_jet_mask]

threshold = 0.5
passing_scores_mask = matched_scores >= threshold
passing_scores = matched_scores[passing_scores_mask]

passing_jet_mask = scores >= threshold
passing_jets = jets[passing_jet_mask]

true_passing_mask = ak.any(passing_jets.genJetIdxG[:, None] == true_tau_jets.genJetIdxG, axis=-1)
true_passing_jets = jets[true_passing_mask]

false_passing_mask = ak.any(passing_jets.genJetIdxG[:, None] == false_tau_jets.genJetIdxG, axis=-1)
false_passing_jets = jets[false_passing_mask]

# --- Totals --- #
total_passing_scores = ak.sum(ak.num(matched_scores))
total_scores = ak.sum(ak.num(scores))
total_true_tau_jets = ak.sum(ak.num(true_tau_jets))
total_true_passing_jets = ak.sum(ak.num(true_passing_jets))
total_false_tau_jets = ak.sum(ak.num(false_tau_jets))
total_false_passing_jets = ak.sum(ak.num(false_passing_jets))

# --- debug --- #
print("jets.genJetIdxG[:, None]")
print(jets.genJetIdxG[:, None])
