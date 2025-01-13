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
#fname = "/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_1-1.root" # background

# Pass dataset info to coffea object
events = NanoEventsFactory.from_root(
    {fname: "Events"},
    schemaclass=NanoAODSchema,
    metadata={"dataset": "signal"},
    delayed = False).events()

# Selection cut parameters
#TODO implement cuts
min_pT = 20
max_eta = 2.4
max_dR2 = 0.3**2
dr_max = 0.4
resolution = 4 # no. of scores checked, 50 is probably a good value once functionality is confirmed

taus = events.GenPart[events.GenVisTau.genPartIdxMother] # hadronically-decaying taus
stau_taus = taus[abs(taus.distinctParent.pdgId) == 1000015] # h-decay taus with stau parents

jets = events.Jet
scores = jets.disTauTag_score1
tau_jets = stau_taus.nearest(events.Jet, threshold = dr_max) # jets dr-matched to stau_taus
matched_scores = tau_jets.disTauTag_score1 # scores of matched jets

fake_rates = []
efficiencies = []

for score_threshold in range(0, resolution+1):
    threshold = score_threshold / resolution
    passing_scores_mask = matched_scores >= threshold
    passing_scores = matched_scores[passing_scores_mask]

    passing_jet_mask = scores >= threshold
    passing_jets = jets[passing_jet_mask]

    true_tau_jet_mask = ak.any(jets.jetId[:, None] == tau_jets.jetId, axis=-1)
    true_tau_jets = jets[true_tau_jet_mask]

    true_passing_mask = ak.any(passing_jets.jetId[:, None] == true_tau_jets.jetId, axis=-1)
    true_passing_jets = jets[true_passing_mask]

    false_tau_jets = jets[~true_tau_jet_mask]
    false_passing_mask = ak.any(passing_jets.jetId[:, None] == false_tau_jets.jetId, axis=-1)
    false_passing_jets = jets[false_passing_mask]

    # --- Totals --- #
    total_passing_scores = ak.sum(ak.num(matched_scores))
    total_scores = ak.sum(ak.num(scores))
    total_true_tau_jets = ak.sum(ak.num(true_tau_jets))
    total_true_passing_jets = ak.sum(ak.num(true_passing_jets))
    total_false_passing_jets = ak.sum(ak.num(false_passing_jets))

    # --- Results --- #
    efficiency = total_true_passing_jets / total_true_tau_jets
    fake_rate = total_false_passing_jets / total_scores

    efficiencies.append(efficiency)
    fake_rates.append(fake_rate)

    # --- debug --- #
    print(f"A threshold of {threshold} yields an efficiency of {efficiency} and a fake rate of {fake_rate}")

print("List of efficiencies", efficiencies)
print("List of fake rates", fake_rates)
