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
signal_fname = "/eos/user/d/dally/DisplacedTauAnalysis/SMS-TStauStau_MStau-100_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_NanoAOD.root" # signal
bg_fname = "/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_1-1.root" # background

# Pass dataset info to coffea objects
signal_events = NanoEventsFactory.from_root(
    {signal_fname: "Events"},
    schemaclass=NanoAODSchema,
    metadata={"dataset": "signal"},
    delayed = False).events()
bg_events = NanoEventsFactory.from_root(
    {bg_fname: "Events"},
    schemaclass=NanoAODSchema,
    metadata={"dataset": "background"},
    delayed = False).events()

# Selection cut parameters
#TODO implement cuts
min_pT = 20
max_eta = 2.4
max_dR2 = 0.3**2
dr_max = 0.4
score_increment_scale_factor = 50

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
signal_jets = signal_events.Jet
cut_signal_jets = apply_cuts(signal_jets)
true_tau_jets = stau_taus.nearest(cut_signal_jets, threshold = dr_max) # jets dr-matched to stau_taus
matched_signal_scores = true_tau_jets.disTauTag_score1

# Background processing
bg_jets = bg_events.Jet
cut_bg_jets = apply_cuts(bg_jets)
bg_scores = cut_bg_jets.disTauTag_score1
false_tau_jets = cut_bg_jets # No staus present in bg
matched_bg_scores = bg_scores

# Jet totals
total_true_tau_jets = ak.sum(ak.num(true_tau_jets))
total_false_tau_jets = ak.sum(ak.num(false_tau_jets))

total_jets = (
    ak.sum(ak.num(cut_signal_jets)) +
    ak.sum(ak.num(cut_bg_jets)) )

# ROC calculations
thresholds = []
fake_rates = []
efficiencies = []

for increment in range(0, score_increment_scale_factor+1):
    threshold = increment / score_increment_scale_factor

    passing_signal_mask = matched_signal_scores >= threshold
    true_passing_jets = true_tau_jets[passing_signal_mask]

    passing_bg_mask = matched_bg_scores >= threshold
    false_passing_jets = false_tau_jets[passing_bg_mask]

    # --- Totals --- #
    total_true_passing_jets = ak.sum(ak.num(true_passing_jets))
    total_false_passing_jets = ak.sum(ak.num(false_passing_jets))

    # --- Results --- #
    efficiency = total_true_passing_jets / total_true_tau_jets
    fake_rate = total_false_passing_jets / total_jets

    thresholds.append(threshold)
    fake_rates.append(fake_rate)
    efficiencies.append(efficiency)

# Plot stuff
fig, ax = plt.subplots()
roc = ax.plot(fake_rates, efficiencies)

plt.xlabel(r"Fake rate $\left(\frac{false\_passing\_jets}{total\_jets}\right)$")
plt.ylabel(r"Tau tagger efficiency $\left(\frac{true\_passing\_jets}{total\_true\_jets}\right)$")

plt.grid()
plt.savefig('tau-tagger-rocc.pdf')
plt.savefig('tau-tagger-rocc.png')
