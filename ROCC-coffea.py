import coffea
import uproot
import scipy
import numpy
import awkward as ak
import matplotlib.pyplot as plt
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, NanoAODSchema

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
min_pT = 20
max_eta = 2.4
max_dR2 = 0.3**2
dr_max = 0.4
# TODO replace threshold variable with loop over range
threshold = 0.5
#tau_selection = taus[
#        (numpy.abs(taus.eta) < max_eta) &
#        (taus.pt > min_pT)]

# Tau stuff
taus = events.GenPart[events.GenVisTau.genPartIdxMother]
stau_taus = taus[abs(taus.distinctParent.pdgId) == 1000015] # collection of all h-decay taus with stau parents

# Jet stuff
jets = events.Jet
scores = jets.disTauTag_score1
tau_jets = stau_taus.nearest(events.Jet, threshold = dr_max)
matched_scores = tau_jets.disTauTag_score1
selected_matched_scores = matched_scores > threshold


# --- debug --- # 
print(f"Jets\n{jets}\n\nTotal scores\n{scores}\n\n")
print(f"Tau jets\n{tau_jets}\n\nMatched scores\n{matched_scores}")
