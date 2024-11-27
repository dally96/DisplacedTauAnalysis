import coffea
import uproot
import scipy
import numpy
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, PFNanoAODSchema
from coffea.analysis_tools import PackedSelection

# --- Selection Cut Parameters --- #
min_pT = 20
max_eta = 2.4

NanoAODSchema.warn_missing_crossrefs = False

fname = "/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215454/0000/nanoaod_output_1.root"

events = NanoEventsFactory.from_root(
    {fname: "Events"},
    schemaclass=PFNanoAODSchema,
    metadata={"dataset": "signal"},
    delayed = False).events()

gpart = events.GenPart
electron_selection = gpart[ (abs(gpart.pdgId) == 11) &\
        (gpart.pt > min_pT) &\
        (gpart.eta < max_eta)&\
        (gpart.hasFlags("isLastCopy")) ]
muon_selection = gpart[ (abs(gpart.pdgId) == 13) &\
        (gpart.pt > min_pT) &\
        (gpart.eta < max_eta)&\
        (gpart.hasFlags("isLastCopy")) ]
tau_selection = gpart[ (abs(gpart.pdgId) == 15) &\
        (gpart.pt > min_pT) &\
        (gpart.eta < max_eta)&\
        (gpart.hasFlags("isLastCopy")) ]
print("ELECTRONS")
print(electron_selection)
print("MUONS")
print(muon_selection)
print("TAUS")
print(tau_selection)
