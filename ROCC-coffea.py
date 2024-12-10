import coffea
import uproot
import scipy
import numpy
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, PFNanoAODSchema
from coffea.analysis_tools import PackedSelection

# Silence obnoxious warning
NanoAODSchema.warn_missing_crossrefs = False

# Import dataset
fname = "/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215454/0000/nanoaod_output_1.root"

# Pass dataset info to coffea object
events = NanoEventsFactory.from_root(
    {fname: "Events"},
    schemaclass=PFNanoAODSchema,
    metadata={"dataset": "signal"},
    delayed = True).events()

# Selection cut parameters
min_pT = 20
max_eta = 2.4
dr2 = 0.3**2

staus = events.GenPart[abs(events.GenPart.pdgID) == 1000015]
stau_taus = staus.distinctChildren


# Prompt and signal selection
dR2_mask = events.Jet.nearest(events.GenPart, threshold = 0.3) 

# Lepton selection
electrons = events.Electron
muons = events.DisMuon
jets = events.Jets

electron_selection = electrons[
        (numpy.abs(electrons.eta) < max_eta) &
        (electrons.pt > min_pT)]
muon_selection = muons[
        (numpy.abs(muons.eta) < max_eta) &
        (muons.pt > min_pT)]

# debug
