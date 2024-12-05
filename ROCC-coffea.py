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

# Prompt and signal selection
#prompt_sig_mask = events[events.GenPart.hasFlags(['isPrompt', 'isLastCopy'])]
#dR2_mask = events[(
#    (events.Jet.phi - events.GenPart.phi)**2 +
#    (events.Jet.eta - events.GenPart.eta)**2)
#    < 0.4**2]

dr2_mask = events.Jet.nearest(events.GenPart, threshold = 0.3) 
print(f"Jet phi {events.Jet.phi[0].compute()}")
print(f"Jet eta {events.Jet.eta[0].compute()}")
print(f"GenPart phi {events.GenPart.phi[0].compute()}")
print(f"GenPart eta {events.GenPart.eta[0].compute()}") 
print(f"Jet score {events.Jet.disTauTag_score1[0].compute()}")
print(dr2_mask.pdgId[0].compute())
print(dr2_mask.genPartIdxMother[0].compute())
print(events.GenPart.pdgId[0][dr2_mask.genPartIdxMother[0]].compute())

# Lepton selection
electrons = events.Electron
muons = events.Muon
taus = events.Tau

electron_selection = electrons[
        (numpy.abs(electrons.eta) < max_eta) &
        (electrons.pt > min_pT)]
muon_selection = muons[
        (numpy.abs(muons.eta) < max_eta) &
        (muons.pt > min_pT)]
tau_selection = taus[
        (numpy.abs(taus.eta) < max_eta) &
        (taus.pt > min_pT)]

# debug
#print("Prompt and signal mask \n", prompt_sig_mask, "\n")
#print("dR^2 mask \n", dR2_mask, "\n")
#print("Electron selection \n", electron_selection, "\n")
#print("Muon selection \n", muon_selection, "\n")
#print("Tau selection \n ", tau_selection)
