import coffea
import uproot
import scipy
import numpy
import awkward as ak
import matplotlib.pyplot as plt
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import PackedSelection

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

jagged_htaus = events.GenPart[events.GenVisTau.genPartIdxMother]
valid_htaus_mask = jagged_htaus >= 0
htaus = jagged_htaus[valid_htaus_mask]

selection = PackedSelection()
selection.add("stau_taus", htaus[abs(htaus.distinctParent.pdgId)] == 1000015)
selection.add("event_cut", selection.require(stau_taus=True))
selected_events = events[selection.all("event_cut")]

#print(f"Taus\n{taus}")
#print(len(ak.flatten(taus)), "total taus")
#print(f"\nHadronically-decaying taus\n{htaus}")
#print(len(ak.flatten(htaus)), "total h-decay taus")
print(f"\nH-decay taus with stau parents\n{selected_events}")
print(len(ak.flatten(selected_events)), "total tau children of staus")
#print(f"\nOther taus\n{neg_taus}")
#print(len(ak.flatten(neg_taus)), "total other taus")
