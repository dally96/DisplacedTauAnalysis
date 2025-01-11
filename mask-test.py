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

taus = events.Tau
htaus = events.GenPart[events.GenVisTau.genPartIdxMother]
stau_taus = htaus[abs(htaus.distinctParent.pdgId) == 1000015]
neg_mask = ~ak.any(taus.pt[:, None] == stau_taus.pt, axis=1)

neg_taus = taus[neg_mask]

print(f"Taus\n{taus}")
print(len(ak.flatten(taus)), "total taus")
print(f"\nHadronically-decaying taus\n{htaus}")
print(len(ak.flatten(htaus)), "total h-decay taus")
print(f"\nH-decay taus with stau parents\n{stau_taus}")
print(len(ak.flatten(stau_taus)), "total tau children of staus")
print(f"\nOther taus\n{neg_taus}")
print(len(ak.flatten(neg_taus)), "total other taus")
