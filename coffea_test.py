import uproot
import scipy
import awkward as ak
import numpy as np
from coffea import processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, PFNanoAODSchema
from coffea.nanoevents.methods.vector import LorentzVector
from hist import Hist
import matplotlib.pyplot as plt
import vector

NanoAODSchema.warn_missing_crossrefs = False

fname = "/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215454/0000/nanoaod_output_1.root"

events = NanoEventsFactory.from_root(
    {fname: "Events"},
    schemaclass=PFNanoAODSchema,
    metadata={"dataset": "signal"},
    delayed = False).events()

# Tau stuff
taus = events.GenPart[events.GenVisTau.genPartIdxMother]
stau_taus = taus[abs(taus.distinctParent.pdgId) == 1000015] # collection of all h-decay taus with stau parents

# Jet stuff
jets = events.Jet
scores = jets.disTauTag_score1
tau_jets = stau_taus.nearest(events.Jet, threshold = 0.4)
matched_scores = tau_jets.disTauTag_score1

print('taus fields =', taus.fields)
print('stau_taus fields =', stau_taus.fields)
print('tau_jets fields =', tau_jets.fields)
print('matched_scores fields =', matched_scores.fields)
