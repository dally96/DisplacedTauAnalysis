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

print('Electron fields =', events.Electron.fields)
print('Muon fields =', events.Muon.fields)
print('Tau fields =', events.Tau.fields)
