import uproot
import scipy
import awkward as ak
import numpy as np
from coffea import processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import PackedSelection
import matplotlib.pyplot as plt
import vector

NanoAODSchema.warn_missing_crossrefs = False

fname = "/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215454/0000/nanoaod_output_1.root"

events = NanoEventsFactory.from_root(
    {fname: "Events"},
    schemaclass=NanoAODSchema,
    metadata={"dataset": "signal"},
    delayed = False).events()

selection = PackedSelection()
tau_pt_vals = events.Tau.pt
selection.add("tau_pt", tau_pt_vals > 2)
selection.add("event_cut", selection.require(tau_pt=true))
selected_events = events[selection.all("event_cut")]

print(f"Tau pt values\n{tau_pt_vals}\n")
print(f"Min tau pt after cut\n{min(selected_events.Tau.pt)}")
