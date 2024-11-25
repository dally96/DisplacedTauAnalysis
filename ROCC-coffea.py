import coffea
import uproot
import scipy
import numpy
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, PFNanoAODSchema
from coffea.analysis_tools import PackedSelection

NanoAODSchema.warn_missing_crossrefs = False

fname = "/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215454/0000/nanoaod_output_1.root"

events = NanoEventsFactory.from_root(
    {fname: "Events"},
    schemaclass=PFNanoAODSchema,
    metadata={"dataset": "signal"},
    delayed = False).events()

selection = PackedSelection()

selection.add("electron", ak.num(events.Electron))
selection.add("muon", ak.num(events.Muon))
selection.add("tau", ak.num(events.Tau))

selection.require(("electron" | "tau" | "muon") = true)

print( len(selection), "leptons out of " )
print( len(events), "total things" )
