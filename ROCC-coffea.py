import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

NanoAODSchema.warn_missing_crossrefs = False

fname = "/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215454/0000/nanoaod_output_1.root"

events = NanoEventsFactory.from_root(
  {fname: "Events"},
  schemaclass=NanoAODSchema,
  metadata={"dataset": "signal"},
).events

events.Generator.fields
