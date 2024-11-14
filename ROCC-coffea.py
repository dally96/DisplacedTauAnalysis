import coffea
import uproot
import scipy
import numpy
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, PFNanoAODSchema

NanoAODSchema.warn_missing_crossrefs = False

fname = "/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215454/0000/nanoaod_output_1.root"

events = NanoEventsFactory.from_root(
    {fname: "Events"},
    schemaclass=PFNanoAODSchema,
    metadata={"dataset": "signal"},
    delayed = False).events()

gpart = events.GenPart # Sara is smart https://github.com/sarafiorendi/displacedTausCoffea/blob/main/study_recojet_to_GEN.py
lepIds = [11, 13, 15]
events['leptons'] = gpart[ (abs(gpart.pdgId) in lepIds) ]

firstPass = events['leptons']

print( len(firstPass), "leptons out of " )
print( len(events), "total things" )
