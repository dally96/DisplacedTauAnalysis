import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema
from hist import Hist, axis, intervals

# Load the file
filename = "/eos/user/d/dally/DisplacedTauAnalysis/SMS-TStauStau_MStau-100_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_NanoAOD.root"
events = NanoEventsFactory.from_root({filename: "Events"}, 
                                         schemaclass=PFNanoAODSchema,
                                         metadata={"dataset": "MC"},
                                         ).events()

## find staus and their tau children  
gpart = events.GenPart
events['staus'] = gpart[(abs(gpart.pdgId) == 1000015) & (gpart.hasFlags("isLastCopy"))] # most likely last copy of stay in the chain

events['staus_taus'] = events.staus.distinctChildren[ (abs(events.staus.distinctChildren.pdgId) == 15) & \
                                                  (events.staus.distinctChildren.hasFlags("isLastCopy")) & \
                                                  (events.staus.distinctChildren.pt > 5) \
                                                 ]
staus_taus = events['staus_taus']

mask_tauh = ak.any(abs(staus_taus.distinctChildren.pdgId) == 211, axis=-1)
one_tauh_evt = (ak.sum(mask_tauh, axis=1) > 0) & (ak.sum(mask_tauh, axis=1) < 3)
flat_one_tauh_evt = ak.flatten(one_tauh_evt, axis=None)

mask_taul = ak.any((abs(staus_taus.distinctChildren.pdgId) == 11) | (abs(staus_taus.distinctChildren.pdgId) == 13), axis=-1)
one_taul_evt = (ak.sum(mask_taul, axis=1) > 0) & (ak.sum(mask_taul, axis=1) < 3)
flat_one_taul_evt = ak.flatten(one_taul_evt, axis=None)

filtered_events = staus_taus[flat_one_tauh_evt & flat_one_taul_evt]  # Apply the combined mask to filter events

for i in range(5):
    print(filtered_events.distinctChildren.pdgId[i].compute())