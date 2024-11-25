
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from hist import Hist, axis, intervals

# Load the file
filename = "/eos/user/d/dally/DisplacedTauAnalysis/SMS-TStauStau_MStau-100_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_NanoAOD.root"
events = NanoEventsFactory.from_root({filename: "Events"}, 
                                         schemaclass=NanoAODSchema,
                                         metadata={"dataset": "MC"},
                                        ).events()

## find staus and their tau children  
gpart = events.GenPart
events['staus'] = gpart[(abs(gpart.pdgId) == 1000015) & (gpart.hasFlags("isLastCopy"))] # most likely last copy of stay in the chain
print(f"Stau array looks like {events['staus'][0].compute()}")

events['staus_taus'] = events.staus.distinctChildren[ (abs(events.staus.distinctChildren.pdgId) == 15) & \
                                                  (events.staus.distinctChildren.hasFlags("isLastCopy")) & \
                                                  (events.staus.distinctChildren.pt > 5) \
                                                 ]
print(f"Staus_taus array looks like {events['staus_taus'][0].compute()}")
staus_taus = events['staus_taus']

mask_tauh = ak.any(abs(staus_taus.distinctChildren.pdgId)==211, axis=-1) 
one_tauh_evt = (ak.sum(mask_tauh, axis = 1) > 0) & (ak.sum(mask_tauh, axis = 1) < 2)
flat_one_tauh_evt = ak.flatten(one_tauh_evt, axis=None)
#one_tauh_mask = ak.sum(flat_one_tauh_evt, axis = 1) < 2 
print(len(flat_one_tauh_evt.compute()))
#events_h = events[one_tauh_mask]

mask_taul = ak.any((abs(staus_taus.distinctChildren.pdgId) == 11) | (abs(staus_taus.distinctChildren.pdgId) == 13), axis=-1)
one_taul_evt = (ak.sum(mask_taul, axis = 1) > 0) & (ak.sum(mask_taul, axis = 1) < 2)
flat_one_taul_evt = ak.flatten(one_taul_evt, axis=None)
#one_taul_mask = ak.sum(flat_one_taul_evt, axis = 1) < 2 
print(len(flat_one_taul_evt.compute()))
print(flat_one_tauh_evt.compute() & flat_one_taul_evt.compute())
#events_l = events[one_taul_mask]

filtered_events = events[flat_one_tauh_evt & flat_one_taul_evt]
print(ak.num(filtered_events, axis = 0).compute())

print(filtered_events.GenPart.pdgId[0].compute())
