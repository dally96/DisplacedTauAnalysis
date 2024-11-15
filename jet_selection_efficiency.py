import coffea
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

mask_tauh = ak.any(abs(staus_taus.distinctChildren.pdgId)==211, axis=-1) 
one_tauh_evt = (ak.sum(mask_tauh, axis = 1) > 0) & (ak.sum(mask_tauh, axis = 1) < 3)
flat_one_tauh_evt = ak.flatten(one_tauh_evt, axis=0)
one_tauh_mask = ak.sum(flat_one_tauh_evt, axis = 1) < 2 
events_h = events[one_tauh_mask]

mask_taul = ak.any((abs(staus_taus.distinctChildren.pdgId) == 11) | (abs(staus_taus.distinctChildren.pdgId) == 13), axis=-1)
one_taul_evt = (ak.sum(mask_taul, axis = 1) > 0) & (ak.sum(mask_taul, axis = 1) < 3)
flat_one_taul_evt = ak.flatten(one_taul_evt, axis=0)
one_taul_mask = ak.sum(flat_one_taul_evt, axis = 1) < 2 
events_l = events[one_taul_mask]

indices_h = ak.local_index(events_h, axis=0).tolist()
indices_l = ak.local_index(events_l, axis=0).tolist()

common_indices = list(set(indices_h) & set(indices_l))

filtered_events = events[common_indices]

print('Number of events with exactly one hadronic tau and one leptonic tau:', len(filtered_events),
      '(%.3f)' % (len(filtered_events) / len(events)))

print("Number of events with exactly one hadronic tau:", ak.sum(one_tauh_evt))
print("Number of events with exactly one leptonic tau:", ak.sum(one_taul_evt))

print("Number of unique events in `events_h`:", len(indices_h))
print("Number of unique events in `events_l`:", len(indices_l))
print("Number of common events:", len(common_indices))

for i in common_indices[:5]:  # Check the first 5 common events
    print("Event index:", i)
    print("Hadronic tau count:", ak.sum(mask_tauh[i]))
    print("Leptonic tau count:", ak.sum(mask_taul[i]))