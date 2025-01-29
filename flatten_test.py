import awkward as ak 
import numpy as np 
import matplotlib.pyplot as plt 
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from hist import Hist, axis, intervals
import warnings
warnings.filterwarnings("ignore", module="coffea.*")
import dask_awkward as dak

# Load the file
filename = "/eos/user/d/dally/DisplacedTauAnalysis/SMS-TStauStau_MStau-100_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_NanoAOD.root"
events = NanoEventsFactory.from_root({filename: "Events"},
                                         schemaclass=NanoAODSchema,
                                         metadata={"dataset": "MC"},
                                         ).events()



## find staus and their tau children
gpart = events.GenPart
events['staus'] = gpart[(abs(gpart.pdgId) == 1000015) & (gpart.hasFlags("isLastCopy"))] # most likely last copy of stay in the chain



# Filter on staus to only include final-state taus with pt>5
events['staus_taus'] = events.staus.distinctChildren[ (abs(events.staus.distinctChildren.pdgId) == 15) & \
                                                  (events.staus.distinctChildren.hasFlags("isLastCopy")) #& \
#                                                  (events.staus.distinctChildren.pt > 5)
                                                 ]

staus_taus = events['staus_taus']
mask_taul = ak.any((abs(staus_taus.distinctChildren.pdgId) == 11) | (abs(staus_taus.distinctChildren.pdgId) == 13), axis=-1)
#print(f"mask_taul is {mask_taul.compute()}")
#print(f"Empty event due to pt cut is at index 54033 {mask_taul[54033].compute()}")
#print(f"The length of mask_taul is {len(mask_taul.compute())}")

sum_mask_taul = ak.sum(mask_taul, axis = 1)
#print(f"sum of mask_taul along axis 1 is {sum_mask_taul.compute()}")
print(f"Is event 54033 still empty {sum_mask_taul[54033].compute()}")
print(f"The lenght of sum_mask_taul is {len(sum_mask_taul.compute())}")
print(f"The length of the flattened sum_mask_taul is {len(ak.flatten(sum_mask_taul).compute())}")
one_taul = (sum_mask_taul == 1)
#print(f"Is there exactly 1 leptonic tau {one_taul.compute()}")
print(f"Is event 54033 still empty {one_taul[54033].compute()}")
print(f"The length of one_taul is {len(one_taul.compute())}")
print(f"The length of the flattened one_taul is {len(ak.flatten(one_taul).compute())}")

#flat_one_taul = dak.flatten(one_taul)
#print(f"The length of flat_one_taul {len(flat_one_taul.compute())}")



