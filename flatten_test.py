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
print(f"Before arg sort {events.staus_taus.distinctChildren.pdgId[56313].compute()}")
print(f"Before arg sort Gen Vis Taus are {events.GenVisTau.genPartIdxMother.compute()[56313]}")
print(f"Printing staus_taus structure {events.staus_taus.pdgId.compute()}")
events['staus_taus'] = ak.firsts(events.staus_taus[ak.argsort(events.staus_taus.pt, ascending=False)], axis = 2)
print(f"Printing staus_taus structure {events.staus_taus.pdgId.compute()}")
print(f"After arg sort {events.staus_taus.pt[71856].compute()}")

staus_taus = events['staus_taus']

print(f"After arg sort and ak.firsts {events.staus_taus.distinctChildren.pdgId.compute()[71856]}")
mask_taul = ak.any((abs(staus_taus.distinctChildren.pdgId) == 11) | (abs(staus_taus.distinctChildren.pdgId) == 13), axis=-1).compute()
print(f"mask_taul[71856] is {mask_taul[71856]}")
#print(f"mask_taul is {mask_taul.compute()}")
#print(f"Empty event due to pt cut is at index 54033 {mask_taul[54033].compute()}")
#print(f"The length of mask_taul is {len(mask_taul.compute())}")

sum_mask_taul = ak.sum(mask_taul, axis = -1)
print(f"sum of mask_taul along axis 1 is at 71856 {sum_mask_taul[71856]}")
#print(f"Is event 54033 still empty {sum_mask_taul[54033].compute()}")
#print(f"The lenght of sum_mask_taul is {len(sum_mask_taul.compute())}")
#print(f"The length of the flattened sum_mask_taul is {len(ak.flatten(sum_mask_taul, axis = 1).compute())}")
one_taul = (sum_mask_taul == 1)
print(f"one_taul[71856] is {one_taul[71856]}")
print(f"The structure of one_taul is {one_taul}")
print(f"The length of one_taul is {len(one_taul)}")
#print(f"Is there exactly 1 leptonic tau {one_taul.compute()}")
#print(f"Is event 54033 still empty {one_taul[54033].compute()}")
#print(f"The length of one_taul is {len(one_taul.compute())}")
#print(f"The length of the flattened one_taul is {len(ak.flatten(one_taul).compute())}")

#flat_one_taul = ak.flatten(one_taul, axis = None)
#print(f"flat_one_taul[71856] is {flat_one_taul[71856]}")
#print(f"The length of flat_one_taul {len(flat_one_taul.compute())}")

#for idx, evt in enumerate(one_taul):
#    if idx < 56314:
#        continue
#    else:
#        if evt[0] != flat_one_taul[idx+1]:
#            print(f"It's event {idx}!")
#            break


