import ROOT
import uproot 
import awkward as ak
import numpy as np 
import math 
import os,sys 


file = "SUS-RunIISummer20UL18GEN-stau100_lsp1_ctau100mm_v6_with-disTauTagScore.root"

f = uproot.open(file)
Events = f["Events"]
nevt = Events.num_entries

print("The number of events is ", nevt)

GenPart_Dict = {}

for branch in Events.keys():
  if "GenPart" == branch.split("_")[0]:
    GenPart_Dict[branch] = Events[branch].array()


#GenPart_ak = ak.zip(GenPart_Dict)

isFirstCopy = ( ((GenPart_Dict["GenPart_statusFlags"] & 4096) == 4096) & (GenPart_Dict["GenPart_statusFlags"] % 2 == 1) & ((abs(GenPart_Dict["GenPart_pdgId"]) == 11) | (abs(GenPart_Dict["GenPart_pdgId"]) == 13) | (abs(GenPart_Dict["GenPart_pdgId"]) == 15)))



GenPart_lep_phi  = GenPart_Dict["GenPart_phi"][isFirstCopy]
print(GenPart_lep_phi)

print(len(GenPart_lep_phi))



