import numpy as np
import math
import scipy
import os
import uproot
import awkward as ak
import matplotlib as mpl
from matplotlib import pyplot as plt 
exec(open("tau_func.py").read())

workingBranches = ["GenVisTau", "Jet", "GenPart"]
Run3_M500_ct100 = uproot.open("041223_FullSample_with-disTauTagScore.root")
events = Run3_M500_ct100["Events"]
eventBranches = events.keys()
nevt = events.num_entries 


Branches = {}
for branch in workingBranches:
  for key in eventBranches:
    if branch in key.split("_")[0]:
      Branches[key] = events[key].array()
for evt in range(100):
  for tau_idx in range(len(Branches["GenVisTau_genPartIdxMother"][evt])):
#  print("on tau ", tau_idx)
#  print(stauIdx(0, Branches["GenVisTau_genPartIdxMother"][evt][tau_idx], Branches["GenPart_pdgId"], Branches["GenPart_genPartIdxMother"]))
    print(Lxy(Branches["GenPart_vertexX"][evt][Branches["GenVisTau_genPartIdxMother"][evt][tau_idx]], Branches["GenPart_vertexX"][evt][stauIdx(evt, Branches["GenVisTau_genPartIdxMother"][evt][tau_idx], Branches["GenPart_pdgId"], Branches["GenPart_genPartIdxMother"])], Branches["GenPart_vertexY"][evt][Branches["GenVisTau_genPartIdxMother"][evt][tau_idx]], Branches["GenPart_vertexY"][evt][stauIdx(evt, Branches["GenVisTau_genPartIdxMother"][evt][tau_idx], Branches["GenPart_pdgId"], Branches["GenPart_genPartIdxMother"])]))
    if Lxy(Branches["GenPart_vertexX"][evt][Branches["GenVisTau_genPartIdxMother"][evt][tau_idx]], Branches["GenPart_vertexX"][evt][stauIdx(evt, Branches["GenVisTau_genPartIdxMother"][evt][tau_idx], Branches["GenPart_pdgId"], Branches["GenPart_genPartIdxMother"])], Branches["GenPart_vertexY"][evt][Branches["GenVisTau_genPartIdxMother"][evt][tau_idx]], Branches["GenPart_vertexY"][evt][stauIdx(evt, Branches["GenVisTau_genPartIdxMother"][evt][tau_idx], Branches["GenPart_pdgId"], Branches["GenPart_genPartIdxMother"])]) > 0.2:
      print("Large Lxy") 



