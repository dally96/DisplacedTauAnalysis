import numpy as np
import uproot 
import awkward as ak
import ROOT 
import scipy
import math
import os
from matplotlib import pyplot as plt
import matplotlib as mpl
from array import array
from latexifier import latexify
import tau_func as tf
import Run3NanoAOD_Dict
import NanoAOD_Dict


Files = NanoAOD_Dict.Nano_Dict 

colors_list = ['#e23f59', '#eab508', '#7fb00c', '#8a15b1', '#57a1e8', '#e88000', '#1c587e', '#b8a91e', '#19cbaf', '#322a2d', '#2a64c5']
workingBranches = ["GenVisTau", "Jet", "GenPart"]
stauMass = 500 #GeV

lifetimeDict = {}

for file in Files:
  Sample_file = uproot.open(Files[file])
  Events = Sample_file["Events"]
  eventBranches = Events.keys()
  nevt = Events.num_entries 
  
  
  Branches = {}
  for branch in workingBranches:
    for key in eventBranches:
      if branch == key.split("_")[0]:
        Branches[key] = Events[key].array()
  lifetimeDict[file] = []

  for event in range(nevt):
    for tau_idx in range(len(Branches["GenVisTau_pt"][event])):
      if Branches["GenVisTau_genPartIdxMother"][event][tau_idx] == -1: continue
      motherTau_idx = Branches["GenVisTau_genPartIdxMother"][event][tau_idx]
      stau_idx = tf.stauIdx(event, motherTau_idx, Branches["GenPart_pdgId"], Branches["GenPart_genPartIdxMother"])  
      if stau_idx == -999: continue
      #lifetimeDict[file].append(tf.ctau(event, motherTau_idx, stau_idx, stauMass, Branches["GenPart_vertexX"], Branches["GenPart_vertexY"], Branches["GenPart_pt"], Branches["GenPart_phi"], Branches["GenPart_eta"]))
      lifetimeDict[file].append(tf.Lxy(event, motherTau_idx, stau_idx, Branches["GenPart_vertexX"], Branches["GenPart_vertexY"]))


ltKeys = list(lifetimeDict.keys())
ltValues = list(lifetimeDict.values()) 

bins = np.linspace(0,40,21)

hist1, bins1, other1 = plt.hist(lifetimeDict[ltKeys[0]], bins = bins, weights = np.ones_like(lifetimeDict[ltKeys[0]])/len(lifetimeDict[ltKeys[0]]), histtype = 'step', color = colors_list[0], label = ltKeys[0])
for sample in range(1, len(ltKeys)):
  hist, bins, other = plt.hist(lifetimeDict[ltKeys[sample]], bins = bins, weights = np.ones_like(lifetimeDict[ltKeys[sample]])/len(lifetimeDict[ltKeys[sample]]), histtype = 'step', color = colors_list[sample], label = ltKeys[sample])
plt.xlabel(r'L_{xy} [cm]')
plt.ylabel("A.U.")
plt.legend()
plt.show()



  
