import numpy as np
import uproot 
import awkward as ak
import ROOT as rt 
import scipy
import math
import os
from matplotlib import pyplot as plt
import matplotlib as mpl
from array import array
from latexifier import latexify
import tau_func as tf

Branches = {}
plotVar = {}

Files = [ "PFCand_tuple_Run2_Stau_M_100_100mm.root",
          #"PFCand_tuple_Run3_Stau_M_100_100mm.root"
        ]

for file in Files:
  Branches[file.split(".")[0]] = {} 
  Branches[file.split(".")[0]]["file"] = uproot.open(file)["tree"]["file_number"].array()  
  Branches[file.split(".")[0]]["event"] = uproot.open(file)["tree"]["event_number"].array() 
  Branches[file.split(".")[0]]["jet"] = uproot.open(file)["tree"]["jet_idx"].array()
  Branches[file.split(".")[0]]["phi"] = uproot.open(file)["tree"]["phi"].array()

  plotVar[file.split(".")[0]] = {}
  
  plotVar[file.split(".")[0]]["phi"] = rt.TH1D("h_ave_phi_" + file.split(".")[0], "Average phi of jet; phi; A.U.", 100, -rt.TMath.Pi(), rt.TMath.Pi())
  
  fileNumbers = []
  for x in Branches[file.split(".")[0]]["file"]:
    if x in fileNumbers: continue
    fileNumbers.append(x)
  fileNumbers.sort()
  

  eventNumbers = []
  eventNumbers_file = []

  for evtIdx, evt in enumerate(Branches[file.split(".")[0]]["event"]):
    if Branches[file.split(".")[0]]["file"][evtIdx] == 0 and evt == 0:
      eventNumbers_file.append(evt)
    elif Branches[file.split(".")[0]]["file"][evtIdx] > 0 and evt == 0:
      eventNumbers.append(eventNumbers_file)
      eventNumbers_file = []
      eventNumbers_file.append(evt)
    elif evt > 0:
      eventNumbers_file.append(evt)
    
  print(len(fileNumbers), len(eventNumbers))
