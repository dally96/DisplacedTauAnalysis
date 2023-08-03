import numpy as np
import uproot 
import awkward as ak
import ROOT
from ROOT import gROOT, TFile, TTree, TBranch, TH1F, TGraph, TMultiGraph
import scipy
import math
import os
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
from array import array

file = "041223_FullSample_with-disTauTagScore.root"
workingBranches = ["GenVisTau", "Jet", "GenPart"]


Run3_M500_ct100 = ROOT.TFile.Open(file)
#Run3_M500_ct100 = ROOT.TFile(file, 'read')
Events =  Run3_M500_ct100.Get("Events")
Branches = Events.GetListOfBranches()
#for branch in Branches:
 # if "dxy" in branch:
  #  print(branch)

#nevt = Events.GetEntries()
#Events.SetBranchAddress("GenVisTau_pt", ntuple)
#print(Events.GetEntriesFast())
#Events.GetEntry(1)

#for imc in range(ntuple.GetEntriesFast()):
#  print(ntuple.At(imc))

#
#class nano_branches(dict) :
#  pass
#
#branchDict = nano_branches()
#
#for branch_names in workingBranches:
for i in range(Branches.GetEntries()):
  branch = Branches.At(i)
  name = branch.GetName()
  if "dxy" in name:
    print(name)
#    branchDict.__setattr__(name, branch)
#
##for ievt in range(nevt):
#  #Events.GetEntry(ievt)
#for evt in Events:
##Events.GetEntry(0)
#  print(getattr(evt,"GenVisTau_pt"))
