import ROOT
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import awkward
import scipy
import uproot

nanoaod_dict = {}



Root_file = ROOT.TFile.Open("Staus_M_500_10mm_13p6TeV_Run3Summer22EE_with-disTauTagScore.root")
Events = Root_file.Events

Branches = Events.GetListOfBranches()
  

Branch_Names = []
#for i in range(len(Branches)):
#  Branch_Names.append(Branches.At(i).GetName())
#
#for name_idx, name in enumerate(Branch_Names):
#  nanoaod_dict[str(name)] = Branch_Names[name_idx][Events.GetEntries(str(name))]

print(Events.Jet_pt.Print())
#print(Events.GetListOfBranches().At(20).GetName())

#def makeDict(root_file):
#        Root_file = ROOT.TFile.Open("root_file")
#        TTree* Events = (TTree*)Root_file.Get("Events")
