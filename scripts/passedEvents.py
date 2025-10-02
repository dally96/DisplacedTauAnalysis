import uproot
import scipy
import matplotlib as mpl
import awkward as ak
import numpy as np
import math
import ROOT
import array
import pandas as pd
import os
import pickle
from leptonPlot import *

ROOT.gStyle.SetOptStat(0)
can1 = ROOT.TCanvas("can1", "can1")

files = [
         "Staus_M_400_100mm_13p6TeV_Run3Summer22_lpcdisptau_NanoAOD_ExtraMuonBranches.root", 
         "Staus_M_100_100mm_13p6TeV_Run3Summer22_lpcdisptau_NanoAOD_ExtraMuonBranches.root",
        ]


triggers = ["HLT_PFMET120_PFMHT120_IDTight", 
            "HLT_PFMET130_PFMHT130_IDTight", 
            "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight", 
            "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight", 
            "HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1", 
            "HLT_DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1",
            "HLT_MET105_IsoTrk50", 
            "HLT_MET120_IsoTrk50", 
            "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1", 
            "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1",]

trigBranches = {}

num_passtrig = {}
num_passtrig_pass = {}
num_passtrig_tot = {}

h_num_passtrig = {}

t = ROOT.TText()
t.SetTextAngle(60)
t.SetTextSize(0.02)
t.SetTextAlign(33)
ROOT.gPad.SetBottomMargin(0.6)

l_passtrig = ROOT.TLegend()
l_passtrig.SetBorderSize(0)
l_passtrig.SetFillStyle(0)
#l_passtrig.SetTextSize(0.025)
color_idx = 1

for file in files:

  Events = uproot.open(file)["Events"]
  print(len(Events["Muon_pt"].array()))
  
  trigBranches[file] = {}  
  num_passtrig[file] = {}

  num_passtrig_pass[file] = []
  num_passtrig_tot[file] = []

  for key in Events.keys():
    for trig in triggers:
      if trig == key:
        print(key)
        trigBranches[file][key] = Events[key].array()
  
  for trig in trigBranches[file]:
    num_passtrig[file][trig] = np.count_nonzero(trigBranches[file][trig] == 1)
    print( np.count_nonzero(trigBranches[file][trig] == 1))
    num_passtrig_pass[file].append(num_passtrig[file][trig])
    num_passtrig_tot[file].append(len(trigBranches[file][trig]))

  h_num_passtrig[file] = ROOT.TH1F("h_num_passtrig"+file, "Run 3 Stau Samples; ; Number of events passed trigger", len(num_passtrig_pass[file]), 0, len(num_passtrig_pass[file]))
  h_num_passtrig[file].GetXaxis().SetLabelOffset(99)
  h_num_passtrig[file].GetXaxis().SetLabelSize(0)

  l_passtrig.AddEntry(h_num_passtrig[file], file.split("_")[2] + "GeV " + file.split("_")[3])  

  if file == list(h_num_passtrig.keys())[0]:
    h_num_passtrig[file].Draw("histe")
  else:
    h_num_passtrig[file].Draw("samehiste")

  y = ROOT.gPad.GetUymin() - 0.2*h_num_passtrig[file].GetYaxis().GetBinWidth(1)

  if file == list(h_num_passtrig.keys())[0]:
    for i in range(len(num_passtrig_pass[file])):
      h_num_passtrig[file].SetBinContent(i + 1, num_passtrig_pass[file][i])
      x = h_num_passtrig[file].GetXaxis().GetBinCenter(i + 1)
      t.DrawText(x, y, list(trigBranches[file].keys())[i])
  else:
    for i in range(len(num_passtrig_pass[file])):
      h_num_passtrig[file].SetBinContent(i + 1, num_passtrig_pass[file][i])
  
  h_num_passtrig[file].Scale(1/len(Events["Muon_pt"].array()))
  h_num_passtrig[file].SetLineColor(color_idx)
  h_num_passtrig[file].GetYaxis().SetRangeUser(1E-3, 1.0)
  
  ROOT.gPad.Update()
  color_idx+=1

l_passtrig.Draw()
can1.SetLogy()
can1.SaveAs("numevents_trig.pdf")
can1.SaveAs("PNG_plots/numevents_trig.png")


