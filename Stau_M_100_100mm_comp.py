import ROOT
import numpy as np
import os 
import uproot
import awkward as ak
import scipy
import math

exec(open("NanoAOD_Dict.py").read())

colors = [ROOT.kBlue, ROOT.kRed, ROOT.kGreen, ROOT.kBlack, ROOT.kViolet, ROOT.kMagenta]
can = ROOT.TCanvas("can", "can", 1000, 600)
score_legend = ROOT.TLegend(0.7,0.6,0.88,0.85)
score_legend.SetBorderSize(0)
Branches = ["Jet_pt", "Jet_eta", "Jet_phi", "Jet_disTauTag_score1"]


Dict = {}
Hist_Dict = {}
Color_Dict = {}
Branch_range = {}
color_index = 0

for file in Nano_Dict:
  sample_file = uproot.open(Nano_Dict[file])
  sample_file_events = sample_file["Events"]

  Branch_range[file+"_"+Branches[0]] = [0, 500]
  Branch_range[file+"_"+Branches[1]] = [-5, 5]
  Branch_range[file+"_"+Branches[2]] = [-math.pi, math.pi]
  Branch_range[file+"_"+Branches[3]] = [0, 1]

  for branch in Branches:
    Dict[file+"_"+branch] = ak.flatten(sample_file_events[branch].array())
  
  
  for branch in Dict:
    if file not in branch: continue
    Hist_Dict[branch] = ROOT.TH1F("h_"+branch, branch+";"+branch+";Fraction of all jets", 100, Branch_range[branch][0], Branch_range[branch][1]) 
    for entry in Dict[branch]:
      Hist_Dict[branch].Fill(entry)
    Hist_Dict[branch].Scale(1/len(Dict[branch]))
    Color_Dict[branch] = color_index
  color_index += 1

print(Dict)
print(Hist_Dict)
print(Color_Dict)  
    
for entry in Branches:
  legend = ROOT.TLegend(0.6,0.6,0.88,0.85)
  legend.SetBorderSize(0)
  for branch in Hist_Dict:
    if entry not in branch: continue
    print(branch)
    Hist_Dict[branch].SetStats(0)
    Hist_Dict[branch].SetLineColor(colors[Color_Dict[branch]])
    legend.AddEntry(Hist_Dict[branch], branch)
    if Color_Dict[branch] <= 0:
      print("Color index of " + branch + " is " + str(Color_Dict[branch]))
      Hist_Dict[branch].Draw("HISTE")
      Hist_Dict[branch].SetTitle(entry)
    else:
      print("Color index of " + branch + " is " + str(Color_Dict[branch]))
      Hist_Dict[branch].Draw("SAMEHISTE")
  legend.Draw()
  can.SaveAs("Run3_M100v500"+entry+"_comp.pdf")
  can.Clear()

