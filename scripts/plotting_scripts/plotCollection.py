import numpy as np 
import awkward as ak
import ROOT as rt 
from ROOT import TMath
import scipy
import math
import os
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
from array import array
import tau_func as tf
import NanoAOD_Dict
import tdrstyle 
from argparse import ArgumentParser
import math
parser = ArgumentParser()
rt.EnableImplicitMT() 

parser.add_argument("collec", help = "which collection of variables to make a plot of", default = "PFCandidate")
#parser.add_argument("units", help = "units of variable", default = "cm")
args = parser.parse_args()

tdrstyle.setTDRStyle()

Files = NanoAOD_Dict.Nano_Dict

colors = [rt.kBlue, rt.kRed, rt.kGreen, rt.kMagenta]

histDict = {}
branchDict = {}
intDict = {}
colorDict = {}

can = rt.TCanvas("can", "can")
color = 0

for file in Files:
  if "Stau" not in file: continue
  df = rt.RDataFrame('Events', Files[file])
  runFile = rt.TFile.Open(Files[file])
  Events = runFile.Get("Events")
  nevt = Events.GetEntriesFast()
  Branches = Events.GetListOfBranches()
  
  for i in range (Branches.GetEntries()):
    if args.collec in Branches.At(i).GetName():
      if Branches.At(i).GetName() in branchDict: continue
      if Branches.At(i).GetName()[0] == "n": continue
      branchDict[Branches.At(i).GetName()] = Branches.At(i).GetName()

  PFCount = 0
  for evt in range(nevt):
    Events.GetEntry(evt)
    for val in range(int(Events.GetBranch("nPFCandidate").GetEntry(evt)/4)):
      PFCount += Events.GetBranch("nPFCandidate").GetLeaf("nPFCandidate").GetValue(val)
  
  for branch in branchDict:
    intDict[file+"_"+branch] = PFCount
    colorDict[file+"_"+branch] = color
    #print(branch, ": Minimum is ", df.Min(branch).GetValue(), " Maximum is : ", df.Max(branch).GetValue())
    filteredDF = df.Filter("@branch > -999 & @branch < 100000")
    #if df.Max(branch).GetValue() > 99999:
      #histDict[file+"_"+branch] = df.Histo1D(("h_"+file+"_"+branch, branch+";"+branch+"; A.U.", 100, rt.TMath.Floor(df.Min(branch).GetValue()/10) * 10, rt.TMath.Ceil(99999/10) * 10), branch) 
    #else:
    histDict[file+"_"+branch] = filteredDF.Histo1D(("h_"+file+"_"+branch, branch+";"+branch+"; A.U.", 100, rt.TMath.Floor(df.Min(branch).GetValue()/10) * 10, rt.TMath.Ceil(df.Max(branch).GetValue()/10) * 10), branch) 
  color += 1

for branch in branchDict:
  print(branch)
  legend = rt.TLegend(0.6,0.6,0.88,0.85)
  first_hist = True
  for hist in histDict:
    if branch.split("_")[-1] == hist.split("_")[-1]:
      histDict[hist].SetStats(0)
      if first_hist:
        histDict[hist].Draw("HISTE")
        first_hist = False
      else:
        histDict[hist].Draw("SAMEHISTE")
      histDict[hist].SetLineColor(colors[colorDict[hist]])
      histDict[hist].Scale(1/intDict[hist])
      legend.AddEntry(histDict[hist].GetPtr(), hist.split("_")[0])
  legend.Draw()
  can.SaveAs(branch+".pdf")
        

 
  
