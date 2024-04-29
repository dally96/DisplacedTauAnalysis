import ROOT 
import numpy as np 
import awkward as ak 
import math
import scipy
import array
import pandas as pd
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from leptonPlot import *
import matplotlib  as mpl
from  matplotlib import pyplot as plt
from xsec import *

BKG = [
      'QCD1000_1400',
      'QCD120_170',
      'QCD1400_1800',
      'QCD170_300',
      'QCD1800_2400',
      'QCD2400_3200',
      'QCD300_470',
      'QCD3200',
      'QCD50_80',
      'QCD600_800',
      'QCD800_1000',
      'QCD80_120',
      'QCD470_600',
      ]

SIG = [
      'Stau_100_100mm',
      ]

can = ROOT.TCanvas("can", "can")

h_LPE_ID_SIG = ROOT.TH1F('h_LPE_ID_SIG', ';BDT ID; A.U.', 100, 0, 10)
h_LPE_ID_BKG = ROOT.TH1F('h_LPE_ID_BKG', ';BDT ID; A.U.', 100, 0, 10)

for file in SIG:
  signal_file = ak.from_parquet("my_skim_electron_" + file + "/part0.parquet")
  LowPtElectron_ID = ak.flatten(signal_file["electron_ID"], axis = None)
  for bdt in LowPtElectron_ID:
    h_LPE_ID_SIG.Fill(bdt, xsecs[file]*38.01*1000/len(signal_file))

for file in BKG:
  background_file = ak.from_parquet("my_skim_electron_" + file + "/part0.parquet")
  LowPtElectron_ID = ak.flatten(background_file["electron_ID"], axis = None)
  for bdt in LowPtElectron_ID:
    h_LPE_ID_BKG.Fill(bdt, xsecs[file]*38.01*1000/len(background_file))

h_LPE_ID_BKG.SetLineColor(ROOT.kBlue)
h_LPE_ID_SIG.SetLineColor(ROOT.kRed)

l_LPE_ID = ROOT.TLegend(0.5,0.5,0.75,0.75)

h_LPE_ID_BKG.Draw("histe")
h_LPE_ID_SIG.Draw("samehiste")

l_LPE_ID.AddEntry(h_LPE_ID_BKG, "QCD bkgd")
l_LPE_ID.AddEntry(h_LPE_ID_SIG, "Stau 100 GeV 100mm")
l_LPE_ID.Draw()
h_LPE_ID_BKG.SetTitle("LowPtElectron BDT_ID, L = 38.01 fb^{-1}")
can.SaveAs("SvB_LowPtElectron_ID.pdf")
  
