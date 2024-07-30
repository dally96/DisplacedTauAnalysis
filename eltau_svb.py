import ROOT 
import numpy as np 
import awkward as ak 
import math
import scipy
import array
import pandas as pd
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
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
      "TTtoLNu2Q", 
      "TTto4Q",   
      "TTto2L2Nu",
      "DYJetsToLL",   
      "WtoLNu2Jets",
      ]

SIG = [
      'Stau_100_100mm',
      ]

can = ROOT.TCanvas("can", "can")

def sig_v_bkg(decay: str, var: str, legend: list, var_low: float, var_hi: float, nbins: int, unit: str, log_set: bool):

    hs_BKG = ROOT.THStack('hs_bkg', '')
    h_SIG = ROOT.TH1F('h_' + decay + '_' + var + '_SIG', ';' + var + ' ' + unit + '; A.U.', nbins, var_low, var_hi)
    h_QCD_BKG = ROOT.TH1F('h_' + decay + '_' + var + '_QCD_BKG', ';' + var + ' ' + unit + '; A.U.', nbins, var_low, var_hi)
    h_TT_BKG = ROOT.TH1F('h_' + decay + '_' + var + '_TT_BKG', ';' + var + ' ' + unit + '; A.U.', nbins, var_low, var_hi)
    h_EWK_BKG = ROOT.TH1F('h_' + decay + '_' + var + '_EWK_BKG', ';' + var + ' ' + unit + '; A.U.', nbins, var_low, var_hi)

    for file in SIG:
        print("Now plotting: " + file)
        signal_file = ak.from_parquet("my_skim_" + decay + "_" + file + "/*.parquet")
        plot_var = ak.flatten(signal_file[var], axis = None)
        for val in plot_var:
            h_SIG.Fill(val, xsecs[file]*38.01*1000*1/num_events[file])

    for file in BKG:
        print("Now plotting: " + file)
        background_file = ak.from_parquet("my_skim_" + decay + "_" + file + "/*.parquet")
        plot_var = ak.flatten(background_file[var], axis = None)
        for val in plot_var:
            if 'QCD' in file:
                h_QCD_BKG.Fill(val, xsecs[file]*38.01*1000*1/num_events[file])
            if 'TT' in file:
                h_TT_BKG.Fill(val, xsecs[file]*38.01*1000*1/num_events[file])
            if 'Jets' in file:
                h_EWK_BKG.Fill(val, xsecs[file]*38.01*1000*1/num_events[file])

    l_svb = ROOT.TLegend()
    
    if len(legend) > 0:
        l_svb = ROOT.TLegend(legend[0], legend[2], legend[1], legend[3])
    
    hs_BKG.Add(h_QCD_BKG)
    hs_BKG.Add(h_TT_BKG)
    hs_BKG.Add(h_EWK_BKG)

    hs_BKG.Draw("histe")
    h_SIG.Draw("samehiste")

    l_svb.AddEntry(h_QCD_BKG, "QCD bkgd")
    l_svb.AddEntry(h_TT_BKG, "TT bkgd")
    l_svb.AddEntry(h_EWK_BKG, "DY + W bkgd")
    l_svb.AddEntry(h_SIG, "Stau 100 GeV 100mm")
    l_svb.Draw()
    hs_BKG.SetTitle(decay + "_" + var + ", L = 38.01 fb^{-1}")
    can.SetLogy(log_set)
    can.SaveAs(decay + "_" + var + "_sigvbkg.pdf")

sig_v_bkg("electron", "electron_pt", [], 0, 100, 25, "[GeV]", 1) 
