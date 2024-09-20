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
#      'QCD50_80',
#      'QCD80_120',
#      'QCD120_170',
#      'QCD170_300',
      'QCD300_470',
      'QCD470_600',
      'QCD600_800',
      'QCD800_1000',
      'QCD1000_1400',
      'QCD1400_1800',
      'QCD1800_2400',
      'QCD2400_3200',
      'QCD3200',
      "TTtoLNu2Q", 
      "TTto4Q",   
      "TTto2L2Nu",
      "DYJetsToLL",   
      "WtoLNu2Jets",
      ]

SIG = [
      'Stau_100_100mm',
      ]
selections = {
              "electron_pt": 30, 
              "electron_eta": 1.44, 
              "electron_cutBased": 4,

              "muon_pt": 30,
              "muon_eta": 1.5,
              "muon_ID": "muon_tightId",
             }
              

can = ROOT.TCanvas("can", "can")
colors = ['#56CBF9', '#BF98A0', '#FDCA40', '#5DFDCB', '#3A5683', '#FF773D']
#ROOT.gStyle.SetOptStat(0)
ROOT.TH1F.SetDefaultSumw2(ROOT.kTRUE);

def getMaximum(h_eff, isLog=False):
    if h_eff.GetNhists() > 0:
        max_y_bkg = 0

    # Iterate over the histograms in the stack
    for hist in h_eff.GetHists():
        if hist:
            for bin in range(1, hist.GetNbinsX() + 1):
                bin_sum = sum([h.GetBinContent(bin) for h in h_eff.GetHists()])
                if bin_sum > max_y_bkg:
                    max_y_bkg = bin_sum
    return max_y_bkg

def getMaximumHist(h_eff, isLog=False):
    max_y_bkg = 0
    for bin in range(1, h_eff.GetNbinsX() + 1):
        if h_eff.GetBinContent(bin) > max_y_bkg:
            max_y_bkg = h_eff.GetBinContent(bin)
    return max_y_bkg

def getMinimum(h_eff, isLog=False):
    if h_eff.GetNhists() > 0:
        min_y_bkg = 1E9

    # Iterate over the histograms in the stack
    for hist in h_eff.GetHists():
        if hist:
            for bin in range(1, hist.GetNbinsX() + 1):
                bin_sum = sum([h.GetBinContent(bin) for h in h_eff.GetHists()])
                if bin_sum < min_y_bkg:
                    min_y_bkg = bin_sum
    return min_y_bkg

def sig_v_bkg(decay: str, var: str, legend: list, var_low: float, var_hi: float, nbins: int, unit: str, log_set: bool):

    hs_BKG = ROOT.THStack('hs_bkg', ';;A.U.')

    if ("el" in decay) or ("mu" in decay):
        hs_BKG.SetTitle("Events with >= 1 " + decay + " and >= 1 jet with pt > 20 GeV and |#eta| < 2.4, L = 38.01 fb^{-1};" + var + ' ' + unit + '; A.U.')
    if ('tau' in decay): 
         hs_BKG.SetTitle("Events with >=2 jets with pt > 20 GeV and |#eta| < 2.4, L = 38.01 fb^{-1};" + var + ' ' + unit + '; A.U.')

    h_SIG = ROOT.TH1F('h_' + decay + '_' + var + '_SIG', ';' + var + ' ' + unit + '; A.U.', nbins, var_low, var_hi)
    h_QCD_BKG = ROOT.TH1F('h_' + decay + '_' + var + '_QCD_BKG', ';' + var + ' ' + unit + '; A.U.', nbins, var_low, var_hi)
    h_TT_BKG = ROOT.TH1F('h_' + decay + '_' + var + '_TT_BKG', ';' + var + ' ' + unit + '; A.U.', nbins, var_low, var_hi)
    h_EWK_BKG = ROOT.TH1F('h_' + decay + '_' + var + '_EWK_BKG', ';' + var + ' ' + unit + '; A.U.', nbins, var_low, var_hi)

    for file in SIG:
        print("Now plotting: " + file)
        signal_file = ak.from_parquet("my_skim_" + decay + "_" + file + "/*.parquet")
        plot_var = signal_file[var]
        if "electron" in var or "muon" in var:
            if decay == "electron":
                plot_var = plot_var[(signal_file["electron_pt"] > selections["electron_pt"])
                                    & (abs(signal_file["electron_eta"]) < selections["electron_eta"])
                                    & (signal_file["electron_cutBased"] == selections["electron_cutBased"])
                                   ]
            if decay == "muon":
                plot_var = plot_var[(signal_file["muon_pt"] > selections["muon_pt"])
                                    & (abs(signal_file["muon_eta"]) < selections["muon_eta"])
                                    & (signal_file["muon_tightId"])
                                   ]
        else:

            if decay == "electron":
                electron_selection = signal_file["electron_pt"][(signal_file["electron_pt"] > selections["electron_pt"])
                                      & (abs(signal_file["electron_eta"]) < selections["electron_eta"])
                                      & (signal_file["electron_cutBased"] == selections["electron_cutBased"])
                                     ]
                num_electrons = ak.num(electron_selection)
                event_mask = num_electrons > 0
                event_mask = ak.flatten(event_mask, axis = None)
                plot_var = plot_var[event_mask]
            if decay == "muon":
                muon_selection = signal_file["muon_pt"][(signal_file["muon_pt"] > selections["muon_pt"])
                                  & (abs(signal_file["muon_eta"]) < selections["muon_eta"])
                                  & (signal_file["muon_tightId"] ==  1)
                                 ]
                num_muons = ak.num(muon_selection)
                event_mask = num_muons > 0
                event_mask = ak.flatten(event_mask, axis = None)
                plot_var = plot_var[event_mask]
        plot_var = ak.flatten(plot_var, axis = None)
        for val in plot_var:
            h_SIG.Fill(val, 1E5*xsecs[file]*38.01*1000*1/num_events[file])

    for file in BKG:
        print("Now plotting: " + file)
        background_file = ak.from_parquet("my_skim_" + decay + "_" + file + "/*.parquet")
        plot_var = background_file[var]
        if "electron" in var or "muon" in var:
            if decay == "electron":
                plot_var = plot_var[(background_file["electron_pt"] > selections["electron_pt"])
                                    & (abs(background_file["electron_eta"]) < selections["electron_eta"])
                                    & (background_file["electron_cutBased"] == selections["electron_cutBased"])
                                   ]
            if decay == "muon":
                plot_var = plot_var[(background_file["muon_pt"] > selections["muon_pt"])
                                    & (abs(background_file["muon_eta"]) < selections["muon_eta"])
                                    & (background_file["muon_tightId"])
                                   ]
        else:
            if decay == "electron":
                electron_selection = background_file["electron_pt"][(background_file["electron_pt"] > selections["electron_pt"])
                                      & (abs(background_file["electron_eta"]) < selections["electron_eta"])
                                      & (background_file["electron_cutBased"] == selections["electron_cutBased"])
                                     ]
                num_electrons = ak.num(electron_selection)
                event_mask = num_electrons > 0
                event_mask = ak.flatten(event_mask, axis = None)
                plot_var = plot_var[event_mask]
            if decay == "muon":
                muon_selection = background_file["muon_pt"][(background_file["muon_pt"] > selections["muon_pt"])
                                  & (abs(background_file["muon_eta"]) < selections["muon_eta"])
                                  & (background_file["muon_tightId"] ==  1)
                                 ]
                num_muons = ak.num(muon_selection)
                event_mask = num_muons > 0
                event_mask = ak.flatten(event_mask, axis = None)
                plot_var = plot_var[event_mask]
            
        plot_var = ak.flatten(plot_var, axis = None)
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
    #l_svb.SetBorderSize(0)
    #l_svb.SetFillStyle(0)    

    h_QCD_BKG.SetLineColor(ROOT.TColor.GetColor(colors[0]))
    h_TT_BKG.SetLineColor(ROOT.TColor.GetColor(colors[1]))
    h_EWK_BKG.SetLineColor(ROOT.TColor.GetColor(colors[2]))
    h_QCD_BKG.SetFillColor(ROOT.TColor.GetColor(colors[0]))
    h_TT_BKG.SetFillColor(ROOT.TColor.GetColor(colors[1]))
    h_EWK_BKG.SetFillColor(ROOT.TColor.GetColor(colors[2]))
    h_SIG.SetLineColor(ROOT.kBlack)    

    #h_QCD_BKG.Sumw2()
    #h_TT_BKG.Sumw2()
    #h_EWK_BKG.Sumw2()

    hs_BKG.Add(h_QCD_BKG)
    hs_BKG.Add(h_TT_BKG)
    hs_BKG.Add(h_EWK_BKG)

    hs_BKG.Draw("histe")
    
    if log_set:
        hs_BKG.SetMinimum(1E-3)
        hs_BKG.SetMaximum(10 * getMaximum(hs_BKG))
    elif not log_set:
        hs_BKG.SetMinimum(getMinimum(hs_BKG) / 10)
        hs_BKG.SetMaximum(2 * getMaximum(hs_BKG))

    can.Update()
    h_SIG.Draw("samehiste")


    l_svb.AddEntry(h_QCD_BKG, "QCD bkgd")
    l_svb.AddEntry(h_TT_BKG, "TT bkgd")
    l_svb.AddEntry(h_EWK_BKG, "DY + W bkgd")
    l_svb.AddEntry(h_SIG, "Stau 100 GeV 100mm * 10^{5}")
    l_svb.Draw()
    #hs_BKG.SetTitle(decay + "_" + var + ", L = 38.01 fb^{-1}")
    can.SetLogy(log_set)
    if log_set:
        can.SaveAs(decay + "_" + var + "_sigvbkg_log_selec.pdf")
    if not log_set:
        can.SaveAs(decay + "_" + var + "_sigvbkg_selec.pdf")

def sample(bkgd: str, decay: str, var: str, legend: list, var_low: float, var_hi: float, nbins: int, unit: str, log_set: bool):
    h_BKG = ROOT.TH1F('h_' + bkgd + '_' + decay + '_' + var, ';' + var + ' ' + unit + '; A.U.', nbins, var_low, var_hi)


    for file in BKG:
        if bkgd in file: 
            background_file = ak.from_parquet("my_skim_" + decay + "_" + file + "/*.parquet")
            plot_var = ak.flatten(background_file[var], axis = None)
            for val in plot_var:
                h_BKG.Fill(val, xsecs[file]*38.01*1000*1/num_events[file])

    l_svb = ROOT.TLegend()

    if len(legend) > 0:
        l_svb = ROOT.TLegend(legend[0], legend[2], legend[1], legend[3])
    
    #h_BKG.Sumw2()
    h_BKG.Draw("histe")

    if log_set:
        h_BKG.SetMinimum(1)
        h_BKG.SetMaximum(10 * getMaximumHist(h_BKG))
    elif not log_set:
        h_BKG.SetMinimum(getMinimum(h_BKG) / 10)
        h_BKG.SetMaximum(2 * getMaximum(h_BKG))
    
    can.Update() 
    
    l_svb.AddEntry(h_BKG, bkgd)
    l_svb.Draw()
    can.SetLogy(log_set)
    if log_set:
        can.SaveAs(bkgd + "_" + decay + "_" + var + "_log.pdf")
    if not log_set:
        can.SaveAs(bkgd + "_" + decay + "_" + var + ".pdf")


#sample("QCD","electron", "electron_dxy", [], -1, 1, 200, "", 1)
#sample("TT", "electron", "electron_dxy", [], -1, 1, 200, "", 1)
#sample("Jets",  "electron", "electron_dxy", [], -1, 1, 200, "", 1)

#sample("QCD","electron", "electron_dz", [], -1, 1, 200, "", 1)
#sample("TT", "electron", "electron_dz", [], -1, 1, 200, "", 1)
#sample("Jets",  "electron", "electron_dz", [], -1, 1, 200, "", 1)

#sample("QCD","muon", "muon_dxy", [], -1, 1, 200, "", 1)
#sample("TT", "muon", "muon_dxy", [], -1, 1, 200, "", 1)
#sample("Jets",  "muon", "muon_dxy", [], -1, 1, 200, "", 1)

#sample("QCD","muon", "muon_dz", [], -1, 1, 200, "", 1)
#sample("TT", "muon", "muon_dz", [], -1, 1, 200, "", 1)
#sample("Jets",  "muon", "muon_dz", [], -1, 1, 200, "", 1)
 
#sig_v_bkg("electron", "electron_pt", [], 20, 500, 120, "[GeV]", 0) 
#sig_v_bkg("electron", "electron_ID", [], 0, 10, 50, "", 1) 
#sig_v_bkg("electron", "jet_pt", [], 20, 100, 20, "[GeV]", 1) 
#sig_v_bkg("electron", "jet_eta", [], -2.4, 2.4, 48, "", 1) 
#sig_v_bkg("electron", "jet_phi", [], -3.2, 3.2, 64, "[rad]", 1) 
#sig_v_bkg("electron", "jet_score", [], 0, 1, 20, "", 1) 
#sig_v_bkg("electron", "deta", [], -5, 5, 100, "", 1) 
#sig_v_bkg("electron", "dphi", [], -3.2, 3.2, 64, "[rad]", 1) 
#sig_v_bkg("electron", "dR", [], 0, 1, 20, "", 1) 
sig_v_bkg("electron", "MET_pT", [], 0, 500, 50, "[GeV]", 1) 
#sig_v_bkg("electron", "electron_eta", [], -2.4, 2.4, 48, "", 0)
#sig_v_bkg("electron", "electron_phi", [], -3.2, 3.2, 64, "[rad]",0) 
#sig_v_bkg("electron", "electron_charge", [], -1, 2, 3,"", 0)
#sig_v_bkg("electron", "electron_dxy", [], -1, 1, 200, "[cm]", 0)
#sig_v_bkg("electron", "electron_dz",[],  -1, 1, 200, "[cm]", 0)
#sig_v_bkg("electron", "electron_eta", [], -2.4, 2.4, 48, "", 1)
#sig_v_bkg("electron", "electron_phi", [], -3.2, 3.2, 64, "[rad]",1) 
#sig_v_bkg("electron", "electron_charge", [], -1, 2, 3,"", 1)
#sig_v_bkg("electron", "electron_dxy", [], -1, 1, 200, "[cm]", 1)
#sig_v_bkg("electron", "electron_dz",[],  -1, 1, 200, "[cm]", 1)
#sig_v_bkg("electron", "generator_scalePDF", [], 0, 3500, 100, "", 1)
#sig_v_bkg("electron", "leadingjet_pt", [], 20, 100, 20, "[GeV]", 1) 
#sig_v_bkg("electron", "leadingjet_eta", [], -2.4, 2.4, 48, "", 1) 
#sig_v_bkg("electron", "leadingjet_phi", [], -3.2, 3.2, 64, "[rad]", 1) 
#sig_v_bkg("electron", "leadingjet_score", [], 0, 1, 20, "", 1) 

#sig_v_bkg("muon", "muon_pt", [], 20, 500, 120, "[GeV]", 0) 
#sig_v_bkg("muon", "muon_pt", [], 20, 500, 120, "[GeV]", 1) 
#sig_v_bkg("muon", "jet_pt", [], 20, 100, 20, "[GeV]", 1) 
#sig_v_bkg("muon", "jet_eta", [], -2.4, 2.4, 48, "", 1) 
#sig_v_bkg("muon", "jet_phi", [], -3.2, 3.2, 64, "[rad]", 1) 
#sig_v_bkg("muon", "jet_score", [], 0, 1, 20, "", 1) 
#sig_v_bkg("muon", "deta", [], -5, 5, 100, "", 1) 
#sig_v_bkg("muon", "dphi", [], -3.2, 3.2, 64, "[rad]", 1) 
#sig_v_bkg("muon", "dR", [], 0, 1, 20, "", 1) 
sig_v_bkg("muon", "MET_pT", [], 0, 500, 50, "[GeV]", 1) 
#sig_v_bkg("muon", "muon_eta", [], -2.4, 2.4, 48, "", 0)
#sig_v_bkg("muon", "muon_phi", [], -3.2, 3.2, 64, "[rad]", 0) 
#sig_v_bkg("muon", "muon_charge", [], -1, 2, 3,"", 0)
#sig_v_bkg("muon", "muon_dxy", [], -1, 1, 200, "", 0)
#sig_v_bkg("muon", "muon_dz",[],  -1, 1, 200, "", 0)
#sig_v_bkg("muon", "muon_eta", [], -2.4, 2.4, 48, "", 1)
#sig_v_bkg("muon", "muon_phi", [], -3.2, 3.2, 64, "[rad]", 1) 
#sig_v_bkg("muon", "muon_charge", [], -1, 2, 3,"", 1)
#sig_v_bkg("muon", "muon_dxy", [], -1, 1, 200, "", 1)
#sig_v_bkg("muon", "muon_dz",[],  -1, 1, 200, "", 1)
#sig_v_bkg("muon", "generator_scalePDF", [], 0, 3500, 100, "", 1)
#sig_v_bkg("muon", "leadingjet_pt", [], 20, 100, 20, "[GeV]", 1) 
#sig_v_bkg("muon", "leadingjet_eta", [], -2.4, 2.4, 48, "", 1) 
#sig_v_bkg("muon", "leadingjet_phi", [], -3.2, 3.2, 64, "[rad]", 1) 
#sig_v_bkg("muon", "leadingjet_score", [], 0, 1, 20, "", 1) 

#sig_v_bkg("ditau", "jet_pt", [], 20, 200, 45, "[GeV]", 1) 
#sig_v_bkg("ditau", "jet_eta", [], -2.4, 2.4, 48, "", 1) 
#sig_v_bkg("ditau", "jet_phi", [], -3.2, 3.2, 64, "[rad]", 1) 
#sig_v_bkg("ditau", "jet_score", [], 0, 1, 20, "", 1) 
#sig_v_bkg("ditau", "MET_pT", [], 0, 500, 50, "[GeV]", 1) 
#sig_v_bkg("ditau", "leadingjet_pt", [], 20, 100, 20, "[GeV]", 1) 
#sig_v_bkg("ditau", "leadingjet_eta", [], -2.4, 2.4, 48, "", 1) 
#sig_v_bkg("ditau", "leadingjet_phi", [], -3.2, 3.2, 64, "[rad]", 1) 
#sig_v_bkg("ditau", "leadingjet_score", [], 0, 1, 20, "", 1) 
#sig_v_bkg("ditau", "generator_scalePDF", [], 0, 3500, 100, "", 1)
