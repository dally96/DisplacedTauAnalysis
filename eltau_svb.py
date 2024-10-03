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
      'QCD50_80',
      'QCD80_120',
      'QCD120_170',
      'QCD170_300',
      'QCD300_470',
      'QCD470_600',
      'QCD600_800',
      'QCD800_1000',
      'QCD1000_1400',
      'QCD1400_1800',
      'QCD1800_2400',
      'QCD2400_3200',
      'QCD3200',
      "DYJetsToLL",   
      "WtoLNu2Jets",
      "TTtoLNu2Q", 
      "TTto4Q",   
      "TTto2L2Nu",
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

TTto2L2Nu_file = {}
for file in BKG:
    if "2L2Nu" in file: 
        TTto2L2Nu_file["electron"] = ak.from_parquet("my_skim_electron_" + file + "/*.parquet")
        TTto2L2Nu_file["muon"] = ak.from_parquet("my_skim_muon_" + file + "/*.parquet")
        TTto2L2Nu_file["ditau"] = ak.from_parquet("my_skim_ditau_" + file + "/*.parquet")

can = ROOT.TCanvas("can", "can")
colors = ['#56CBF9', '#6abecc', '#64b0bf', '#5da2b3', '#5694a8', '#4f879d', '#477a93', '#3f6d89', '#386180', '#2f5477', '#27486e', '#1d3c65', '#13315c', '#BF98A0', '#FDCA40', '#5DFDCB', '#3A5683', '#FF773D']
ROOT.gStyle.SetOptStat(0)
ROOT.TH1F.SetDefaultSumw2(ROOT.kTRUE);
ROOT.gStyle.SetPalette(ROOT.kColorPrintableOnGrey)

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

def num_particles(decay: str, particle: str, legend: list, var_low: float, var_hi: float, nbins: int):

    hs_BKG = ROOT.THStack('hs_bkg', ';;A.U.')

    if ("el" in decay):
        hs_BKG.SetTitle("Events with >= 1 " + decay + " and >= 1 jet with pt > 30 GeV and |#eta| < 1.44, tightID; Number of " + particle + "/event ; A.U.") 
    if ("mu" in decay):
        hs_BKG.SetTitle("Events with >= 1 " + decay + " and >= 1 jet with pt > 30 GeV and |#eta| < 1.5, tightID; Number of " + particle + "/event ; A.U.") 
    if ('tau' in decay): 
         hs_BKG.SetTitle("Events with >=2 jets with pt > 20 GeV and |#eta| < 2.4, L = 38.01 fb^{-1};; A.U.")

    h_SIG = ROOT.TH1F('h_' + decay + '_SIG', ';Number of ' + particle +'/event; A.U.', nbins, var_low, var_hi)
    h_QCD_BKG = ROOT.TH1F('h_' + decay + '_QCD_BKG', ';Number of ' + particle + '/event; A.U.', nbins, var_low, var_hi)
    h_TT_BKG = ROOT.TH1F('h_' + decay + '_TT_BKG', ';Number of ' + particle + '/event; A.U.', nbins, var_low, var_hi)
    h_EWK_BKG = ROOT.TH1F('h_' + decay + '_EWK_BKG', ';Number of ' +particle + '/event; A.U.', nbins, var_low, var_hi)

    for file in SIG:
        print("Now plotting: " + file)
        signal_file = ak.from_parquet("my_skim_" + decay + "_" + file + "/*.parquet")
        if decay == "electron":
            electron_selection = signal_file["electron_pt"][(signal_file["electron_pt"] > selections["electron_pt"])
                                  & (abs(signal_file["electron_eta"]) < selections["electron_eta"])
                                  & (signal_file["electron_cutBased"] == selections["electron_cutBased"])
                                 ]
            num_electrons = ak.flatten(ak.num(electron_selection), axis = None)
            event_mask = num_electrons > 0
            event_mask = ak.flatten(event_mask, axis = None)
            num_jets = ak.flatten(ak.num(signal_file["jet_pt"][event_mask]), axis = None)
            print(num_jets)
            if particle == "electron":
                hist_SIG, bins_SIG = np.histogram(num_electrons, bins = np.linspace(var_low, var_hi, nbins+1), weights = [1/len(num_electrons),]*len(num_electrons))
                for i in range(nbins):
                    h_SIG.SetBinContent(i + 1, hist_SIG[i])
            if particle == "jet":
                hist_SIG, bins_SIG = np.histogram(num_jets, bins = np.linspace(var_low, var_hi, nbins+1), weights = [1/len(num_jets),]*len(num_jets))
                for i in range(nbins):
                    h_SIG.SetBinContent(i + 1, hist_SIG[i])
            
        if decay == "muon":
            muon_selection = signal_file["muon_pt"][(signal_file["muon_pt"] > selections["muon_pt"])
                              & (abs(signal_file["muon_eta"]) < selections["muon_eta"])
                              & (signal_file["muon_tightId"] ==  1)
                             ]
            num_muons = ak.num(muon_selection)
            event_mask = num_muons > 0 
            event_mask = ak.flatten(event_mask, axis = None)
            num_jets = ak.flatten(ak.num(signal_file["jet_pt"][event_mask]), axis = None)
            if particle == "muon":
                hist_SIG, bins_SIG = np.histogram(num_muons, bins = np.linspace(var_low, var_hi, nbins+1), weights = [1/len(num_muons),]*len(num_muons))
                for i in range(nbins):
                    h_SIG.SetBinContent(i + 1, hist_SIG[i])
            if particle == "jet":
                hist_SIG, bins_SIG = np.histogram(num_jets, bins = np.linspace(var_low, var_hi, nbins+1), weights = [1/len(num_jets),]*len(num_jets))
                for i in range(nbins):
                    h_SIG.SetBinContent(i + 1, hist_SIG[i])

    for file in BKG:
        print("Now plotting: " + file)
        background_file = ak.from_parquet("my_skim_" + decay + "_" + file + "/*.parquet")
        if decay == "electron":
            electron_selection = background_file["electron_pt"][(background_file["electron_pt"] > selections["electron_pt"])
                                  & (abs(background_file["electron_eta"]) < selections["electron_eta"])
                                  & (background_file["electron_cutBased"] == selections["electron_cutBased"])
                                 ]
            num_electrons = ak.flatten(ak.num(electron_selection), axis = None)
            event_mask = num_electrons > 0
            event_mask = ak.flatten(event_mask, axis = None)
            num_jets = ak.flatten(ak.num(background_file["jet_pt"][event_mask]), axis = None)
            if particle == "electron":
                if 'TT' in file:
                    h_TT, bins_TT = np.histogram(num_electrons, bins = np.linspace(var_low, var_hi, nbins+1), weights = [1/len(num_electrons),]*len(num_electrons))
                    for i in range(nbins):
                        h_TT_BKG.SetBinContent(i + 1, h_TT[i])
                if 'QCD' in file:
                    h_QCD, bins_QCD = np.histogram(num_electrons, bins = np.linspace(var_low, var_hi, nbins+1), weights = [1/len(num_electrons),]*len(num_electrons))
                    for i in range(nbins):
                        h_QCD_BKG.SetBinContent(i + 1, h_QCD[i])
                if 'Jets' in file:
                    h_EWK, bins_EWK = np.histogram(num_electrons, bins = np.linspace(var_low, var_hi, nbins+1), weights = [1/len(num_electrons),]*len(num_electrons))
                    for i in range(nbins): 
                        h_EWK_BKG.SetBinContent(i + 1, h_EWK[i])
            if particle == "jet":
                if 'TT' in file:
                    h_TT, bins_TT = np.histogram(num_jets, bins = np.linspace(var_low, var_hi, nbins+1), weights = [1/len(num_jets),]*len(num_jets))
                    for i in range(nbins):
                        h_TT_BKG.SetBinContent(i + 1, h_TT[i])
                if 'QCD' in file:
                    h_QCD, bins_QCD = np.histogram(num_jets, bins = np.linspace(var_low, var_hi, nbins+1), weights = [1/len(num_jets),]*len(num_jets))
                    for i in range(nbins):
                        h_QCD_BKG.SetBinContent(i + 1, h_QCD[i])
                if 'Jets' in file:
                    h_EWK, bins_EWK = np.histogram(num_jets, bins = np.linspace(var_low, var_hi, nbins+1), weights = [1/len(num_jets),]*len(num_jets))
                    for i in range(nbins): 
                        h_EWK_BKG.SetBinContent(i + 1, h_EWK[i])
            
        if decay == "muon":
            muon_selection = background_file["muon_pt"][(background_file["muon_pt"] > selections["muon_pt"])
                              & (abs(background_file["muon_eta"]) < selections["muon_eta"])
                              & (background_file["muon_tightId"] ==  1)
                             ]
            num_muons = ak.num(muon_selection)
            event_mask = num_muons > 0 
            event_mask = ak.flatten(event_mask, axis = None)
            num_jets = ak.flatten(ak.num(background_file["jet_pt"][event_mask]), axis = None)
            if particle == "muon":
                if 'TT' in file:
                    h_TT, bins_TT = np.histogram(num_muons, bins = np.linspace(var_low, var_hi, nbins+1), weights = [1/len(num_muons),]*len(num_muons))
                    for i in range(nbins):
                        h_TT_BKG.SetBinContent(i + 1, h_TT[i])
                if 'QCD' in file:
                    h_QCD, bins_QCD = np.histogram(num_muons, bins = np.linspace(var_low, var_hi, nbins+1), weights = [1/len(num_muons),]*len(num_muons))
                    for i in range(nbins):
                        h_QCD_BKG.SetBinContent(i + 1, h_QCD[i])
                if 'Jets' in file:
                    h_EWK, bins_EWK = np.histogram(num_muons, bins = np.linspace(var_low, var_hi, nbins+1), weights = [1/len(num_muons),]*len(num_muons))
                    for i in range(nbins): 
                        h_EWK_BKG.SetBinContent(i + 1, h_EWK[i])
            if particle == "jet":
                if 'TT' in file:
                    h_TT, bins_TT = np.histogram(num_jets, bins = np.linspace(var_low, var_hi, nbins+1), weights = [1/len(num_jets),]*len(num_jets))
                    for i in range(nbins):
                        h_TT_BKG.SetBinContent(i + 1, h_TT[i])
                if 'QCD' in file:
                    h_QCD, bins_QCD = np.histogram(num_jets, bins = np.linspace(var_low, var_hi, nbins+1), weights = [1/len(num_jets),]*len(num_jets))
                    for i in range(nbins):
                        h_QCD_BKG.SetBinContent(i + 1, h_QCD[i])
                if 'Jets' in file:
                    h_EWK, bins_EWK = np.histogram(num_jets, bins = np.linspace(var_low, var_hi, nbins+1), weights = [1/len(num_jets),]*len(num_jets))
                    for i in range(nbins): 
                        h_EWK_BKG.SetBinContent(i + 1, h_EWK[i])

    l_svb = ROOT.TLegend()
    
    if len(legend) > 0:
        l_svb = ROOT.TLegend(legend[0], legend[2], legend[1], legend[3])

    h_SIG.SetLineColor(ROOT.kBlack)    
    h_QCD_BKG.SetLineColor(ROOT.TColor.GetColor(colors[0]))
    h_TT_BKG.SetLineColor(ROOT.TColor.GetColor(colors[1]))
    h_EWK_BKG.SetLineColor(ROOT.TColor.GetColor(colors[2]))
    #h_QCD_BKG.SetFillColor(ROOT.TColor.GetColor(colors[0]))
    #h_TT_BKG.SetFillColor(ROOT.TColor.GetColor(colors[1]))
    #h_EWK_BKG.SetFillColor(ROOT.TColor.GetColor(colors[2]))

    h_QCD_BKG.Draw("histe")
    
    h_QCD_BKG.SetMaximum(1)

    h_TT_BKG.Draw("samehiste")
    h_EWK_BKG.Draw("samehiste")  
    h_SIG.Draw("samehiste")  

    l_svb.AddEntry(h_QCD_BKG, "QCD bkgd")
    l_svb.AddEntry(h_TT_BKG, "TT bkgd")
    l_svb.AddEntry(h_EWK_BKG, "DY + W bkgd")
    l_svb.AddEntry(h_SIG, "Stau 100 GeV 100mm")
    l_svb.Draw()

    can.SaveAs(decay + "_num_" + particle + "_sigvbkg_select.pdf")

def sig_v_bkg(decay: str, var: str, legend: list, var_low: float, var_hi: float, nbins: int, unit: str, log_set: bool):

    hs_BKG = ROOT.THStack('hs_bkg', ';;A.U.')
    h_QCD = {}


    if ("el" in decay) or ("mu" in decay):
        hs_BKG.SetTitle("Events with >= 1 " + decay + " and >= 1 jet with pt > 20 GeV and |#eta| < 2.4, L = 38.01 fb^{-1};" + var + ' ' + unit + '; A.U.')
    if ('tau' in decay): 
         hs_BKG.SetTitle("Events with >=2 jets with pt > 20 GeV and |#eta| < 2.4, L = 38.01 fb^{-1};" + var + ' ' + unit + '; A.U.')

    h_SIG = ROOT.TH1F('h_' + decay + '_' + var + '_SIG', ';' + var + ' ' + unit + '; A.U.', nbins, var_low, var_hi)
    h_QCD_BKG = ROOT.TH1F('h_' + decay + '_' + var + '_QCD_BKG', ';' + var + ' ' + unit + '; A.U.', nbins, var_low, var_hi)
    for file in BKG:
        if 'QCD' in file:
            h_QCD[file] = ROOT.TH1F('h_' + decay + '_' + var + '_' + file, ';' + var + ' ' + unit + '; A.U.', nbins, var_low, var_hi)
    h_TT_BKG = ROOT.TH1F('h_' + decay + '_' + var + '_TT_BKG', ';' + var + ' ' + unit + '; A.U.', nbins, var_low, var_hi)
    h_EWK_BKG = ROOT.TH1F('h_' + decay + '_' + var + '_EWK_BKG', ';' + var + ' ' + unit + '; A.U.', nbins, var_low, var_hi)

    for file in SIG:
        print("Now plotting: " + file)
        signal_file = ak.from_parquet("my_skim_" + decay + "_" + file + "/*.parquet")
        plot_var = signal_file[var]
        if "electron" in var or "muon" in var:
            if decay == "electron":
                plot_var = plot_var[(signal_file["electron_pt"] > selections["electron_pt"])
                                    #& (abs(signal_file["electron_eta"]) < selections["electron_eta"])
                                    & (signal_file["electron_cutBased"] == selections["electron_cutBased"])
                                    #& (abs(signal_file["electron_dxy"]) > 100E-4)
                                    & (abs(signal_file["electron_dxy"]) < 50E-4)
                                   ]
            if decay == "muon":
                plot_var = plot_var[(signal_file["muon_pt"] > selections["muon_pt"])
                                    #& (abs(signal_file["muon_eta"]) < selections["muon_eta"])
                                    & (signal_file["muon_tightId"])
                                    #& (abs(signal_file["muon_dxy"]) > 100E-4)
                                    & (abs(signal_file["muon_dxy"]) < 50E-4)
                                   ]
        else:

            if decay == "electron":
                electron_selection = signal_file["electron_pt"][(signal_file["electron_pt"] > selections["electron_pt"])
                                      #& (abs(signal_file["electron_eta"]) < selections["electron_eta"])
                                      & (signal_file["electron_cutBased"] == selections["electron_cutBased"])
                                      #& (abs(signal_file["electron_dxy"]) > 100E-4)
                                      & (abs(signal_file["electron_dxy"]) < 50E-4)
                                     ]
                num_electrons = ak.num(electron_selection)
                event_mask = num_electrons > 0
                event_mask = ak.flatten(event_mask, axis = None)
                plot_var = plot_var[event_mask]
            if decay == "muon":
                muon_selection = signal_file["muon_pt"][(signal_file["muon_pt"] > selections["muon_pt"])
                                  #& (abs(signal_file["muon_eta"]) < selections["muon_eta"])
                                  & (signal_file["muon_tightId"] ==  1)
                                  #& (abs(signal_file["muon_dxy"]) > 100E-4)
                                  & (abs(signal_file["muon_dxy"]) < 50E-4)
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
        if "2L2Nu" in file:
            background_file = TTto2L2Nu_file[decay]
        else:
            background_file = ak.from_parquet("my_skim_" + decay + "_" + file + "/*.parquet")
        plot_var = background_file[var]
        if "electron" in var or "muon" in var:
            if decay == "electron":
                plot_var = plot_var[(background_file["electron_pt"] > selections["electron_pt"])
                                    #& (abs(background_file["electron_eta"]) < selections["electron_eta"])
                                    & (background_file["electron_cutBased"] == selections["electron_cutBased"])
                                    #& (abs(background_file["electron_dxy"]) > 100E-4)
                                    & (abs(background_file["electron_dxy"]) < 50E-4)
                                   ]
            if decay == "muon":
                plot_var = plot_var[(background_file["muon_pt"] > selections["muon_pt"])
                                    #& (abs(background_file["muon_eta"]) < selections["muon_eta"])
                                    & (background_file["muon_tightId"])
                                    #& (abs(background_file["muon_dxy"]) > 100E-4)
                                    & (abs(background_file["muon_dxy"]) < 50E-4)
                                   ]
        else:
            if decay == "electron":
                electron_selection = background_file["electron_pt"][(background_file["electron_pt"] > selections["electron_pt"])
                                     # & (abs(background_file["electron_eta"]) < selections["electron_eta"])
                                      & (background_file["electron_cutBased"] == selections["electron_cutBased"])
                                      #& (abs(background_file["electron_dxy"]) < 100E-4)
                                      & (abs(background_file["electron_dxy"]) < 50E-4)
                                     ]
                num_electrons = ak.num(electron_selection)
                event_mask = num_electrons > 0
                event_mask = ak.flatten(event_mask, axis = None)
                plot_var = plot_var[event_mask]
            if decay == "muon":
                muon_selection = background_file["muon_pt"][(background_file["muon_pt"] > selections["muon_pt"])
                                  #& (abs(background_file["muon_eta"]) < selections["muon_eta"])
                                  & (background_file["muon_tightId"] ==  1)
                                  #& (abs(background_file["muon_dxy"]) > 100E-4)
                                  & (abs(background_file["muon_dxy"]) < 50E-4)
                                 ]
                num_muons = ak.num(muon_selection)
                event_mask = num_muons > 0
                event_mask = ak.flatten(event_mask, axis = None)
                plot_var = plot_var[event_mask]
            
        plot_var = ak.flatten(plot_var, axis = None)
        for val in plot_var:
            if 'QCD' in file:
                h_QCD_BKG.Fill(val, xsecs[file]*38.01*1000*1/num_events[file])
            #if file in h_QCD.keys():
            #    h_QCD[file].Fill(val, xsecs[file]*38.01*1000*1/num_events[file])
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

    #color = 0
    #for file in h_QCD.keys():
    #    h_QCD[file].SetLineColor(ROOT.gStyle.GetColorPalette(int((color + 1)*17)))
    #    h_QCD[file].SetFillColor(ROOT.gStyle.GetColorPalette(int((color + 1)*17)))
    #    color += 1 
    h_TT_BKG.SetLineColor(ROOT.TColor.GetColor(colors[14]))
    h_EWK_BKG.SetLineColor(ROOT.TColor.GetColor(colors[15]))
    h_QCD_BKG.SetFillColor(ROOT.TColor.GetColor(colors[0]))
    h_TT_BKG.SetFillColor(ROOT.TColor.GetColor(colors[14]))
    h_EWK_BKG.SetFillColor(ROOT.TColor.GetColor(colors[15]))
    h_SIG.SetLineColor(ROOT.kRed)    

    #h_QCD_BKG.Sumw2()
    #h_TT_BKG.Sumw2()
    #h_EWK_BKG.Sumw2()

    hs_BKG.Add(h_QCD_BKG)
    #for file in h_QCD.keys():
    #    hs_BKG.Add(h_QCD[file])
    hs_BKG.Add(h_TT_BKG)
    hs_BKG.Add(h_EWK_BKG)

    hs_BKG.Draw("histe")
    
    if log_set:
        hs_BKG.SetMinimum(1)
        hs_BKG.SetMaximum(10 * getMaximum(hs_BKG))
    elif not log_set:
        hs_BKG.SetMinimum(getMinimum(hs_BKG) / 10)
        hs_BKG.SetMaximum(getMaximum(hs_BKG))

    can.Update()
    h_SIG.Draw("samehiste")


    l_svb.AddEntry(h_QCD_BKG, "QCD bkgd")
    #for file in h_QCD.keys():
    #    l_svb.AddEntry(h_QCD[file], file)
    l_svb.AddEntry(h_TT_BKG, "TT bkgd")
    l_svb.AddEntry(h_EWK_BKG, "DY + W bkgd")
    l_svb.AddEntry(h_SIG, "Stau 100 GeV 100mm * 10^{5}")
    l_svb.Draw()
    #hs_BKG.SetTitle(decay + "_" + var + ", L = 38.01 fb^{-1}")
    can.SetLogy(log_set)
    if log_set:
        can.SaveAs(decay + "_" + var + "_sigvbkg_log_selection_prompt.pdf")
    if not log_set:
        can.SaveAs(decay + "_" + var + "_sigvbkg_selection_prompt.pdf")

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

#num_particles("electron", "electron", [0.73, 0.88, 0.73, 0.83], 0, 5, 5)
#num_particles("electron", "jet", [0.73, 0.88, 0.73, 0.83], 0, 10, 10)
#num_particles("muon", "muon", [0.73, 0.88, 0.73, 0.83], 0, 5, 5)
#num_particles("muon", "jet", [0.73, 0.88, 0.73, 0.83], 0, 10, 10)
 
sig_v_bkg("electron", "electron_pt", [], 20, 1000, 245, "[GeV]", 1) 
sig_v_bkg("electron", "jet_pt", [], 20, 1000, 245, "[GeV]", 1) 
sig_v_bkg("electron", "electron_pt", [], 20, 1000, 245, "[GeV]", 0) 
sig_v_bkg("electron", "jet_pt", [], 20, 1000, 245, "[GeV]", 0) 
sig_v_bkg("electron", "jet_eta", [], -2.4, 2.4, 48, "", 1) 
sig_v_bkg("electron", "jet_phi", [], -3.2, 3.2, 64, "[rad]", 1) 
sig_v_bkg("electron", "jet_score", [], 0, 1, 20, "", 1) 
sig_v_bkg("electron", "jet_score", [], 0, 1, 20, "", 0) 
sig_v_bkg("electron", "deta", [], -5, 5, 100, "", 1) 
sig_v_bkg("electron", "dphi", [], -3.2, 3.2, 64, "[rad]", 1) 
sig_v_bkg("electron", "dR", [], 0, 1, 20, "", 1) 
sig_v_bkg("electron", "MET_pT", [], 0, 500, 50, "[GeV]", 1) 
sig_v_bkg("electron", "MET_pT", [], 0, 500, 50, "[GeV]", 0) 
sig_v_bkg("electron", "electron_eta", [], -2.4, 2.4, 48, "", 0)
sig_v_bkg("electron", "electron_phi", [], -3.2, 3.2, 64, "[rad]",0) 
#sig_v_bkg("electron", "electron_charge", [], -1, 2, 3,"", 0)
sig_v_bkg("electron", "electron_dxy", [], -1, 1, 200, "[cm]", 0)
sig_v_bkg("electron", "electron_dz",[],  -1, 1, 200, "[cm]", 0)
sig_v_bkg("electron", "electron_eta", [], -2.4, 2.4, 48, "", 1)
sig_v_bkg("electron", "electron_phi", [], -3.2, 3.2, 64, "[rad]",1) 
#sig_v_bkg("electron", "electron_charge", [], -1, 2, 3,"", 1)
sig_v_bkg("electron", "electron_dxy", [], -1, 1, 200, "[cm]", 1)
sig_v_bkg("electron", "electron_dz",[],  -1, 1, 200, "[cm]", 1)
#sig_v_bkg("electron", "generator_scalePDF", [], 0, 3500, 100, "", 1)
sig_v_bkg("electron", "leadingjet_pt", [], 20, 1000, 245, "[GeV]", 1) 
sig_v_bkg("electron", "leadingjet_pt", [], 20, 1000, 245, "[GeV]", 0) 
sig_v_bkg("electron", "leadingjet_eta", [], -2.4, 2.4, 48, "", 1) 
sig_v_bkg("electron", "leadingjet_phi", [], -3.2, 3.2, 64, "[rad]", 1) 
sig_v_bkg("electron", "leadingjet_score", [], 0, 1, 20, "", 1) 
sig_v_bkg("electron", "leadingjet_eta", [], -2.4, 2.4, 48, "", 0) 
sig_v_bkg("electron", "leadingjet_phi", [], -3.2, 3.2, 64, "[rad]", 0) 
sig_v_bkg("electron", "leadingjet_score", [], 0, 1, 20, "", 0) 

sig_v_bkg("muon", "muon_pt", [], 20, 1000, 245, "[GeV]", 0) 
sig_v_bkg("muon", "muon_pt", [], 20, 1000, 245, "[GeV]", 1) 
sig_v_bkg("muon", "muon_pfRelIso03_all", [], 0, 1, 25, "", 1)
sig_v_bkg("muon", "muon_pfRelIso03_chg", [], 0, 1, 25, "", 1)
sig_v_bkg("muon", "muon_pfRelIso04_all", [], 0, 1, 25, "", 1)
sig_v_bkg("muon", "jet_pt", [], 20, 1000, 245, "[GeV]", 1) 
sig_v_bkg("muon", "jet_pt", [], 20, 1000, 245, "[GeV]", 0) 
sig_v_bkg("muon", "jet_eta", [], -2.4, 2.4, 48, "", 1) 
sig_v_bkg("muon", "jet_phi", [], -3.2, 3.2, 64, "[rad]", 1) 
sig_v_bkg("muon", "jet_eta", [], -2.4, 2.4, 48, "", 0) 
sig_v_bkg("muon", "jet_phi", [], -3.2, 3.2, 64, "[rad]", 0) 
sig_v_bkg("muon", "jet_score", [], 0, 1, 20, "", 1) 
sig_v_bkg("muon", "jet_score", [], 0, 1, 20, "", 0) 
sig_v_bkg("muon", "deta", [], -5, 5, 100, "", 1) 
sig_v_bkg("muon", "dphi", [], -3.2, 3.2, 64, "[rad]", 1) 
sig_v_bkg("muon", "dR", [], 0, 1, 20, "", 1) 
sig_v_bkg("muon", "MET_pT", [], 0, 500, 50, "[GeV]", 1) 
sig_v_bkg("muon", "MET_pT", [], 0, 500, 50, "[GeV]", 0) 
sig_v_bkg("muon", "muon_eta", [], -2.4, 2.4, 48, "", 0)
sig_v_bkg("muon", "muon_phi", [], -3.2, 3.2, 64, "[rad]", 0) 
sig_v_bkg("muon", "muon_charge", [], -1, 2, 3,"", 0)
sig_v_bkg("muon", "muon_dxy", [], -1, 1, 200, "", 0)
sig_v_bkg("muon", "muon_dz",[],  -1, 1, 200, "", 0)
sig_v_bkg("muon", "muon_eta", [], -2.4, 2.4, 48, "", 1)
sig_v_bkg("muon", "muon_phi", [], -3.2, 3.2, 64, "[rad]", 1) 
sig_v_bkg("muon", "muon_charge", [], -1, 2, 3,"", 1)
sig_v_bkg("muon", "muon_dxy", [], -1, 1, 200, "", 1)
sig_v_bkg("muon", "muon_dz",[],  -1, 1, 200, "", 1)
#sig_v_bkg("muon", "generator_scalePDF", [], 0, 3500, 100, "", 1)
sig_v_bkg("muon", "leadingjet_pt", [], 20, 1000, 245, "[GeV]", 1) 
sig_v_bkg("muon", "leadingjet_pt", [], 20, 1000, 245, "[GeV]", 0) 
sig_v_bkg("muon", "leadingjet_eta", [], -2.4, 2.4, 48, "", 1) 
sig_v_bkg("muon", "leadingjet_phi", [], -3.2, 3.2, 64, "[rad]", 1) 
sig_v_bkg("muon", "leadingjet_score", [], 0, 1, 20, "", 1) 
sig_v_bkg("muon", "leadingjet_eta", [], -2.4, 2.4, 48, "", 0) 
sig_v_bkg("muon", "leadingjet_phi", [], -3.2, 3.2, 64, "[rad]", 0) 
sig_v_bkg("muon", "leadingjet_score", [], 0, 1, 20, "", 0) 

#sig_v_bkg("ditau", "jet_pt", [], 20, 1000, 245, "[GeV]", 1) 
#sig_v_bkg("ditau", "jet_pt", [], 20, 1000, 245, "[GeV]", 0) 
#sig_v_bkg("ditau", "jet_eta", [], -2.4, 2.4, 48, "", 1) 
#sig_v_bkg("ditau", "jet_phi", [], -3.2, 3.2, 64, "[rad]", 1) 
#sig_v_bkg("ditau", "jet_score", [], 0, 1, 20, "", 1) 
#sig_v_bkg("ditau", "MET_pT", [], 0, 500, 50, "[GeV]", 1) 
#sig_v_bkg("ditau", "leadingjet_pt", [], 20, 1000, 245, "[GeV]", 1) 
#sig_v_bkg("ditau", "leadingjet_pt", [], 20, 1000, 245, "[GeV]", 0) 
#sig_v_bkg("ditau", "leadingjet_eta", [], -2.4, 2.4, 48, "", 1) 
#sig_v_bkg("ditau", "leadingjet_phi", [], -3.2, 3.2, 64, "[rad]", 1) 
#sig_v_bkg("ditau", "leadingjet_score", [], 0, 1, 20, "", 1) 
#sig_v_bkg("ditau", "leadingjet_score", [], 0, 1, 20, "", 0) 
#sig_v_bkg("ditau", "generator_scalePDF", [], 0, 3500, 100, "", 1)
