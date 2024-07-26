import uproot
import scipy
import matplotlib as mpl
import awkward as ak
import numpy as np
import math
import ROOT
import array
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt

GenPtMin = 0
GenEtaMax = 2.5

can = ROOT.TCanvas("can", "can")


ptRANGE = ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30-35", "35-40", "40-45", "45-50",
           "50-55", "55-60", "60-65", "65-70", "70-75", "75-80", "80-85", "85-90", "90-95", "95-100"]

etaRANGE = ["-3--2.8", "-2,8--2.6", "-2.6--2.4","-2.4--2.2", "-2.2--2.0", "-2.0--1.8", "-1.8--1.6", "-1.6--1.4", "-1.4--1.2", "-1.2--1.0", "-1.0--0.8", "-0.8--0.6", "-0.6--0.4", "-0.4--0.2", "-0.2-0",
            "0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0", "1.0-1.2", "1.2-1.4", "1.4-1.6", "1.6-1.8", "1.8-2.0", "2.0-2.2", "2.2-2.4", "2.4-2.6", "2.6-2.8", "2.8-3.0"]

dxyRANGE = ["0-0.001", "0.001-0.002", "0.002-0.003", "0.003-0.004", "0.004-0.005", "0.005-0.006", "0.006-0.007", "0.007-0.008", "0.008-0.009", "0.009-0.01",
            "0.01-0.011", "0.011-0.012", "0.012-0.013", "0.013-0.014", "0.014-0.015", "0.015-0.016", "0.016-0.017", "0.017-0.018", "0.018-0.019", "0.019-0.02",
            "0.02-0.021", "0.021-0.022", "0.022-0.023", "0.023-0.024", "0.024-0.025", "0.025-0.026", "0.026-0.027", "0.027-0.028", "0.028-0.029", "0.029-0.03",
            "0.03-0.031", "0.031-0.032", "0.032-0.033", "0.033-0.034", "0.034-0.035", "0.035-0.036"]

lxyRANGE = ["0-0.001", "0.001-0.002", "0.002-0.003", "0.003-0.004", "0.004-0.005", "0.005-0.006", "0.006-0.007", "0.007-0.008", "0.008-0.009", "0.009-0.01",
            "0.01-0.011", "0.011-0.012", "0.012-0.013", "0.013-0.014", "0.014-0.015", "0.015-0.016", "0.016-0.017", "0.017-0.018", "0.018-0.019", "0.019-0.02",
            "0.02-0.021", "0.021-0.022", "0.022-0.023", "0.023-0.024", "0.024-0.025", "0.025-0.026", "0.026-0.027", "0.027-0.028", "0.028-0.029", "0.029-0.03",
            "0.03-0.031", "0.031-0.032", "0.032-0.033", "0.033-0.034", "0.034-0.035", "0.035-0.036"]

ROOT.gStyle.SetOptStat(0)

file = "DYJetsToLL_M-50_TuneCP5_13p6TeV-madgraphMLM-pythia8_Run3Summer22MiniAODv4-forPOG_130X_mcRun3_2022_realistic_v5-v2_NanoAOD.root" 

rootFile = uproot.open(file)
Events = rootFile["Events"]

lepBranches = {}
genBranches = {}

for key in Events.keys():
  if ("Muon_" in key) or ("Electron_" in key) or ("Tau_" in key):
    lepBranches[key] = Events[key].array()
  if "GenPart_" in key:
    genBranches[key] = Events[key].array()

genBranches["GenMuon_dxy"] = ak.from_parquet("GenMuon_dxy.parquet")
genBranches["GenMuon_lxy"] = ak.from_parquet("GenMuon_lxy.parquet")

genBranches["GenElectron_dxy"] = ak.from_parquet("GenElectron_dxy.parquet")
genBranches["GenElectron_lxy"] = ak.from_parquet("GenElectron_lxy.parquet")
  

muFromZ = ak.from_parquet("muFromZ.parquet")
allMuFromZ = ak.from_parquet("allMuFromZ.parquet")

eFromZ = ak.from_parquet("eFromZ.parquet")
allEFromZ =  ak.from_parquet("allEFromZ.parquet")

#for i in range(20,40):
#  print("Event:", i)
#  print("Gen electrons:", eFromZ[i])
#  print("Reconstructed electrons:", lepBranches["Electron_genPartIdx"][i])


GenMuFromZ_pt = genBranches["GenPart_pt"][muFromZ] 
GenMuFromZ_eta = genBranches["GenPart_eta"][muFromZ]
GenMuFromZ_dxy = np.abs(genBranches["GenMuon_dxy"][muFromZ])
GenMuFromZ_lxy = genBranches["GenMuon_lxy"][muFromZ]

GenMuEtaCut = (abs(GenMuFromZ_eta) < 2.4)

GenMuFromZ_pt = GenMuFromZ_pt[GenMuEtaCut]
GenMuFromZ_eta = GenMuFromZ_eta[GenMuEtaCut]
GenMuFromZ_dxy = GenMuFromZ_dxy[GenMuEtaCut]
GenMuFromZ_lxy = GenMuFromZ_lxy[GenMuEtaCut]

GenEFromZ_pt = genBranches["GenPart_pt"][eFromZ]
GenEFromZ_eta = genBranches["GenPart_eta"][eFromZ]
GenEFromZ_dxy = np.abs(genBranches["GenElectron_dxy"][eFromZ])
GenEFromZ_lxy = genBranches["GenElectron_lxy"][eFromZ]

GenEEtaCut = (abs(GenEFromZ_eta) < 2.4)

GenEFromZ_pt = GenEFromZ_pt[GenEEtaCut]
GenEFromZ_eta = GenEFromZ_eta[GenEEtaCut] 
GenEFromZ_dxy = GenEFromZ_dxy[GenEEtaCut] 
GenEFromZ_lxy = GenEFromZ_lxy[GenEEtaCut] 

RecoMuonsFromGen = [np.array([val for val in subarr if val in values]) for subarr, values in zip(lepBranches["Muon_genPartIdx"], allMuFromZ)]
RecoMuonsFromGenIndices = [np.where(np.isin(subarr, values))[0] for subarr, values in zip(lepBranches["Muon_genPartIdx"], allMuFromZ)]


RecoElectronsFromGen = [np.array([val for val in subarr if val in values]) for subarr, values in zip(lepBranches["Electron_genPartIdx"], allEFromZ)]
RecoElectronsFromGenIndices = [np.where(np.isin(subarr, values))[0] for subarr, values in zip(lepBranches["Electron_genPartIdx"], allEFromZ)]

RecoMuons_pt = lepBranches["Muon_pt"][RecoMuonsFromGenIndices]
RecoMuons_eta = lepBranches["Muon_eta"][RecoMuonsFromGenIndices]
RecoMuons_dxy = np.abs(lepBranches["Muon_dxy"][RecoMuonsFromGenIndices])
RecoMuons_looseId = lepBranches["Muon_looseId"][RecoMuonsFromGenIndices]
RecoMuons_tightId = lepBranches["Muon_tightId"][RecoMuonsFromGenIndices]

RecoElectrons_pt =  lepBranches["Electron_pt"][RecoElectronsFromGenIndices]   
RecoElectrons_eta = lepBranches["Electron_eta"][RecoElectronsFromGenIndices]
RecoElectrons_dxy = np.abs(lepBranches["Electron_dxy"][RecoElectronsFromGenIndices])
RecoElectrons_cutBased = lepBranches["Electron_cutBased"][RecoElectronsFromGenIndices]

RecoMuons_isStandalone = lepBranches["Muon_isStandalone"][RecoMuonsFromGenIndices]
RecoMuons_isGlobal = lepBranches["Muon_isGlobal"][RecoMuonsFromGenIndices]


RecoMuonsFromGen_pt  = genBranches["GenPart_pt"][RecoMuonsFromGen]
RecoMuonsFromGen_eta = genBranches["GenPart_eta"][RecoMuonsFromGen]
RecoMuonsFromGen_dxy = np.abs(genBranches["GenMuon_dxy"][RecoMuonsFromGen])
RecoMuonsFromGen_lxy = genBranches["GenMuon_lxy"][RecoMuonsFromGen]

RecoMuonsFromGenEtaCut = (abs(RecoMuonsFromGen_eta) < 2.4)

RecoMuonsFromGen_pt = RecoMuonsFromGen_pt[RecoMuonsFromGenEtaCut]
RecoMuonsFromGen_eta = RecoMuonsFromGen_eta[RecoMuonsFromGenEtaCut]
RecoMuonsFromGen_dxy = RecoMuonsFromGen_dxy[RecoMuonsFromGenEtaCut]
RecoMuonsFromGen_lxy = RecoMuonsFromGen_lxy[RecoMuonsFromGenEtaCut]

RecoElectronsFromGen_pt  = genBranches["GenPart_pt"][RecoElectronsFromGen]
RecoElectronsFromGen_eta = genBranches["GenPart_eta"][RecoElectronsFromGen]
RecoElectronsFromGen_dxy = np.abs(genBranches["GenElectron_dxy"][RecoElectronsFromGen])
RecoElectronsFromGen_lxy = genBranches["GenElectron_lxy"][RecoElectronsFromGen]

RecoElectronsFromGenEtaCut = (abs(RecoElectronsFromGen_eta) < 2.4)

RecoElectronsFromGen_pt  = RecoElectronsFromGen_pt[RecoElectronsFromGenEtaCut]
RecoElectronsFromGen_eta = RecoElectronsFromGen_eta[RecoElectronsFromGenEtaCut]
RecoElectronsFromGen_dxy = RecoElectronsFromGen_dxy[RecoElectronsFromGenEtaCut]
RecoElectronsFromGen_lxy = RecoElectronsFromGen_lxy[RecoElectronsFromGenEtaCut]

RecoMuons_pt = RecoMuons_pt[RecoMuonsFromGenEtaCut]
RecoMuons_eta = RecoMuons_eta[RecoMuonsFromGenEtaCut]
RecoMuons_dxy = RecoMuons_dxy[RecoMuonsFromGenEtaCut]
RecoMuons_looseId = RecoMuons_looseId[RecoMuonsFromGenEtaCut]
RecoMuons_tightId = RecoMuons_tightId[RecoMuonsFromGenEtaCut]

RecoElectrons_pt =  RecoElectrons_pt[RecoElectronsFromGenEtaCut] 
RecoElectrons_eta = RecoElectrons_eta[RecoElectronsFromGenEtaCut]
RecoElectrons_dxy = RecoElectrons_dxy[RecoElectronsFromGenEtaCut]
RecoElectrons_cutBased = RecoElectrons_cutBased[RecoElectronsFromGenEtaCut]


RecoMuons_isStandalone = RecoMuons_isStandalone[RecoMuonsFromGenEtaCut]
RecoMuons_isGlobal = RecoMuons_isGlobal[RecoMuonsFromGenEtaCut]

SAMuons = ((RecoMuons_isStandalone == 1) & (RecoMuons_isGlobal != 1))
GlobalMuons = (RecoMuons_isGlobal == 1)
TightMuons = (RecoMuons_tightId == 1)
LooseMuons = (RecoMuons_looseId == 1)

TightElectrons = (RecoElectrons_cutBased >= 4)
MediumElectrons = (RecoElectrons_cutBased >= 3)
LooseElectrons = (RecoElectrons_cutBased >= 2)
VetoElectrons = (RecoElectrons_cutBased >= 1)

SARecoMuons_pt = RecoMuons_pt[SAMuons]
SARecoMuons_eta = RecoMuons_eta[SAMuons]
SARecoMuons_dxy = RecoMuons_dxy[SAMuons]

GlobalRecoMuons_pt = RecoMuons_pt[GlobalMuons]
GlobalRecoMuons_eta = RecoMuons_eta[GlobalMuons]
GlobalRecoMuons_dxy = RecoMuons_dxy[GlobalMuons]

LooseRecoMuons_pt = RecoMuons_pt[LooseMuons]
LooseRecoMuons_eta = RecoMuons_eta[LooseMuons]
LooseRecoMuons_dxy = RecoMuons_dxy[LooseMuons]

TightRecoMuons_pt = RecoMuons_pt[TightMuons]
TightRecoMuons_eta = RecoMuons_eta[TightMuons]
TightRecoMuons_dxy = RecoMuons_dxy[TightMuons]

VetoRecoElectrons_pt =  RecoElectrons_pt[VetoElectrons]
VetoRecoElectrons_eta = RecoElectrons_eta[VetoElectrons]
VetoRecoElectrons_dxy = RecoElectrons_dxy[VetoElectrons]

LooseRecoElectrons_pt =  RecoElectrons_pt[LooseElectrons]
LooseRecoElectrons_eta = RecoElectrons_eta[LooseElectrons]
LooseRecoElectrons_dxy = RecoElectrons_dxy[LooseElectrons]

MediumRecoElectrons_pt =  RecoElectrons_pt[MediumElectrons]
MediumRecoElectrons_eta = RecoElectrons_eta[MediumElectrons]
MediumRecoElectrons_dxy = RecoElectrons_dxy[MediumElectrons]

TightRecoElectrons_pt =  RecoElectrons_pt[TightElectrons]
TightRecoElectrons_eta = RecoElectrons_eta[TightElectrons]
TightRecoElectrons_dxy = RecoElectrons_dxy[TightElectrons]

SARecoMuonsFromGen_pt = RecoMuonsFromGen_pt[SAMuons]
SARecoMuonsFromGen_eta = RecoMuonsFromGen_eta[SAMuons]
SARecoMuonsFromGen_dxy = RecoMuonsFromGen_dxy[SAMuons]
SARecoMuonsFromGen_lxy = RecoMuonsFromGen_lxy[SAMuons]

GlobalRecoMuonsFromGen_pt = RecoMuonsFromGen_pt[GlobalMuons]
GlobalRecoMuonsFromGen_eta = RecoMuonsFromGen_eta[GlobalMuons]
GlobalRecoMuonsFromGen_dxy = RecoMuonsFromGen_dxy[GlobalMuons]
GlobalRecoMuonsFromGen_lxy = RecoMuonsFromGen_lxy[GlobalMuons]

LooseRecoMuonsFromGen_pt =  RecoMuonsFromGen_pt[LooseMuons]
LooseRecoMuonsFromGen_eta = RecoMuonsFromGen_eta[LooseMuons]
LooseRecoMuonsFromGen_dxy = RecoMuonsFromGen_dxy[LooseMuons]
LooseRecoMuonsFromGen_lxy = RecoMuonsFromGen_lxy[LooseMuons]

TightRecoMuonsFromGen_pt =  RecoMuonsFromGen_pt[TightMuons]
TightRecoMuonsFromGen_eta = RecoMuonsFromGen_eta[TightMuons]
TightRecoMuonsFromGen_dxy = RecoMuonsFromGen_dxy[TightMuons]
TightRecoMuonsFromGen_lxy = RecoMuonsFromGen_lxy[TightMuons]

VetoRecoElectronsFromGen_pt =  RecoElectronsFromGen_pt[VetoElectrons]
VetoRecoElectronsFromGen_eta = RecoElectronsFromGen_eta[VetoElectrons]
VetoRecoElectronsFromGen_dxy = RecoElectronsFromGen_dxy[VetoElectrons]

LooseRecoElectronsFromGen_pt =  RecoElectronsFromGen_pt[LooseElectrons]
LooseRecoElectronsFromGen_eta = RecoElectronsFromGen_eta[LooseElectrons]
LooseRecoElectronsFromGen_dxy = RecoElectronsFromGen_dxy[LooseElectrons]

MediumRecoElectronsFromGen_pt =  RecoElectronsFromGen_pt[MediumElectrons]
MediumRecoElectronsFromGen_eta = RecoElectronsFromGen_eta[MediumElectrons]
MediumRecoElectronsFromGen_dxy = RecoElectronsFromGen_dxy[MediumElectrons]

TightRecoElectronsFromGen_pt =  RecoElectronsFromGen_pt[TightElectrons]
TightRecoElectronsFromGen_eta = RecoElectronsFromGen_eta[TightElectrons]
TightRecoElectronsFromGen_dxy = RecoElectronsFromGen_dxy[TightElectrons]

RecoMuon_ptDiff = ak.Array(np.subtract(RecoMuons_pt, RecoMuonsFromGen_pt))
RecoMuon_etaDiff = ak.Array(np.subtract(RecoMuons_eta, RecoMuonsFromGen_eta))
RecoMuon_dxyDiff = ak.Array(np.subtract(RecoMuons_dxy, RecoMuonsFromGen_dxy))

SARecoMuon_ptDiff = ak.Array(np.subtract(RecoMuons_pt, RecoMuonsFromGen_pt))[SAMuons]
SARecoMuon_etaDiff = ak.Array(np.subtract(RecoMuons_eta, RecoMuonsFromGen_eta))[SAMuons]
SARecoMuon_dxyDiff = ak.Array(np.subtract(RecoMuons_dxy, RecoMuonsFromGen_dxy))[SAMuons]

GlobalRecoMuon_ptDiff = ak.Array(np.subtract(RecoMuons_pt, RecoMuonsFromGen_pt))[GlobalMuons]
GlobalRecoMuon_etaDiff = ak.Array(np.subtract(RecoMuons_eta, RecoMuonsFromGen_eta))[GlobalMuons]
GlobalRecoMuon_dxyDiff = ak.Array(np.subtract(RecoMuons_dxy, RecoMuonsFromGen_dxy))[GlobalMuons]

RecoElectron_ptDiff  = ak.Array(np.subtract(RecoElectrons_pt, RecoElectronsFromGen_pt))
RecoElectron_etaDiff = ak.Array(np.subtract(RecoElectrons_eta, RecoElectronsFromGen_eta))
RecoElectron_dxyDiff = ak.Array(np.subtract(RecoElectrons_dxy, RecoElectronsFromGen_dxy))

def makeResPlot(lepton, dict_entries, xvar, yvar, xrange, xmin, xmax, yresmin, yresmax, xbinsize, xvararr, yvardiff, xunit, yunit): 
  h_resVsX_y_dict = {}
  
  for ent in range(len(dict_entries)):
    h_resVsX_y_dict[dict_entries[ent]] = []
  
  for ent in range(len(dict_entries)):
    for i in range(len(xrange)):
      hist = ROOT.TH1F("h_resVs"+xvar+"_"+yvar+"_"+dict_entries[ent]+"_"+str(xrange[i]), ";"+yvar+"residual (reco - gen) "+yunit+";Number of leptons", 100, yresmin, yresmax)
      h_resVsX_y_dict[dict_entries[ent]].append(hist)

  #for ent in range(len(dict_entries)):
  #  for mu in range(len(ak.flatten(xvararr[ent]))):
  #    for x in range(len(xrange)):
  #      if ((ak.flatten(xvararr[ent])[mu] > (xmin + x * xbinsize)) & (ak.flatten(xvararr[ent])[mu] < (xmin + (x + 1) * xbinsize))):
  #        h_resVsX_y_dict[dict_entries[ent]][x].Fill(ak.flatten(yvardiff[ent])[mu])

  #for ent in range(len(dict_entries)):
  #  for x in range(len(xrange)):
  #    for mu in range(len(ak.flatten(xvararr[ent][(xvararr[ent] > (xmin + x * xbinsize)) & (xvararr[ent] < (xmin + (x + 1) * xbinsize))]))):
  #      h_resVsX_y_dict[dict_entries[ent]][x].Fill(ak.flatten(yvardiff[ent][(xvararr[ent] > (xmin + x * xbinsize)) & (xvararr[ent] < (xmin + (x + 1) * xbinsize))])[mu])

  for ent in range(len(dict_entries)):
    for x in range(len(xrange)):
      hist, bins, other = plt.hist(ak.flatten(yvardiff[ent][(xvararr[ent] > (xmin + x * xbinsize)) & (xvararr[ent] < (xmin + (x + 1) * xbinsize))]), bins=np.linspace(yresmin, yresmax, 101))
      for i in range(len(hist)):
        h_resVsX_y_dict[dict_entries[ent]][x].SetBinContent(i+1, hist[i])
  
  h2_resVsX_y_dict = {}
  for ent in range(len(dict_entries)):
    h2_resVsX_y_dict[dict_entries[ent]] = ROOT.TH1F("h2_resVsX_y_"+dict_entries[ent], "Muons from Z to"+lepton+lepton+" decay;gen "+lepton+" "+xvar+" "+xunit+";"+yvar+" resolution "+yunit, len(xrange), xmin, xmax)

  for ent in range(len(dict_entries)):
    for i in range(len(xrange)): 
      h2_resVsX_y_dict[dict_entries[ent]].SetBinContent(i + 1, h_resVsX_y_dict[dict_entries[ent]][i].GetRMS())
      h2_resVsX_y_dict[dict_entries[ent]].SetBinError(i + 1, h_resVsX_y_dict[dict_entries[ent]][i].GetRMSError())

  l_resVsX_y = ROOT.TLegend()
  l_resVsX_y.SetFillStyle(0)

  for ent in range(len(dict_entries)):
    h2_resVsX_y_dict[dict_entries[ent]].SetMinimum(0)
    h2_resVsX_y_dict[dict_entries[ent]].SetMarkerStyle(20)
    h2_resVsX_y_dict[dict_entries[ent]].SetMarkerColor(ent + 1)
    l_resVsX_y.AddEntry(h2_resVsX_y_dict[dict_entries[ent]], dict_entries[ent])
    if ent == 0:
      h2_resVsX_y_dict[dict_entries[ent]].Draw("p")
    else:
      h2_resVsX_y_dict[dict_entries[ent]].Draw("psame")
  
  l_resVsX_y.Draw()
  can.SaveAs("Zto"+lepton+lepton+"_resVs"+xvar+"_"+yvar+".pdf")


def makeEffPlot(lepton, dict_entries, xvar, bins, xmin, xmax, xbinsize, xunit, tot_arr, pass_arr, log_set):
  h_eff_dict = {}
  h_eff_num_dict = {}
  h_eff_den_dict = {}
  
  for ent in range(len(dict_entries)):
    h_eff_dict[dict_entries[ent]] = ROOT.TEfficiency("h_eff_"+xvar+"_"+dict_entries[ent], file.split("_")[0]+file.split("_")[1]+file.split("_")[2]+";"+xvar+" "+xunit+" ; Fraction of gen "+lepton+" which are recon'd", bins, xmin, xmax)
    h_eff_num_dict[dict_entries[ent]] = ROOT.TH1F("h_eff_num"+xvar+"_"+dict_entries[ent], file.split("_")[0]+file.split("_")[1]+file.split("_")[2]+";"+xvar+" "+xunit+" ; Number of gen "+lepton+" which are recon'd", bins, xmin, xmax)
    h_eff_den_dict[dict_entries[ent]] = ROOT.TH1F("h_eff_den"+xvar+"_"+dict_entries[ent], file.split("_")[0]+file.split("_")[1]+file.split("_")[2]+";"+xvar+" "+xunit+" ; Number of gen "+lepton, bins, xmin, xmax)

  for ent in range(len(dict_entries)):
    for i in range(bins):
      h_eff_dict[dict_entries[ent]].SetTotalEvents(i + 1, len(ak.flatten(tot_arr[(tot_arr > (xmin + (i * xbinsize))) & (tot_arr < (xmin + ((i + 1) * xbinsize)))])))
      h_eff_den_dict[dict_entries[ent]].SetBinContent(i + 1, len(ak.flatten(tot_arr[(tot_arr > (xmin + (i * xbinsize))) & (tot_arr < (xmin + ((i + 1) * xbinsize)))])))
      h_eff_dict[dict_entries[ent]].SetPassedEvents(i + 1, len(ak.flatten(pass_arr[ent][(pass_arr[ent] > (xmin + (i * xbinsize))) & (pass_arr[ent] < (xmin + ((i + 1) * xbinsize)))])))
      h_eff_num_dict[dict_entries[ent]].SetBinContent(i + 1, len(ak.flatten(pass_arr[ent][(pass_arr[ent] > (xmin + (i * xbinsize))) & (pass_arr[ent] < (xmin + ((i + 1) * xbinsize)))])))      

  can.SetLogy(log_set)
  l_eff = ROOT.TLegend()
  l_eff.SetBorderSize(0)
  for ent in range(len(dict_entries)):
    h_eff_dict[dict_entries[ent]].SetLineColor(ent + 1)
    l_eff.AddEntry(h_eff_dict[dict_entries[ent]], dict_entries[ent])
    if ent == 0:
      h_eff_dict[dict_entries[ent]].Draw()
      ROOT.gPad.Update()
      h_eff_dict[dict_entries[ent]].GetPaintedGraph().GetYaxis().SetRangeUser(0, 1.1)
    else:
      h_eff_dict[dict_entries[ent]].Draw("same")
  if len(dict_entries) > 1:
    l_eff.Draw()
  can.SaveAs("Zto"+lepton+lepton+"_eff_"+xvar+".pdf")

  l_eff_num = ROOT.TLegend()
  for ent in range(len(dict_entries)):
    h_eff_num_dict[dict_entries[ent]].SetLineColor(ent + 1)
    l_eff_num.AddEntry(h_eff_num_dict[dict_entries[ent]], dict_entries[ent])
    if ent == 0:
      h_eff_num_dict[dict_entries[ent]].Draw("hist")
    else:
      h_eff_num_dict[dict_entries[ent]].Draw("samehist")
  if len(dict_entries) > 1:
    l_eff_num.Draw()
  can.SaveAs("Zto"+lepton+lepton+"_eff_"+xvar+"_num.pdf")

  l_eff_den = ROOT.TLegend()
  for ent in range(len(dict_entries)):
    h_eff_den_dict[dict_entries[ent]].SetLineColor(ent + 1)
    l_eff_den.AddEntry(h_eff_den_dict[dict_entries[ent]], dict_entries[ent])
    if ent == 0:
      h_eff_den_dict[dict_entries[ent]].Draw("hist")
    else:
      h_eff_den_dict[dict_entries[ent]].Draw("samehist")
  if len(dict_entries) > 1:
    l_eff_den.Draw()
  can.SaveAs("Zto"+lepton+lepton+"_eff_"+xvar+"_den.pdf")

def makeEffPlotEta(lepton, dict_entries, xvar, xunit, tot_arr, etatot_arr, pass_arr, etapass_arr, log_set):
  h_eff_dict = {}
  h_eff_num_dict = {}
  h_eff_den_dict = {}

  ptBins = np.concatenate((np.linspace(0,100,51), np.linspace(105, 150, 10), np.linspace(160, 200, 5)))
  print(ptBins) 
  ptBins = array.array('d', ptBins)

  etaBins = [0, 1.479, 2.4]
  etaRegions = ["0-1.479", "1.479-2.4"]

  for i in etaRegions:
    h_eff_dict[i] = {}
    h_eff_num_dict[i] = {}
    h_eff_den_dict[i] = {} 
  
  for reg in range(len(etaRegions)):
    for ent in range(len(dict_entries)):
      h_eff_dict[etaRegions[reg]][dict_entries[ent]] = ROOT.TEfficiency("h_eff_"+xvar+"_eta"+etaRegions[reg]+"_"+dict_entries[ent], file.split("_")[0]+file.split("_")[1]+file.split("_")[2]+" eta "+etaRegions[reg]+";"+xvar+" "+xunit+" ; Fraction of gen "+lepton+" which are recon'd", len(ptBins) - 1, ptBins)
      h_eff_num_dict[etaRegions[reg]][dict_entries[ent]] = ROOT.TH1F("h_eff_num"+xvar+"_eta"+etaRegions[reg]+"_"+dict_entries[ent], file.split("_")[0]+file.split("_")[1]+file.split("_")[2]+" eta "+etaRegions[reg]+";"+xvar+" "+xunit+" ; Number of gen "+lepton+" which are recon'd", len(ptBins) - 1, ptBins)
      h_eff_den_dict[etaRegions[reg]][dict_entries[ent]] = ROOT.TH1F("h_eff_den"+xvar+"_eta"+etaRegions[reg]+"_"+dict_entries[ent], file.split("_")[0]+file.split("_")[1]+file.split("_")[2]+" eta "+etaRegions[reg]+";"+xvar+" "+xunit+" ; Number of gen "+lepton, len(ptBins) - 1, ptBins)

  for reg in range(len(etaRegions)):
    for ent in range(len(dict_entries)):
      for i in range(len(ptBins) - 1):
        h_eff_dict[etaRegions[reg]][dict_entries[ent]].SetTotalEvents(i + 1, len(ak.flatten(tot_arr[(abs(etatot_arr) > etaBins[reg]) & (abs(etatot_arr) <= etaBins[reg + 1]) & (tot_arr > ptBins[i]) & (tot_arr <= ptBins[i + 1])])))
        h_eff_den_dict[etaRegions[reg]][dict_entries[ent]].SetBinContent(i + 1, len(ak.flatten(tot_arr[(abs(etatot_arr) > etaBins[reg]) & (abs(etatot_arr) <= etaBins[reg + 1]) & (tot_arr > ptBins[i]) & (tot_arr < ptBins[i + 1])])))
        h_eff_dict[etaRegions[reg]][dict_entries[ent]].SetPassedEvents(i + 1, len(ak.flatten(pass_arr[ent][(abs(etapass_arr[ent]) > etaBins[reg]) & (abs(etapass_arr[ent]) <= etaBins[reg + 1]) & (pass_arr[ent] > ptBins[i]) & (pass_arr[ent] <= ptBins[i + 1])])))
        h_eff_num_dict[etaRegions[reg]][dict_entries[ent]].SetBinContent(i + 1, len(ak.flatten(pass_arr[ent][(abs(etapass_arr[ent]) > etaBins[reg]) & (abs(etapass_arr[ent]) <= etaBins[reg + 1]) & (pass_arr[ent] > ptBins[i]) & (pass_arr[ent] <= ptBins[i + 1])])))      

  for reg in range(len(etaRegions)):
    can.SetLogy(log_set)
    l_eff = ROOT.TLegend()
    l_eff.SetBorderSize(0)
    for ent in range(len(dict_entries)):
      h_eff_dict[etaRegions[reg]][dict_entries[ent]].SetLineColor(ent + 1)
      l_eff.AddEntry(h_eff_dict[etaRegions[reg]][dict_entries[ent]], dict_entries[ent])
      if ent == 0:
        h_eff_dict[etaRegions[reg]][dict_entries[ent]].Draw()
        ROOT.gPad.Update()
        h_eff_dict[etaRegions[reg]][dict_entries[ent]].GetPaintedGraph().GetYaxis().SetRangeUser(0, 1.1)
      else:
        h_eff_dict[etaRegions[reg]][dict_entries[ent]].Draw("same")
    if len(dict_entries) > 1:
      l_eff.Draw()
    can.SaveAs("Zto"+lepton+lepton+"_eff_"+xvar+"_eta"+etaRegions[reg]+".png")

    l_eff_num = ROOT.TLegend()
    for ent in range(len(dict_entries)):
      h_eff_num_dict[etaRegions[reg]][dict_entries[ent]].SetLineColor(ent + 1)
      l_eff_num.AddEntry(h_eff_num_dict[etaRegions[reg]][dict_entries[ent]], dict_entries[ent])
      if ent == 0:
        h_eff_num_dict[etaRegions[reg]][dict_entries[ent]].Draw("hist")
      else:
        h_eff_num_dict[etaRegions[reg]][dict_entries[ent]].Draw("samehist")
    if len(dict_entries) > 1:
      l_eff_num.Draw()
    can.SaveAs("Zto"+lepton+lepton+"_eff_"+xvar+"_eta"+etaRegions[reg]+"_num.pdf")

    l_eff_den = ROOT.TLegend()
    for ent in range(len(dict_entries)):
      h_eff_den_dict[etaRegions[reg]][dict_entries[ent]].SetLineColor(ent + 1)
      l_eff_den.AddEntry(h_eff_den_dict[etaRegions[reg]][dict_entries[ent]], dict_entries[ent])
      if ent == 0:
        h_eff_den_dict[etaRegions[reg]][dict_entries[ent]].Draw("hist")
      else:
        h_eff_den_dict[etaRegions[reg]][dict_entries[ent]].Draw("samehist")
    if len(dict_entries) > 1:
      l_eff_den.Draw()
    can.SaveAs("Zto"+lepton+lepton+"_eff_"+xvar+"_eta"+etaRegions[reg]+"_den.pdf")




print("The length of pt array is", len(ak.flatten(RecoMuonsFromGen_pt)))
print("The length of eta array is", len(ak.flatten(RecoMuonsFromGen_eta)))

#makeEffPlot("mu", ["all", "STA", "Global"], "pt", 200, 0, 100, 0.5, "[GeV]", GenMuFromZ_pt, [RecoMuonsFromGen_pt, SARecoMuonsFromGen_pt, GlobalRecoMuonsFromGen_pt], 0)
#makeEffPlot("mu", ["all", "STA", "Global"], "eta", 30, -3, 3, 0.2, " ", GenMuFromZ_eta, [RecoMuonsFromGen_eta, SARecoMuonsFromGen_eta, GlobalRecoMuonsFromGen_eta], 0)
#makeEffPlot("mu", ["all", "STA", "Global"], "dxy", 36, 0, 0.036E-3, 0.001E-3, "[cm]", GenMuFromZ_dxy, [RecoMuonsFromGen_dxy, SARecoMuonsFromGen_dxy, GlobalRecoMuonsFromGen_dxy], 0)
#makeEffPlot("mu", ["all", "STA", "Global"], "lxy", 36, 0, 0.036E-3, 0.001E-3, "[cm]", GenMuFromZ_lxy, [RecoMuonsFromGen_lxy, SARecoMuonsFromGen_lxy, GlobalRecoMuonsFromGen_lxy], 0)
#makeEffPlot("e", ["no ID", "Loose ID", "Tight ID"], "pt", 200, 0, 100, 0.5, "[GeV]", GenEFromZ_pt, [RecoElectronsFromGen_pt, LooseRecoElectronsFromGen_pt, TightRecoElectronsFromGen_pt],  0)

#makeEffPlotEta("mu", ["no ID", "Loose ID", "Tight ID"], "pt", "[GeV]", GenMuFromZ_pt, [RecoMuonsFromGen_pt, LooseRecoMuonsFromGen_pt, TightRecoMuonsFromGen_pt], [RecoMuonsFromGen_eta, LooseRecoMuonsFromGen_eta, TightRecoMuonsFromGen_eta], 0)
makeEffPlotEta("e", ["no ID", "Veto", "Loose ID", "Medium ID", "Tight ID"], "pt", "[GeV]", GenEFromZ_pt, GenEFromZ_eta,[RecoElectronsFromGen_pt, VetoRecoElectronsFromGen_pt, LooseRecoElectronsFromGen_pt, MediumRecoElectronsFromGen_pt, TightRecoElectronsFromGen_pt], [RecoElectronsFromGen_eta, VetoRecoElectronsFromGen_eta, LooseRecoElectronsFromGen_eta, MediumRecoElectronsFromGen_eta, TightRecoElectronsFromGen_eta], 0)
#makeEffPlot("e", ["all"], "pt", 20, 0, 100, 5, "[GeV]", GenEFromZ_pt, [RecoElectronsFromGen_pt], 0)
#makeEffPlot("e", ["all"], "eta", 24, -2.4, 2.4, 0.2, " ", GenEFromZ_eta, [RecoElectronsFromGen_eta], 0)
#makeEffPlot("e", ["all"], "dxy", 36, 0, 0.036E-3, 0.001E-3, "[cm]", GenEFromZ_dxy, [RecoElectronsFromGen_dxy], 0)
#makeEffPlot("e", ["all"], "lxy", 36, 0, 0.036E-3, 0.001E-3, "[cm]", GenEFromZ_lxy, [RecoElectronsFromGen_lxy], 0)
#
#makeResPlot("mu", ["all", "STA", "Global"], "Pt", "pt", ptRANGE, 0, 100, -5, 5, 5, [RecoMuonsFromGen_pt, SARecoMuonsFromGen_pt, GlobalRecoMuonsFromGen_pt], [RecoMuon_ptDiff, SARecoMuon_ptDiff, GlobalRecoMuon_ptDiff], "[GeV]", "[GeV]")
#makeResPlot("mu", ["all", "STA", "Global"], "Eta", "pt", etaRANGE, -3, 3, -5, 5, 0.2, [RecoMuonsFromGen_eta, SARecoMuonsFromGen_eta, GlobalRecoMuonsFromGen_eta], [RecoMuon_ptDiff, SARecoMuon_ptDiff, GlobalRecoMuon_ptDiff], " " , "[GeV]")
#makeResPlot("mu", ["all", "STA", "Global"], "Dxy", "pt", dxyRANGE, 0, 3.6E-5, -5, 5, 0.1E-5, [RecoMuonsFromGen_dxy, SARecoMuonsFromGen_dxy, GlobalRecoMuonsFromGen_dxy], [RecoMuon_ptDiff, SARecoMuon_ptDiff, GlobalRecoMuon_ptDiff], "[cm] " , "[GeV]")
#makeResPlot("mu", ["all", "STA", "Global"], "Lxy", "pt", lxyRANGE, 0, 3.6E-5, -5, 5, 0.1E-5, [RecoMuonsFromGen_lxy, SARecoMuonsFromGen_lxy, GlobalRecoMuonsFromGen_lxy], [RecoMuon_ptDiff, SARecoMuon_ptDiff, GlobalRecoMuon_ptDiff], "[cm] " , "[GeV]")
#makeResPlot("mu", ["all", "STA", "Global"], "Pt", "dxy", ptRANGE, 0, 100, -0.5, 0.5, 5, [RecoMuonsFromGen_pt, SARecoMuonsFromGen_pt, GlobalRecoMuonsFromGen_pt], [RecoMuon_dxyDiff, SARecoMuon_dxyDiff, GlobalRecoMuon_dxyDiff], "[GeV]", "[cm]")
#makeResPlot("mu", ["all", "STA", "Global"], "Eta", "dxy", etaRANGE, -3, 3, -0.5, 0.5, 0.2, [RecoMuonsFromGen_eta, SARecoMuonsFromGen_eta, GlobalRecoMuonsFromGen_eta], [RecoMuon_dxyDiff, SARecoMuon_dxyDiff, GlobalRecoMuon_dxyDiff], " ", "[cm]")
#makeResPlot("mu", ["all", "STA", "Global"], "Dxy", "dxy", dxyRANGE, 0, 3.6E-5, -0.5, 0.5, 0.1E-5, [RecoMuonsFromGen_dxy, SARecoMuonsFromGen_dxy, GlobalRecoMuonsFromGen_dxy], [RecoMuon_dxyDiff, SARecoMuon_dxyDiff, GlobalRecoMuon_dxyDiff], "[cm]", "[cm]")
#makeResPlot("mu", ["all", "STA", "Global"], "Lxy",  "dxy", lxyRANGE, 0, 3.6E-5, -0.5, 0.5, 0.1E-5, [RecoMuonsFromGen_lxy, SARecoMuonsFromGen_lxy, GlobalRecoMuonsFromGen_lxy], [RecoMuon_dxyDiff, SARecoMuon_dxyDiff, GlobalRecoMuon_dxyDiff], "[cm]", "[cm]")
#
#makeResPlot("e", ["all"], "Pt", "pt", ptRANGE, 0, 100, -5, 5, 5, [RecoElectronsFromGen_pt], [RecoElectron_ptDiff], "[GeV]", "[GeV]")
#makeResPlot("e", ["all"], "Eta", "pt", etaRANGE, -3, 3, -5, 5, 0.2, [RecoElectronsFromGen_eta], [RecoElectron_ptDiff], " " , "[GeV]")
#makeResPlot("e", ["all"], "Dxy", "pt", dxyRANGE, 0, 3.6E-5, -5, 5, 0.1E-5, [RecoElectronsFromGen_dxy], [RecoElectron_ptDiff], "[cm]", "[GeV]")
#makeResPlot("e", ["all"], "Lxy", "pt", lxyRANGE, 0, 3.6E-5, -5, 5, 0.1E-5, [RecoElectronsFromGen_lxy], [RecoElectron_ptDiff], "[cm]", "[GeV]")
#makeResPlot("e", ["all"], "Pt", "dxy", ptRANGE, 0, 100, -0.5, 0.5, 5, [RecoElectronsFromGen_pt], [RecoElectron_dxyDiff], "[GeV]", "[cm]")
#makeResPlot("e", ["all"], "Eta", "dxy", etaRANGE, -3, 3, -0.5, 0.5, 0.2, [RecoElectronsFromGen_eta], [RecoElectron_dxyDiff], " ", "[cm]")
#makeResPlot("e", ["all"], "Dxy", "dxy", dxyRANGE, 0, 3.6E-5, -0.5, 0.5, 0.1E-5, [RecoElectronsFromGen_dxy], [RecoElectron_dxyDiff], "[cm]", "[cm]")
#makeResPlot("e", ["all"], "Lxy",  "dxy", lxyRANGE, 0, 3.6E-5, -0.5, 0.5, 0.1E-5, [RecoElectronsFromGen_lxy], [RecoElectron_dxyDiff], "[cm]", "[cm]")
