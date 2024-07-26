import uproot
import scipy
import matplotlib as mpl
import awkward as ak
import numpy as np
import math
import ROOT
import array
import pandas as pd

MuGenPtMin = 20
MuGenEtaMax = 2.4

ElGenPtMin = 20
ElGenEtaMax = 1.44

ptbins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 70, 90, 110, 130, 150, 200]
ptbins = array.array('d', ptbins)

ROOT.gStyle.SetOptStat(0)
stauFile = uproot.open("Staus_M_100_100mm_13p6TeV_Run3Summer22_DisMuon_GenPartMatch.root")
Events = stauFile["Events"]
lepBranches = {}
genPartBranches = {}

can = ROOT.TCanvas("can", "can")

for key in Events.keys():
  if ("Muon_" in key) or ("Electron_" in key):
    lepBranches[key] = Events[key].array()
  if "GenPart_" in key:
    genPartBranches[key] = Events[key].array()

MuonFakePt = lepBranches["Muon_pt"][(lepBranches["Muon_genPartIdx"] < 0) & (lepBranches["Muon_pt"] > MuGenPtMin) & (abs(lepBranches["Muon_eta"]) < MuGenEtaMax)]
MuonFakeEta = lepBranches["Muon_eta"][(lepBranches["Muon_genPartIdx"] < 0) & (lepBranches["Muon_pt"] > MuGenPtMin) & (abs(lepBranches["Muon_eta"]) < MuGenEtaMax)]
MuonFakeDxy = lepBranches["Muon_dxy"][(lepBranches["Muon_genPartIdx"] < 0) & (lepBranches["Muon_pt"] > MuGenPtMin) & (abs(lepBranches["Muon_eta"]) < MuGenEtaMax)]

DisMuonFakePt = lepBranches["DisMuon_pt"][(lepBranches["DisMuon_genPartIdx"] < 0) & (lepBranches["DisMuon_pt"] > MuGenPtMin) & (abs(lepBranches["DisMuon_eta"]) < MuGenEtaMax)]
DisMuonFakeEta = lepBranches["DisMuon_eta"][(lepBranches["DisMuon_genPartIdx"] < 0) & (lepBranches["DisMuon_pt"] > MuGenPtMin) & (abs(lepBranches["DisMuon_eta"]) < MuGenEtaMax)]
DisMuonFakeDxy = lepBranches["DisMuon_dxy"][(lepBranches["DisMuon_genPartIdx"] < 0) & (lepBranches["DisMuon_pt"] > MuGenPtMin) & (abs(lepBranches["DisMuon_eta"]) < MuGenEtaMax)]


def isMotherStau(evt, part_idx):
  #Find its mother
  mother_idx = genPartBranches["GenPart_genPartIdxMother"][evt][part_idx]
  #If it's a stau, we're done
  if (abs(genPartBranches["GenPart_pdgId"][evt][mother_idx]) == 1000015):
    return 1
  #If it's a tau, go further up the chain
  if (abs(genPartBranches["GenPart_pdgId"][evt][mother_idx]) == 15):
    return isMotherStau(evt, mother_idx)
  #Else, it's  not what we're looking for
  else:
    return 0        

def isMotherStauIdx(evt, part_idx):
  mother_idx = genPartBranches["GenPart_genPartIdxMother"][evt][part_idx]
  if (abs(genPartBranches["GenPart_pdgId"][evt][mother_idx]) == 1000015):
    return mother_idx
  else:
    return isMotherStauIdx(evt, mother_idx)

     
# This chooses the gen particles which are muons which are decayed from displaced taus
genDisMuon = ak.from_parquet("genDisMuon.parquet")
genDisMuonPt = genPartBranches["GenPart_pt"][genDisMuon]
genDisMuonEta = genPartBranches["GenPart_eta"][genDisMuon]
genDisMuonPhi = genPartBranches["GenPart_phi"][genDisMuon]

genDisEle = ak.from_parquet("genDisElectron.parquet")
genDisElePt = genPartBranches["GenPart_pt"][genDisEle]
genDisEleEta = genPartBranches["GenPart_eta"][genDisEle]
genDisElePhi = genPartBranches["GenPart_phi"][genDisEle]

RecoMuon = [np.array([val for val in subarr if val in values]) for subarr, values in zip(lepBranches["Muon_genPartIdx"], genDisMuon)]
RecoMuonIndices = [np.where(np.isin(subarr, values))[0] for subarr, values in zip(lepBranches["Muon_genPartIdx"], genDisMuon)]

RecoDisMuon = [np.array([val for val in subarr if val in values]) for subarr, values in zip(lepBranches["DisMuon_genPartIdx"], genDisMuon)]
RecoDisMuonIndices = [np.where(np.isin(subarr, values))[0] for subarr, values in zip(lepBranches["DisMuon_genPartIdx"], genDisMuon)]

RecoEle = [np.array([val for val in subarr if val in values]) for subarr, values in zip(lepBranches["Electron_genPartIdx"], genDisEle)]
RecoEleIndices = [np.where(np.isin(subarr, values))[0] for subarr, values in zip(lepBranches["Electron_genPartIdx"], genDisEle)]

RecoMuonPt = lepBranches["Muon_pt"][RecoMuonIndices]
RecoDisMuonPt = lepBranches["DisMuon_pt"][RecoDisMuonIndices]
RecoElePt = lepBranches["Electron_pt"][RecoEleIndices]

RecoMuonEta = lepBranches["Muon_eta"][RecoMuonIndices]
RecoDisMuonEta = lepBranches["DisMuon_eta"][RecoDisMuonIndices]
RecoEleEta = lepBranches["Electron_eta"][RecoEleIndices]

RecoMuonGenPartIdx = lepBranches["Muon_genPartIdx"][RecoMuonIndices]
RecoDisMuonGenPartIdx= lepBranches["DisMuon_genPartIdx"][RecoDisMuonIndices]

RecoMuonDxy = lepBranches["Muon_dxy"][RecoMuonIndices]
RecoDisMuonDxy = lepBranches["DisMuon_dxy"][RecoDisMuonIndices]

MuonPtCut = (RecoMuonPt > MuGenPtMin)
MuonEtaCut = (abs(RecoMuonEta) < MuGenEtaMax)
MuonFakeCut = (RecoMuonGenPartIdx < 0)

DisMuonPtCut = (RecoDisMuonPt > MuGenPtMin)
DisMuonEtaCut = (abs(RecoDisMuonEta) < MuGenEtaMax)
DisMuonFakeCut = (RecoDisMuonGenPartIdx < 0)

ElePtCut = (RecoElePt > ElGenPtMin)
EleEtaCut = (abs(RecoEleEta) < ElGenEtaMax)

CutMuonPt = lepBranches["Muon_pt"][(lepBranches["Muon_pt"] > MuGenPtMin) & (abs(lepBranches["Muon_eta"]) < MuGenEtaMax)]
CutMuonEta = lepBranches["Muon_eta"][(lepBranches["Muon_pt"] > MuGenPtMin) & (abs(lepBranches["Muon_eta"]) < MuGenEtaMax)]
CutMuonDxy = lepBranches["Muon_dxy"][(lepBranches["Muon_pt"] > MuGenPtMin) & (abs(lepBranches["Muon_eta"]) < MuGenEtaMax)]

CutDisMuonPt = lepBranches["DisMuon_pt"][(lepBranches["DisMuon_pt"] > MuGenPtMin) & (abs(lepBranches["DisMuon_eta"]) < MuGenEtaMax)]
CutDisMuonEta = lepBranches["DisMuon_eta"][(lepBranches["DisMuon_pt"] > MuGenPtMin) & (abs(lepBranches["DisMuon_eta"]) < MuGenEtaMax)]
CutDisMuonDxy = lepBranches["DisMuon_dxy"][(lepBranches["DisMuon_pt"] > MuGenPtMin) & (abs(lepBranches["DisMuon_eta"]) < MuGenEtaMax)]



CutRecoMuonPt = RecoMuonPt[MuonPtCut & MuonEtaCut]
CutRecoDisMuonPt = RecoDisMuonPt[DisMuonPtCut & DisMuonEtaCut] 
CutRecoElePt = RecoElePt[ElePtCut & EleEtaCut]

CutRecoMuonEta = RecoMuonEta[MuonPtCut & MuonEtaCut]
CutRecoDisMuonEta = RecoDisMuonEta[DisMuonPtCut & DisMuonEtaCut] 
CutRecoEleEta = RecoEleEta[ElePtCut & EleEtaCut]

h_reco_muon_pt = ROOT.TH1F("h_reco_muon_pt", "Staus_M_100_100mm_13p6TeV_Run3Summer22;#mu p_{T} [GeV]; Entries/5.0 GeV", 101, 0, 505)
h_reco_dis_muon_pt = ROOT.TH1F("h_reco_dis_muon_pt", "Staus_M_100_100mm_13p6TeV_Run3Summer22;#mu p_{T} [GeV]; Entries/5.0 GeV", 101, 0, 505)
h_reco_ele_pt = ROOT.TH1F("h_reco_ele_pt", "Staus_M_100_100mm_13p6TeV_Run3Summer22;e p_{T} [GeV]; Entries/5.0 GeV", 101, 0, 505)

h_reco_muon_eta = ROOT.TH1F("h_reco_muon_eta", "Staus_M_100_100mm_13p6TeV_Run3Summer22;#mu #eta; Entries/0.1", 50, -2.5, 2.5)
h_reco_dis_muon_eta = ROOT.TH1F("h_reco_dis_muon_eta", "Staus_M_100_100mm_13p6TeV_Run3Summer22;#mu #eta; Entries/0.1", 50, -2.5, 2.5)
h_reco_ele_eta = ROOT.TH1F("h_reco_ele_eta", "Staus_M_100_100mm_13p6TeV_Run3Summer22;e #eta [GeV]; Entries/0.1", 50, -2.5, 2.5)

h_fake_muon_pt = ROOT.TH1F("h_fake_muon_pt", "Staus_M_100_100mm_13p6TeV_Run3Summer22;#mu p_{T} [GeV]; Entries/5.0 GeV", len(ptbins) - 1, ptbins)
h_fake_dis_muon_pt = ROOT.TH1F("h_fake_dis_muon_pt", "Staus_M_100_100mm_13p6TeV_Run3Summer22;#mu p_{T} [GeV]; Entries/5.0 GeV", len(ptbins) - 1, ptbins)

h_fake_muon_eta = ROOT.TH1F("h_fake_muon_eta", "Staus_M_100_100mm_13p6TeV_Run3Summer22;#mu #eta; Entries/0.1", 50, -2.5, 2.5)
h_fake_dis_muon_eta = ROOT.TH1F("h_fake_dis_muon_eta", "Staus_M_100_100mm_13p6TeV_Run3Summer22;#mu #eta; Entries/0.1", 50, -2.5, 2.5)

h_fake_muon_dxy = ROOT.TH1F("h_fake_muon_dxy", "Staus_M_100_100mm_13p6TeV_Run3Summer22;#mu d_{xy} [cm]; Entries/0.1 cm", 100, -5, 5)
h_fake_dis_muon_dxy = ROOT.TH1F("h_fake_dis_muon_dxy", "Staus_M_100_100mm_13p6TeV_Run3Summer22;#mu d_{xy} [cm]; Entries/0.1 cm", 100, -5, 5)

for mu in range(len(ak.flatten(MuonFakePt))):
  h_fake_muon_pt.Fill(ak.flatten(MuonFakePt)[mu])
  h_fake_muon_eta.Fill(ak.flatten(MuonFakeEta)[mu])
  h_fake_muon_dxy.Fill(ak.flatten(MuonFakeDxy)[mu])

for mu in range(len(ak.flatten(DisMuonFakePt))):
  h_fake_dis_muon_pt.Fill(ak.flatten(DisMuonFakePt)[mu])
  h_fake_dis_muon_eta.Fill(ak.flatten(DisMuonFakeEta)[mu])
  h_fake_dis_muon_dxy.Fill(ak.flatten(DisMuonFakeDxy)[mu])

for mu in range(len(ak.flatten(CutRecoMuonPt))):
  h_reco_muon_pt.Fill(ak.flatten(CutRecoMuonPt)[mu])
  h_reco_muon_eta.Fill(ak.flatten(CutRecoMuonEta)[mu])

for mu in range(len(ak.flatten(CutRecoDisMuonPt))):
  h_reco_dis_muon_pt.Fill(ak.flatten(CutRecoDisMuonPt)[mu])
  h_reco_dis_muon_eta.Fill(ak.flatten(CutRecoDisMuonEta)[mu])

for e in range(len(ak.flatten(CutRecoElePt))):
  h_reco_ele_pt.Fill(ak.flatten(CutRecoElePt)[e])
  h_reco_ele_eta.Fill(ak.flatten(CutRecoEleEta)[e])

h_reco_muon_pt.SetBinContent(h_reco_muon_pt.GetNbinsX(), h_reco_muon_pt.GetBinContent(h_reco_muon_pt.GetNbinsX()) + h_reco_muon_pt.GetBinContent(h_reco_muon_pt.GetNbinsX() + 1))  
h_reco_dis_muon_pt.SetBinContent(h_reco_dis_muon_pt.GetNbinsX(), h_reco_dis_muon_pt.GetBinContent(h_reco_dis_muon_pt.GetNbinsX()) + h_reco_dis_muon_pt.GetBinContent(h_reco_dis_muon_pt.GetNbinsX() + 1))  
h_reco_ele_pt.SetBinContent(h_reco_ele_pt.GetNbinsX(), h_reco_ele_pt.GetBinContent(h_reco_ele_pt.GetNbinsX()) + h_reco_ele_pt.GetBinContent(h_reco_ele_pt.GetNbinsX() + 1))

h_reco_muon_pt.Scale(1./ h_reco_muon_pt.Integral())
h_reco_dis_muon_pt.Scale(1./ h_reco_dis_muon_pt.Integral())
h_reco_ele_pt.Scale(1./ h_reco_ele_pt.Integral())

h_reco_muon_eta.Scale(1./ h_reco_muon_eta.Integral())
h_reco_dis_muon_eta.Scale(1./ h_reco_dis_muon_eta.Integral())
h_reco_ele_eta.Scale(1./ h_reco_ele_eta.Integral())

h_fake_muon_pteff = ROOT.TEfficiency("h_fake_muon_pteff", "Staus_M_100_100mm_13p6TeV_Run3Summer22; p_{T} [GeV]; Fraction of reco muons w/ no gen", len(ptbins) - 1, ptbins)
h_fake_muon_etaeff = ROOT.TEfficiency("h_fake_muon_etaeff", "Staus_M_100_100mm_13p6TeV_Run3Summer22; #eta;  Fraction of reco muons w/ no gen", 50, -2.5, 2.5)
h_fake_muon_dxyeff = ROOT.TEfficiency("h_fake_muon_dxyeff", "Staus_M_100_100mm_13p6TeV_Run3Summer22; d_{xy} [cm]; Fraction of reco muons w/ no gen", 100, -5, 5)

h_fake_dis_muon_pteff = ROOT.TEfficiency("h_fake_dis_muon_pteff", "Staus_M_100_100mm_13p6TeV_Run3Summer22; p_{T} [GeV]; Fraction of reco muons w/ no gen", len(ptbins) - 1, ptbins)
h_fake_dis_muon_etaeff = ROOT.TEfficiency("h_fake_dis_muon_etaeff", "Staus_M_100_100mm_13p6TeV_Run3Summer22; #eta;  Fraction of reco muons w/ no gen", 50, -2.5, 2.5)
h_fake_dis_muon_dxyeff = ROOT.TEfficiency("h_fake_dis_muon_dxyeff", "Staus_M_100_100mm_13p6TeV_Run3Summer22; d_{xy} [cm]; Fraction of reco muons w/ no gen", 100, -5, 5)

for i in range(h_fake_muon_pt.GetNbinsX()):
  h_fake_muon_pteff.SetTotalEvents(i + 1, len(ak.flatten(CutMuonPt[(CutMuonPt > ptbins[i]) & (CutMuonPt < ptbins[i+1])])))
  h_fake_muon_pteff.SetPassedEvents(i + 1, int(h_fake_muon_pt.GetBinContent(i + 1))) 
  h_fake_dis_muon_pteff.SetTotalEvents(i + 1, len(ak.flatten(CutDisMuonPt[(CutDisMuonPt > ptbins[i]) & (CutDisMuonPt < ptbins[i+1])])))
  h_fake_dis_muon_pteff.SetPassedEvents(i + 1, int(h_fake_dis_muon_pt.GetBinContent(i + 1))) 
  
for i in range(h_fake_muon_eta.GetNbinsX()):
  h_fake_muon_etaeff.SetTotalEvents(i + 1, len(ak.flatten(CutMuonEta[(CutMuonEta > (-2.5 + i * 0.1)) & (CutMuonEta < (-2.5 + (i + 1) * 0.1))])))
  h_fake_muon_etaeff.SetPassedEvents(i + 1, int(h_fake_muon_eta.GetBinContent(i + 1)))
  h_fake_dis_muon_etaeff.SetTotalEvents(i + 1, len(ak.flatten(CutDisMuonEta[(CutDisMuonEta > (-2.5 + i * 0.1)) & (CutDisMuonEta < (-2.5 + (i + 1) * 0.1))])))
  h_fake_dis_muon_etaeff.SetPassedEvents(i + 1, int(h_fake_dis_muon_eta.GetBinContent(i + 1)))  

for i in range(h_fake_muon_dxy.GetNbinsX()):
  h_fake_muon_dxyeff.SetTotalEvents(i + 1, len(ak.flatten(CutMuonDxy[(CutMuonDxy > (-5 + i * 0.1)) & (CutMuonDxy < (-5 + (i + 1) * 0.1))])))
  h_fake_muon_dxyeff.SetPassedEvents(i + 1, int(h_fake_muon_dxy.GetBinContent(i + 1)))
  h_fake_dis_muon_dxyeff.SetTotalEvents(i + 1, len(ak.flatten(CutDisMuonDxy[(CutDisMuonDxy > (-5 + i * 0.1)) & (CutDisMuonDxy < (-5 + (i + 1) * 0.1))])))
  h_fake_dis_muon_dxyeff.SetPassedEvents(i + 1, int(h_fake_dis_muon_dxy.GetBinContent(i + 1)))
   

def plotHisto(h_prompt, h_dis, var):
  legend = ROOT.TLegend()
  legend.SetFillStyle(0)
  h_prompt.SetLineColor(1)
  h_prompt.SetMarkerColor(1)
  h_dis.SetLineColor(2)
  h_dis.SetMarkerColor(2)
  legend.AddEntry(h_prompt, "prompt muon reco")
  legend.AddEntry(h_dis, "dis muon reco")
  h_prompt.Draw()
  h_dis.Draw("same")
  ROOT.gPad.Update()
  h_prompt.GetPaintedGraph().GetYaxis().SetRangeUser(0, 1)
  legend.Draw()
  can.SaveAs(str(h_prompt).split("_")[1] + "Muon_" + var + ".pdf")

plotHisto(h_fake_muon_pteff, h_fake_dis_muon_pteff, "pt") 
plotHisto(h_fake_muon_etaeff, h_fake_dis_muon_etaeff, "eta")
plotHisto(h_fake_muon_dxyeff, h_fake_dis_muon_dxyeff, "dxy")

can.SetLogy()

l_reco_muon_pt = ROOT.TLegend()
h_reco_muon_pt.SetLineColor(1)
l_reco_muon_pt.AddEntry(h_reco_muon_pt, "prompt muon reco")
h_reco_dis_muon_pt.SetLineColor(2)
l_reco_muon_pt.AddEntry(h_reco_dis_muon_pt, "dis muon reco")
h_reco_muon_pt.Draw("histe")
h_reco_dis_muon_pt.Draw("samehiste")
l_reco_muon_pt.Draw()
can.SaveAs("RecoMuon_pt.pdf")


l_reco_muon_eta = ROOT.TLegend()
h_reco_muon_eta.SetLineColor(1)
h_reco_muon_eta.SetMinimum(1E-3)
l_reco_muon_eta.AddEntry(h_reco_muon_eta, "prompt muon reco")
h_reco_dis_muon_eta.SetLineColor(2)
h_reco_dis_muon_eta.SetMinimum(1E-3)
l_reco_muon_eta.AddEntry(h_reco_dis_muon_eta, "dis muon reco")
h_reco_muon_eta.Draw("histe")
h_reco_dis_muon_eta.Draw("samehiste")
l_reco_muon_eta.Draw()
can.SaveAs("RecoMuon_eta.pdf")

h_reco_ele_pt.SetLineColor(1)
h_reco_ele_pt.Draw("histe")
can.SaveAs("RecoEle_pt.pdf")

h_reco_ele_eta.SetLineColor(1)
h_reco_ele_eta.Draw("histe")
can.SaveAs("RecoEle_eta.pdf")


