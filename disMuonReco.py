import uproot
import scipy
import matplotlib as mpl
import awkward as ak
import numpy as np
import math
import ROOT
import array
import pandas as pd

GenPtMin = 20
GenEtaMax = 2.4

ROOT.gStyle.SetOptStat(0)
stauFile = uproot.open("Staus_M_100_100mm_13p6TeV_Run3Summer22_DisMuon_GenPartMatch.root")
Events = stauFile["Events"]
muonBranches = {}
genPartBranches = {}

can = ROOT.TCanvas("can", "can")
  
dxy_bin = [-30., -15., -10., -5., -4.5, -4., -3.5, -3., -2.5, -2., -1.5, -1., -0.5, 0,
           0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 10., 15., 30.]
dxy_bin = array.array('d', dxy_bin)

dxyRANGE = ["-30--15", "-15--10", "-10--5", "-5--4.5", "-4.5--4", "-4--3.5", "-3.5--3", "-3--2.5", "-2.5--2", "-2--1.5", "-1.5--1", "-1--0.5", "-0.5-0",
            "0-0.5", "0.5-1", "1-1.5", "1.5-2", "2-2.5", "2.5-3", "3-3.5", "3.5-4", "4-4.5", "4.5-5", "5-10", "10-15", "15-30"]

ptRANGE = ["20-25", "25-30", "30-35", "35-40", "40-45", "45-50", "50-55", "55-60",
           "60-65", "65-70", "70-75", "75-80", "80-85", "85-90", "90-95", "95-100"]

etaRANGE = ["-2.4--2.2", "-2.2--2.0", "-2.0--1.8", "-1.8--1.6", "-1.6--1.4", "-1.4--1.2", "-1.2--1.0", "-1.0--0.8", "-0.8--0.6", "-0.6--0.4", "-0.4--0.2", "-0.2-0",
            "0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0", "1.0-1.2", "1.2-1.4", "1.4-1.6", "1.6-1.8", "1.8-2.0", "2.0-2.2", "2.2-2.4"]

for key in Events.keys():
  if "Muon_" in key:
    muonBranches[key] = Events[key].array()
  if "GenPart_" in key:
    genPartBranches[key] = Events[key].array()

def isMotherStau(evt, part_idx, lepId):
  #Find its mother
  mother_idx = genPartBranches["GenPart_genPartIdxMother"][evt][part_idx]
  #If it's a stau, we're done
  if (abs(genPartBranches["GenPart_pdgId"][evt][mother_idx]) == 1000015):
    return 1
  #If it's a tau, go further up the chain
  if (abs(genPartBranches["GenPart_pdgId"][evt][mother_idx]) == 15):
    return isMotherStau(evt, mother_idx, lepId)
  #If it's a lepton of the same type, go further up the chain
  if (abs(genPartBranches["GenPart_pdgId"][evt][mother_idx]) == lepId):
    return isMotherStau(evt, mother_idx, lepId)
  #Else, it's  not what we're looking for
  else:
    return 0        

def isMotherStauIdx(evt, part_idx):
  mother_idx = genPartBranches["GenPart_genPartIdxMother"][evt][part_idx]
  if (abs(genPartBranches["GenPart_pdgId"][evt][mother_idx]) == 1000015):
    return mother_idx
  else:
    return isMotherStauIdx(evt, mother_idx)

def isFinalLepton(evt, part_idx, lepId):
  middleLep = False
  for mother_idx in range(len(genBranches["GenPart_genPartIdxMother"][evt])):
    if (genBranches["GenPart_genPartIdxMother"][evt][mother_idx] == part_idx) and (abs(genBranches["GenPart_pdgId"][evt][mother_idx]) == lepId):
      middleLep = True
      return 0
  if middleLep == False:
    if isMotherStau(evt, part_idx, lepId):
      return 1
    else:
      return 0 
     
# This chooses the gen particles which are muons which are decayed from displaced taus
genDisMuon = []
allGenDisMuon = []
genStauIdx = []
for evt in range(len(genPartBranches["GenPart_pdgId"])):
  genDisMuon_evt = []
  allGenDisMuon_evt = []
  genStauIdx_evt = []
  if (len(genPartBranches["GenPart_pdgId"][evt]) != 0):
    for part_idx in range(len(genPartBranches["GenPart_pdgId"][evt])):
      if (abs(genPartBranches["GenPart_pdgId"][evt][part_idx]) == 13):
        if ((genPartBranches["GenPart_pt"][evt][part_idx] < GenPtMin) and (abs(genPartBranches["GenPart_eta"][evt][part_idx]) > GenEtaMax)): continue 
        if (isMotherStau(evt, part_idx, 13)):
          allGenDisMuon_evt.append(part_idx)
        if (isFinalLepton(evt, part_idx, 13):
          genStauIdx_evt.append(isMotherStauIdx(evt, part_idx))
          genDisMuon_evt.append(part_idx)
  genDisMuon.append(genDisMuon_evt)
  allGenDisMuon.append(allGenDisMuon_evt)
  genStauIdx.append(genStauIdx_evt)

genDisMuonPt = genPartBranches["GenPart_pt"][genDisMuon]
genDisMuonEta = genPartBranches["GenPart_eta"][genDisMuon]
genDisMuonPhi = genPartBranches["GenPart_phi"][genDisMuon]

def dxy(evt, muon_idx, stau_idx):
  return (genPartBranches["GenPart_vertexY"][evt][muon_idx] - genPartBranches["GenPart_vertexY"][evt][stau_idx]) * np.cos(genPartBranches["GenPart_phi"][evt][muon_idx]) - (genPartBranches["GenPart_vertexX"][evt][muon_idx] - genPartBranches["GenPart_vertexX"][evt][stau_idx]) * np.sin(genPartBranches["GenPart_phi"][evt][muon_idx])
  

genPartBranches["GenMuon_dxy"] = []

for evt in range(len(genPartBranches["GenPart_pdgId"])):
  genDisMuondxy_evt = np.zeros(len(genPartBranches["GenPart_pdgId"][evt]))
  if (len(genPartBranches["GenPart_pdgId"][evt]) != 0):
    for idx in range(len(genDisMuon[evt])):
      dxy_val = dxy(evt, genDisMuon[evt][idx], genStauIdx[evt][idx]) 
      genDisMuondxy_evt[genDisMuon[evt][idx]] = dxy_val
  genPartBranches["GenMuon_dxy"].append(ak.Array(genDisMuondxy_evt))
genPartBranches["GenMuon_dxy"] = ak.Array(genPartBranches["GenMuon_dxy"])
#print(genPartBranches["GenMuon_dxy"])

genDisMuondxy = genPartBranches["GenMuon_dxy"][genDisMuon]
#print(genDisMuondxy)
#print(min(ak.flatten(genDisMuondxy))) 
#print(max(ak.flatten(genDisMuondxy))) 

filteredRecoMuon = [np.array([val for val in subarr if val in values]) for subarr, values in zip(muonBranches["Muon_genPartIdx"], genDisMuon)]
filteredRecoMuonIndices = [np.where(np.isin(subarr, values))[0] for subarr, values in zip(muonBranches["Muon_genPartIdx"], genDisMuon)]

filteredRecoMuonPt_recon = muonBranches["Muon_pt"][filteredRecoMuonIndices]
filteredRecoMuonEta_recon = muonBranches["Muon_eta"][filteredRecoMuonIndices]
filteredRecoMuondxy_recon = muonBranches["Muon_dxy"][filteredRecoMuonIndices]

filteredRecoMuonPt =  genPartBranches["GenPart_pt"][filteredRecoMuon]
filteredRecoMuonEta = genPartBranches["GenPart_eta"][filteredRecoMuon]
filteredRecoMuondxy = genPartBranches["GenMuon_dxy"][filteredRecoMuon]

filteredRecoMuonPt_Diff = ak.Array(np.subtract(filteredRecoMuonPt_recon, filteredRecoMuonPt))
filteredRecoMuondxy_Diff = ak.Array(np.subtract(filteredRecoMuondxy_recon, filteredRecoMuondxy))
filteredRecoMuonEta_Diff = ak.Array(np.subtract(filteredRecoMuonEta_recon, filteredRecoMuonEta))

h_resVsPt_pt_prompt = []

for i in range(len(ptRANGE)):
  pt_hist = ROOT.TH1F("prompt_resVsPt_pt_"+ str(ptRANGE[i]), ";p_{T} residual (reco - gen) [GeV]; Number of muons", 100, -5, 5)
  h_resVsPt_pt_prompt.append(pt_hist)

for mu in range(len(ak.flatten(filteredRecoMuonPt))):
  for pt in range(len(ptRANGE)):
    if ((ak.flatten(filteredRecoMuonPt)[mu] > (20 + pt * 5)) & (ak.flatten(filteredRecoMuonPt)[mu] < (20 + (pt + 1) * 5))):
      h_resVsPt_pt_prompt[pt].Fill(ak.flatten(filteredRecoMuonPt_Diff)[mu])
h2_resVsPt_pt_prompt = ROOT.TH1F("prompt_resVsPt2_pt", "Staus_M_100_100mm_13p6TeV_Run3Summer22;gen muon p_{T} [GeV]; p_{T} resolution [GeV]", len(ptRANGE), 20, 100)

for i in range(len(ptRANGE)):
  h2_resVsPt_pt_prompt.SetBinContent(i + 1, h_resVsPt_pt_prompt[i].GetRMS())
  h2_resVsPt_pt_prompt.SetBinError(i + 1, h_resVsPt_pt_prompt[i].GetRMSError())

h_resVsDxy_dxy_prompt = []

for i in range(len(dxyRANGE)):
  if abs(dxy_bin[i]) < 5: 
    dxy_hist = ROOT.TH1F("h_prompt_resVsDxy_dxy_"+ str(dxyRANGE[i]), "h_prompt_resVsDxy_dxy_"+ str(dxyRANGE[i])+";d_{xy} residual (reco - gen) [cm]; Number of muons", 100, -0.5, 0.5)
  if abs(dxy_bin[i]) >= 5: 
    dxy_hist = ROOT.TH1F("h_prompt_resVsDxy_dxy_"+ str(dxyRANGE[i]), "h_prompt_resVsDxy_dxy_"+ str(dxyRANGE[i])+";d_{xy} residual (reco - gen) [cm]; Number of muons", 100, -5, 5)
  h_resVsDxy_dxy_prompt.append(dxy_hist)

for mu in range(len(ak.flatten(filteredRecoMuondxy))):
  for dxy in range(len(dxyRANGE)):
    if ((ak.flatten(filteredRecoMuondxy)[mu] > dxy_bin[dxy]) & (ak.flatten(filteredRecoMuondxy)[mu] < dxy_bin[dxy + 1])):
      h_resVsDxy_dxy_prompt[dxy].Fill(ak.flatten(filteredRecoMuondxy_Diff)[mu])

h2_resVsDxy_dxy_prompt = ROOT.TH1F("prompt_resVsDxy2_dxy", "Staus_M_100_100mm_13p6TeV_Run3Summer22;gen muon d_{xy} [cm];d_{xy} resolution [cm]", len(dxyRANGE), dxy_bin)

for i in range(len(dxyRANGE)):
  #dxyLegend = ROOT.TLegend()
  #h_resVsDxy_dxy_prompt[i].Draw('histe')
  #dxyLegend.AddEntry(h_resVsDxy_dxy_prompt[i], "RMS = " + str(h_resVsDxy_dxy_prompt[i].GetRMS()))
  #dxyLegend.Draw()
  #can.SaveAs("MuonIdPlots/h_prompt_resVsDxy_dxy_"+ str(dxyRANGE[i]) + ".pdf")
  h2_resVsDxy_dxy_prompt.SetBinContent(i + 1, h_resVsDxy_dxy_prompt[i].GetRMS())
  h2_resVsDxy_dxy_prompt.SetBinError(i + 1, h_resVsDxy_dxy_prompt[i].GetRMSError())

h_resVsEta_eta_prompt = []

for i in range(len(etaRANGE)):
    eta_hist = ROOT.TH1F("prompt_resVsEta_eta_"+ str(etaRANGE[i]), "prompt_resVsEta_eta_"+ str(etaRANGE[i])+";#eta residual (reco - gen); Number of muons", 100, -0.2, 0.2)
    h_resVsEta_eta_prompt.append(eta_hist)

for mu in range(len(ak.flatten(filteredRecoMuonEta))):
  for eta in range(len(etaRANGE)):
    if ((ak.flatten(filteredRecoMuonEta)[mu] > (-2.4 + eta * 0.2)) & (ak.flatten(filteredRecoMuonEta)[mu] < (-2.4 + (eta + 1) * 0.2))):
      h_resVsEta_eta_prompt[eta].Fill(ak.flatten(filteredRecoMuonEta_Diff)[mu])

h2_resVsEta_eta_prompt = ROOT.TH1F("prompt_resVsEta2_eta", "Staus_M_100_100mm_13p6TeV_Run3Summer22;gen muon #eta;#eta resolution", len(etaRANGE), -2.4, 2.4)

for i in range(len(etaRANGE)):
  #etaLegend = ROOT.TLegend()
  #h_resVsEta_eta_prompt[i].Draw('histe')
  #etaLegend.AddEntry(h_resVsEta_eta_prompt[i], "RMS = " + str(h_resVsEta_eta_prompt[i].GetRMS()))
  #etaLegend.Draw()
  #can.SaveAs("MuonIdPlots/h_prompt_resVsEta_eta_"+ str(etaRANGE[i]) + ".pdf")
  

  h2_resVsEta_eta_prompt.SetBinContent(i + 1, h_resVsEta_eta_prompt[i].GetRMS())
  h2_resVsEta_eta_prompt.SetBinError(i + 1, h_resVsEta_eta_prompt[i].GetRMSError())


promptDiffPt = ROOT.TH2F("h2_promptptdiff", "Staus_M_100_100mm_13p6TeV_Run3Summer22; gen muon p_{T} [GeV]; prompt reco muon p_{T} - gen muon p_{T} [GeV]", 100, 0, 100, 20, -10, 10)
for i in range(len(ak.flatten(filteredRecoMuon))):
  promptDiffPt.Fill(ak.flatten(filteredRecoMuonPt)[i], ak.flatten(filteredRecoMuonPt_Diff)[i])
promptDiffDxy = ROOT.TH2F("h2_promptdxydiff", "Staus_M_100_100mm_13p6TeV_Run3Summer22; gen muon d_{xy} [cm]; prompt reco muon d_{xy} - gen muon d_{xy} [cm]", len(dxy_bin) - 1, dxy_bin, 40, -1, 1)
for i in range(len(ak.flatten(filteredRecoMuon))):
  promptDiffDxy.Fill(ak.flatten(filteredRecoMuondxy)[i], ak.flatten(filteredRecoMuondxy_Diff)[i])

pt_cut = (filteredRecoMuonPt > GenPtMin)
eta_cut = (abs(filteredRecoMuonEta) < GenEtaMax)

filteredRecoMuonPt = filteredRecoMuonPt[pt_cut & eta_cut]
filteredRecoMuonEta = filteredRecoMuonEta[pt_cut & eta_cut]
filteredRecoMuondxy = filteredRecoMuondxy[pt_cut & eta_cut] 

filteredRecoDisMuon = [np.array([val for val in subarr if val in values]) for subarr, values in zip(muonBranches["DisMuon_genPartIdx"], genDisMuon)]
filteredRecoDisMuonIndices = [np.where(np.isin(subarr, values))[0] for subarr, values in zip(muonBranches["DisMuon_genPartIdx"], genDisMuon)]

filteredRecoDisMuonPt_recon = muonBranches["DisMuon_pt"][filteredRecoDisMuonIndices]
filteredRecoDisMuondxy_recon = muonBranches["DisMuon_dxy"][filteredRecoDisMuonIndices]
filteredRecoDisMuonEta_recon = muonBranches["DisMuon_eta"][filteredRecoDisMuonIndices]

filteredRecoDisMuonPt  = genPartBranches["GenPart_pt"][filteredRecoDisMuon]
filteredRecoDisMuonEta = genPartBranches["GenPart_eta"][filteredRecoDisMuon]
filteredRecoDisMuondxy = genPartBranches["GenMuon_dxy"][filteredRecoDisMuon]

filteredRecoDisMuonPt_Diff = ak.Array(np.subtract(filteredRecoDisMuonPt_recon, filteredRecoDisMuonPt))
filteredRecoDisMuondxy_Diff = ak.Array(np.subtract(filteredRecoDisMuondxy_recon, filteredRecoDisMuondxy))
filteredRecoDisMuonEta_Diff = ak.Array(np.subtract(filteredRecoDisMuonEta_recon, filteredRecoDisMuonEta))

h_resVsPt_pt_dis = []

for i in range(len(ptRANGE)):
  pt_hist = ROOT.TH1F("resVsPt_pt_"+ str(ptRANGE[i]), ";p_{T} residual (reco - gen) [GeV]; Number of muons", 100, -5, 5)
  h_resVsPt_pt_dis.append(pt_hist)

for mu in range(len(ak.flatten(filteredRecoDisMuonPt))):
  for pt in range(len(ptRANGE)):
    if ((ak.flatten(filteredRecoDisMuonPt)[mu] > (20 + pt * 5)) & (ak.flatten(filteredRecoDisMuonPt)[mu] < (20 + (pt + 1) * 5))):
      h_resVsPt_pt_dis[pt].Fill(ak.flatten(filteredRecoDisMuonPt_Diff)[mu])
h2_resVsPt_pt_dis = ROOT.TH1F("resVsPt2_pt", ";gen muon p_{T} [GeV]; p_{T} resolution [GeV]", len(ptRANGE), 20, 100)

for i in range(len(ptRANGE)):
  h2_resVsPt_pt_dis.SetBinContent(i + 1, h_resVsPt_pt_dis[i].GetRMS())
  h2_resVsPt_pt_dis.SetBinError(i + 1, h_resVsPt_pt_dis[i].GetRMSError())

resVsPt_ptLegend = ROOT.TLegend()

h2_resVsPt_pt_prompt.SetMinimum(0)
h2_resVsPt_pt_prompt.SetMarkerStyle(20)
h2_resVsPt_pt_prompt.SetMarkerColor(1)
resVsPt_ptLegend.AddEntry(h2_resVsPt_pt_prompt, "prompt muon recon")
h2_resVsPt_pt_prompt.Draw('p')
h2_resVsPt_pt_dis.SetMinimum(0)
h2_resVsPt_pt_dis.SetMarkerStyle(20)
h2_resVsPt_pt_dis.SetMarkerColor(2)
resVsPt_ptLegend.AddEntry(h2_resVsPt_pt_dis, "dis muon recon")
h2_resVsPt_pt_dis.Draw('psame')
resVsPt_ptLegend.Draw()
can.SaveAs("eta2p4cut/h2_resVsPt_pt.pdf")


h_resVsDxy_dxy_dis = []

for i in range(len(dxyRANGE)):
  if abs(dxy_bin[i]) < 5: 
    dxy_hist = ROOT.TH1F("h_dis_resVsDxy_dxy_"+ str(dxyRANGE[i]), "h_dis_resVsDxy_dxy_"+ str(dxyRANGE[i])+";d_{xy} residual (reco - gen) [cm]; Number of muons", 100, -0.5, 0.5)
  if abs(dxy_bin[i]) >= 5: 
    dxy_hist = ROOT.TH1F("h_dis_resVsDxy_dxy_"+ str(dxyRANGE[i]), "h_dis_resVsDxy_dxy_"+ str(dxyRANGE[i])+";d_{xy} residual (reco - gen) [cm]; Number of muons", 100, -5, 5)
  h_resVsDxy_dxy_dis.append(dxy_hist)

for mu in range(len(ak.flatten(filteredRecoDisMuondxy))):
  for dxy in range(len(dxyRANGE)):
    if ((ak.flatten(filteredRecoDisMuondxy)[mu] > dxy_bin[dxy]) & (ak.flatten(filteredRecoDisMuondxy)[mu] < dxy_bin[dxy + 1])):
      h_resVsDxy_dxy_dis[dxy].Fill(ak.flatten(filteredRecoDisMuondxy_Diff)[mu])

h2_resVsDxy_dxy_dis = ROOT.TH1F("resVsDxy2_dxy", ";gen muon d_{xy} [cm]; d_{xy} resolution [cm]", len(dxyRANGE), dxy_bin)

for i in range(len(dxyRANGE)):
  #disDxyLegend = ROOT.TLegend()
  #h_resVsDxy_dxy_dis[i].Draw("histe")
  #disDxyLegend.AddEntry(h_resVsDxy_dxy_dis[i], "RMS = " + str(h_resVsDxy_dxy_dis[i].GetRMS()))  
  #disDxyLegend.Draw()
  #can.SaveAs("MuonIdPlots/h_dis_resVsDxy_dxy_"+ str(dxyRANGE[i])+".pdf")
  h2_resVsDxy_dxy_dis.SetBinContent(i + 1, h_resVsDxy_dxy_dis[i].GetRMS())
  h2_resVsDxy_dxy_dis.SetBinError(i + 1, h_resVsDxy_dxy_dis[i].GetRMSError())

h_resVsEta_eta_dis = []

for i in range(len(etaRANGE)):
    eta_hist = ROOT.TH1F("resVsEta_eta_"+ str(etaRANGE[i]), "h_resVsEta_eta_"+ str(etaRANGE[i])+";#eta residual (reco - gen); Number of muons", 100, -0.2, 0.2)
    h_resVsEta_eta_dis.append(eta_hist)

for mu in range(len(ak.flatten(filteredRecoDisMuonEta))):
  for eta in range(len(etaRANGE)):
    if ((ak.flatten(filteredRecoDisMuonEta)[mu] > (-2.4 + eta * 0.2)) & (ak.flatten(filteredRecoDisMuonEta)[mu] < (-2.4 + (eta + 1)*0.2))):
      h_resVsEta_eta_dis[eta].Fill(ak.flatten(filteredRecoDisMuonEta_Diff)[mu])

h2_resVsEta_eta_dis = ROOT.TH1F("resVsEta2_eta", "Staus_M_100_100mm_13p6TeV_Run3Summer22;gen muon #eta;#eta resolution", len(etaRANGE), -2.4, 2.4)

for i in range(len(etaRANGE)):
  #disEtaLegend = ROOT.TLegend()
  #h_resVsEta_eta_dis[i].Draw("histe")
  #disEtaLegend.AddEntry(h_resVsEta_eta_dis[i], "RMS = " + str(h_resVsEta_eta_dis[i].GetRMS()))  
  #disEtaLegend.Draw()
  #can.SaveAs("MuonIdPlots/h_dis_resVsEta_eta_"+ str(etaRANGE[i])+".pdf")

  h2_resVsEta_eta_dis.SetBinContent(i + 1, h_resVsEta_eta_dis[i].GetRMS())
  h2_resVsEta_eta_dis.SetBinError(i + 1, h_resVsEta_eta_dis[i].GetRMSError())

resVsDxy_dxyLegend = ROOT.TLegend()

h2_resVsDxy_dxy_prompt.SetMinimum(0)
h2_resVsDxy_dxy_prompt.SetMaximum(2.3)
h2_resVsDxy_dxy_prompt.SetMarkerStyle(20)
h2_resVsDxy_dxy_prompt.SetMarkerColor(1)
resVsDxy_dxyLegend.AddEntry(h2_resVsDxy_dxy_prompt, "prompt muon reco")
h2_resVsDxy_dxy_prompt.Draw('p')
h2_resVsDxy_dxy_dis.SetMinimum(0)
h2_resVsDxy_dxy_dis.SetMarkerStyle(20)
h2_resVsDxy_dxy_dis.SetMarkerColor(2)
resVsDxy_dxyLegend.AddEntry(h2_resVsDxy_dxy_dis, "dis muon reco")
h2_resVsDxy_dxy_dis.Draw('psame')
#h2_resVsDxy_dxy_prompt.GetXaxis().SetRangeUser(-4.5, 5)
#h2_resVsDxy_dxy_dis.GetXaxis().SetRangeUser(-4.5, 5)
resVsDxy_dxyLegend.Draw()
can.SaveAs("eta2p4cut/h2_resVsDxy_dxy.pdf")

resVsEta_etaLegend = ROOT.TLegend()
h2_resVsEta_eta_prompt.SetMinimum(0)
h2_resVsEta_eta_prompt.SetMarkerStyle(20)
h2_resVsEta_eta_prompt.SetMaximum(0.1)
h2_resVsEta_eta_prompt.SetMarkerColor(1)
resVsEta_etaLegend.AddEntry(h2_resVsEta_eta_prompt, "prompt muon reco")
h2_resVsEta_eta_prompt.Draw('p')
h2_resVsEta_eta_dis.SetMinimum(0)
h2_resVsEta_eta_dis.SetMaximum(0.1)
h2_resVsEta_eta_dis.SetMarkerStyle(20)
h2_resVsEta_eta_dis.SetMarkerColor(2)
resVsEta_etaLegend.AddEntry(h2_resVsEta_eta_dis, "dis muon reco")
h2_resVsEta_eta_dis.Draw('psame')
#h2_resVsDxy_dxy_prompt.GetXaxis().SetRangeUser(-4.5, 5)
#h2_resVsDxy_dxy_dis.GetXaxis().SetRangeUser(-4.5, 5)
resVsEta_etaLegend.Draw()
can.SaveAs("eta2p4cut/h2_resVsEta_eta.pdf")







disDiffPt = ROOT.TH2F("h2_disptdiff", "Staus_M_100_100mm_13p6TeV_Run3Summer22; gen muon p_{T} [GeV]; dis reco muon p_{T} - gen muon p_{T} [GeV]", 100, 0, 100, 20, -10, 10)
for i in range(len(ak.flatten(filteredRecoDisMuon))):
  disDiffPt.Fill(ak.flatten(filteredRecoDisMuonPt)[i], ak.flatten(filteredRecoDisMuonPt_Diff)[i])
disDiffDxy = ROOT.TH2F("h2_disdxydiff", "Staus_M_100_100mm_13p6TeV_Run3Summer22; gen muon d_{xy} [cm]; dis reco muon d_{xy} - gen muon d_{xy} [cm]", len(dxy_bin) - 1, dxy_bin, 40, -1, 1)
for i in range(len(ak.flatten(filteredRecoDisMuon))):
  disDiffDxy.Fill(ak.flatten(filteredRecoDisMuondxy)[i], ak.flatten(filteredRecoDisMuondxy_Diff)[i])

dis_pt_cut = (filteredRecoDisMuonPt > GenPtMin)
dis_eta_cut = (abs(filteredRecoDisMuonEta) < GenEtaMax)

filteredRecoDisMuonPt = filteredRecoDisMuonPt[dis_pt_cut & dis_eta_cut]
filteredRecoDisMuonEta = filteredRecoDisMuonEta[dis_pt_cut & dis_eta_cut]
filteredRecoDisMuondxy = filteredRecoDisMuondxy[dis_pt_cut & dis_eta_cut] 

OrDisMuon = ak.Array([np.array(np.union1d(mu,dismu)) for mu, dismu in zip(filteredRecoMuon, filteredRecoDisMuon)])
OrDisMuon = ak.values_astype(OrDisMuon, "int64")
filteredRecoOrMuonPt = genPartBranches["GenPart_pt"][OrDisMuon]
filteredRecoOrMuonEta = genPartBranches["GenPart_eta"][OrDisMuon]
filteredRecoOrMuondxy = genPartBranches["GenMuon_dxy"][OrDisMuon]

or_pt_cut = (filteredRecoOrMuonPt > GenPtMin)
or_eta_cut = (abs(filteredRecoOrMuonEta) < GenEtaMax)

filteredRecoOrMuonPt =  filteredRecoOrMuonPt[or_pt_cut & or_eta_cut]
filteredRecoOrMuonEta = filteredRecoOrMuonEta[or_pt_cut & or_eta_cut]
filteredRecoOrMuondxy = filteredRecoOrMuondxy[or_pt_cut & or_eta_cut] 

#AndDisMuon, AndPrompMuonIndx, AndDisMuonIndx = [np.array(np.intersect1d(mu, dismu, return_indices=True)) for mu, dismu in zip(filteredRecoMuon, filteredRecoDisMuon)]
AndDisMuon_zip = [np.array(np.intersect1d(mu, dismu, return_indices=True)) for mu, dismu in zip(filteredRecoMuon, filteredRecoDisMuon)]

AndDisMuon_zip = list(zip(*AndDisMuon_zip))
AndDisMuon, AndPrompMuonIndx, AndDisMuonIndx = AndDisMuon_zip
AndDisMuon = ak.values_astype(AndDisMuon, "int64")
AndPrompMuonIndx = ak.values_astype(AndPrompMuonIndx, "int64")
AndDisMuonIndx = ak.values_astype(AndDisMuonIndx, "int64")
print(len(ak.flatten(AndDisMuon)))

filteredRecoAndDisMuonPt =  genPartBranches["GenPart_pt"][AndDisMuon]
filteredRecoAndDisMuonEta = genPartBranches["GenPart_eta"][AndDisMuon]
filteredRecoAndDisMuondxy = genPartBranches["GenMuon_dxy"][AndDisMuon]

filteredRecoAndMuonPt_recon = muonBranches["Muon_pt"][AndPrompMuonIndx]
filteredRecoAndMuonEta_recon = muonBranches["Muon_eta"][AndPrompMuonIndx]
filteredRecoAndMuondxy_recon = muonBranches["Muon_dxy"][AndPrompMuonIndx]

filteredRecoAndMuonPt_Diff = ak.Array(np.subtract(filteredRecoAndMuonPt_recon, filteredRecoAndDisMuonPt))
filteredRecoAndMuondxy_Diff = ak.Array(np.subtract(filteredRecoAndMuondxy_recon, filteredRecoAndDisMuondxy))
filteredRecoAndMuonEta_Diff = ak.Array(np.subtract(filteredRecoAndMuonEta_recon, filteredRecoAndDisMuonEta))

filteredRecoAndDisMuonPt_recon = muonBranches["Muon_pt"][AndDisMuonIndx]
filteredRecoAndDisMuonEta_recon = muonBranches["Muon_eta"][AndDisMuonIndx]
filteredRecoAndDisMuondxy_recon = muonBranches["Muon_dxy"][AndDisMuonIndx]

filteredRecoAndDisMuonPt_Diff = ak.Array(np.subtract(filteredRecoAndDisMuonPt_recon, filteredRecoAndDisMuonPt))
filteredRecoAndDisMuondxy_Diff = ak.Array(np.subtract(filteredRecoAndDisMuondxy_recon, filteredRecoAndDisMuondxy))
filteredRecoAndDisMuonEta_Diff = ak.Array(np.subtract(filteredRecoAndDisMuonEta_recon, filteredRecoAndDisMuonEta))

h_resVsPt_pt_and_dis = []

for i in range(len(ptRANGE)):
  pt_hist = ROOT.TH1F("and_dis_resVsPt_pt_"+ str(ptRANGE[i]), ";p_{T} residual (reco - gen) [GeV]; Number of muons", 100, -5, 5)
  h_resVsPt_pt_and_dis.append(pt_hist)

for mu in range(len(ak.flatten(filteredRecoAndDisMuonPt))):
  for pt in range(len(ptRANGE)):
    if ((ak.flatten(filteredRecoAndDisMuonPt)[mu] > (20 + pt * 5)) & (ak.flatten(filteredRecoAndDisMuonPt)[mu] < (20 + (pt + 1) * 5))):
      h_resVsPt_pt_and_dis[pt].Fill(ak.flatten(filteredRecoAndDisMuonPt_Diff)[mu])
h2_resVsPt_pt_and_dis = ROOT.TH1F("and_dis_resVsPt2_pt", ";gen muon p_{T} [GeV]; p_{T} resolution [GeV]", len(ptRANGE), 20, 100)

for i in range(len(ptRANGE)):

  #andDisPtLegend = ROOT.TLegend()
  #h_resVsPt_pt_and_dis[i].Draw("histe")
  #andDisPtLegend.AddEntry(h_resVsPt_pt_and_dis[i], "RMS = " + str(h_resVsPt_pt_and_dis[i].GetRMS()))  
  #andDisPtLegend.Draw()
  #can.SaveAs("MuonIdPlots/h_and_dis_resVsPt_pt_"+ str(ptRANGE[i])+".pdf")

  h2_resVsPt_pt_and_dis.SetBinContent(i + 1, h_resVsPt_pt_and_dis[i].GetRMS())
  h2_resVsPt_pt_and_dis.SetBinError(i + 1, h_resVsPt_pt_and_dis[i].GetRMSError())

h_resVsPt_pt_and_prompt = []

for i in range(len(ptRANGE)):
  pt_hist = ROOT.TH1F("and_prompt_resVsPt_pt_"+ str(ptRANGE[i]), ";p_{T} residual (reco - gen) [GeV]; Number of muons", 100, -5, 5)
  h_resVsPt_pt_and_prompt.append(pt_hist)

for mu in range(len(ak.flatten(filteredRecoAndDisMuonPt))):
  for pt in range(len(ptRANGE)):
    if ((ak.flatten(filteredRecoAndDisMuonPt)[mu] > (20 + pt * 5)) & (ak.flatten(filteredRecoAndDisMuonPt)[mu] < (20 + (pt + 1) * 5))):
      h_resVsPt_pt_and_prompt[pt].Fill(ak.flatten(filteredRecoAndMuonPt_Diff)[mu])
h2_resVsPt_pt_and_prompt = ROOT.TH1F("and_prompt_resVsPt2_pt", ";gen muon p_{T} [GeV]; p_{T} resolution [GeV]", len(ptRANGE), 20, 100)

for i in range(len(ptRANGE)):

  #andPromptPtLegend = ROOT.TLegend()
  #h_resVsPt_pt_and_prompt[i].Draw("histe")
  #andPromptPtLegend.AddEntry(h_resVsPt_pt_and_prompt[i], "RMS = " + str(h_resVsPt_pt_and_prompt[i].GetRMS()))  
  #andPromptPtLegend.Draw()
  #can.SaveAs("MuonIdPlots/h_and_prompt_resVsPt_pt_"+ str(ptRANGE[i])+".pdf")

  h2_resVsPt_pt_and_prompt.SetBinContent(i + 1, h_resVsPt_pt_and_prompt[i].GetRMS())
  h2_resVsPt_pt_and_prompt.SetBinError(i + 1, h_resVsPt_pt_and_prompt[i].GetRMSError())

h_resVsDxy_dxy_and_dis = []

for i in range(len(dxyRANGE)):
  if abs(dxy_bin[i]) < 5: 
    dxy_hist = ROOT.TH1F("h_and_dis_resVsDxy_dxy_"+ str(dxyRANGE[i]), "h_and_dis_resVsDxy_dxy_"+ str(dxyRANGE[i])+";d_{xy} residual (reco - gen) [cm]; Number of muons", 100, -0.5, 0.5)
  if abs(dxy_bin[i]) >= 5: 
    dxy_hist = ROOT.TH1F("h_and_dis_resVsDxy_dxy_"+ str(dxyRANGE[i]), "h_and_dis_resVsDxy_dxy_"+ str(dxyRANGE[i])+";d_{xy} residual (reco - gen) [cm]; Number of muons", 100, -5, 5)
  h_resVsDxy_dxy_and_dis.append(dxy_hist)

for mu in range(len(ak.flatten(filteredRecoAndDisMuondxy))):
  for dxy in range(len(dxyRANGE)):
    if ((ak.flatten(filteredRecoAndDisMuondxy)[mu] > dxy_bin[dxy]) & (ak.flatten(filteredRecoAndDisMuondxy)[mu] < dxy_bin[dxy + 1])):
      h_resVsDxy_dxy_and_dis[dxy].Fill(ak.flatten(filteredRecoAndDisMuondxy_Diff)[mu])

h2_resVsDxy_dxy_and_dis = ROOT.TH1F("and_dis_resVsDxy2_dxy", ";gen muon d_{xy} [cm]; d_{xy} resolution [cm]", len(dxyRANGE), dxy_bin)

for i in range(len(dxyRANGE)):
  h2_resVsDxy_dxy_and_dis.SetBinContent(i + 1, h_resVsDxy_dxy_and_dis[i].GetRMS())
  h2_resVsDxy_dxy_and_dis.SetBinError(i + 1, h_resVsDxy_dxy_and_dis[i].GetRMSError())

h_resVsDxy_dxy_and_prompt = []

for i in range(len(dxyRANGE)):
  if abs(dxy_bin[i]) < 5: 
    dxy_hist = ROOT.TH1F("h_and_prompt_resVsDxy_dxy_"+ str(dxyRANGE[i]), "h_and_prompt_resVsDxy_dxy_"+ str(dxyRANGE[i])+";d_{xy} residual (reco - gen) [cm]; Number of muons", 100, -0.5, 0.5)
  if abs(dxy_bin[i]) >= 5: 
    dxy_hist = ROOT.TH1F("h_and_prompt_resVsDxy_dxy_"+ str(dxyRANGE[i]), "h_and_prompt_resVsDxy_dxy_"+ str(dxyRANGE[i])+";d_{xy} residual (reco - gen) [cm]; Number of muons", 100, -5, 5)
  h_resVsDxy_dxy_and_prompt.append(dxy_hist)

for mu in range(len(ak.flatten(filteredRecoAndDisMuondxy))):
  for dxy in range(len(dxyRANGE)):
    if ((ak.flatten(filteredRecoAndDisMuondxy)[mu] > dxy_bin[dxy]) & (ak.flatten(filteredRecoAndDisMuondxy)[mu] < dxy_bin[dxy + 1])):
      h_resVsDxy_dxy_and_prompt[dxy].Fill(ak.flatten(filteredRecoAndMuondxy_Diff)[mu])

h2_resVsDxy_dxy_and_prompt = ROOT.TH1F("and_prompt_resVsDxy2_dxy", ";gen muon d_{xy} [cm]; d_{xy} resolution [cm]", len(dxyRANGE), dxy_bin)

for i in range(len(dxyRANGE)):
  h2_resVsDxy_dxy_and_prompt.SetBinContent(i + 1, h_resVsDxy_dxy_and_prompt[i].GetRMS())
  h2_resVsDxy_dxy_and_prompt.SetBinError(i + 1, h_resVsDxy_dxy_and_prompt[i].GetRMSError())

resVsDxy_and_dxyLegend = ROOT.TLegend()

h2_resVsDxy_dxy_and_prompt.SetMinimum(0)
h2_resVsDxy_dxy_and_prompt.SetMarkerStyle(20)
h2_resVsDxy_dxy_and_prompt.SetMarkerColor(1)
resVsDxy_and_dxyLegend.AddEntry(h2_resVsDxy_dxy_and_prompt, "prompt muon recon")
h2_resVsDxy_dxy_and_prompt.Draw('p')
h2_resVsDxy_dxy_and_dis.SetMinimum(0)
h2_resVsDxy_dxy_and_dis.SetMarkerStyle(20)
h2_resVsDxy_dxy_and_dis.SetMarkerColor(2)
resVsDxy_and_dxyLegend.AddEntry(h2_resVsDxy_dxy_and_dis, "dis muon recon")
h2_resVsDxy_dxy_and_dis.Draw('psame')
#h2_resVsDxy_dxy_and_prompt.GetXaxis().SetRangeUser(-4.5, 4.5)
#h2_resVsDxy_dxy_and_dis.GetXaxis().SetRangeUser(-4.5, 4.5)
resVsDxy_and_dxyLegend.Draw()
can.SaveAs("eta2p4cut/h2_resVsDxy_dxy_and.pdf")

h_resVsEta_eta_and_dis = []
h_resVsEta_pt_and_dis = []
h_resVsEta_dxy_and_dis = []

for i in range(len(etaRANGE)):
    eta_hist = ROOT.TH1F("and_dis_resVsEta_eta_"+ str(etaRANGE[i]), "h_and_dis_resVsEta_eta_"+ str(etaRANGE[i])+";#eta residual (reco - gen); Number of muons", 100, -0.2, 0.2)
    pt_hist = ROOT.TH1F("and_dis_resVsEta_pt_"+ str(etaRANGE[i]), "h_and_dis_resVsEta_pt_"+ str(etaRANGE[i])+";p_{T} residual (reco - gen) [GeV]; Number of muons", 100, -5, 5)
    dxy_hist = ROOT.TH1F("and_dis_resVsEta_dxy_"+ str(etaRANGE[i]), "h_and_dis_resVsEta_dxy_"+ str(etaRANGE[i])+";d_{xy} residual (reco - gen) [cm]; Number of muons", 100, -5, 5)
    h_resVsEta_eta_and_dis.append(eta_hist)
    h_resVsEta_pt_and_dis.append(pt_hist)
    h_resVsEta_dxy_and_dis.append(dxy_hist)

for mu in range(len(ak.flatten(filteredRecoAndDisMuonEta))):
  for eta in range(len(etaRANGE)):
    if ((ak.flatten(filteredRecoAndDisMuonEta)[mu] > (-2.0 + eta * 0.2)) & (ak.flatten(filteredRecoAndDisMuonEta)[mu] < (-2.0 + (eta + 1)*0.2))):
      h_resVsEta_eta_and_dis[eta].Fill(ak.flatten(filteredRecoAndDisMuonEta_Diff)[mu])
      h_resVsEta_pt_and_dis[eta].Fill(ak.flatten(filteredRecoAndDisMuonPt_Diff)[mu])
      h_resVsEta_dxy_and_dis[eta].Fill(ak.flatten(filteredRecoAndDisMuondxy_Diff)[mu])

h2_resVsEta_eta_and_dis = ROOT.TH1F("and_dis_resVsEta2_eta", "Staus_M_100_100mm_13p6TeV_Run3Summer22;gen muon #eta;#eta resolution", len(etaRANGE), -2.4, 2.4)
h2_resVsEta_pt_and_dis = ROOT.TH1F("and_dis_resVsEta2_pt", "Staus_M_100_100mm_13p6TeV_Run3Summer22;gen muon #eta;p_{T} resolution", len(etaRANGE), -2.4, 2.4)
h2_resVsEta_dxy_and_dis = ROOT.TH1F("and_dis_resVsEta2_dxy", "Staus_M_100_100mm_13p6TeV_Run3Summer22;gen muon #eta;d_{xy} resolution", len(etaRANGE), -2.4, 2.4)

for i in range(len(etaRANGE)):

  #andDisPtLegend = ROOT.TLegend()
  #h_resVsEta_pt_and_dis[i].Draw("histe")
  #andDisPtLegend.AddEntry(h_resVsEta_pt_and_dis[i], "RMS = " + str(h_resVsEta_pt_and_dis[i].GetRMS()))  
  #andDisPtLegend.Draw()
  #can.SaveAs("MuonIdPlots/h_and_dis_resVsEta_pt_"+ str(etaRANGE[i])+".pdf")

  h2_resVsEta_eta_and_dis.SetBinContent(i + 1, h_resVsEta_eta_and_dis[i].GetRMS())
  h2_resVsEta_eta_and_dis.SetBinError(i + 1, h_resVsEta_eta_and_dis[i].GetRMSError())
  h2_resVsEta_pt_and_dis.SetBinContent(i + 1, h_resVsEta_pt_and_dis[i].GetRMS())
  h2_resVsEta_pt_and_dis.SetBinError(i + 1, h_resVsEta_pt_and_dis[i].GetRMSError())
  h2_resVsEta_dxy_and_dis.SetBinContent(i + 1, h_resVsEta_dxy_and_dis[i].GetRMS())
  h2_resVsEta_dxy_and_dis.SetBinError(i + 1, h_resVsEta_dxy_and_dis[i].GetRMSError())

h_resVsEta_eta_and_prompt = []
h_resVsEta_pt_and_prompt = []
h_resVsEta_dxy_and_prompt = []

for i in range(len(etaRANGE)):
    eta_hist = ROOT.TH1F("and_prompt_resVsEta_eta_"+ str(etaRANGE[i]), "h_and_prompt_resVsEta_eta_"+ str(etaRANGE[i])+";#eta residual (reco - gen); Number of muons", 100, -0.2, 0.2)
    pt_hist = ROOT.TH1F("and_prompt_resVsEta_pt_"+ str(etaRANGE[i]), "h_and_prompt_resVsEta_pt_"+ str(etaRANGE[i])+";p_{T} residual (reco - gen) [GeV]; Number of muons", 100, -5, 5)
    dxy_hist = ROOT.TH1F("and_prompt_resVsEta_dxy_"+ str(etaRANGE[i]), "h_and_prompt_resVsEta_dxy_"+ str(etaRANGE[i])+";d_{xy} residual (reco - gen) [cm]; Number of muons", 100, -5, 5)
    h_resVsEta_eta_and_prompt.append(eta_hist)
    h_resVsEta_pt_and_prompt.append(pt_hist)
    h_resVsEta_dxy_and_prompt.append(dxy_hist)

for mu in range(len(ak.flatten(filteredRecoAndDisMuonEta))):
  for eta in range(len(etaRANGE)):
    if ((ak.flatten(filteredRecoAndDisMuonEta)[mu] > (-2.0 + eta * 0.2)) & (ak.flatten(filteredRecoAndDisMuonEta)[mu] < (-2.0 + (eta + 1)*0.2))):
      h_resVsEta_eta_and_prompt[eta].Fill(ak.flatten(filteredRecoAndMuonEta_Diff)[mu])
      h_resVsEta_pt_and_prompt[eta].Fill(ak.flatten(filteredRecoAndMuonPt_Diff)[mu])
      h_resVsEta_dxy_and_prompt[eta].Fill(ak.flatten(filteredRecoAndMuondxy_Diff)[mu])

h2_resVsEta_eta_and_prompt = ROOT.TH1F("and_prompt_resVsEta2_eta", "Staus_M_100_100mm_13p6TeV_Run3Summer22;gen muon #eta;#eta resolution", len(etaRANGE), -2.4, 2.4)
h2_resVsEta_pt_and_prompt = ROOT.TH1F("and_prompt_resVsEta2_pt", "Staus_M_100_100mm_13p6TeV_Run3Summer22;gen muon #eta;p_{T} resolution", len(etaRANGE), -2.4, 2.4)
h2_resVsEta_dxy_and_prompt = ROOT.TH1F("and_prompt_resVsEta2_dxy", "Staus_M_100_100mm_13p6TeV_Run3Summer22;gen muon #eta;d_{xy} resolution", len(etaRANGE), -2.4, 2.4)


for i in range(len(etaRANGE)):

  #andPromptPtLegend = ROOT.TLegend()
  #h_resVsEta_pt_and_prompt[i].Draw("histe")
  #andPromptPtLegend.AddEntry(h_resVsEta_pt_and_prompt[i], "RMS = " + str(h_resVsEta_pt_and_prompt[i].GetRMS()))  
  #andPromptPtLegend.Draw()
  #can.SaveAs("MuonIdPlots/h_and_prompt_resVsEta_pt_"+ str(etaRANGE[i])+".pdf")

  h2_resVsEta_eta_and_prompt.SetBinContent(i + 1, h_resVsEta_eta_and_prompt[i].GetRMS())
  h2_resVsEta_eta_and_prompt.SetBinError(i + 1, h_resVsEta_eta_and_prompt[i].GetRMSError())
  h2_resVsEta_pt_and_prompt.SetBinContent(i + 1, h_resVsEta_pt_and_prompt[i].GetRMS())
  h2_resVsEta_pt_and_prompt.SetBinError(i + 1, h_resVsEta_pt_and_prompt[i].GetRMSError())
  h2_resVsEta_dxy_and_prompt.SetBinContent(i + 1, h_resVsEta_dxy_and_prompt[i].GetRMS())
  h2_resVsEta_dxy_and_prompt.SetBinError(i + 1, h_resVsEta_dxy_and_prompt[i].GetRMSError())

resVsEta_and_etaLegend = ROOT.TLegend()
resVsEta_and_ptLegend = ROOT.TLegend()
resVsEta_and_dxyLegend = ROOT.TLegend()

h2_resVsEta_eta_and_prompt.SetMinimum(0)
h2_resVsEta_eta_and_prompt.SetMarkerStyle(20)
h2_resVsEta_eta_and_prompt.SetMarkerColor(1)
resVsEta_and_etaLegend.AddEntry(h2_resVsEta_eta_and_prompt, "prompt muon reco")
h2_resVsEta_eta_and_prompt.Draw('p')
h2_resVsEta_eta_and_dis.SetMinimum(0)
h2_resVsEta_eta_and_dis.SetMarkerStyle(20)
h2_resVsEta_eta_and_dis.SetMarkerColor(2)
resVsEta_and_etaLegend.AddEntry(h2_resVsEta_eta_and_dis, "dis muon reco")
h2_resVsEta_eta_and_dis.Draw('psame')
resVsEta_and_etaLegend.Draw()
can.SaveAs("eta2p4cut/h2_resVsEta_eta_and.pdf")


h2_resVsEta_pt_and_prompt.SetMinimum(0)
#h2_resVsEta_pt_and_prompt.SetMaximum(2.3)
h2_resVsEta_pt_and_prompt.SetMarkerStyle(20)
h2_resVsEta_pt_and_prompt.SetMarkerColor(1)
resVsEta_and_ptLegend.AddEntry(h2_resVsEta_pt_and_prompt, "prompt muon reco")
h2_resVsEta_pt_and_prompt.Draw('p')
h2_resVsEta_pt_and_dis.SetMinimum(0)
h2_resVsEta_pt_and_dis.SetMarkerStyle(20)
h2_resVsEta_pt_and_dis.SetMarkerColor(2)
resVsEta_and_ptLegend.AddEntry(h2_resVsEta_pt_and_dis, "dis muon reco")
h2_resVsEta_pt_and_dis.Draw('psame')
#h2_resVsDxy_dxy_prompt.GetXaxis().SetRangeUser(-4.5, 5)
#h2_resVsDxy_dxy_dis.GetXaxis().SetRangeUser(-4.5, 5)
resVsEta_and_ptLegend.Draw()
can.SaveAs("eta2p4cut/h2_resVsEta_pt_and.pdf")


h2_resVsEta_dxy_and_prompt.SetMinimum(0)
#h2_resVsEta_dxy_and_prompt.SetMaximum(2.3)
h2_resVsEta_dxy_and_prompt.SetMarkerStyle(20)
h2_resVsEta_dxy_and_prompt.SetMarkerColor(1)
resVsEta_and_dxyLegend.AddEntry(h2_resVsEta_dxy_and_prompt, "prompt muon reco")
h2_resVsEta_dxy_and_prompt.Draw('p')
h2_resVsEta_dxy_and_dis.SetMinimum(0)
h2_resVsEta_dxy_and_dis.SetMarkerStyle(20)
h2_resVsEta_dxy_and_dis.SetMarkerColor(2)
resVsEta_and_dxyLegend.AddEntry(h2_resVsEta_dxy_and_dis, "dis muon reco")
h2_resVsEta_dxy_and_dis.Draw('psame')
#h2_resVsDxy_dxy_prompt.GetXaxis().SetRangeUser(-4.5, 5)
#h2_resVsDxy_dxy_dis.GetXaxis().SetRangeUser(-4.5, 5)
resVsEta_and_dxyLegend.Draw()
can.SaveAs("eta2p4cut/h2_resVsEta_dxy_and.pdf")

resVsPt_and_ptLegend = ROOT.TLegend()

h2_resVsPt_pt_and_prompt.SetMinimum(0)
h2_resVsPt_pt_and_prompt.SetMarkerStyle(20)
h2_resVsPt_pt_and_prompt.SetMarkerColor(1)
resVsPt_and_ptLegend.AddEntry(h2_resVsPt_pt_and_prompt, "prompt muon recon")
h2_resVsPt_pt_and_prompt.Draw('p')
h2_resVsPt_pt_and_dis.SetMinimum(0)
h2_resVsPt_pt_and_dis.SetMarkerStyle(20)
h2_resVsPt_pt_and_dis.SetMarkerColor(2)
resVsPt_and_ptLegend.AddEntry(h2_resVsPt_pt_and_dis, "dis muon recon")
h2_resVsPt_pt_and_dis.Draw('psame')
resVsPt_and_ptLegend.Draw()
can.SaveAs("eta2p4cut/h2_resVsPt_pt_and.pdf")

disMuon_pteff = ROOT.TEfficiency("disMuon_pteff", "Staus_M_100_100mm_13p6TeV_Run3Summer22; pt [GeV]; Fraction of gen dis muons which are recon'd", 20, 20, 100)
disMuonReco_pteff = ROOT.TEfficiency("disMuonReco_pteff", "Staus_M_100_100mm_13p6TeV_Run3Summer22; pt [GeV]; Fraction of gen dis muons which are recon'd", 20, 20, 100)
disMuonOr_pteff = ROOT.TEfficiency("disMuonOr_pteff", "Staus_M_100_100mm_13p6TeV_Run3Summer22; pt [GeV]; Fraction of gen dis muons which are recon'd", 20, 20, 100)
for i in range(20):
  disMuon_pteff.SetTotalEvents(i + 1, len(ak.flatten(genDisMuonPt[(genDisMuonPt > (20 + i * 4)) & (genDisMuonPt < (20 + (i + 1) * 4))])))
  disMuon_pteff.SetPassedEvents(i + 1, len(ak.flatten(filteredRecoMuonPt[(filteredRecoMuonPt > (20 + i * 4)) & (filteredRecoMuonPt < (20 + (i + 1) * 4))])))
  disMuonReco_pteff.SetTotalEvents(i + 1, len(ak.flatten(genDisMuonPt[(genDisMuonPt > (20 + i * 4)) & (genDisMuonPt < (20 + (i + 1) * 4))])))
  disMuonReco_pteff.SetPassedEvents(i + 1, len(ak.flatten(filteredRecoDisMuonPt[(filteredRecoDisMuonPt > (20 + i * 4)) & (filteredRecoDisMuonPt < (20 + (i + 1) * 4))])))
  disMuonOr_pteff.SetTotalEvents(i + 1, len(ak.flatten(genDisMuonPt[(genDisMuonPt > (20 + i * 4)) & (genDisMuonPt < (20 + (i + 1) * 4))])))
  disMuonOr_pteff.SetPassedEvents(i + 1, len(ak.flatten(filteredRecoOrMuonPt[(filteredRecoOrMuonPt > (20 + i * 4)) & (filteredRecoOrMuonPt < (20 + (i + 1) * 4))])))

disMuon_etaeff = ROOT.TEfficiency("disMuon_etaeff", "Staus_M_100_100mm_13p6TeV_Run3Summer22; #eta; Fraction of gen dis muons which are recon'd", 48, -2.4, 2.4)
disMuonReco_etaeff = ROOT.TEfficiency("disMuonReco_etaeff", "Staus_M_100_100mm_13p6TeV_Run3Summer22; #eta; Fraction of gen dis muons which are recon'd", 48, -2.4, 2.4)
disMuonOr_etaeff = ROOT.TEfficiency("disMuonOr_etaeff", "Staus_M_100_100mm_13p6TeV_Run3Summer22; #eta; Fraction of gen dis muons which are recon'd", 48, -2.4, 2.4)
for i in range(48):
  disMuon_etaeff.SetTotalEvents(i + 1, len(ak.flatten(genDisMuonEta[(genDisMuonEta > (-2.4 + 0.1 * i)) & (genDisMuonEta < (-2.4 + (i + 1) * 0.1))])))
  disMuon_etaeff.SetPassedEvents(i + 1, len(ak.flatten(filteredRecoMuonEta[(filteredRecoMuonEta > (i * 0.1 - 2.4)) & (filteredRecoMuonEta < ((i + 1) * 0.1 - 2.4))])))
  disMuonReco_etaeff.SetTotalEvents(i + 1, len(ak.flatten(genDisMuonEta[(genDisMuonEta > (-2.4 + 0.1 * i)) & (genDisMuonEta < (-2.4 + (i + 1) * 0.1))])))
  disMuonReco_etaeff.SetPassedEvents(i + 1, len(ak.flatten(filteredRecoDisMuonEta[(filteredRecoDisMuonEta > (i * 0.1 - 2.4)) & (filteredRecoDisMuonEta < ((i + 1) * 0.1 - 2.4))])))
  disMuonOr_etaeff.SetTotalEvents(i + 1, len(ak.flatten(genDisMuonEta[(genDisMuonEta > (-2.4 + 0.1 * i)) & (genDisMuonEta < (-2.4 + (i + 1) * 0.1))])))
  disMuonOr_etaeff.SetPassedEvents(i + 1, len(ak.flatten(filteredRecoOrMuonEta[(filteredRecoOrMuonEta > (i * 0.1 - 2.4)) & (filteredRecoOrMuonEta < ((i + 1) * 0.1 - 2.4))])))

disMuon_dxyeff = ROOT.TEfficiency("disMuon_dxyeff", "Staus_M_100_100mm_13p6TeV_Run3Summer22; d_{xy} [cm]; Fraction of gen dis muons which are recon'd", len(dxy_bin) - 1, dxy_bin)
disMuonReco_dxyeff = ROOT.TEfficiency("disMuonReco_dxyeff", "Staus_M_100_100mm_13p6TeV_Run3Summer22; d_{xy} [cm]; Fraction of gen dis muons which are recon'd", len(dxy_bin) - 1, dxy_bin)
disMuonOr_dxyeff = ROOT.TEfficiency("disMuonOr_dxyeff", "Staus_M_100_100mm_13p6TeV_Run3Summer22; d_{xy} [cm]; Fraction of gen dis muons which are recon'd", len(dxy_bin) - 1, dxy_bin)

disMuon_dxyeff_num = ROOT.TH1F("disMuon_dxyeff_num", "Staus_M_100_100mm_13p6TeV_Run3Summer22; d_{xy} [cm]; Number of gen dis muons which are recon'd", len(dxy_bin) - 1, dxy_bin)
disMuonReco_dxyeff_num = ROOT.TH1F("disMuonReco_dxyeff_num", "Staus_M_100_100mm_13p6TeV_Run3Summer22; d_{xy} [cm]; Number of gen dis muons which are recon'd", len(dxy_bin) - 1, dxy_bin)
disMuonOr_dxyeff_num = ROOT.TH1F("disMuonOr_dxyeff_num", "Staus_M_100_100mm_13p6TeV_Run3Summer22; d_{xy} [cm]; Number of gen dis muons which are recon'd", len(dxy_bin) - 1, dxy_bin)

disMuon_dxyeff_denom = ROOT.TH1F("disMuon_dxyeff_denom", "Staus_M_100_100mm_13p6TeV_Run3Summer22; d_{xy} [cm]; Number of gen dis muons", len(dxy_bin) - 1, dxy_bin)
disMuonReco_dxyeff_denom = ROOT.TH1F("disMuonReco_dxyeff_denom", "Staus_M_100_100mm_13p6TeV_Run3Summer22; d_{xy} [cm]; Number of gen dis muons", len(dxy_bin) - 1, dxy_bin)
disMuonOr_dxyeff_denom = ROOT.TH1F("disMuonOr_dxyeff_denom", "Staus_M_100_100mm_13p6TeV_Run3Summer22; d_{xy} [cm]; Number of gen dis muons", len(dxy_bin) - 1, dxy_bin)
for i in range(len(dxy_bin) - 1):
  disMuon_dxyeff.SetTotalEvents(i + 1, len(ak.flatten(genDisMuondxy[(genDisMuondxy >= dxy_bin[i]) & (genDisMuondxy < dxy_bin[i + 1])])))
  disMuon_dxyeff_denom.SetBinContent(i + 1, len(ak.flatten(genDisMuondxy[(genDisMuondxy >= dxy_bin[i]) & (genDisMuondxy < dxy_bin[i + 1])])))
  disMuon_dxyeff.SetPassedEvents(i + 1, len(ak.flatten(filteredRecoMuondxy[(filteredRecoMuondxy >= dxy_bin[i]) & (filteredRecoMuondxy < dxy_bin[i + 1])])))
  disMuon_dxyeff_num.SetBinContent(i + 1, len(ak.flatten(filteredRecoMuondxy[(filteredRecoMuondxy >= dxy_bin[i]) & (filteredRecoMuondxy < dxy_bin[i + 1])])))

  disMuonReco_dxyeff.SetTotalEvents(i + 1, len(ak.flatten(genDisMuondxy[(genDisMuondxy >= dxy_bin[i]) & (genDisMuondxy < dxy_bin[i + 1])])))
  disMuonReco_dxyeff_denom.SetBinContent(i + 1, len(ak.flatten(genDisMuondxy[(genDisMuondxy >= dxy_bin[i]) & (genDisMuondxy < dxy_bin[i + 1])])))
  disMuonReco_dxyeff.SetPassedEvents(i + 1, len(ak.flatten(filteredRecoDisMuondxy[(filteredRecoDisMuondxy >= dxy_bin[i]) & (filteredRecoDisMuondxy< dxy_bin[i + 1])])))
  disMuonReco_dxyeff_num.SetBinContent(i + 1, len(ak.flatten(filteredRecoDisMuondxy[(filteredRecoDisMuondxy >= dxy_bin[i]) & (filteredRecoDisMuondxy< dxy_bin[i + 1])])))  

  disMuonOr_dxyeff.SetTotalEvents(i + 1, len(ak.flatten(genDisMuondxy[(genDisMuondxy >= dxy_bin[i]) & (genDisMuondxy < dxy_bin[i + 1])])))
  disMuonOr_dxyeff_denom.SetBinContent(i + 1, len(ak.flatten(genDisMuondxy[(genDisMuondxy >= dxy_bin[i]) & (genDisMuondxy < dxy_bin[i + 1])])))
  disMuonOr_dxyeff.SetPassedEvents(i + 1, len(ak.flatten(filteredRecoOrMuondxy[(filteredRecoOrMuondxy >= dxy_bin[i]) & (filteredRecoOrMuondxy < dxy_bin[i + 1])])))
  disMuonOr_dxyeff_num.SetBinContent(i + 1, len(ak.flatten(filteredRecoOrMuondxy[(filteredRecoOrMuondxy >= dxy_bin[i]) & (filteredRecoOrMuondxy < dxy_bin[i + 1])])))  

#for i in range(200):
#  disMuon_dxyeff.SetTotalEvents(i + 1, len(ak.flatten(genDisMuondxy[(genDisMuondxy > (-30 + (100/200) * i)) & (genDisMuondxy < (-30 + (i + 1) * (100/200)))])))
#  disMuon_dxyeff.SetPassedEvents(i + 1, len(ak.flatten(filteredRecoMuondxy[(filteredRecoMuondxy > (i * (100/200) - 30)) & (filteredRecoMuondxy < ((i + 1) * (100/200) - 30))])))
#  disMuonReco_dxyeff.SetTotalEvents(i + 1, len(ak.flatten(genDisMuondxy[(genDisMuondxy > (-30 + (100/200) * i)) & (genDisMuondxy < (-30 + (i + 1) * (100/200)))])))
#  disMuonReco_dxyeff.SetPassedEvents(i + 1, len(ak.flatten(filteredRecoDisMuondxy[(filteredRecoDisMuondxy > (i * (100/200) - 30)) & (filteredRecoDisMuondxy < ((i + 1) * (100/200) - 30))])))
#  disMuonOr_dxyeff.SetTotalEvents(i + 1, len(ak.flatten(genDisMuondxy[(genDisMuondxy > (-30 + (100/200) * i)) & (genDisMuondxy < (-30 + (i + 1) * (100/200)))])))
#  disMuonOr_dxyeff.SetPassedEvents(i + 1, len(ak.flatten(filteredRecoOrMuondxy[(filteredRecoOrMuondxy > (i * (100/200) - 30)) & (filteredRecoOrMuondxy < ((i + 1) * (100/200) - 30))])))

PtEffLegend = ROOT.TLegend()
PtEffLegend.SetFillStyle(0)
disMuon_pteff.SetMarkerColor(1)
disMuonReco_pteff.SetMarkerColor(2)
disMuonOr_pteff.SetMarkerColor(3)
disMuon_pteff.SetLineColor(1)
disMuonReco_pteff.SetLineColor(2)
disMuonOr_pteff.SetLineColor(3)
PtEffLegend.AddEntry(disMuon_pteff, "Prompt muon reco")
PtEffLegend.AddEntry(disMuonReco_pteff, "Dis muon reco")
PtEffLegend.AddEntry(disMuonOr_pteff, "Combined muon reco")
disMuonReco_pteff.Draw()
disMuon_pteff.Draw("SAME")
disMuonOr_pteff.Draw("SAME")
ROOT.gPad.Update()
disMuon_pteff.GetPaintedGraph().GetYaxis().SetRangeUser(0, 1)
disMuonReco_pteff.GetPaintedGraph().GetYaxis().SetRangeUser(0, 1)
PtEffLegend.Draw()
can.SaveAs("eta2p4cut/DisMuonReconEff_promptVsDis_ptEff.pdf")

EtaEffLegend = ROOT.TLegend()
EtaEffLegend.SetFillStyle(0)
disMuon_etaeff.SetMarkerColor(1)
disMuonReco_etaeff.SetMarkerColor(2)
disMuonOr_etaeff.SetMarkerColor(3)
disMuon_etaeff.SetLineColor(1)
disMuonReco_etaeff.SetLineColor(2)
disMuonOr_etaeff.SetLineColor(3)
EtaEffLegend.AddEntry(disMuon_etaeff, "Prompt muon reco")
EtaEffLegend.AddEntry(disMuonReco_etaeff, "Dis muon reco")
EtaEffLegend.AddEntry(disMuonOr_etaeff, "Combined muon reco")
disMuonReco_etaeff.Draw()
disMuon_etaeff.Draw("SAME")
disMuonOr_etaeff.Draw("SAME")
ROOT.gPad.Update()
disMuon_etaeff.GetPaintedGraph().GetYaxis().SetRangeUser(0, 1)
disMuonReco_etaeff.GetPaintedGraph().GetYaxis().SetRangeUser(0, 1)
EtaEffLegend.Draw()
can.SaveAs("eta2p4cut/DisMuonReconEff_promptVsDis_etaEff.pdf")

DxyEffLegend = ROOT.TLegend()
DxyEffLegend.SetFillStyle(0)
disMuon_dxyeff.SetMarkerColor(1)
disMuonReco_dxyeff.SetMarkerColor(2)
disMuonOr_dxyeff.SetMarkerColor(3)
disMuon_dxyeff.SetLineColor(1)
disMuonReco_dxyeff.SetLineColor(2)
disMuonOr_dxyeff.SetLineColor(3)
DxyEffLegend.AddEntry(disMuon_dxyeff, "Prompt muon reco")
DxyEffLegend.AddEntry(disMuonReco_dxyeff, "Dis muon reco")
DxyEffLegend.AddEntry(disMuonOr_dxyeff, "Combined muon reco")
disMuonReco_dxyeff.Draw()
disMuon_dxyeff.Draw("SAME")
disMuonOr_dxyeff.Draw("SAME")
ROOT.gPad.Update()
disMuon_dxyeff.GetPaintedGraph().GetYaxis().SetRangeUser(0, 1)
disMuonReco_dxyeff.GetPaintedGraph().GetYaxis().SetRangeUser(0, 1)
disMuonOr_dxyeff.GetPaintedGraph().GetYaxis().SetRangeUser(0, 1)
DxyEffLegend.Draw()
can.SaveAs("eta2p4cut/DisMuonReconEff_promptVsDis_dxyEff.pdf")

DxyEffNumLegend = ROOT.TLegend()
DxyEffNumLegend.SetFillStyle(0)
disMuon_dxyeff_num.SetMarkerColor(1)
disMuonReco_dxyeff_num.SetMarkerColor(2)
disMuonOr_dxyeff_num.SetMarkerColor(3)
disMuon_dxyeff_num.SetLineColor(1)
disMuonReco_dxyeff_num.SetLineColor(2)
disMuonOr_dxyeff_num.SetLineColor(3)
DxyEffNumLegend.AddEntry(disMuon_dxyeff_num, "Prompt muon reco")
DxyEffNumLegend.AddEntry(disMuonReco_dxyeff_num, "Dis muon reco")
DxyEffNumLegend.AddEntry(disMuonOr_dxyeff_num, "Combined muon reco")
disMuonOr_dxyeff_num.Draw()
disMuonReco_dxyeff_num.Draw("SAME")
disMuon_dxyeff_num.Draw("SAME")
#ROOT.gPad.Update()
#disMuon_dxyeff.GetPaintedGraph().GetYaxis().SetRangeUser(0, 1)
#disMuonReco_dxyeff.GetPaintedGraph().GetYaxis().SetRangeUser(0, 1)
#disMuonOr_dxyeff.GetPaintedGraph().GetYaxis().SetRangeUser(0, 1)
DxyEffNumLegend.Draw()
can.SaveAs("eta2p4cut/DisMuonReconEff_promptVsDis_dxyEff_num.pdf")

DxyEffDenomLegend = ROOT.TLegend()
DxyEffDenomLegend.SetFillStyle(0)
disMuon_dxyeff_denom.SetMarkerColor(1)
disMuonReco_dxyeff_denom.SetMarkerColor(2)
disMuonOr_dxyeff_denom.SetMarkerColor(3)
disMuon_dxyeff_denom.SetLineColor(1)
disMuonReco_dxyeff_denom.SetLineColor(2)
disMuonOr_dxyeff_denom.SetLineColor(3)
DxyEffDenomLegend.AddEntry(disMuon_dxyeff_denom, "Prompt muon reco")
DxyEffDenomLegend.AddEntry(disMuonReco_dxyeff_denom, "Dis muon reco")
DxyEffDenomLegend.AddEntry(disMuonOr_dxyeff_denom, "Combined muon reco")
disMuonOr_dxyeff_denom.Draw()
disMuonReco_dxyeff_denom.Draw("SAME")
disMuon_dxyeff_denom.Draw("SAME")
#ROOT.gPad.Update()
#disMuon_dxyeff.GetPaintedGraph().GetYaxis().SetRangeUser(0, 1)
#disMuonReco_dxyeff.GetPaintedGraph().GetYaxis().SetRangeUser(0, 1)
#disMuonOr_dxyeff.GetPaintedGraph().GetYaxis().SetRangeUser(0, 1)
DxyEffDenomLegend.Draw()
can.SaveAs("eta2p4cut/DisMuonReconEff_promptVsDis_dxyEff_denom.pdf")
#disMuon_pteff.Draw()
#can.SaveAs("disMuon_recoPtEff.pdf")
#
#disMuon_etaeff.Draw()
#can.SaveAs("disMuon_recoEtaEff.pdf")

disMuon_dxyeff.Draw()
can.SaveAs("eta2p4cut/disMuon_recodxyEff.pdf")

promptDiffPt.Draw("COLZ")
can.SaveAs("eta2p4cut/disMuon_promptPtDiff.pdf")

promptDiffDxy.Draw("COLZ")
can.SaveAs("eta2p4cut/disMuon_promptDxyDiff.pdf")

disDiffPt.Draw("COLZ")
can.SaveAs("eta2p4cut/disMuon_disPtDiff.pdf")

disDiffDxy.Draw("COLZ")
can.SaveAs("eta2p4cut/disMuon_disDxyDiff.pdf")
