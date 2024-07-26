import uproot
import scipy
import matplotlib as mpl
import awkward as ak
import numpy as np
import math
import ROOT
import array
import pandas as pd

stauFile = uproot.open("Staus_M_100_100mm_13p6TeV_Run3Summer22_NanoAOD.root")
Events = stauFile["Events"]
eleBranches = {}
genPartBranches = {}

can = ROOT.TCanvas("can", "can")
  
dxy_bin = [-30., -25., -20., -15., -10., -5., -4.5, -4., -3.5, -3., -2.5, -2., -1.5, -1., -0.5, 0,
           0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 10., 15., 20., 25., 30.]
dxy_bin = array.array('d', dxy_bin)

for key in Events.keys():
  if "Electron_" in key:
    eleBranches[key] = Events[key].array()
  if "GenPart_" in key:
    genPartBranches[key] = Events[key].array()

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
  

# This chooses the gen particles which are electrons which are decayed from displaced taus
genDisEle = []
genStauIdx = []
for evt in range(len(genPartBranches["GenPart_pdgId"])):
  genDisEle_evt = []
  genStauIdx_evt = []
  if (len(genPartBranches["GenPart_pdgId"][evt]) != 0):
    for part_idx in range(len(genPartBranches["GenPart_pdgId"][evt])):
      if (abs(genPartBranches["GenPart_pdgId"][evt][part_idx]) == 11):
        if (isMotherStau(evt, part_idx)):
          genDisEle_evt.append(part_idx)
          genStauIdx_evt.append(isMotherStauIdx(evt, part_idx))
  genDisEle.append(genDisEle_evt)
  genStauIdx.append(genStauIdx_evt)

genDisElePt = genPartBranches["GenPart_pt"][genDisEle]
genDisEleEta = genPartBranches["GenPart_eta"][genDisEle]
genDisElePhi = genPartBranches["GenPart_phi"][genDisEle]

def dxy(evt, muon_idx, stau_idx):
  return (genPartBranches["GenPart_vertexY"][evt][muon_idx] - genPartBranches["GenPart_vertexY"][evt][stau_idx]) * np.cos(genPartBranches["GenPart_phi"][evt][muon_idx]) - (genPartBranches["GenPart_vertexX"][evt][muon_idx] - genPartBranches["GenPart_vertexX"][evt][stau_idx]) * np.sin(genPartBranches["GenPart_phi"][evt][muon_idx])


genPartBranches["GenEle_dxy"] = []

for evt in range(len(genPartBranches["GenPart_pdgId"])):
  genDisEledxy_evt = np.zeros(len(genPartBranches["GenPart_pdgId"][evt]))
  if (len(genPartBranches["GenPart_pdgId"][evt]) != 0):
    for idx in range(len(genDisEle[evt])):
      dxy_val = dxy(evt, genDisEle[evt][idx], genStauIdx[evt][idx]) 
      genDisEledxy_evt[genDisEle[evt][idx]] = dxy_val
  genPartBranches["GenEle_dxy"].append(ak.Array(genDisEledxy_evt))
genPartBranches["GenEle_dxy"] = ak.Array(genPartBranches["GenEle_dxy"])
#print(genPartBranches["GenMuon_dxy"])

genDisEledxy = genPartBranches["GenEle_dxy"][genDisEle]
genDisElePt =  genDisElePt[genDisElePt > 20]
genDisEleEta = genDisEleEta[genDisElePt > 20]
genDisEledxy = genDisEledxy[genDisElePt > 20]

filteredRecoEle = [np.array([val for val in subarr if val in values]) for subarr, values in zip(eleBranches["Electron_genPartIdx"], genDisEle)]
filteredRecoEleIndices = [np.where(np.isin(subarr, values))[0] for subarr, values in zip(eleBranches["Electron_genPartIdx"], genDisEle)]

filteredRecoElePt  = genPartBranches["GenPart_pt"][filteredRecoEle]
filteredRecoEleEta = genPartBranches["GenPart_eta"][filteredRecoEle]
filteredRecoEledxy = genPartBranches["GenEle_dxy"][filteredRecoEle]

filteredRecoElePt  = filteredRecoElePt[filteredRecoElePt > 20]
filteredRecoEleEta = filteredRecoEleEta[filteredRecoElePt > 20]
filteredRecoEledxy = filteredRecoEledxy[filteredRecoElePt > 20]

disEle_pteff = ROOT.TEfficiency("disEle_pteff", "Staus_M_100_100mm_13p6TeV_Run3Summer22; pt [GeV]; Fraction of gen dis ele which are recon'd", 25, 0, 100)
for i in range(25):
  disEle_pteff.SetTotalEvents(i + 1, len(ak.flatten(genDisElePt[(genDisElePt > (i * 4)) & (genDisElePt < ((i + 1) * 4))])))
  disEle_pteff.SetPassedEvents(i + 1, len(ak.flatten(filteredRecoElePt[(filteredRecoElePt > (i * 4)) & (filteredRecoElePt < ((i + 1) * 4))])))

disEle_etaeff = ROOT.TEfficiency("disEle_etaeff", "Staus_M_100_100mm_13p6TeV_Run3Summer22; #eta; Fraction of gen dis ele which are recon'd", 16, -2.4, 2.4)
for i in range(16):
  disEle_etaeff.SetTotalEvents(i + 1, len(ak.flatten(genDisEleEta[(genDisEleEta > (-2.4 + 0.3 * i)) & (genDisEleEta < (-2.4 + (i + 1) * 0.3))])))
  disEle_etaeff.SetPassedEvents(i + 1, len(ak.flatten(filteredRecoEleEta[(filteredRecoEleEta > (i * 0.3 - 2.4)) & (filteredRecoEleEta < ((i + 1) * 0.3 - 2.4))])))

disEle_dxyeff = ROOT.TEfficiency("disEle_dxyeff", "Staus_M_100_100mm_13p6TeV_Run3Summer22; d_{xy} [cm]; Fraction of gen dis ele which are recon'd", len(dxy_bin) - 1, dxy_bin)
for i in range(len(dxy_bin) - 1):
  disEle_dxyeff.SetTotalEvents(i + 1, len(ak.flatten(genDisEledxy[(genDisEledxy >= dxy_bin[i]) & (genDisEledxy < dxy_bin[i + 1])])))
  disEle_dxyeff.SetPassedEvents(i + 1, len(ak.flatten(filteredRecoEledxy[(filteredRecoEledxy >= dxy_bin[i]) & (filteredRecoEledxy < dxy_bin[i + 1])])))

disEle_pteff.Draw()
can.SaveAs("disEle_recoPtEff.pdf")

disEle_etaeff.Draw()
can.SaveAs("disEle_recoEtaEff.pdf")

disEle_dxyeff.Draw()
can.SaveAs("disEle_recoDxyEff.pdf")


