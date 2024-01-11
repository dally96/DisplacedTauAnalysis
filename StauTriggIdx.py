import uproot
import scipy
import matplotlib as mpl
import awkward as ak
import numpy as np
import math
import ROOT
import array
import pandas as pd
import os
import pickle

GenPtMin = 20
GenEtaMax = 2.4

can = ROOT.TCanvas("can", "can")

ROOT.gStyle.SetOptStat(0)

#file = "DYJetsToLL_M-50_TuneCP5_13p6TeV-madgraphMLM-pythia8_Run3Summer22MiniAODv4-forPOG_130X_mcRun3_2022_realistic_v5-v2_NanoAOD.root"
#file = "Staus_M_100_100mm_13p6TeV_Run3Summer22_DisMuon_GenPartMatch.root"
file = "Staus_M_100_100mm_13p6TeV_Run3Summer22_lpcdisptau_NanoAOD_ExtraMuonBranches.root"

data_file_path = "data_" + file.split(".")[0] + ".pkl"

rootFile = uproot.open(file)
Events = rootFile["Events"]

triggers = ["HLT_PFMET120_PFMHT120_IDTight", 
            "HLT_PFMET130_PFMHT130_IDTight", 
            "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight", 
            "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight", 
            "HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg", 
            "HLT_MET105_IsoTrk50", 
            "HLT_MET120_IsoTrk50", 
            "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1", 
            "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1",]

variables = ["pt", "eta", "dxy", "lxy"]

lepBranches = {}
genBranches = {}
triBranches = {}

for key in Events.keys():
  if ("Muon_" in key) or ("Electron_" in key) or ("Tau_" in key):
    lepBranches[key] = Events[key].array()
  if ("GenPart_" in key) or ("GenVtx_" in key):
    genBranches[key] = Events[key].array()
  for trig in triggers:
    if trig == key:
      print(key)
      triBranches[key] = Events[key].array() 

passtrig = {}
failtrig = {}

for trig in triBranches:
  passtrig[trig] = (triBranches[trig] == 1)
  failtrig[trig] = (triBranches[trig] == 0)

def isMotherZ(evt, part_idx, lepId):
  mother_idx = genBranches["GenPart_genPartIdxMother"][evt][part_idx]
  if (mother_idx == -1):
    return 0
  if (abs(genBranches["GenPart_pdgId"][evt][mother_idx]) == 23):
    return 1
  if (abs(genBranches["GenPart_pdgId"][evt][mother_idx]) == lepId):
    return isMotherZ(evt, mother_idx, lepId)
  else:
    return 0

def isMotherStau(evt, part_idx, lepId):
  #Find its mother
  mother_idx = genBranches["GenPart_genPartIdxMother"][evt][part_idx]
  #If it's a stau, we're done
  if (abs(genBranches["GenPart_pdgId"][evt][mother_idx]) == 1000015):
    return 1
  #If it's a tau, go further up the chain
  if (abs(genBranches["GenPart_pdgId"][evt][mother_idx]) == 15):
    return isMotherStau(evt, mother_idx, lepId)
  #If it's a lepton of the same type, go further up the chain
  if (abs(genBranches["GenPart_pdgId"][evt][mother_idx]) == lepId):
    return isMotherStau(evt, mother_idx, lepId)
  #Else, it's  not what we're looking for
  else:
    return 0        

#Function only to be used if we already know the particle decayed from a stau
def isMotherStauIdx(evt, part_idx):
  mother_idx = genBranches["GenPart_genPartIdxMother"][evt][part_idx]
  if (abs(genBranches["GenPart_pdgId"][evt][mother_idx]) == 1000015):
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
    if isMotherZ(evt, part_idx, lepId):
      return 1
    else:
      return 0 

def isFinalLeptonStau(evt, part_idx, lepId):
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

def dxy(evt, lep_idx):
  return (genBranches["GenPart_vertexY"][evt][lep_idx] - genBranches["GenVtx_y"][evt]) * np.cos(genBranches["GenPart_phi"][evt][lep_idx]) - (genBranches["GenPart_vertexX"][evt][lep_idx] - genBranches["GenVtx_x"][evt]) * np.sin(genBranches["GenPart_phi"][evt][lep_idx])

def Z_Lxy(evt, lep_idx):
  motherIdx = genBranches["GenPart_genPartIdxMother"][evt][lep_idx]
  L_squ = (genBranches["GenPart_vertexY"][evt][motherIdx] - genBranches["GenVtx_y"][evt]) ** 2 + (genBranches["GenPart_vertexX"][evt][motherIdx] - genBranches["GenVtx_x"][evt]) ** 2
  return np.sqrt(L_squ)

if not os.path.exists(data_file_path):
  genDisMuon = []
  allGenDisMuon = []
  genDisElectron = []
  allGenDisElectron = []
  for evt in range(len(genBranches["GenPart_pdgId"])):
    genDisMuon_evt = []
    allGenDisMuon_evt = []
    genDisElectron_evt = []
    allGenDisElectron_evt = []
    if (len(genBranches["GenPart_pdgId"][evt]) != 0):
      for part_idx in range(len(genBranches["GenPart_pdgId"][evt])):
        if (abs(genBranches["GenPart_pdgId"][evt][part_idx]) == 13):
          if ((genBranches["GenPart_pt"][evt][part_idx] < GenPtMin) and (abs(genBranches["GenPart_eta"][evt][part_idx]) > GenEtaMax)): continue 
          if (isMotherStau(evt, part_idx, 13)):
            allGenDisMuon_evt.append(part_idx)
          if (isFinalLeptonStau(evt, part_idx, 13)):
            genDisMuon_evt.append(part_idx)
        if (abs(genBranches["GenPart_pdgId"][evt][part_idx]) == 11) and (genBranches["GenPart_genPartIdxMother"][evt][part_idx] >= 0):
          if (isMotherStau(evt, part_idx, 11)):
            allGenDisElectron_evt.append(part_idx)
          if (isFinalLeptonStau(evt, part_idx, 11)):
            genDisElectron_evt.append(part_idx)
    genDisMuon.append(genDisMuon_evt)
    allGenDisMuon.append(allGenDisMuon_evt)
    genDisElectron.append(genDisElectron_evt)
    allGenDisElectron.append(allGenDisElectron_evt)
    
  GenMuon_dxy = []
  GenElectron_dxy = []
  GenMuon_lxy = []
  GenElectron_lxy = []
  for evt in range(len(genBranches["GenPart_pdgId"])):
    GenMuon_dxy_evt = np.zeros(len(genBranches["GenPart_pdgId"][evt]))
    GenElectron_dxy_evt = np.zeros(len(genBranches["GenPart_pdgId"][evt]))
    GenMuon_lxy_evt = np.zeros(len(genBranches["GenPart_pdgId"][evt]))
    GenElectron_lxy_evt = np.zeros(len(genBranches["GenPart_pdgId"][evt]))
    if (len(genBranches["GenPart_pdgId"][evt]) != 0):
      for idx in range(len(allGenDisMuon[evt])):
        GenMuon_dxy_evt[allGenDisMuon[evt][idx]] = dxy(evt, allGenDisMuon[evt][idx])
        GenMuon_lxy_evt[allGenDisMuon[evt][idx]] = Z_Lxy(evt, allGenDisMuon[evt][idx])
      
      for idx in range(len(allGenDisElectron[evt])):
        GenElectron_dxy_evt[allGenDisElectron[evt][idx]] = dxy(evt, allGenDisElectron[evt][idx])
        GenElectron_lxy_evt[allGenDisElectron[evt][idx]] = Z_Lxy(evt, allGenDisElectron[evt][idx])
  
    GenMuon_dxy.append(ak.Array(GenMuon_dxy_evt))
    GenElectron_dxy.append(ak.Array(GenElectron_dxy_evt))
    GenMuon_lxy.append(ak.Array(GenMuon_lxy_evt))
    GenElectron_lxy.append(ak.Array(GenElectron_lxy_evt))
  
  genBranches["GenMuon_dxy"] = ak.Array(GenMuon_dxy)
  genBranches["GenElectron_dxy"] = ak.Array(GenElectron_dxy)
  genBranches["GenMuon_lxy"] = ak.Array(GenMuon_lxy)
  genBranches["GenElectron_lxy"] = ak.Array(GenElectron_lxy)

  with open(data_file_path, 'wb') as data_file:
    pickle.dump({"genDisMuon": genDisMuon, 
                 "allGenDisMuon": allGenDisMuon,
                 "genDisElectron": genDisElectron,
                 "allGenDisElectron": allGenDisElectron,
                 "GenMuon_dxy": genBranches["GenMuon_dxy"],
                 "GenElectron_dxy": genBranches["GenElectron_dxy"],
                 "GenMuon_lxy": genBranches["GenMuon_lxy"],
                 "GenElectron_lxy": genBranches["GenElectron_lxy"]}, data_file)
else: 
  with open(data_file_path, 'rb') as data_file:
    data = pickle.load(data_file)
    genDisMuon = data["genDisMuon"]
    allGenDisMuon = data["allGenDisMuon"]
    genDisElectron = data["genDisElectron"]
    allGenDisElectron = data["allGenDisElectron"]
    genBranches["GenMuon_dxy"] = data["GenMuon_dxy"]
    genBranches["GenElectron_dxy"] = data["GenElectron_dxy"]
    genBranches["GenMuon_lxy"] = data["GenMuon_lxy"]
    genBranches["GenElectron_lxy"] = data["GenElectron_lxy"]

GenMu_pt  = genBranches["GenPart_pt"][genDisMuon] 
GenMu_eta = genBranches["GenPart_eta"][genDisMuon]
GenMu_dxy = np.abs(genBranches["GenMuon_dxy"][genDisMuon])
GenMu_lxy = genBranches["GenMuon_lxy"][genDisMuon]

GenMuEtaCut = (abs(GenMu_eta) < GenEtaMax)
GenMuPtCut  = (GenMu_pt > GenPtMin)

GenMu_pt  = GenMu_pt[GenMuEtaCut & GenMuPtCut]
GenMu_eta = GenMu_eta[GenMuEtaCut & GenMuPtCut]
GenMu_dxy = GenMu_dxy[GenMuEtaCut & GenMuPtCut]
GenMu_lxy = GenMu_lxy[GenMuEtaCut & GenMuPtCut]

pass_gen_mu = {}
fail_gen_mu = {}

for trig in triBranches:
  pass_gen_mu[trig] = {}
  fail_gen_mu[trig] = {}
  
  pass_gen_mu[trig]["GenMu_pt"]  = GenMu_pt[passtrig[trig]]
  pass_gen_mu[trig]["GenMu_eta"] = GenMu_eta[passtrig[trig]]
  pass_gen_mu[trig]["GenMu_dxy"] = GenMu_dxy[passtrig[trig]]
  pass_gen_mu[trig]["GenMu_lxy"] = GenMu_lxy[passtrig[trig]]

  fail_gen_mu[trig]["GenMu_pt"]  = GenMu_pt[failtrig[trig]]
  fail_gen_mu[trig]["GenMu_eta"] = GenMu_eta[failtrig[trig]]
  fail_gen_mu[trig]["GenMu_dxy"] = GenMu_dxy[failtrig[trig]]
  fail_gen_mu[trig]["GenMu_lxy"] = GenMu_lxy[failtrig[trig]]
   

RecoMuonsFromGen = [np.array([val for val in subarr if val in values]) for subarr, values in zip(lepBranches["Muon_genPartIdx"], allGenDisMuon)]
RecoMuonsFromGenIndices = [np.where(np.isin(subarr, values))[0] for subarr, values in zip(lepBranches["Muon_genPartIdx"], allGenDisMuon)]

RecoMuonsFromGen_pt  = genBranches["GenPart_pt"][RecoMuonsFromGen]
RecoMuonsFromGen_eta = genBranches["GenPart_eta"][RecoMuonsFromGen]
RecoMuonsFromGen_dxy = np.abs(genBranches["GenMuon_dxy"][RecoMuonsFromGen])
RecoMuonsFromGen_lxy = genBranches["GenMuon_lxy"][RecoMuonsFromGen]

RecoMuonsFromGenEtaCut = (abs(RecoMuonsFromGen_eta) < GenEtaMax)
RecoMuonsFromGenPtCut  = (RecoMuonsFromGen_pt > GenPtMin)

RecoMuonsFromGen_pt  = RecoMuonsFromGen_pt[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut] 
RecoMuonsFromGen_eta = RecoMuonsFromGen_eta[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut]
RecoMuonsFromGen_dxy = RecoMuonsFromGen_dxy[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut]
RecoMuonsFromGen_lxy = RecoMuonsFromGen_lxy[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut]

pass_reco_gen_mu = {}
fail_reco_gen_mu = {}

for trig in triBranches:
  pass_reco_gen_mu[trig] = {}
  fail_reco_gen_mu[trig] = {}

  pass_reco_gen_mu[trig]["RecoMuonsFromGen_pt"]  = RecoMuonsFromGen_pt[passtrig[trig]]
  pass_reco_gen_mu[trig]["RecoMuonsFromGen_eta"] = RecoMuonsFromGen_eta[passtrig[trig]]
  pass_reco_gen_mu[trig]["RecoMuonsFromGen_dxy"] = RecoMuonsFromGen_dxy[passtrig[trig]]
  pass_reco_gen_mu[trig]["RecoMuonsFromGen_lxy"] = RecoMuonsFromGen_lxy[passtrig[trig]]

  fail_reco_gen_mu[trig]["RecoMuonsFromGen_pt"]  = RecoMuonsFromGen_pt[failtrig[trig]]
  fail_reco_gen_mu[trig]["RecoMuonsFromGen_eta"] = RecoMuonsFromGen_eta[failtrig[trig]]
  fail_reco_gen_mu[trig]["RecoMuonsFromGen_dxy"] = RecoMuonsFromGen_dxy[failtrig[trig]]
  fail_reco_gen_mu[trig]["RecoMuonsFromGen_lxy"] = RecoMuonsFromGen_lxy[failtrig[trig]]
  
for var in variables:
  for trig in triBranches:
    


