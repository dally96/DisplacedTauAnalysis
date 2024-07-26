import uproot
import scipy
import matplotlib as mpl
import awkward as ak
import numpy as np
import math
import ROOT
import array
import pandas as pd
from coffea.nanoevents import NanoEventsFactory


GenPtMin = 0
GenEtaMax = 2.5

can = ROOT.TCanvas("can", "can")

ROOT.gStyle.SetOptStat(0)

#file = "DYJetsToLL_M-50_TuneCP5_13p6TeV-madgraphMLM-pythia8_Run3Summer22MiniAODv4-forPOG_130X_mcRun3_2022_realistic_v5-v2_NanoAOD.root"
#file = "Staus_M_100_100mm_13p6TeV_Run3Summer22_DisMuon_GenPartMatch.root"
file = "Staus_M_100_100mm_13p6TeV_Run3Summer22_lpcdisptau_NanoAOD_ExtraDisMuonBranches.root"
#file = "Staus_M_400_100mm_13p6TeV_Run3Summer22_lpcdisptau_NanoAOD_ExtraDisMuonBranches.root"

rootFile = uproot.open(file)
Events = rootFile["Events"]
events = NanoEventsFactory.from_root({file:"Events"}).events()[0:21]

#gpart = events.GenPart
#staus_taus = staus.distinctChildren[(abs(staus.distinctChildren.pdgId) == 15) & (staus.distinctChildren.hasFlags("isLastCopy"))]
#gen_mus = staus_taus.distinctChildren[(abs(staus_taus.children.pdgId) == 13)]
#gen_electrons = staus_taus.distinctChildren[(abs(staus_taus.children.pdgId) == 11)]


lepBranches = {}
genBranches = {}

for key in Events.keys():
  if ("Muon_" in key) or ("Electron_" in key) or ("Tau_" in key):
    lepBranches[key] = Events[key].array()
  if ("GenPart_" in key) or ("GenVtx_" in key):
    genBranches[key] = Events[key].array()

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


muFromZ = []
allMuFromZ = []
eFromZ = []
allEFromZ = []
if "DY" in file:
  for evt in range(len(genBranches["GenPart_pdgId"])):
    muFromZ_evt = []
    allMuFromZ_evt = []
    eFromZ_evt = []
    allEFromZ_evt = []
    if (13 in genBranches["GenPart_pdgId"][evt]) and (-13 in genBranches["GenPart_pdgId"][evt]) and (23 in genBranches["GenPart_pdgId"][evt]):
      for part in range(len(genBranches["GenPart_pdgId"][evt])):
        if (abs(genBranches["GenPart_pdgId"][evt][part]) == 13) and (genBranches["GenPart_genPartIdxMother"][evt][part] >= 0): 
          if isFinalLepton(evt, part, 13):
            muFromZ_evt.append(part)
          if isMotherZ(evt, part, 13):
            allMuFromZ_evt.append(part)
  
    if (11 in genBranches["GenPart_pdgId"][evt]) and (-11 in genBranches["GenPart_pdgId"][evt]) and (23 in genBranches["GenPart_pdgId"][evt]):
      for part in range(len(genBranches["GenPart_pdgId"][evt])):
        if (abs(genBranches["GenPart_pdgId"][evt][part]) == 11) and (genBranches["GenPart_genPartIdxMother"][evt][part] >= 0):
          if isFinalLepton(evt, part, 11):
            eFromZ_evt.append(part)
          if isMotherZ(evt, part, 11):
            allEFromZ_evt.append(part)
    muFromZ.append(muFromZ_evt)
    allMuFromZ.append(allMuFromZ_evt)
    eFromZ.append(eFromZ_evt)
    allEFromZ.append(allEFromZ_evt)

genDisMuon = []
allGenDisMuon = []
genDisElectron = []
allGenDisElectron = []
if "Stau" in file:
  #for evt in range(len(genBranches["GenPart_pdgId"])):
  for evt in range(11):
    genDisMuon_evt = []
    allGenDisMuon_evt = []
    genDisElectron_evt = []
    allGenDisElectron_evt = []
    if (len(genBranches["GenPart_pdgId"][evt]) != 0):
      for part_idx in range(len(genBranches["GenPart_pdgId"][evt])):
        if (abs(genBranches["GenPart_pdgId"][evt][part_idx]) == 13):
          if ((genBranches["GenPart_pt"][evt][part_idx] < GenPtMin) and (abs(genBranches["GenPart_eta"][evt][part_idx]) > GenEtaMax)): continue 
          if (isMotherStau(evt, part_idx, 13)):
            print("For event", evt, "the pt of the muon is", genBranches["GenPart_pt"][evt][part_idx])
            allGenDisMuon_evt.append(part_idx)
          if (isFinalLeptonStau(evt, part_idx, 13)):
            genDisMuon_evt.append(part_idx)
        if (abs(genBranches["GenPart_pdgId"][evt][part_idx]) == 11) and (genBranches["GenPart_genPartIdxMother"][evt][part_idx] >= 0):
          if (isMotherStau(evt, part_idx, 11)):
            print("For event", evt, "the pt of the electron is", genBranches["GenPart_pt"][evt][part_idx])
            allGenDisElectron_evt.append(part_idx)
          if (isFinalLeptonStau(evt, part_idx, 11)):
            genDisElectron_evt.append(part_idx)
    genDisMuon.append(genDisMuon_evt)
    allGenDisMuon.append(allGenDisMuon_evt)
    genDisElectron.append(genDisElectron_evt)
    allGenDisElectron.append(allGenDisElectron_evt)
  




def dxy(evt, lep_idx):
  return (genBranches["GenPart_vertexY"][evt][lep_idx] - genBranches["GenVtx_y"][evt]) * np.cos(genBranches["GenPart_phi"][evt][lep_idx]) - (genBranches["GenPart_vertexX"][evt][lep_idx] - genBranches["GenVtx_x"][evt]) * np.sin(genBranches["GenPart_phi"][evt][lep_idx])

def Z_Lxy(evt, lep_idx):
  motherIdx = genBranches["GenPart_genPartIdxMother"][evt][lep_idx]
  L_squ = (genBranches["GenPart_vertexY"][evt][motherIdx] - genBranches["GenVtx_y"][evt]) ** 2 + (genBranches["GenPart_vertexX"][evt][motherIdx] - genBranches["GenVtx_x"][evt]) ** 2
  return np.sqrt(L_squ)

GenMuon_dxy = []
GenElectron_dxy = []
GenMuon_lxy = []
GenElectron_lxy = []
#for evt in range(len(genBranches["GenPart_pdgId"])):
#  GenMuon_dxy_evt = np.zeros(len(genBranches["GenPart_pdgId"][evt]))
#  GenElectron_dxy_evt = np.zeros(len(genBranches["GenPart_pdgId"][evt]))
#  GenMuon_lxy_evt = np.zeros(len(genBranches["GenPart_pdgId"][evt]))
#  GenElectron_lxy_evt = np.zeros(len(genBranches["GenPart_pdgId"][evt]))
#  if (len(genBranches["GenPart_pdgId"][evt]) != 0):
#    if "DY" in file:
#      for idx in range(len(allMuFromZ[evt])):
#        dxy_val = dxy(evt, allMuFromZ[evt][idx]) 
#        GenMuon_dxy_evt[allMuFromZ[evt][idx]] = dxy_val
#        GenMuon_lxy_evt[allMuFromZ[evt][idx]] = Z_Lxy(evt, allMuFromZ[evt][idx])
#  
#      for idx in range(len(allEFromZ[evt])):
#        dxy_val = dxy(evt, allEFromZ[evt][idx])
#        GenElectron_dxy_evt[allEFromZ[evt][idx]] = dxy_val
#        GenElectron_lxy_evt[allEFromZ[evt][idx]] = Z_Lxy(evt, allEFromZ[evt][idx])
#
#    if "Stau" in file:
#      for idx in range(len(allGenDisMuon[evt])):
#        GenMuon_dxy_evt[allGenDisMuon[evt][idx]] = dxy(evt, allGenDisMuon[evt][idx])
#        GenMuon_lxy_evt[allGenDisMuon[evt][idx]] = Z_Lxy(evt, allGenDisMuon[evt][idx])
#      
#      for idx in range(len(allGenDisElectron[evt])):
#        GenElectron_dxy_evt[allGenDisElectron[evt][idx]] = dxy(evt, allGenDisElectron[evt][idx])
#        GenElectron_lxy_evt[allGenDisElectron[evt][idx]] = Z_Lxy(evt, allGenDisElectron[evt][idx])
#
#  GenMuon_dxy.append(ak.Array(GenMuon_dxy_evt))
#  GenElectron_dxy.append(ak.Array(GenElectron_dxy_evt))
#  GenMuon_lxy.append(ak.Array(GenMuon_lxy_evt))
#  GenElectron_lxy.append(ak.Array(GenElectron_lxy_evt))
#
#GenMuon_dxy = ak.Array(GenMuon_dxy)
#GenElectron_dxy = ak.Array(GenElectron_dxy)
#GenMuon_lxy = ak.Array(GenMuon_lxy)
#GenElectron_lxy = ak.Array(GenElectron_lxy)
#
#if "DY" in file:
#  ak.to_parquet(muFromZ, "muFromZ.parquet")
#  ak.to_parquet(allMuFromZ, "allMuFromZ.parquet")
#  ak.to_parquet(eFromZ, "eFromZ.parquet")
#  ak.to_parquet(allEFromZ, "allEFromZ.parquet")
#
#  ak.to_parquet(GenMuon_dxy, "DY_GenMuon_dxy.parquet")
#  ak.to_parquet(GenMuon_lxy, "DY_GenMuon_lxy.parquet")
#  ak.to_parquet(GenElectron_dxy, "DY_GenElectron_dxy.parquet")
#  ak.to_parquet(GenElectron_lxy, "DY_GenElectron_lxy.parquet")
#
#if "Stau" in file:
#  ak.to_parquet(genDisMuon, "genDisMuon_"+file.split(".")[0]+".parquet")
#  ak.to_parquet(allGenDisMuon, "allGenDisMuon_"+file.split(".")[0]+".parquet")
#  ak.to_parquet(genDisElectron, "genDisElectron_"+file.split(".")[0]+".parquet")
#  ak.to_parquet(allGenDisElectron, "allGenDisElectron_"+file.split(".")[0]+".parquet")
#
#  ak.to_parquet(GenMuon_dxy, "Stau_GenMuon_dxy_"+file.split(".")[0]+".parquet")
#  ak.to_parquet(GenMuon_lxy, "Stau_GenMuon_lxy_"+file.split(".")[0]+".parquet")
#  ak.to_parquet(GenElectron_dxy, "Stau_GenElectron_dxy_"+file.split(".")[0]+".parquet")
#  ak.to_parquet(GenElectron_lxy, "Stau_GenElectron_lxy_"+file.split(".")[0]+".parquet")
#
