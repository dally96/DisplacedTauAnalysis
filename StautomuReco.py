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

GenPtMin = 20
GenEtaMax = 2.4

can = ROOT.TCanvas("can", "can")


ptRANGE = ["20-25", "25-30", "30-35", "35-40", "40-45", "45-50", "50-55", "55-60", 
           "60-65", "65-70", "70-75", "75-80", "80-85", "85-90", "90-95", "95-100"]

etaRANGE = ["-2.4--2.2", "-2.2--2.0", "-2.0--1.8", "-1.8--1.6", "-1.6--1.4", "-1.4--1.2", "-1.2--1.0", "-1.0--0.8", "-0.8--0.6", "-0.6--0.4", "-0.4--0.2", "-0.2-0",
            "0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0", "1.0-1.2", "1.2-1.4", "1.4-1.6", "1.6-1.8", "1.8-2.0", "2.0-2.2", "2.2-2.4"]

dxyRANGE = ["0-0.5", "0.5-1", "1-1.5", "1.5-2", "2-2.5","2.5-3", "3-3.5", "3.5-4", "4-4.5", "4.5-5",
            "5-5.5", "5.5-6", "6-6.5", "6.5-7", "7-7.5","7.5-8", "8-8.5", "8.5-9", "9-9.5", "9.5-10",
            "10-10.5", "10.5-11", "11-11.5", "11.5-12", "12-12.5","12.5-13", "13-13.5", "13.5-14", "14-14.5", "14.5-15"]
            
lxyRANGE = ["0-0.5", "0.5-1", "1-1.5", "1.5-2", "2-2.5","2.5-3", "3-3.5", "3.5-4", "4-4.5", "4.5-5",
            "5-5.5", "5.5-6", "6-6.5", "6.5-7", "7-7.5","7.5-8", "8-8.5", "8.5-9", "9-9.5", "9.5-10",
            "10-10.5", "10.5-11", "11-11.5", "11.5-12", "12-12.5","12.5-13", "13-13.5", "13.5-14", "14-14.5", "14.5-15"]


ROOT.gStyle.SetOptStat(0)

file = "Staus_M_100_100mm_13p6TeV_Run3Summer22_DisMuon_GenPartMatch.root" 

rootFile = uproot.open(file)
Events = rootFile["Events"]

lepBranches = {}
genBranches = {}

for key in Events.keys():
  if ("Muon_" in key) or ("Electron_" in key) or ("Tau_" in key):
    lepBranches[key] = Events[key].array()
  if "GenPart_" in key:
    genBranches[key] = Events[key].array()

genBranches["GenMuon_dxy"] = ak.from_parquet("Stau_GenMuon_dxy.parquet")
genBranches["GenMuon_lxy"] = ak.from_parquet("Stau_GenMuon_lxy.parquet")

genBranches["GenElectron_dxy"] = ak.from_parquet("Stau_GenElectron_dxy.parquet")
genBranches["GenElectron_lxy"] = ak.from_parquet("Stau_GenElectron_lxy.parquet")
  

muFromZ = ak.from_parquet("genDisMuon.parquet")
allMuFromZ = ak.from_parquet("allGenDisMuon.parquet")

eFromZ = ak.from_parquet("genDisElectron.parquet")
allEFromZ =  ak.from_parquet("allGenDisElectron.parquet")

#for i in range(20,40):
#  print("Event:", i)
#  print("Gen electrons:", eFromZ[i])
#  print("Reconstructed electrons:", lepBranches["Electron_genPartIdx"][i])


GenMu_pt = genBranches["GenPart_pt"][muFromZ] 
GenMu_eta = genBranches["GenPart_eta"][muFromZ]
GenMu_dxy = np.abs(genBranches["GenMuon_dxy"][muFromZ])
GenMu_lxy = genBranches["GenMuon_lxy"][muFromZ]

GenMuEtaCut = (abs(GenMu_eta) < GenEtaMax)
GenMuPtCut  = (GenMu_pt > GenPtMin)

GenMu_pt = GenMu_pt[GenMuEtaCut & GenMuPtCut]
GenMu_eta = GenMu_eta[GenMuEtaCut & GenMuPtCut]
GenMu_dxy = GenMu_dxy[GenMuEtaCut & GenMuPtCut]
GenMu_lxy = GenMu_lxy[GenMuEtaCut & GenMuPtCut]

GenE_pt = genBranches["GenPart_pt"][eFromZ]
GenE_eta = genBranches["GenPart_eta"][eFromZ]
GenE_dxy = np.abs(genBranches["GenElectron_dxy"][eFromZ])
GenE_lxy = genBranches["GenElectron_lxy"][eFromZ]

GenEEtaCut = (abs(GenE_eta) < GenEtaMax)
GenEPtCut  = (GenE_pt > GenPtMin)

GenE_pt =  GenE_pt[GenEEtaCut & GenEPtCut]
GenE_eta = GenE_eta[GenEEtaCut & GenEPtCut] 
GenE_dxy = GenE_dxy[GenEEtaCut & GenEPtCut] 
GenE_lxy = GenE_lxy[GenEEtaCut & GenEPtCut] 

RecoMuonsFromGen = [np.array([val for val in subarr if val in values]) for subarr, values in zip(lepBranches["Muon_genPartIdx"], allMuFromZ)]
RecoMuonsFromGenIndices = [np.where(np.isin(subarr, values))[0] for subarr, values in zip(lepBranches["Muon_genPartIdx"], allMuFromZ)]

RecoDisMuonsFromGen = [np.array([val for val in subarr if val in values]) for subarr, values in zip(lepBranches["DisMuon_genPartIdx"], allMuFromZ)]
RecoDisMuonsFromGenIndices = [np.where(np.isin(subarr, values))[0] for subarr, values in zip(lepBranches["DisMuon_genPartIdx"], allMuFromZ)]

RecoOrMuonsFromGen = ak.Array([np.array(np.union1d(mu,dismu)) for mu, dismu in zip(RecoMuonsFromGen, RecoDisMuonsFromGen)])
RecoOrMuonsFromGen = ak.values_astype(RecoOrMuonsFromGen, "int64")

RecoAndMuonsFromGen_zip = [np.array(np.intersect1d(mu, dismu, return_indices=True)) for mu, dismu in zip(RecoMuonsFromGen, RecoDisMuonsFromGen)]
RecoAndMuonsFromGen_unzip = list(zip(*RecoAndMuonsFromGen_zip))
RecoAndMuonsFromGen, RecoAndPromptMuonsFromGenIndices, RecoAndDisMuonsFromGenIndices = RecoAndMuonsFromGen_unzip
RecoAndMuonsFromGen = ak.values_astype(RecoAndMuonsFromGen, "int64")
RecoAndPromptMuonsFromGenIndices = ak.values_astype(RecoAndPromptMuonsFromGenIndices, "int64")
RecoAndDisMuonsFromGenIndices = ak.values_astype(RecoAndDisMuonsFromGenIndices, "int64")   

RecoElectronsFromGen = [np.array([val for val in subarr if val in values]) for subarr, values in zip(lepBranches["Electron_genPartIdx"], allEFromZ)]
RecoElectronsFromGenIndices = [np.where(np.isin(subarr, values))[0] for subarr, values in zip(lepBranches["Electron_genPartIdx"], allEFromZ)]

RecoMuons_pt = lepBranches["Muon_pt"][RecoMuonsFromGenIndices]
RecoMuons_eta = lepBranches["Muon_eta"][RecoMuonsFromGenIndices]
RecoMuons_dxy = np.abs(lepBranches["Muon_dxy"][RecoMuonsFromGenIndices])

RecoDisMuons_pt = lepBranches["DisMuon_pt"][RecoDisMuonsFromGenIndices]
RecoDisMuons_eta = lepBranches["DisMuon_eta"][RecoDisMuonsFromGenIndices]
RecoDisMuons_dxy = np.abs(lepBranches["DisMuon_dxy"][RecoDisMuonsFromGenIndices])

RecoAndPromptMuons_pt = lepBranches["Muon_pt"][RecoAndPromptMuonsFromGenIndices]
RecoAndPromptMuons_eta = lepBranches["Muon_eta"][RecoAndPromptMuonsFromGenIndices]
RecoAndPromptMuons_dxy = np.abs(lepBranches["Muon_dxy"][RecoAndPromptMuonsFromGenIndices])

RecoAndDisMuons_pt = lepBranches["DisMuon_pt"][RecoAndDisMuonsFromGenIndices]
RecoAndDisMuons_eta = lepBranches["DisMuon_eta"][RecoAndDisMuonsFromGenIndices]
RecoAndDisMuons_dxy = np.abs(lepBranches["DisMuon_dxy"][RecoAndDisMuonsFromGenIndices])

RecoElectrons_pt =  lepBranches["Electron_pt"][RecoElectronsFromGenIndices]   
RecoElectrons_eta = lepBranches["Electron_eta"][RecoElectronsFromGenIndices]
RecoElectrons_dxy = np.abs(lepBranches["Electron_dxy"][RecoElectronsFromGenIndices])

RecoMuons_isStandalone = lepBranches["Muon_isStandalone"][RecoMuonsFromGenIndices]
RecoMuons_isGlobal = lepBranches["Muon_isGlobal"][RecoMuonsFromGenIndices]

RecoDisMuons_isStandalone = lepBranches["DisMuon_isStandalone"][RecoDisMuonsFromGenIndices]
RecoDisMuons_isGlobal = lepBranches["DisMuon_isGlobal"][RecoDisMuonsFromGenIndices]

RecoMuonsFromGen_pt  = genBranches["GenPart_pt"][RecoMuonsFromGen]
RecoMuonsFromGen_eta = genBranches["GenPart_eta"][RecoMuonsFromGen]
RecoMuonsFromGen_dxy = np.abs(genBranches["GenMuon_dxy"][RecoMuonsFromGen])
RecoMuonsFromGen_lxy = genBranches["GenMuon_lxy"][RecoMuonsFromGen]

RecoDisMuonsFromGen_pt  = genBranches["GenPart_pt"][RecoDisMuonsFromGen]
RecoDisMuonsFromGen_eta = genBranches["GenPart_eta"][RecoDisMuonsFromGen]
RecoDisMuonsFromGen_dxy = np.abs(genBranches["GenMuon_dxy"][RecoDisMuonsFromGen])
RecoDisMuonsFromGen_lxy = genBranches["GenMuon_lxy"][RecoDisMuonsFromGen]

RecoOrMuonsFromGen_pt  = genBranches["GenPart_pt"][RecoOrMuonsFromGen]
RecoOrMuonsFromGen_eta = genBranches["GenPart_eta"][RecoOrMuonsFromGen]
RecoOrMuonsFromGen_dxy = np.abs(genBranches["GenMuon_dxy"][RecoOrMuonsFromGen])
RecoOrMuonsFromGen_lxy = genBranches["GenMuon_lxy"][RecoOrMuonsFromGen]

RecoAndMuonsFromGen_pt  = genBranches["GenPart_pt"][RecoAndMuonsFromGen]
RecoAndMuonsFromGen_eta = genBranches["GenPart_eta"][RecoAndMuonsFromGen]
RecoAndMuonsFromGen_dxy = np.abs(genBranches["GenMuon_dxy"][RecoAndMuonsFromGen])
RecoAndMuonsFromGen_lxy = genBranches["GenMuon_lxy"][RecoAndMuonsFromGen]

RecoMuonsFromGenEtaCut = (abs(RecoMuonsFromGen_eta) < GenEtaMax)
RecoMuonsFromGenPtCut  = (RecoMuonsFromGen_pt > GenPtMin)

RecoDisMuonsFromGenEtaCut = (abs(RecoDisMuonsFromGen_eta) < GenEtaMax)
RecoDisMuonsFromGenPtCut  = (RecoDisMuonsFromGen_pt > GenPtMin)

RecoOrMuonsFromGenEtaCut = (abs(RecoOrMuonsFromGen_eta) < GenEtaMax)
RecoOrMuonsFromGenPtCut  = (RecoOrMuonsFromGen_pt > GenPtMin)

RecoAndMuonsFromGenEtaCut = (abs(RecoAndMuonsFromGen_eta) < GenEtaMax)
RecoAndMuonsFromGenPtCut  = (RecoAndMuonsFromGen_pt > GenPtMin)

RecoMuonsFromGen_pt = RecoMuonsFromGen_pt[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut]
RecoMuonsFromGen_eta = RecoMuonsFromGen_eta[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut]
RecoMuonsFromGen_dxy = RecoMuonsFromGen_dxy[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut]
RecoMuonsFromGen_lxy = RecoMuonsFromGen_lxy[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut]

RecoDisMuonsFromGen_pt = RecoDisMuonsFromGen_pt[RecoDisMuonsFromGenEtaCut & RecoDisMuonsFromGenPtCut]
RecoDisMuonsFromGen_eta = RecoDisMuonsFromGen_eta[RecoDisMuonsFromGenEtaCut & RecoDisMuonsFromGenPtCut]
RecoDisMuonsFromGen_dxy = RecoDisMuonsFromGen_dxy[RecoDisMuonsFromGenEtaCut & RecoDisMuonsFromGenPtCut]
RecoDisMuonsFromGen_lxy = RecoDisMuonsFromGen_lxy[RecoDisMuonsFromGenEtaCut & RecoDisMuonsFromGenPtCut]

RecoOrMuonsFromGen_pt = RecoOrMuonsFromGen_pt[RecoOrMuonsFromGenEtaCut & RecoOrMuonsFromGenPtCut]
RecoOrMuonsFromGen_eta = RecoOrMuonsFromGen_eta[RecoOrMuonsFromGenEtaCut & RecoOrMuonsFromGenPtCut]
RecoOrMuonsFromGen_dxy = RecoOrMuonsFromGen_dxy[RecoOrMuonsFromGenEtaCut & RecoOrMuonsFromGenPtCut]
RecoOrMuonsFromGen_lxy = RecoOrMuonsFromGen_lxy[RecoOrMuonsFromGenEtaCut & RecoOrMuonsFromGenPtCut]

RecoAndMuonsFromGen_pt =  RecoAndMuonsFromGen_pt[RecoAndMuonsFromGenEtaCut & RecoAndMuonsFromGenPtCut]
RecoAndMuonsFromGen_eta = RecoAndMuonsFromGen_eta[RecoAndMuonsFromGenEtaCut & RecoAndMuonsFromGenPtCut]
RecoAndMuonsFromGen_dxy = RecoAndMuonsFromGen_dxy[RecoAndMuonsFromGenEtaCut & RecoAndMuonsFromGenPtCut]
RecoAndMuonsFromGen_lxy = RecoAndMuonsFromGen_lxy[RecoAndMuonsFromGenEtaCut & RecoAndMuonsFromGenPtCut]

RecoElectronsFromGen_pt  = genBranches["GenPart_pt"][RecoElectronsFromGen]
RecoElectronsFromGen_eta = genBranches["GenPart_eta"][RecoElectronsFromGen]
RecoElectronsFromGen_dxy = np.abs(genBranches["GenElectron_dxy"][RecoElectronsFromGen])
RecoElectronsFromGen_lxy = genBranches["GenElectron_lxy"][RecoElectronsFromGen]

RecoElectronsFromGenEtaCut = (abs(RecoElectronsFromGen_eta) < GenEtaMax)
RecoElectronsFromGenPtCut  = (RecoElectronsFromGen_pt > GenPtMin)

RecoElectronsFromGen_pt  = RecoElectronsFromGen_pt[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut]
RecoElectronsFromGen_eta = RecoElectronsFromGen_eta[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut]
RecoElectronsFromGen_dxy = RecoElectronsFromGen_dxy[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut]
RecoElectronsFromGen_lxy = RecoElectronsFromGen_lxy[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut]

RecoMuons_pt = RecoMuons_pt[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut]
RecoMuons_eta = RecoMuons_eta[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut]
RecoMuons_dxy = RecoMuons_dxy[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut]

RecoDisMuons_pt =  RecoDisMuons_pt[RecoDisMuonsFromGenEtaCut & RecoDisMuonsFromGenPtCut]
RecoDisMuons_eta = RecoDisMuons_eta[RecoDisMuonsFromGenEtaCut & RecoDisMuonsFromGenPtCut]
RecoDisMuons_dxy = RecoDisMuons_dxy[RecoDisMuonsFromGenEtaCut & RecoDisMuonsFromGenPtCut]

RecoAndPromptMuons_pt = RecoAndPromptMuons_pt[RecoAndMuonsFromGenEtaCut & RecoAndMuonsFromGenPtCut]
RecoAndPromptMuons_eta = RecoAndPromptMuons_eta[RecoAndMuonsFromGenEtaCut & RecoAndMuonsFromGenPtCut]
RecoAndPromptMuons_dxy = RecoAndPromptMuons_dxy[RecoAndMuonsFromGenEtaCut & RecoAndMuonsFromGenPtCut]

RecoAndDisMuons_pt =  RecoAndDisMuons_pt[RecoAndMuonsFromGenEtaCut & RecoAndMuonsFromGenPtCut]
RecoAndDisMuons_eta = RecoAndDisMuons_eta[RecoAndMuonsFromGenEtaCut & RecoAndMuonsFromGenPtCut]
RecoAndDisMuons_dxy = RecoAndDisMuons_dxy[RecoAndMuonsFromGenEtaCut & RecoAndMuonsFromGenPtCut]

RecoElectrons_pt =  RecoElectrons_pt[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut] 
RecoElectrons_eta = RecoElectrons_eta[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut]
RecoElectrons_dxy = RecoElectrons_dxy[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut]

RecoMuons_isStandalone = RecoMuons_isStandalone[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut]
RecoMuons_isGlobal = RecoMuons_isGlobal[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut]

RecoDisMuons_isStandalone = RecoDisMuons_isStandalone[RecoDisMuonsFromGenEtaCut & RecoDisMuonsFromGenPtCut]
RecoDisMuons_isGlobal = RecoDisMuons_isGlobal[RecoDisMuonsFromGenEtaCut & RecoDisMuonsFromGenPtCut]

SAMuons = ((RecoMuons_isStandalone == 1) & (RecoMuons_isGlobal != 1))
GlobalMuons = (RecoMuons_isGlobal == 1)

SADisMuons = ((RecoDisMuons_isStandalone == 1) & (RecoDisMuons_isGlobal != 1))
GlobalDisMuons = (RecoDisMuons_isGlobal == 1)

SARecoMuons_pt = RecoMuons_pt[SAMuons]
SARecoMuons_eta = RecoMuons_eta[SAMuons]
SARecoMuons_dxy = RecoMuons_dxy[SAMuons]

SARecoDisMuons_pt =  RecoDisMuons_pt[SADisMuons]
SARecoDisMuons_eta = RecoDisMuons_eta[SADisMuons]
SARecoDisMuons_dxy = RecoDisMuons_dxy[SADisMuons]

GlobalRecoMuons_pt = RecoMuons_pt[GlobalMuons]
GlobalRecoMuons_eta = RecoMuons_eta[GlobalMuons]
GlobalRecoMuons_dxy = RecoMuons_dxy[GlobalMuons]

GlobalRecoDisMuons_pt = RecoDisMuons_pt[GlobalDisMuons]
GlobalRecoDisMuons_eta = RecoDisMuons_eta[GlobalDisMuons]
GlobalRecoDisMuons_dxy = RecoDisMuons_dxy[GlobalDisMuons]

SARecoMuonsFromGen_pt = RecoMuonsFromGen_pt[SAMuons]
SARecoMuonsFromGen_eta = RecoMuonsFromGen_eta[SAMuons]
SARecoMuonsFromGen_dxy = RecoMuonsFromGen_dxy[SAMuons]
SARecoMuonsFromGen_lxy = RecoMuonsFromGen_lxy[SAMuons]

SARecoDisMuonsFromGen_pt = RecoDisMuonsFromGen_pt[SADisMuons]
SARecoDisMuonsFromGen_eta = RecoDisMuonsFromGen_eta[SADisMuons]
SARecoDisMuonsFromGen_dxy = RecoDisMuonsFromGen_dxy[SADisMuons]
SARecoDisMuonsFromGen_lxy = RecoDisMuonsFromGen_lxy[SADisMuons]

GlobalRecoMuonsFromGen_pt  = RecoMuonsFromGen_pt[GlobalMuons]
GlobalRecoMuonsFromGen_eta = RecoMuonsFromGen_eta[GlobalMuons]
GlobalRecoMuonsFromGen_dxy = RecoMuonsFromGen_dxy[GlobalMuons]
GlobalRecoMuonsFromGen_lxy = RecoMuonsFromGen_lxy[GlobalMuons]

GlobalRecoDisMuonsFromGen_pt = RecoDisMuonsFromGen_pt[GlobalDisMuons]
GlobalRecoDisMuonsFromGen_eta = RecoDisMuonsFromGen_eta[GlobalDisMuons]
GlobalRecoDisMuonsFromGen_dxy = RecoDisMuonsFromGen_dxy[GlobalDisMuons]
GlobalRecoDisMuonsFromGen_lxy = RecoDisMuonsFromGen_lxy[GlobalDisMuons]

RecoMuon_ptDiff = ak.Array(np.subtract(RecoMuons_pt, RecoMuonsFromGen_pt))
RecoMuon_etaDiff = ak.Array(np.subtract(RecoMuons_eta, RecoMuonsFromGen_eta))
RecoMuon_dxyDiff = ak.Array(np.subtract(RecoMuons_dxy, RecoMuonsFromGen_dxy))

RecoAndPromptMuon_ptDiff  = ak.Array(np.subtract(RecoAndPromptMuons_pt,  RecoAndMuonsFromGen_pt))
RecoAndPromptMuon_etaDiff = ak.Array(np.subtract(RecoAndPromptMuons_eta, RecoAndMuonsFromGen_eta))
RecoAndPromptMuon_dxyDiff = ak.Array(np.subtract(RecoAndPromptMuons_dxy, RecoAndMuonsFromGen_dxy))

RecoAndDisMuon_ptDiff  = ak.Array(np.subtract(RecoAndDisMuons_pt,  RecoAndMuonsFromGen_pt))
RecoAndDisMuon_etaDiff = ak.Array(np.subtract(RecoAndDisMuons_eta, RecoAndMuonsFromGen_eta))
RecoAndDisMuon_dxyDiff = ak.Array(np.subtract(RecoAndDisMuons_dxy, RecoAndMuonsFromGen_dxy))

SARecoMuon_ptDiff = ak.Array(np.subtract(RecoMuons_pt, RecoMuonsFromGen_pt))[SAMuons]
SARecoMuon_etaDiff = ak.Array(np.subtract(RecoMuons_eta, RecoMuonsFromGen_eta))[SAMuons]
SARecoMuon_dxyDiff = ak.Array(np.subtract(RecoMuons_dxy, RecoMuonsFromGen_dxy))[SAMuons]

GlobalRecoMuon_ptDiff = ak.Array(np.subtract(RecoMuons_pt, RecoMuonsFromGen_pt))[GlobalMuons]
GlobalRecoMuon_etaDiff = ak.Array(np.subtract(RecoMuons_eta, RecoMuonsFromGen_eta))[GlobalMuons]
GlobalRecoMuon_dxyDiff = ak.Array(np.subtract(RecoMuons_dxy, RecoMuonsFromGen_dxy))[GlobalMuons]

RecoDisMuon_ptDiff = ak.Array(np.subtract(RecoDisMuons_pt, RecoDisMuonsFromGen_pt))
RecoDisMuon_etaDiff = ak.Array(np.subtract(RecoDisMuons_eta, RecoDisMuonsFromGen_eta))
RecoDisMuon_dxyDiff = ak.Array(np.subtract(RecoDisMuons_dxy, RecoDisMuonsFromGen_dxy))

SARecoDisMuon_ptDiff = ak.Array(np.subtract(RecoDisMuons_pt, RecoDisMuonsFromGen_pt))[SADisMuons]
SARecoDisMuon_etaDiff = ak.Array(np.subtract(RecoDisMuons_eta, RecoDisMuonsFromGen_eta))[SADisMuons]
SARecoDisMuon_dxyDiff = ak.Array(np.subtract(RecoDisMuons_dxy, RecoDisMuonsFromGen_dxy))[SADisMuons]

GlobalRecoDisMuon_ptDiff = ak.Array(np.subtract(RecoDisMuons_pt, RecoDisMuonsFromGen_pt))[GlobalDisMuons]
GlobalRecoDisMuon_etaDiff = ak.Array(np.subtract(RecoDisMuons_eta, RecoDisMuonsFromGen_eta))[GlobalDisMuons]
GlobalRecoDisMuon_dxyDiff = ak.Array(np.subtract(RecoDisMuons_dxy, RecoDisMuonsFromGen_dxy))[GlobalDisMuons]

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
    h2_resVsX_y_dict[dict_entries[ent]] = ROOT.TH1F("h2_resVsX_y_"+dict_entries[ent], file.split(".")[0]+";gen "+lepton+" "+xvar+" "+xunit+";"+yvar+" resolution "+yunit, len(xrange), xmin, xmax)

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
  can.SaveAs("Stauto"+lepton+"_resVs"+xvar+"_"+yvar+".pdf")


def makeEffPlot(lepton, plot_type, dict_entries, xvar, bins, xmin, xmax, xbinsize, xunit, tot_arr, pass_arr, log_set):
  h_eff_dict = {}
  h_eff_num_dict = {}
  h_eff_den_dict = {}
  
  for ent in range(len(dict_entries)):
    h_eff_dict[dict_entries[ent]] = ROOT.TEfficiency("h_eff_"+xvar+"_"+dict_entries[ent], file.split(".")[0]+";"+xvar+" "+xunit+" ; Fraction of gen "+lepton+" which are recon'd", bins, xmin, xmax)
    h_eff_num_dict[dict_entries[ent]] = ROOT.TH1F("h_eff_num"+xvar+"_"+dict_entries[ent], file.split(".")[0]+";"+xvar+" "+xunit+" ; Number of gen "+lepton+" which are recon'd", bins, xmin, xmax)
    h_eff_den_dict[dict_entries[ent]] = ROOT.TH1F("h_eff_den"+xvar+"_"+dict_entries[ent], file.split(".")[0]+";"+xvar+" "+xunit+" ; Number of gen "+lepton, bins, xmin, xmax)

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
  can.SaveAs("Stauto"+lepton+"_eff_"+plot_type+"_"+xvar+".pdf")

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
  can.SaveAs("Stauto"+lepton+"_eff_"+xvar+"_num.pdf")

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
  can.SaveAs("Stauto"+lepton+"_eff_"+xvar+"_den.pdf")

#makeEffPlot("mu", ["Prompt", "Dis", "Either"], "pt", 80, 20, 100, 1, "[GeV]", GenMu_pt, [RecoMuonsFromGen_pt, RecoDisMuonsFromGen_pt, RecoOrMuonsFromGen_pt], 0) 
#makeEffPlot("mu", ["Prompt", "Dis", "Either"], "eta", 24, -2.4, 2.4, 0.2, " ", GenMu_eta, [RecoMuonsFromGen_eta, RecoDisMuonsFromGen_eta, RecoOrMuonsFromGen_eta], 0) 
#makeEffPlot("mu", ["Prompt", "Dis", "Either"], "dxy", 30, 0, 15, 0.5, "[cm]", GenMu_dxy, [RecoMuonsFromGen_dxy, RecoDisMuonsFromGen_dxy, RecoOrMuonsFromGen_dxy], 0) 
#makeEffPlot("mu", ["Prompt", "Dis", "Either"], "lxy", 30, 0, 15, 0.5, "[cm]", GenMu_lxy, [RecoMuonsFromGen_lxy, RecoDisMuonsFromGen_lxy, RecoOrMuonsFromGen_lxy], 0) 

makeEffPlot("mu", "promptmutype", ["All", "Global", "STA only"], "pt", 80, 20, 100, 1, "[GeV]", GenMu_pt, [RecoMuonsFromGen_pt, GlobalRecoMuonsFromGen_pt, SARecoMuonsFromGen_pt],0) 
makeEffPlot("mu", "promptmutype", ["All", "Global", "STA only"], "eta", 24, -2.4, 2.4, 0.2, " ", GenMu_eta, [RecoMuonsFromGen_eta, GlobalRecoMuonsFromGen_eta, SARecoMuonsFromGen_eta],0) 
makeEffPlot("mu", "promptmutype", ["All", "Global", "STA only"], "dxy", 30, 0, 15, 0.5, "[cm]", GenMu_dxy, [RecoMuonsFromGen_dxy, GlobalRecoMuonsFromGen_dxy, SARecoMuonsFromGen_dxy],0) 
makeEffPlot("mu", "promptmutype", ["All", "Global", "STA only"], "lxy", 30, 0, 15, 0.5, "[cm]", GenMu_lxy, [RecoMuonsFromGen_lxy, GlobalRecoMuonsFromGen_lxy, SARecoMuonsFromGen_lxy],0) 

makeEffPlot("mu", "dismutype", ["All", "Global", "STA only"], "pt", 80, 20, 100, 1, "[GeV]", GenMu_pt, [RecoDisMuonsFromGen_pt, GlobalRecoDisMuonsFromGen_pt, SARecoDisMuonsFromGen_pt],0) 
makeEffPlot("mu", "dismutype", ["All", "Global", "STA only"], "eta", 24, -2.4, 2.4, 0.2, " ", GenMu_eta, [RecoDisMuonsFromGen_eta, GlobalRecoDisMuonsFromGen_eta, SARecoDisMuonsFromGen_eta],0) 
makeEffPlot("mu", "dismutype", ["All", "Global", "STA only"], "dxy", 30, 0, 15, 0.5, "[cm]", GenMu_dxy, [RecoDisMuonsFromGen_dxy, GlobalRecoDisMuonsFromGen_dxy, SARecoDisMuonsFromGen_dxy],0) 
makeEffPlot("mu", "dismutype", ["All", "Global", "STA only"], "lxy", 30, 0, 15, 0.5, "[cm]", GenMu_lxy, [RecoDisMuonsFromGen_lxy, GlobalRecoDisMuonsFromGen_lxy, SARecoDisMuonsFromGen_lxy],0) 

#makeEffPlot("e", ["Electrons"], "pt", 80, 20, 100, 1, "[GeV]", GenE_pt,  [RecoElectronsFromGen_pt] , 0) 
#makeEffPlot("e", ["Electrons"], "eta", 24, -2.4, 2.4, 0.2, " ",GenE_eta, [RecoElectronsFromGen_eta], 0) 
#makeEffPlot("e", ["Electrons"], "dxy", 30, 0, 15, 0.5, "[cm]", GenE_dxy, [RecoElectronsFromGen_dxy], 0) 
#makeEffPlot("e", ["Electrons"], "lxy", 30, 0, 15, 0.5, "[cm]", GenE_lxy, [RecoElectronsFromGen_lxy], 0) 
#
#makeResPlot("mu", ["Prompt", "Dis"], "Pt", "pt", ptRANGE, 20, 100, -5, 5, 5, [RecoAndMuonsFromGen_pt, RecoAndMuonsFromGen_pt], [RecoAndPromptMuon_ptDiff,  RecoAndDisMuon_ptDiff], "[GeV]", "[GeV]")
#makeResPlot("mu", ["Prompt", "Dis"], "Eta", "pt", etaRANGE, -2.4, 2.4, -5, 5, 0.2, [RecoAndMuonsFromGen_eta, RecoAndMuonsFromGen_eta], [RecoAndPromptMuon_ptDiff,  RecoAndDisMuon_ptDiff], " ", "[GeV]")
#makeResPlot("mu", ["Prompt", "Dis"], "Dxy", "pt", dxyRANGE, 0, 15, -5, 5, 0.5, [RecoAndMuonsFromGen_dxy, RecoAndMuonsFromGen_dxy], [RecoAndPromptMuon_ptDiff,  RecoAndDisMuon_ptDiff], "[cm]", "[GeV]")
#makeResPlot("mu", ["Prompt", "Dis"], "Lxy", "pt", lxyRANGE, 0, 15, -5, 5, 0.5, [RecoAndMuonsFromGen_lxy, RecoAndMuonsFromGen_lxy], [RecoAndPromptMuon_ptDiff,  RecoAndDisMuon_ptDiff], "[cm]", "[GeV]")
#
#makeResPlot("mu", ["Prompt", "Dis"], "Pt", "dxy", ptRANGE, 20, 100, -0.5, 0.5, 5, [RecoAndMuonsFromGen_pt, RecoAndMuonsFromGen_pt], [RecoAndPromptMuon_dxyDiff,  RecoAndDisMuon_dxyDiff], "[GeV]", "[cm]")
#makeResPlot("mu", ["Prompt", "Dis"], "Eta", "dxy", etaRANGE, -2.4, 2.4, -0.5, 0.5, 0.2, [RecoAndMuonsFromGen_eta, RecoAndMuonsFromGen_eta], [RecoAndPromptMuon_dxyDiff,  RecoAndDisMuon_dxyDiff], " ", "[cm]")
#makeResPlot("mu", ["Prompt", "Dis"], "Dxy", "dxy", dxyRANGE, 0, 15, -0.5, 0.5, 0.5, [RecoAndMuonsFromGen_dxy, RecoAndMuonsFromGen_dxy], [RecoAndPromptMuon_dxyDiff,  RecoAndDisMuon_dxyDiff], "[cm]", "[cm]")
#makeResPlot("mu", ["Prompt", "Dis"], "Lxy", "dxy", ptRANGE, 0, 15, -0.5, 0.5, 0.5, [RecoAndMuonsFromGen_lxy, RecoAndMuonsFromGen_lxy], [RecoAndPromptMuon_dxyDiff,  RecoAndDisMuon_dxyDiff], "[cm]", "[cm]")
