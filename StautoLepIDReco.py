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
from leptonPlot import *

GenPtMin = 20
GenEtaMax = 2.4



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

#file = "Staus_M_100_100mm_13p6TeV_Run3Summer22_DisMuon_GenPartMatch.root" 
file = "Staus_M_100_100mm_13p6TeV_Run3Summer22_lpcdisptau_NanoAOD_ExtraDisMuonBranches.root"

rootFile = uproot.open(file)
Events = rootFile["Events"]

lepBranches = {}
genBranches = {}

for key in Events.keys():
  if ("Muon_" in key) or ("Electron_" in key) or ("Tau_" in key):
    lepBranches[key] = Events[key].array()
  if "GenPart_" in key:
    genBranches[key] = Events[key].array()

genBranches["GenMuon_dxy"] = ak.from_parquet("Stau_GenMuon_dxy_"+file.split('.')[0]+".parquet")
genBranches["GenMuon_lxy"] = ak.from_parquet("Stau_GenMuon_lxy_"+file.split('.')[0]+".parquet")

muFromZ    = ak.from_parquet("genDisMuon_"+file.split('.')[0]+".parquet")
allMuFromZ = ak.from_parquet("allGenDisMuon_"+file.split('.')[0]+".parquet")

eFromZ = ak.from_parquet("genDisElectron_"+file.split('.')[0]+".parquet")
allEFromZ =  ak.from_parquet("allGenDisElectron_"+file.split('.')[0]+".parquet")

genBranches["GenElectron_dxy"] = ak.from_parquet("Stau_GenElectron_dxy_"+file.split('.')[0]+".parquet")
genBranches["GenElectron_lxy"] = ak.from_parquet("Stau_GenElectron_lxy_"+file.split('.')[0]+".parquet")

GenMu_pt  = genBranches["GenPart_pt"][muFromZ] 
GenMu_eta = genBranches["GenPart_eta"][muFromZ]
GenMu_dxy = np.abs(genBranches["GenMuon_dxy"][muFromZ])
GenMu_lxy = genBranches["GenMuon_lxy"][muFromZ]

GenMuEtaCut = (abs(GenMu_eta) < GenEtaMax)
GenMuPtCut  = (GenMu_pt > GenPtMin)

GenMu_pt  = GenMu_pt[GenMuEtaCut & GenMuPtCut]
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

RecoElectronsFromGen = [np.array([val for val in subarr if val in values]) for subarr, values in zip(lepBranches["Electron_genPartIdx"], allEFromZ)]
RecoElectronsFromGenIndices = [np.where(np.isin(subarr, values))[0] for subarr, values in zip(lepBranches["Electron_genPartIdx"], allEFromZ)]

RecoMuons_pt             = lepBranches["Muon_pt"][RecoMuonsFromGenIndices]
RecoMuons_eta            = lepBranches["Muon_eta"][RecoMuonsFromGenIndices]
RecoMuons_dxy            = np.abs(lepBranches["Muon_dxy"][RecoMuonsFromGenIndices])

RecoMuons_looseId = lepBranches["Muon_looseId"][RecoMuonsFromGenIndices]
RecoMuons_mediumId = lepBranches["Muon_mediumId"][RecoMuonsFromGenIndices] 

RecoMuons_isGlobal       = lepBranches["Muon_isGlobal"][RecoMuonsFromGenIndices]
RecoMuons_isPFCand       = lepBranches["Muon_isPFcand"][RecoMuonsFromGenIndices]
RecoMuons_nStations      = lepBranches["Muon_nStations"][RecoMuonsFromGenIndices]
RecoMuons_nTrackerLayers = lepBranches["Muon_nTrackerLayers"][RecoMuonsFromGenIndices]
RecoMuons_trkChi2        = lepBranches["Muon_trkChi2"][RecoMuonsFromGenIndices]
RecoMuons_muonHits       = lepBranches["Muon_muonHits"][RecoMuonsFromGenIndices]
RecoMuons_pixelHits      = lepBranches["Muon_pixelHits"][RecoMuonsFromGenIndices]

RecoMuonsFromGen_pt  = genBranches["GenPart_pt"][RecoMuonsFromGen]
RecoMuonsFromGen_eta = genBranches["GenPart_eta"][RecoMuonsFromGen]
RecoMuonsFromGen_dxy = np.abs(genBranches["GenMuon_dxy"][RecoMuonsFromGen])
RecoMuonsFromGen_lxy = genBranches["GenMuon_lxy"][RecoMuonsFromGen]

RecoElectrons_pt =  lepBranches["Electron_pt"][RecoElectronsFromGenIndices]   
RecoElectrons_eta = lepBranches["Electron_eta"][RecoElectronsFromGenIndices]
RecoElectrons_dxy = np.abs(lepBranches["Electron_dxy"][RecoElectronsFromGenIndices])

RecoElectrons_cutBased  = lepBranches["Electron_cutBased"][RecoElectronsFromGenIndices]  
RecoElectrons_convVeto  = lepBranches["Electron_convVeto"][RecoElectronsFromGenIndices]
RecoElectrons_lostHits  =  lepBranches["Electron_lostHits"][RecoElectronsFromGenIndices]

RecoElectronsFromGen_pt  = genBranches["GenPart_pt"][RecoElectronsFromGen]
RecoElectronsFromGen_eta = genBranches["GenPart_eta"][RecoElectronsFromGen]
RecoElectronsFromGen_dxy = np.abs(genBranches["GenElectron_dxy"][RecoElectronsFromGen])
RecoElectronsFromGen_lxy = genBranches["GenElectron_lxy"][RecoElectronsFromGen]


### Kineamtic Cuts
RecoMuonsFromGenEtaCut = (abs(RecoMuonsFromGen_eta) < GenEtaMax)
RecoMuonsFromGenPtCut  = (RecoMuonsFromGen_pt > GenPtMin)

RecoElectronsFromGenEtaCut = (abs(RecoElectronsFromGen_eta) < GenEtaMax)
RecoElectronsFromGenPtCut  = (RecoElectronsFromGen_pt > GenPtMin)

### Start Tight ID w/o dxy/dz requirements
Global        = (RecoMuons_isGlobal == 1)
PFCand        = (RecoMuons_isPFCand == 1)
Stations      = (RecoMuons_nStations > 1)
TrackerLayers = (RecoMuons_nTrackerLayers > 5)
Chi2          = (RecoMuons_trkChi2 < 10)
MuonHits      = (RecoMuons_muonHits > 0)
PixelHits     = (RecoMuons_pixelHits > 0)

### Loose and Med ID, we can use the boolean since there are no dxy/dz requirements 
Loose         = (RecoMuons_looseId == 1)
Medium        = (RecoMuons_mediumId == 1)

EleVeto       = (RecoElectrons_cutBased >= 1)
EleLoose      = (RecoElectrons_cutBased >= 2)
EleMedium     = (RecoElectrons_cutBased >= 3)
EleTight      = (RecoElectrons_cutBased >= 4)

### IDs by hand for electrons
ConvVeto      = (RecoElectrons_convVeto == 1)
LostHits2     = (RecoElectrons_lostHits <= 2)
LostHits3     = (RecoElectrons_lostHits <= 3)

### Applying cuts to our muons 
RecoMuonsFromGen_pt  = RecoMuonsFromGen_pt[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut] 
RecoMuonsFromGen_eta = RecoMuonsFromGen_eta[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut]
RecoMuonsFromGen_dxy = RecoMuonsFromGen_dxy[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut]
RecoMuonsFromGen_lxy = RecoMuonsFromGen_lxy[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut]

RecoMuons_pt   = RecoMuons_pt[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut] 
RecoMuons_eta  = RecoMuons_eta[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut]
RecoMuons_dxy  = RecoMuons_dxy[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut]

TightRecoMuonsFromGen_pt  = RecoMuonsFromGen_pt[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut & Global & PFCand & Stations & TrackerLayers & Chi2 & MuonHits & PixelHits]
TightRecoMuonsFromGen_eta = RecoMuonsFromGen_eta[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut & Global & PFCand & Stations & TrackerLayers & Chi2 & MuonHits & PixelHits]
TightRecoMuonsFromGen_dxy = RecoMuonsFromGen_dxy[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut & Global & PFCand & Stations & TrackerLayers & Chi2 & MuonHits & PixelHits]
TightRecoMuonsFromGen_lxy = RecoMuonsFromGen_lxy[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut & Global & PFCand & Stations & TrackerLayers & Chi2 & MuonHits & PixelHits]

TightRecoMuons_pt   = RecoMuons_pt[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut & Global & PFCand & Stations & TrackerLayers & Chi2 & MuonHits & PixelHits]
TightRecoMuons_eta  = RecoMuons_eta[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut & Global & PFCand & Stations & TrackerLayers & Chi2 & MuonHits & PixelHits]
TightRecoMuons_dxy  = RecoMuons_dxy[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut & Global & PFCand & Stations & TrackerLayers & Chi2 & MuonHits & PixelHits]

MediumRecoMuonsFromGen_pt  = RecoMuonsFromGen_pt[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut & Medium]
MediumRecoMuonsFromGen_eta = RecoMuonsFromGen_eta[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut & Medium]
MediumRecoMuonsFromGen_dxy = RecoMuonsFromGen_dxy[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut & Medium]
MediumRecoMuonsFromGen_lxy = RecoMuonsFromGen_lxy[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut & Medium]

MediumRecoMuons_pt   = RecoMuons_pt[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut & Medium]
MediumRecoMuons_eta  = RecoMuons_eta[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut & Medium]
MediumRecoMuons_dxy  = RecoMuons_dxy[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut & Medium]

LooseRecoMuonsFromGen_pt  = RecoMuonsFromGen_pt[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut & Loose]
LooseRecoMuonsFromGen_eta = RecoMuonsFromGen_eta[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut & Loose]
LooseRecoMuonsFromGen_dxy = RecoMuonsFromGen_dxy[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut & Loose]
LooseRecoMuonsFromGen_lxy = RecoMuonsFromGen_lxy[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut & Loose]

LooseRecoMuons_pt   = RecoMuons_pt[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut & Loose]
LooseRecoMuons_eta  = RecoMuons_eta[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut & Loose]
LooseRecoMuons_dxy  = RecoMuons_dxy[RecoMuonsFromGenEtaCut & RecoMuonsFromGenPtCut & Loose]

RecoElectronsFromGen_pt  = RecoElectronsFromGen_pt[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut]
RecoElectronsFromGen_eta = RecoElectronsFromGen_eta[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut]
RecoElectronsFromGen_dxy = RecoElectronsFromGen_dxy[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut]
RecoElectronsFromGen_lxy = RecoElectronsFromGen_lxy[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut]

RecoElectrons_pt  = RecoElectrons_pt[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut]
RecoElectrons_eta = RecoElectrons_eta[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut]
RecoElectrons_dxy = RecoElectrons_dxy[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut]

TightRecoElectronsFromGen_pt = RecoElectronsFromGen_pt[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut & EleTight]
TightRecoElectronsFromGen_eta = RecoElectronsFromGen_eta[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut & EleTight]
TightRecoElectronsFromGen_dxy = RecoElectronsFromGen_dxy[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut & EleTight]
TightRecoElectronsFromGen_lxy = RecoElectronsFromGen_lxy[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut & EleTight]
  
TightRecoElectrons_pt  = RecoElectrons_pt[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut & EleTight]
TightRecoElectrons_eta = RecoElectrons_eta[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut & EleTight]
TightRecoElectrons_dxy = RecoElectrons_dxy[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut & EleTight]

MediumRecoElectronsFromGen_pt = RecoElectronsFromGen_pt[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut & EleMedium]
MediumRecoElectronsFromGen_eta = RecoElectronsFromGen_eta[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut & EleMedium]
MediumRecoElectronsFromGen_dxy = RecoElectronsFromGen_dxy[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut & EleMedium]
MediumRecoElectronsFromGen_lxy = RecoElectronsFromGen_lxy[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut & EleMedium]

MediumRecoElectrons_pt  = RecoElectrons_pt[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut & EleMedium]
MediumRecoElectrons_eta = RecoElectrons_eta[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut & EleMedium]
MediumRecoElectrons_dxy = RecoElectrons_dxy[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut & EleMedium]

LooseRecoElectronsFromGen_pt =  RecoElectronsFromGen_pt[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut &  EleLoose]
LooseRecoElectronsFromGen_eta = RecoElectronsFromGen_eta[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut & EleLoose]
LooseRecoElectronsFromGen_dxy = RecoElectronsFromGen_dxy[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut & EleLoose]
LooseRecoElectronsFromGen_lxy = RecoElectronsFromGen_lxy[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut & EleLoose]

LooseRecoElectrons_pt  = RecoElectrons_pt[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut &  EleLoose]
LooseRecoElectrons_eta = RecoElectrons_eta[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut & EleLoose]
LooseRecoElectrons_dxy = RecoElectrons_dxy[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut & EleLoose]

VetoRecoElectronsFromGen_pt =  RecoElectronsFromGen_pt[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut &  EleVeto]
VetoRecoElectronsFromGen_eta = RecoElectronsFromGen_eta[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut & EleVeto]
VetoRecoElectronsFromGen_dxy = RecoElectronsFromGen_dxy[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut & EleVeto]
VetoRecoElectronsFromGen_lxy = RecoElectronsFromGen_lxy[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut & EleVeto]

VetoRecoElectrons_pt  = RecoElectrons_pt[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut &  EleVeto]
VetoRecoElectrons_eta = RecoElectrons_eta[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut & EleVeto]
VetoRecoElectrons_dxy = RecoElectrons_dxy[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut & EleVeto]

Hand2RecoElectronsFromGen_pt =  RecoElectronsFromGen_pt[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut &  LostHits2 & ConvVeto]
Hand2RecoElectronsFromGen_eta = RecoElectronsFromGen_eta[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut & LostHits2 & ConvVeto]
Hand2RecoElectronsFromGen_dxy = RecoElectronsFromGen_dxy[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut & LostHits2 & ConvVeto]
Hand2RecoElectronsFromGen_lxy = RecoElectronsFromGen_lxy[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut & LostHits2 & ConvVeto]

Hand2RecoElectrons_pt  = RecoElectrons_pt[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut &  LostHits2 & ConvVeto]
Hand2RecoElectrons_eta = RecoElectrons_eta[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut & LostHits2 & ConvVeto]
Hand2RecoElectrons_dxy = RecoElectrons_dxy[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut & LostHits2 & ConvVeto]

Hand3RecoElectronsFromGen_pt =  RecoElectronsFromGen_pt[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut &  LostHits3 & ConvVeto]
Hand3RecoElectronsFromGen_eta = RecoElectronsFromGen_eta[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut & LostHits3 & ConvVeto]
Hand3RecoElectronsFromGen_dxy = RecoElectronsFromGen_dxy[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut & LostHits3 & ConvVeto]
Hand3RecoElectronsFromGen_lxy = RecoElectronsFromGen_lxy[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut & LostHits3 & ConvVeto]

Hand3RecoElectrons_pt  = RecoElectrons_pt[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut &  LostHits3 & ConvVeto]
Hand3RecoElectrons_eta = RecoElectrons_eta[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut & LostHits3 & ConvVeto]
Hand3RecoElectrons_dxy = RecoElectrons_dxy[RecoElectronsFromGenEtaCut & RecoElectronsFromGenPtCut & LostHits3 & ConvVeto]

### Efficiency Plots
#makeEffPlot("mu", "ID", ["No ID", "Tight", "Med", "Loose"], "pt", 16, 20, 100, 5, "[GeV]", GenMu_pt, [RecoMuonsFromGen_pt, TightRecoMuonsFromGen_pt, MediumRecoMuonsFromGen_pt, LooseRecoMuonsFromGen_pt], 0, file)
#makeEffPlot("mu", "ID", ["No ID", "Tight", "Med", "Loose"], "eta", 24, -2.4, 2.4, 0.2, " ", GenMu_eta, [RecoMuonsFromGen_eta, TightRecoMuonsFromGen_eta, MediumRecoMuonsFromGen_eta, LooseRecoMuonsFromGen_eta], 0, file)
makeEffPlot("mu", "ID", ["No ID", "Tight", "Med", "Loose"], "dxy", 30, 0, 15, 0.5, "[cm]", GenMu_dxy, [RecoMuonsFromGen_dxy, TightRecoMuonsFromGen_dxy, MediumRecoMuonsFromGen_dxy, LooseRecoMuonsFromGen_dxy], 0, file)
#makeEffPlot("mu", "ID", ["No ID", "Tight", "Med", "Loose"], "lxy", 30, 0, 15, 0.5, "[cm]", GenMu_lxy, [RecoMuonsFromGen_lxy, TightRecoMuonsFromGen_lxy, MediumRecoMuonsFromGen_lxy, LooseRecoMuonsFromGen_lxy], 0, file) 
#
#makeEffPlot("e", "ID", ["No ID", "Tight", "Med", "Loose"], "pt", 16, 20, 100, 5, "[GeV]", GenE_pt, [RecoElectronsFromGen_pt, TightRecoElectronsFromGen_pt, MediumRecoElectronsFromGen_pt, LooseRecoElectronsFromGen_pt], 0, file)
#makeEffPlot("e", "ID", ["No ID", "Tight", "Med", "Loose"], "eta", 24, -2.4, 2.4, 0.2, " ", GenE_eta, [RecoElectronsFromGen_eta, TightRecoElectronsFromGen_eta, MediumRecoElectronsFromGen_eta, LooseRecoElectronsFromGen_eta], 0, file)
#makeEffPlot("e", "ID", ["No ID", "Tight", "Med", "Loose", "Veto"], "dxy", 30, 0, 15, 0.5, "[cm]", GenE_dxy, [RecoElectronsFromGen_dxy, TightRecoElectronsFromGen_dxy, MediumRecoElectronsFromGen_dxy, LooseRecoElectronsFromGen_dxy, VetoRecoElectronsFromGen_dxy], 0, file)
#makeEffPlot("e", "ID", ["No ID", "Tight", "Med", "Loose"], "lxy", 30, 0, 15, 0.5, "[cm]", GenE_lxy, [RecoElectronsFromGen_lxy, TightRecoElectronsFromGen_lxy, MediumRecoElectronsFromGen_lxy, LooseRecoElectronsFromGen_lxy], 0, file) 

#makeEffPlotEta("e", ["no ID", "convVeto & lostHits","Veto", "Loose ID", "Medium ID", "Tight ID"], "pt", "[GeV]", GenE_pt, GenE_eta,[RecoElectronsFromGen_pt, Hand2RecoElectronsFromGen_pt, VetoRecoElectronsFromGen_pt, LooseRecoElectronsFromGen_pt, MediumRecoElectronsFromGen_pt, TightRecoElectronsFromGen_pt], [RecoElectronsFromGen_eta, Hand2RecoElectronsFromGen_eta, VetoRecoElectronsFromGen_eta, LooseRecoElectronsFromGen_eta, MediumRecoElectronsFromGen_eta, TightRecoElectronsFromGen_eta], [RecoElectronsFromGen_pt, Hand3RecoElectronsFromGen_pt, VetoRecoElectronsFromGen_pt, LooseRecoElectronsFromGen_pt, MediumRecoElectronsFromGen_pt, TightRecoElectronsFromGen_pt], [RecoElectronsFromGen_eta, Hand3RecoElectronsFromGen_eta, VetoRecoElectronsFromGen_eta, LooseRecoElectronsFromGen_eta, MediumRecoElectronsFromGen_eta, TightRecoElectronsFromGen_eta], 0, file)
#makeEffPlotEta("e", ["no ID", "convVeto & lostHits","Veto", "Loose ID", "Medium ID", "Tight ID"], "dxy", "[cm]", GenE_dxy, GenE_eta,[RecoElectronsFromGen_dxy, Hand2RecoElectronsFromGen_dxy, VetoRecoElectronsFromGen_dxy, LooseRecoElectronsFromGen_dxy, MediumRecoElectronsFromGen_dxy, TightRecoElectronsFromGen_dxy], [RecoElectronsFromGen_eta, Hand2RecoElectronsFromGen_eta, VetoRecoElectronsFromGen_eta, LooseRecoElectronsFromGen_eta, MediumRecoElectronsFromGen_eta, TightRecoElectronsFromGen_eta], [RecoElectronsFromGen_dxy, Hand3RecoElectronsFromGen_dxy, VetoRecoElectronsFromGen_dxy, LooseRecoElectronsFromGen_dxy, MediumRecoElectronsFromGen_dxy, TightRecoElectronsFromGen_dxy], [RecoElectronsFromGen_eta, Hand3RecoElectronsFromGen_eta, VetoRecoElectronsFromGen_eta, LooseRecoElectronsFromGen_eta, MediumRecoElectronsFromGen_eta, TightRecoElectronsFromGen_eta], 0, file, np.linspace(0,15,16))
