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
from leptonPlot import makeEffPlot

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
file = "Staus_M_100_100mm_13p6TeV_Run3Summer22_lpcdisptau_NanoAOD_ExtraMuonBranches.root"

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

muFromZ    = ak.from_parquet("genDisMuon.parquet")
allMuFromZ = ak.from_parquet("allGenDisMuon.parquet")

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

RecoMuonsFromGen = [np.array([val for val in subarr if val in values]) for subarr, values in zip(lepBranches["Muon_genPartIdx"], allMuFromZ)]
RecoMuonsFromGenIndices = [np.where(np.isin(subarr, values))[0] for subarr, values in zip(lepBranches["Muon_genPartIdx"], allMuFromZ)]

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

### Kineamtic Cuts
RecoMuonsFromGenEtaCut = (abs(RecoMuonsFromGen_eta) < GenEtaMax)
RecoMuonsFromGenPtCut  = (RecoMuonsFromGen_pt > GenPtMin)

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

makeEffPlot("mu", "ID", ["No ID", "Tight", "Med", "Loose"], "pt", 16, 20, 100, 5, "[GeV]", GenMu_pt, [RecoMuonsFromGen_pt, TightRecoMuonsFromGen_pt, MediumRecoMuonsFromGen_pt, LooseRecoMuonsFromGen_pt], 0, file)
makeEffPlot("mu", "ID", ["No ID", "Tight", "Med", "Loose"], "eta", 24, -2.4, 2.4, 0.2, " ", GenMu_eta, [RecoMuonsFromGen_eta, TightRecoMuonsFromGen_eta, MediumRecoMuonsFromGen_eta, LooseRecoMuonsFromGen_eta], 0, file)
makeEffPlot("mu", "ID", ["No ID", "Tight", "Med", "Loose"], "dxy", 30, 0, 15, 0.5, "[cm]", GenMu_dxy, [RecoMuonsFromGen_dxy, TightRecoMuonsFromGen_dxy, MediumRecoMuonsFromGen_dxy, LooseRecoMuonsFromGen_dxy], 0, file)
makeEffPlot("mu", "ID", ["No ID", "Tight", "Med", "Loose"], "lxy", 30, 0, 15, 0.5, "[cm]", GenMu_lxy, [RecoMuonsFromGen_lxy, TightRecoMuonsFromGen_lxy, MediumRecoMuonsFromGen_lxy, LooseRecoMuonsFromGen_lxy], 0, file) 


