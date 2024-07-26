import ROOT
import sys
import os 
import numpy as np
import awkward as ak
from array import array
from collections import OrderedDict
from DataFormats.FWLite import Events, Handle
import FWCore.ParameterSet.Config as cms
import math
from copy import deepcopy as dc
from Staus_M_100_100mm_13p6TeV_Run3Summer22_array import *
from leptonPlot import *

events = Events(MiniFiles)

maxevents = -1 
totevents = events.size()
handles = {}
handle_disMuon = Handle('std::vector<pat::Muon>')
handle_genpart = Handle('std::vector<reco::GenParticle>')
handle_bs      = Handle('reco::BeamSpot')
handle_photon  = Handle('std::vector<pat::Photon>')


handles["photons"]  = [("slimmedPhotons", '', 'PAT'), handle_photon, False]
handles["dismuon"]  = [("slimmedDisplacedMuons", '', 'PAT'), handle_disMuon, False]
handles["genpart"]  = [("prunedGenParticles", '', 'PAT'), handle_genpart, False]
handles["beamspot"] = [("offlineBeamSpot", '', 'RECO'), handle_bs, False]

DisMuon_genPartIdx = []
GenMu_pt = []
GenMu_eta = []
GenMu_dxy = []
GenMu_dxyTrack = []

IDs = ["No ID", "Tight", "Medium", "Loose"]

ID_pt = {}
ID_eta = {}
ID_dxy = {}

for ids in IDs:
  ID_pt[ids] = []
  ID_eta[ids] = []
  ID_dxy[ids] = []

def dxy(part, ev):
  dxy = ((part.vertex().Y() - ev.beamspot.position().y()) * np.cos(part.phi()) - (part.vertex().X() - ev.beamspot.position().x()) * np.sin(part.phi()))
  return dxy 

def findDecayedMother(part, abspdgId):
  if part.numberOfMothers() == 0:
    return False
  mother = part.motherRef(0)
  if abs(mother.pdgId()) == abspdgId:
    return True
  if abs(mother.pdgId()) != abspdgId:
    return findDecayedMother(mother, abspdgId)
   

def isMotherMu(part):
  if abs(part.motherRef(0).pdgId()) == 13:
    return 1
  else:
    return 0

def isLoose(mu):
  if mu.isGlobalMuon():
    return True
  else: 
    return False

def isMedium(mu):
  if  mu.globalTrack().isNull():
    return False
  else:
    goodGlob = mu.isGlobalMuon() & (mu.globalTrack().normalizedChi2() < 3) & (mu.combinedQuality().chi2LocalPosition < 12) & (mu.combinedQuality().trkKink < 20)
    isMedium = isLoose(mu) & (mu.innerTrack().validFraction() > 0.8) & (mu.segmentCompatibility() > (0.303 if goodGlob else 0.451)) 
    return isMedium

def isTight(mu):
  if not mu.isGlobalMuon or mu.globalTrack().isNull() or mu.innerTrack().isNull():
    return False
  muID = (mu.globalTrack().normalizedChi2() < 10) & (mu.numberOfMatchedStations() > 1)
  hits = (mu.globalTrack().hitPattern().numberOfValidMuonHits() > 0) & (mu.innerTrack().hitPattern().numberOfValidPixelHits() > 0) & (mu.innerTrack().hitPattern().trackerLayersWithMeasurement() > 5)
  return muID & hits
h_mu_chi2 = ROOT.TH1F("h_mu_chi2", ";chi 2; number of muons", 11, 0, 11)
for i, ev in enumerate(events):

  if maxevents>0 and i>maxevents:
    break
  if i%100==0:
    print(('===> processing %d / %d event' %(i, totevents)))

  for k, v in list(handles.items()):
    setattr(ev, k, None)
    v[2] = False
    try:
        ev.getByLabel(v[0], v[1])
        setattr(ev, k, v[1].product())
        v[2] = True
    except:    
        v[2] = False
    
  DisMuon_genPartIdx_evt = []  
  GenMu_pt_evt  = []
  GenMu_eta_evt = []
  GenMu_dxy_evt = []
  GenMu_dxyTrack_evt = []

  RecoMuonsFromGen_pt_evt  = []
  RecoMuonsFromGen_eta_evt = []
  RecoMuonsFromGen_dxy_evt = []
  cutID_evt = []

  ID_pt_evt = {}
  ID_eta_evt = {}
  ID_dxy_evt = {}
  
  for ids in IDs:
    ID_pt_evt[ids]  = [] 
    ID_eta_evt[ids] = []
    ID_dxy_evt[ids] = []

  for pp, part in enumerate(ev.genpart):
    if (abs(part.pdgId()) != 13): continue
    if (findDecayedMother(part, 1000015) == False): continue
    if (part.pt() < 20) or (abs(part.eta() > 2.4)): continue
    if not isMotherMu(part):
      GenMu_pt_evt.append(part.pt())
      GenMu_eta_evt.append(part.eta())
      GenMu_dxy_evt.append((part.vertex().Y() - ev.beamspot.y0()) * np.cos(part.phi()) - (part.vertex().X() - ev.beamspot.x0()) * np.sin(part.phi()))
      #GenMu_dxyTrack_evt.append(part.track().dxy())

  for m,mu in enumerate(ev.dismuon):
    #if not mu.globalTrack().isNull():
      #h_mu_chi2.Fill(mu.globalTrack().normalizedChi2())  
    

    muon_p4 = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<double>')(mu.pt(), mu.eta(), mu.phi(), mu.mass())
    dR_cut = False
    for pp, part in enumerate(ev.genpart):
      if dR_cut == True: continue 
      if (abs(part.pdgId()) != 13) : continue
      if (findDecayedMother(part, 1000015) == False): continue
      if (part.pt() < 20) or (abs(part.eta() > 2.4)): continue
      if pp in DisMuon_genPartIdx_evt: continue
      part_p4 = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<double>')(part.pt(), part.eta(), part.phi(), part.mass()) 
      dR = ROOT.Math.VectorUtil.DeltaR(part_p4, muon_p4)
      
      if dR < 0.3:
        dR_cut = True
        DisMuon_genPartIdx_evt.append(pp)
        RecoMuonsFromGen_pt_evt.append(part_p4.pt())  
        RecoMuonsFromGen_eta_evt.append(part_p4.eta())
        RecoMuonsFromGen_dxy_evt.append((part.vertex().Y() - ev.beamspot.y0()) * np.cos(part.phi()) - (part.vertex().X() - ev.beamspot.x0()) * np.sin(part.phi()))
        if isTight(mu) and isMedium(mu):
          cutID_evt.append(4)
        elif isTight(mu) and not isMedium(mu):
          cutID_evt.append(3)
        elif isMedium(mu) and not isTight(mu):
          cutID_evt.append(2)
        elif isLoose(mu) and not isMedium(mu) and not isTight(mu):
          cutID_evt.append(1)
        elif not isLoose(mu) and not isMedium(mu) and not isTight(mu):
          cutID_evt.append(0)
  
  for k, ids in enumerate(cutID_evt):
    #print("The id of the muon", k, " recon'd in evt", i, "is", ids)
    if ids >= 0:
      ID_pt_evt["No ID"].append(RecoMuonsFromGen_pt_evt[k])
      ID_eta_evt["No ID"].append(RecoMuonsFromGen_eta_evt[k])
      ID_dxy_evt["No ID"].append(RecoMuonsFromGen_dxy_evt[k])
    if ids >= 1:
      ID_pt_evt["Loose"].append(RecoMuonsFromGen_pt_evt[k])
      ID_eta_evt["Loose"].append(RecoMuonsFromGen_eta_evt[k])
      ID_dxy_evt["Loose"].append(RecoMuonsFromGen_dxy_evt[k])
    if ids == 2 or ids == 4:
      ID_pt_evt["Medium"].append(RecoMuonsFromGen_pt_evt[k])
      ID_eta_evt["Medium"].append(RecoMuonsFromGen_eta_evt[k])
      ID_dxy_evt["Medium"].append(RecoMuonsFromGen_dxy_evt[k])
    if ids >= 3:
      ID_pt_evt["Tight"].append(RecoMuonsFromGen_pt_evt[k])
      ID_eta_evt["Tight"].append(RecoMuonsFromGen_eta_evt[k])
      ID_dxy_evt["Tight"].append(RecoMuonsFromGen_dxy_evt[k])

  ID_pt["No ID"].append(ID_pt_evt["No ID"])
  ID_eta["No ID"].append(ID_eta_evt["No ID"])
  ID_dxy["No ID"].append(ID_dxy_evt["No ID"])
  ID_pt["Loose"].append(ID_pt_evt["Loose"])
  ID_eta["Loose"].append(ID_eta_evt["Loose"])
  ID_dxy["Loose"].append(ID_dxy_evt["Loose"])
  ID_pt["Medium"].append(ID_pt_evt["Medium"])
  ID_eta["Medium"].append(ID_eta_evt["Medium"])
  ID_dxy["Medium"].append(ID_dxy_evt["Medium"])
  ID_pt["Tight"].append(ID_pt_evt["Tight"])
  ID_eta["Tight"].append(ID_eta_evt["Tight"])
  ID_dxy["Tight"].append(ID_dxy_evt["Tight"])

  GenMu_pt.append(GenMu_pt_evt)
  GenMu_eta.append(GenMu_eta_evt)
  GenMu_dxy.append(GenMu_dxy_evt)
  GenMu_dxyTrack.append(GenMu_dxyTrack_evt)

#print(GenMu_pt)
for ids in IDs:
  ID_pt[ids] = ak.Array(ID_pt[ids])
  ID_eta[ids] = ak.Array(ID_eta[ids])
  ID_dxy[ids] = ak.Array(ID_dxy[ids])

GenMu_pt = ak.Array(GenMu_pt)
GenMu_eta = ak.Array(GenMu_eta)
GenMu_dxy = ak.Array(GenMu_dxy)
#GenMu_dxyTrack = ak.Array(GenMu_dxyTrack)

makeEffPlot("mu", "DisID", IDs, "pt", 16, 20, 100, 5, "[GeV]", [GenMu_pt,]*len(IDs), [ID_pt["No ID"], ID_pt["Tight"], ID_pt["Medium"], ID_pt["Loose"]], 0, MiniFiles[0].split("/")[-1]) 
makeEffPlot("mu", "DisID", IDs, "eta", 24, -2.4, 2.4, 0.2, "", [GenMu_eta,]*len(IDs), [ID_eta["No ID"], ID_eta["Tight"], ID_eta["Medium"], ID_eta["Loose"]], 0, MiniFiles[0].split("/")[-1]) 
makeEffPlot("mu", "DisID", IDs, "dxy", 30, 0, 15, 0.5, "[cm]", [GenMu_dxy,]*len(IDs), [ID_dxy["No ID"], ID_dxy["Tight"], ID_dxy["Medium"], ID_dxy["Loose"]], 0, MiniFiles[0].split("/")[-1]) 

