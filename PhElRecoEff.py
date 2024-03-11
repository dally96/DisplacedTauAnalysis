import numpy as np
import awkward as ak
import ROOT 
import math
import scipy
import array
import pandas as pd
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from leptonPlot import *

file = "Staus_M_100_100mm_13p6TeV_Run3Summer22_lpcdisptau_NanoAOD_ExtraDisMuonBranches.root"

events = NanoEventsFactory.from_root({file:"Events"}).events()


gpart = events.GenPart
gvistau = events.GenVisTau
electrons = events.Electron
photons = events.Photon
lowptelectrons = events.LowPtElectron

staus = gpart[(abs(gpart.pdgId) == 1000015) & (gpart.hasFlags("isLastCopy"))]
staus_taus = staus.distinctChildren[(abs(staus.distinctChildren.pdgId) == 15) & (staus.distinctChildren.hasFlags("isLastCopy"))]
gen_electrons = staus_taus.distinctChildren[(abs(staus_taus.distinctChildren.pdgId) == 11)]
gen_electrons = gen_electrons[(gen_electrons.pt > GenPtMin) & (abs(gen_electrons.eta) < GenEtaMax)]

RecoElectronsFromGen = electrons[(abs(electrons.matched_gen.distinctParent.distinctParent.pdgId) == 1000015)]
RecoElectronsFromGen = RecoElectronsFromGen[(RecoElectronsFromGen.pt > GenPtMin) & (abs(RecoElectronsFromGen.eta) < GenEtaMax)]

photons = photons[photons.pixelSeed == True]
RecoPhotons = photons[(abs(gpart[photons.genPartIdx.compute()].pdgId) == 11)
                  & (abs(gpart[photons.genPartIdx.compute()].distinctParent.distinctParent.pdgId) == 1000015)
                  & (gpart[photons.genPartIdx.compute()].pt > GenPtMin)
                  & (abs(gpart[photons.genPartIdx.compute()].eta) < GenEtaMax)]
RecoPhotonsFromGen   = gpart[photons.genPartIdx.compute()]
RecoPhotonsFromGen   = RecoPhotonsFromGen[abs(RecoPhotonsFromGen.pdgId) == 11]
RecoPhotonsFromGen   = RecoPhotonsFromGen[abs(RecoPhotonsFromGen.distinctParent.distinctParent.pdgId) == 1000015]
RecoPhotonsFromGen   = RecoPhotonsFromGen[(RecoPhotonsFromGen.pt > GenPtMin) & (abs(RecoPhotonsFromGen.eta) < GenEtaMax)]
dpt_pt = abs(ak.flatten(RecoPhotonsFromGen.pt, axis = None) - ak.flatten(RecoPhotons.pt, axis = None))/ak.flatten(RecoPhotonsFromGen.pt, axis = None)
dpt_pt_mask = dpt_pt < 0.3
RecoPhotonsFromGen = RecoPhotonsFromGen[dpt_pt_mask]

RecoLowPtElecFromGen = gpart[lowptelectrons.genPartIdx.compute()]
RecoLowPtElecFromGen = RecoLowPtElecFromGen[abs(RecoLowPtElecFromGen.distinctParent.distinctParent.pdgId) == 1000015]
RecoLowPtElecFromGen = RecoLowPtElecFromGen[(RecoLowPtElecFromGen.pt > GenPtMin) & (abs(RecoLowPtElecFromGen.eta) < GenEtaMax)]

#print("Number of gen electrons:", len(ak.flatten(gen_electrons.pt.compute(), axis = None)))
#print("Number of reco electrons from Electron collection:", len(ak.flatten(RecoElectronsFromGen.pt.compute(), axis = None)))
#print("Number of reco electrons from LowPtElectron collection:", len(ak.flatten(RecoLowPtElecFromGen.pt.compute(), axis = None)))
#print("Number of reco electrons from Photon collection:", len(ak.flatten(RecoPhotonsFromGen.pt.compute(), axis = None)))

makeEffPlot("e", "gammacomp_pixelseed_energyassoc", ["Electrons", "LowPtElectrons", "Photons"], "pt", 16, 20, 100, 5, "[GeV]", [gen_electrons.pt.compute(),] * 3, [RecoElectronsFromGen.pt.compute(), RecoLowPtElecFromGen.pt.compute(), RecoPhotonsFromGen.pt.compute()], 0, file)
