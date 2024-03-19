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

#electrons = electrons[(electrons.hoe < 0.15)]
RecoElectronsFromGen = electrons[(abs(electrons.matched_gen.distinctParent.distinctParent.pdgId) == 1000015)]
RecoElectronsFromGen = RecoElectronsFromGen[(RecoElectronsFromGen.pt > GenPtMin) & (abs(RecoElectronsFromGen.eta) < GenEtaMax)]

#photons = photons[photons.pixelSeed == True]
photons = photons[(photons.electronIdx != -1) & (photons.electronIdx < ak.num(electrons))]
phelectrons = electrons[photons.electronIdx.compute()]
phelectrons = phelectrons[(phelectrons.photonIdx != -1) & (phelectrons.photonIdx < ak.num(photons))]
nonphelectrons = phelectrons[(phelectrons.photonIdx == -1) | (phelectrons.photonIdx >= ak.num(photons))]
phelphotons = photons[phelectrons.photonIdx.compute()]
nonphelphotons = photons[(electrons[photons.electronIdx.compute()].photonIdx == -1) | (electrons[photons.electronIdx.compute()].photonIdx>= ak.num(photons))]

photons_pt = ak.flatten(photons.pt.compute(), axis = None)
phelphotons_pt = ak.flatten(phelphotons.pt.compute(), axis = None)
nonphelphotons_pt = ak.flatten(nonphelphotons.pt.compute(), axis = None)

photons_eta = ak.flatten(photons.eta.compute(), axis = None)
phelphotons_eta = ak.flatten(phelphotons.eta.compute(), axis = None)
nonphelphotons_eta = ak.flatten(nonphelphotons.eta.compute(), axis = None)

photons_hoe = ak.flatten(photons.hoe.compute(), axis = None)
phelphotons_hoe = ak.flatten(phelphotons.hoe.compute(), axis = None)
nonphelphotons_hoe = ak.flatten(nonphelphotons.hoe.compute(), axis = None)

for pt in phelphotons_pt:
  if pt not in photons_pt:
    nonphelphotons_pt.append(pt)

for eta in phelphotons_eta:
  if eta not in photons_eta:
    nonphelphotons_eta.append(eta)

for hoe in phelphotons_hoe:
  if hoe not in photons_hoe:
    nonphelphotons_hoe.append(hoe)



#print("Photon selection", photons[(electrons[photons.electronIdx.compute()].photonIdx == -1) | (electrons[photons.electronIdx.compute()].photonIdx > ak.num(photons))].compute())
RecoPhotonsFromGen   = gpart[photons.genPartIdx.compute()]
RecoPhotonsFromGen   = RecoPhotonsFromGen[abs(RecoPhotonsFromGen.pdgId) == 11]
RecoPhotonsFromGen   = RecoPhotonsFromGen[abs(RecoPhotonsFromGen.distinctParent.distinctParent.pdgId) == 1000015]
RecoPhotonsFromGen   = RecoPhotonsFromGen[(RecoPhotonsFromGen.pt > GenPtMin) & (abs(RecoPhotonsFromGen.eta) < GenEtaMax)]

RecoLowPtElecFromGen = gpart[lowptelectrons.genPartIdx.compute()]
RecoLowPtElecFromGen = RecoLowPtElecFromGen[abs(RecoLowPtElecFromGen.distinctParent.distinctParent.pdgId) == 1000015]
RecoLowPtElecFromGen = RecoLowPtElecFromGen[(RecoLowPtElecFromGen.pt > GenPtMin) & (abs(RecoLowPtElecFromGen.eta) < GenEtaMax)]

print("Number of gen electrons:", len(ak.flatten(gen_electrons.pt.compute(), axis = None)))
print("Number of reco electrons from Electron collection:", len(ak.flatten(RecoElectronsFromGen.pt.compute(), axis = None)))
print("Number of reco electrons from LowPtElectron collection:", len(ak.flatten(RecoLowPtElecFromGen.pt.compute(), axis = None)))
print("Number of reco electrons from Photon collection:", len(ak.flatten(RecoPhotonsFromGen.pt.compute(), axis = None)))

#makeEffPlot("e", "gammacomp_enotgam", ["Electrons", "LowPtElectrons", "Photons"], "pt", 16, 20, 100, 5, "[GeV]", [gen_electrons.pt.compute(),] * 3, [RecoElectronsFromGen.pt.compute(), RecoLowPtElecFromGen.pt.compute(), RecoPhotonsFromGen.pt.compute()], 0, file)
#makeEffPlot("e", "gammacomp_matchedgam", ["Electrons", "LowPtElectrons", "Photons"], "hoe", 0, 1, 20, 0.05, "", [gen_electrons.hoe.compute(),] * 3, [RecoElectronsFromGen.eta.compute(), RecoLowPtElecFromGen.pt.compute(), RecoPhotonsFromGen.pt.compute()], 0, file)
makeEffPlot("photons", "e_not_photon", ["#gamma associated to e associated to same #gamma",  "#gamma that weren't associated to e"] , "pt", 20, 0, 100, 5, "[GeV]", [photons_pt,] * 2, [phelphotons_pt, nonphelphotons_pt], 0, file)
makeEffPlot("photons", "e_not_photon", ["#gamma associated to e associated to same #gamma",  "#gamma that weren't associated to e"] , "#eta", 24, -2.4, 2.4, 0.2, "", [photons_eta,] * 2, [phelphotons_eta, nonphelphotons_eta], 0, file) 
makeEffPlot("photons", "e_not_photon", ["#gamma associated to e associated to same #gamma",  "#gamma that weren't associated to e"] , "hoe", 20, 0, 1, 0.05, "", [photons_hoe,] * 2, [phelphotons_hoe, nonphelphotons_hoe], 0, file) 
