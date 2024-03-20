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
ROOT.gStyle.SetOptStat(0)

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
RecoElectronsFromGen = RecoElectronsFromGen.matched_gen[(RecoElectronsFromGen.matched_gen.pt > GenPtMin) & (abs(RecoElectronsFromGen.matched_gen.eta) < GenEtaMax)]

#photons = photons[photons.pixelSeed == True]
photons = photons[(photons.electronIdx != -1) & (photons.electronIdx < ak.num(electrons))]
phelectrons = electrons[photons.electronIdx.compute()]
#phelectrons = phelectrons[(phelectrons.photonIdx != -1) & (phelectrons.photonIdx < ak.num(photons))]
nonphelectrons = phelectrons[(phelectrons.photonIdx == -1) | (phelectrons.photonIdx >= ak.num(photons))]
phelphotons = photons[phelectrons.photonIdx.compute()]
nonphelphotons = photons[(electrons[photons.electronIdx.compute()].photonIdx == -1) | (electrons[photons.electronIdx.compute()].photonIdx>= ak.num(photons))]

#photons_pt = ak.flatten(photons.pt.compute(), axis = None)
#phelphotons_pt = ak.flatten(phelphotons.pt.compute(), axis = None)
#nonphelphotons_pt = ak.flatten(nonphelphotons.pt.compute(), axis = None)
#
#photons_eta = ak.flatten(photons.eta.compute(), axis = None)
#phelphotons_eta = ak.flatten(phelphotons.eta.compute(), axis = None)
#nonphelphotons_eta = ak.flatten(nonphelphotons.eta.compute(), axis = None)
#
#photons_hoe = ak.flatten(photons.hoe.compute(), axis = None)
#phelphotons_hoe = ak.flatten(phelphotons.hoe.compute(), axis = None)
#nonphelphotons_hoe = ak.flatten(nonphelphotons.hoe.compute(), axis = None)
#
#for pt in phelphotons_pt:
#  if pt not in photons_pt:
#    nonphelphotons_pt.append(pt)
#
#for eta in phelphotons_eta:
#  if eta not in photons_eta:
#    nonphelphotons_eta.append(eta)
#
#for hoe in phelphotons_hoe:
#  if hoe not in photons_hoe:
#    nonphelphotons_hoe.append(hoe)

phelectrons = phelectrons[(abs(phelectrons.matched_gen.distinctParent.distinctParent.pdgId) == 1000015)]
phelectrons = phelectrons.matched_gen[(phelectrons.matched_gen.pt > GenPtMin) & (abs(phelectrons.matched_gen.eta) < GenEtaMax)]

RecoElectronsFromGenList = ak.drop_none(RecoElectronsFromGen.pt).compute().tolist()
phelectronsList = ak.drop_none(phelectrons.pt).compute().tolist()

photoelectrons = electrons[photons.electronIdx.compute()]
photoelectrons = photoelectrons[(photoelectrons.pt > GenPtMin) & (abs(photoelectrons.eta) < GenEtaMax)]

electrons = electrons[(electrons.pt > GenPtMin) & (abs(electrons.eta) < GenEtaMax)]

electronList_pt = ak.drop_none(electrons.pt).compute().tolist()
photoelectronList_pt = ak.drop_none(photoelectrons.pt).compute().tolist()

electronList_eta = ak.drop_none(electrons.eta).compute().tolist()
photoelectronList_eta = ak.drop_none(photoelectrons.eta).compute().tolist()

electronList_hoe = ak.drop_none(electrons.hoe).compute().tolist()
photoelectronList_hoe = ak.drop_none(photoelectrons.hoe).compute().tolist()

electronList_dxy = abs(ak.drop_none(electrons.dxy)).compute().tolist()
photoelectronList_dxy = abs(ak.drop_none(photoelectrons.dxy)).compute().tolist()

electronList_dz = abs(ak.drop_none(electrons.dz)).compute().tolist()
photoelectronList_dz = abs(ak.drop_none(photoelectrons.dz)).compute().tolist()

electronList_convVeto = ak.drop_none(electrons.convVeto).compute().tolist()
photoelectronList_convVeto = ak.drop_none(photoelectrons.convVeto).compute().tolist()

electronList_eInvMinusPInv = ak.drop_none(electrons.eInvMinusPInv).compute().tolist()
photoelectronList_eInvMinusPInv = ak.drop_none(photoelectrons.eInvMinusPInv).compute().tolist()

electronList_r9 = ak.drop_none(electrons.r9).compute().tolist()
photoelectronList_r9 = ak.drop_none(photoelectrons.r9).compute().tolist()

electronList_sieie = ak.drop_none(electrons.sieie).compute().tolist()
photoelectronList_sieie = ak.drop_none(photoelectrons.sieie).compute().tolist()

diff_pt = [np.setdiff1d(subarr1, subarr2) for subarr1, subarr2 in zip(electronList_pt, photoelectronList_pt)]
diff_eta = [np.setdiff1d(subarr1, subarr2) for subarr1, subarr2 in zip(electronList_eta, photoelectronList_eta)]
diff_hoe = [np.setdiff1d(subarr1, subarr2) for subarr1, subarr2 in zip(electronList_hoe, photoelectronList_hoe)]
diff_dxy = [np.setdiff1d(subarr1, subarr2) for subarr1, subarr2 in zip(electronList_dxy, photoelectronList_dxy)]
diff_dz = [np.setdiff1d(subarr1, subarr2) for subarr1, subarr2 in zip(electronList_dz, photoelectronList_dz)]
diff_convVeto = [np.setdiff1d(subarr1, subarr2) for subarr1, subarr2 in zip(electronList_convVeto, photoelectronList_convVeto)]
diff_eInvMinusPInv = [np.setdiff1d(subarr1, subarr2) for subarr1, subarr2 in zip(electronList_eInvMinusPInv, photoelectronList_eInvMinusPInv)]
diff_r9 = [np.setdiff1d(subarr1, subarr2) for subarr1, subarr2 in zip(electronList_r9, photoelectronList_r9)]
diff_sieie = [np.setdiff1d(subarr1, subarr2) for subarr1, subarr2 in zip(electronList_sieie, photoelectronList_sieie)]


l_ele_pt = ROOT.TLegend()
h_ele_pt = ROOT.TH1F('h_ele_pt', ';pt [GeV]; Number of electrons', 40, 20, 100)
h_diff_pt = ROOT.TH1F('h_diff_pt', ';pt [GeV]; Number of electrons', 40, 20, 100)
for i in ak.flatten(photoelectronList_pt, axis = None):
  h_ele_pt.Fill(i)
for i in ak.flatten(diff_pt, axis = None):
  h_diff_pt.Fill(i)
h_ele_pt.Scale(1/len(ak.flatten(photoelectronList_pt, axis = None)))
h_diff_pt.Scale(1/len(ak.flatten(diff_pt, axis = None)))
h_diff_pt.SetLineColor(3)
l_ele_pt.AddEntry(h_ele_pt, "matched electrons")
l_ele_pt.AddEntry(h_diff_pt, "electrons not part of photon collection")
h_diff_pt.Draw("HISTE")
h_ele_pt.Draw("SAMEHISTE")
l_ele_pt.Draw()
can.SaveAs("h_ele_pt.pdf")

l_ele_eta = ROOT.TLegend()
h_ele_eta = ROOT.TH1F('h_ele_eta', ';#eta; Number of electrons', 48, -2.4, 2.4)
h_diff_eta = ROOT.TH1F('h_diff_eta', ';#eta; Number of electrons', 48, -2.4, 2.4)
for i in ak.flatten(photoelectronList_eta, axis = None):
  h_ele_eta.Fill(i)
for i in ak.flatten(diff_eta, axis = None):
  h_diff_eta.Fill(i)
h_ele_eta.Scale(1/len(ak.flatten(photoelectronList_eta, axis = None)))
h_diff_eta.Scale(1/len(ak.flatten(diff_eta, axis = None)))
h_diff_eta.SetLineColor(3)
l_ele_eta.AddEntry(h_ele_eta, "matched electrons")
l_ele_eta.AddEntry(h_diff_eta, "electrons not part of photon collection")
h_diff_eta.Draw("HISTE")
h_ele_eta.Draw("SAMEHISTE")
l_ele_eta.Draw()
can.SaveAs("h_ele_eta.pdf")

l_ele_hoe = ROOT.TLegend()
h_ele_hoe = ROOT.TH1F('h_ele_hoe', ';H/E; Number of electrons', 25, 0, 1)
h_diff_hoe = ROOT.TH1F('h_diff_hoe', ';H/E; Number of electrons', 25, 0, 1)
for i in ak.flatten(photoelectronList_hoe, axis = None):
  h_ele_hoe.Fill(i)
for i in ak.flatten(diff_hoe, axis = None):
  h_diff_hoe.Fill(i)
h_ele_hoe.Scale(1/len(ak.flatten(photoelectronList_hoe, axis = None)))
h_diff_hoe.Scale(1/len(ak.flatten(diff_hoe, axis = None)))
h_diff_hoe.SetLineColor(3)
l_ele_hoe.AddEntry(h_ele_hoe, "matched electrons")
l_ele_hoe.AddEntry(h_diff_hoe, "electrons not part of photon collection")
h_ele_hoe.Draw("HISTE")
h_diff_hoe.Draw("SAMEHISTE")
l_ele_hoe.Draw()
can.SaveAs("h_ele_hoe.pdf")

l_ele_dxy = ROOT.TLegend()
h_ele_dxy = ROOT.TH1F('h_ele_dxy', ';d_{xy} [cm]; Number of electrons', 30, 0, 15)
l_ele_dxy.SetBorderSize(0)
l_ele_dxy.SetFillStyle(0)
h_diff_dxy = ROOT.TH1F('h_diff_dxy', ';d_{xy} [cm]; Number of electrons', 30, 0, 15)
for i in ak.flatten(photoelectronList_dxy, axis = None):
  h_ele_dxy.Fill(i)
for i in ak.flatten(diff_dxy, axis = None):
  h_diff_dxy.Fill(i)
h_ele_dxy.Scale(1/len(ak.flatten(photoelectronList_dxy, axis = None)))
h_diff_dxy.Scale(1/len(ak.flatten(diff_dxy, axis = None)))
h_diff_dxy.SetLineColor(3)
l_ele_dxy.AddEntry(h_ele_dxy, "matched electrons")
l_ele_dxy.AddEntry(h_diff_dxy, "electrons not part of photon collection")
h_ele_dxy.Draw("HISTE")
h_diff_dxy.Draw("SAMEHISTE")
l_ele_dxy.Draw()
can.SaveAs("h_ele_dxy.pdf")

l_ele_dz = ROOT.TLegend()
h_ele_dz = ROOT.TH1F('h_ele_dz', ';d_{z} [cm]; Number of electrons', 30, 0, 15)
l_ele_dz.SetBorderSize(0)
l_ele_dz.SetFillStyle(0)
h_diff_dz = ROOT.TH1F('h_diff_dz', ';d_{z} [cm]; Number of electrons', 30, 0, 15)
for i in ak.flatten(photoelectronList_dz, axis = None):
  h_ele_dz.Fill(i)
for i in ak.flatten(diff_dz, axis = None):
  h_diff_dz.Fill(i)
h_ele_dz.Scale(1/len(ak.flatten(photoelectronList_dz, axis = None)))
h_diff_dz.Scale(1/len(ak.flatten(diff_dz, axis = None)))
h_diff_dz.SetLineColor(3)
l_ele_dz.AddEntry(h_ele_dz, "matched electrons")
l_ele_dz.AddEntry(h_diff_dz, "electrons not part of photon collection")
h_ele_dz.Draw("HISTE")
h_diff_dz.Draw("SAMEHISTE")
l_ele_dz.Draw()
can.SaveAs("h_ele_dz.pdf")


l_ele_convVeto = ROOT.TLegend()
h_ele_convVeto = ROOT.TH1F('h_ele_convVeto', ';convVeto; Number of electrons', 2, 0, 2)
h_diff_convVeto = ROOT.TH1F('h_diff_convVeto', ';convVeto; Number of electrons', 2, 0, 2)
for i in ak.flatten(photoelectronList_convVeto, axis = None):
  h_ele_convVeto.Fill(i)
for i in ak.flatten(diff_convVeto, axis = None):
  h_diff_convVeto.Fill(i)
h_ele_convVeto.Scale(1/len(ak.flatten(photoelectronList_convVeto, axis = None)))
h_diff_convVeto.Scale(1/len(ak.flatten(diff_convVeto, axis = None)))
h_diff_convVeto.SetLineColor(3)
l_ele_convVeto.AddEntry(h_ele_convVeto, "matched electrons")
l_ele_convVeto.AddEntry(h_diff_convVeto, "electrons not part of photon collection")
h_diff_convVeto.Draw("HISTE")
h_ele_convVeto.Draw("SAMEHISTE")
l_ele_convVeto.Draw()
can.SaveAs("h_ele_convVeto.pdf")

l_ele_eInvMinusPInv = ROOT.TLegend()
h_ele_eInvMinusPInv = ROOT.TH1F('h_ele_eInvMinusPInv', ';eInvMinusPInv; Number of electrons', 28, 0, 7)
h_diff_eInvMinusPInv = ROOT.TH1F('h_diff_eInvMinusPInv', ';eInvMinusPInv; Number of electrons', 28, 0, 7)
for i in ak.flatten(photoelectronList_eInvMinusPInv, axis = None):
  h_ele_eInvMinusPInv.Fill(abs(i))
for i in ak.flatten(diff_eInvMinusPInv, axis = None):
  h_diff_eInvMinusPInv.Fill(abs(i))
h_ele_eInvMinusPInv.Scale(1/len(ak.flatten(photoelectronList_eInvMinusPInv, axis = None)))
h_diff_eInvMinusPInv.Scale(1/len(ak.flatten(diff_eInvMinusPInv, axis = None)))
h_diff_eInvMinusPInv.SetLineColor(3)
l_ele_eInvMinusPInv.AddEntry(h_ele_eInvMinusPInv, "matched electrons")
l_ele_eInvMinusPInv.AddEntry(h_diff_eInvMinusPInv, "electrons not part of photon collection")
h_diff_eInvMinusPInv.Draw("HISTE")
h_ele_eInvMinusPInv.Draw("SAMEHISTE")
l_ele_eInvMinusPInv.Draw()
can.SaveAs("h_ele_eInvMinusPInv.pdf")

l_ele_r9 = ROOT.TLegend()
h_ele_r9 = ROOT.TH1F('h_ele_r9', ';r9; Number of electrons', 120, 0, 30)
h_diff_r9 = ROOT.TH1F('h_diff_r9', ';r9; Number of electrons', 120, 0, 30)
for i in ak.flatten(photoelectronList_r9, axis = None):
  h_ele_r9.Fill(i)
for i in ak.flatten(diff_r9, axis = None):
  h_diff_r9.Fill(i)
h_ele_r9.Scale(1/len(ak.flatten(photoelectronList_r9, axis = None)))
h_diff_r9.Scale(1/len(ak.flatten(diff_r9, axis = None)))
h_diff_r9.SetLineColor(3)
l_ele_r9.AddEntry(h_ele_r9, "matched electrons")
l_ele_r9.AddEntry(h_diff_r9, "electrons not part of photon collection")
h_diff_r9.Draw("HISTE")
h_ele_r9.Draw("SAMEHISTE")
l_ele_r9.Draw()
can.SaveAs("h_ele_r9.pdf")

l_ele_sieie = ROOT.TLegend()
h_ele_sieie = ROOT.TH1F('h_ele_sieie', ';sieie; Number of electrons', 70, 0, 35)
h_diff_sieie = ROOT.TH1F('h_diff_sieie', ';sieie; Number of electrons', 70, 0, 35)
for i in ak.flatten(photoelectronList_sieie, axis = None):
  h_ele_sieie.Fill(i)
for i in ak.flatten(diff_sieie, axis = None):
  h_diff_sieie.Fill(i)
h_ele_sieie.Scale(1/len(ak.flatten(photoelectronList_sieie, axis = None)))
h_diff_sieie.Scale(1/len(ak.flatten(diff_sieie, axis = None)))
h_diff_sieie.SetLineColor(3)
l_ele_sieie.AddEntry(h_ele_sieie, "matched electrons")
l_ele_sieie.AddEntry(h_diff_sieie, "electrons not part of photon collection")
h_diff_sieie.Draw("HISTE")
h_ele_sieie.Draw("SAMEHISTE")
l_ele_sieie.Draw()
can.SaveAs("h_ele_sieie.pdf")


RecoPhotonsFromGen   = gpart[photons.genPartIdx.compute()]
RecoPhotonsFromGen   = RecoPhotonsFromGen[abs(RecoPhotonsFromGen.pdgId) == 11]
RecoPhotonsFromGen   = RecoPhotonsFromGen[abs(RecoPhotonsFromGen.distinctParent.distinctParent.pdgId) == 1000015]
RecoPhotonsFromGen   = RecoPhotonsFromGen[(RecoPhotonsFromGen.pt > GenPtMin) & (abs(RecoPhotonsFromGen.eta) < GenEtaMax)]

RecoPhotons_pt = ak.flatten(RecoPhotonsFromGen.pt.compute(), axis = None)
photoelectrons_pt = ak.flatten(photoelectrons.pt.compute(), axis = None)



#for i in RecoPhotons_pt:
#  if i not in photoelectrons_pt:
#    print("RecoPhotons is not a subset of photoelectrons")


RecoLowPtElecFromGen = gpart[lowptelectrons.genPartIdx.compute()]
RecoLowPtElecFromGen = RecoLowPtElecFromGen[abs(RecoLowPtElecFromGen.distinctParent.distinctParent.pdgId) == 1000015]
RecoLowPtElecFromGen = RecoLowPtElecFromGen[(RecoLowPtElecFromGen.pt > GenPtMin) & (abs(RecoLowPtElecFromGen.eta) < GenEtaMax)]

print("Number of gen electrons:", len(ak.flatten(gen_electrons.pt.compute(), axis = None)))
print("Number of reco electrons from Electron collection:", len(ak.flatten(RecoElectronsFromGen.pt.compute(), axis = None)))
print("Number of reco electrons from LowPtElectron collection:", len(ak.flatten(RecoLowPtElecFromGen.pt.compute(), axis = None)))
print("Number of reco electrons from Photon collection:", len(ak.flatten(phelectrons.pt.compute(), axis = None)))

makeEffPlot("e", "gammacomp_gammaMatchedToGenE", ["Electrons", "Photon[Photon_electronIdx] which are matched to a dis gen electron"], "pt", 16, 20, 100, 5, "[GeV]", [gen_electrons.pt.compute(),] * 2, [RecoElectronsFromGen.pt.compute(), RecoPhotonsFromGen.pt.compute()], 0, file)
makeEffPlot("e", "gammacomp_gammaMatchedToEMatchedToGenE", ["Electrons", "Electron[Photon_electronIdx] which are matched to a dis gen electron"], "pt", 16, 20, 100, 5, "[GeV]", [gen_electrons.pt.compute(),] * 2, [RecoElectronsFromGen.pt.compute(), phelectrons.pt.compute()], 0, file)
#makeEffPlot("e", "gammacomp_matchedgam", ["Electrons", "LowPtElectrons", "Photons"], "hoe", 0, 1, 20, 0.05, "", [gen_electrons.hoe.compute(),] * 3, [RecoElectronsFromGen.eta.compute(), RecoLowPtElecFromGen.pt.compute(), RecoPhotonsFromGen.pt.compute()], 0, file)
#makeEffPlot("photons", "e_not_photon", ["#gamma associated to e associated to same #gamma",  "#gamma that weren't associated to e"] , "pt", 20, 0, 100, 5, "[GeV]", [photons_pt,] * 2, [phelphotons_pt, nonphelphotons_pt], 0, file)
#makeEffPlot("photons", "e_not_photon", ["#gamma associated to e associated to same #gamma",  "#gamma that weren't associated to e"] , "#eta", 24, -2.4, 2.4, 0.2, "", [photons_eta,] * 2, [phelphotons_eta, nonphelphotons_eta], 0, file) 
#makeEffPlot("photons", "e_not_photon", ["#gamma associated to e associated to same #gamma",  "#gamma that weren't associated to e"] , "hoe", 20, 0, 1, 0.05, "", [photons_hoe,] * 2, [phelphotons_hoe, nonphelphotons_hoe], 0, file) 
