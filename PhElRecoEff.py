import numpy as np
import awkward as ak
import ROOT 
import math
import scipy
import array
import pandas as pd
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from leptonPlot import *

ptRANGE = ["20-25", "25-30", "30-35", "35-40", "40-45", "45-50", "50-55", "55-60", 
           "60-65", "65-70", "70-75", "75-80", "80-85", "85-90", "90-95", "95-100"]

etaRANGE = ["-2.4--2.2", "-2.2--2.0", "-2.0--1.8", "-1.8--1.6", "-1.6--1.4", "-1.4--1.2", "-1.2--1.0", "-1.0--0.8", "-0.8--0.6", "-0.6--0.4", "-0.4--0.2", "-0.2-0",
            "0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0", "1.0-1.2", "1.2-1.4", "1.4-1.6", "1.6-1.8", "1.8-2.0", "2.0-2.2", "2.2-2.4"]

#dxyRANGE = ["0-0.5", "0.5-1", "1-1.5", "1.5-2", "2-2.5","2.5-3", "3-3.5", "3.5-4", "4-4.5", "4.5-5",
#            "5-5.5", "5.5-6", "6-6.5", "6.5-7", "7-7.5","7.5-8", "8-8.5", "8.5-9", "9-9.5", "9.5-10",
#            "10-10.5", "10.5-11", "11-11.5", "11.5-12", "12-12.5","12.5-13", "13-13.5", "13.5-14", "14-14.5", "14.5-15"]
dxyRANGE = ["0-0.5", "0.5-1", "1-1.5", "1.5-2", "2-2.5", "2.5-3", "3-5", "5-15"]
dxyBins = [0,0.5,1,1.5,2,2.5,3,5,15]
dxyBins = array.array('d', dxyBins)
lxyRANGE = ["0-0.5", "0.5-1", "1-1.5", "1.5-2", "2-2.5","2.5-3", "3-3.5", "3.5-4", "4-4.5", "4.5-5",
            "5-5.5", "5.5-6", "6-6.5", "6.5-7", "7-7.5","7.5-8", "8-8.5", "8.5-9", "9-9.5", "9.5-10",
            "10-10.5", "10.5-11", "11-11.5", "11.5-12", "12-12.5","12.5-13", "13-13.5", "13.5-14", "14-14.5", "14.5-15"]

file = "Staus_M_100_100mm_13p6TeV_Run3Summer22_lpcdisptau_NanoAOD_ExtraDisMuonBranches.root"

events = NanoEventsFactory.from_root({file:"Events"}).events()
ROOT.gStyle.SetOptStat(0)

gpart = events.GenPart
gvtx = events.GenVtx
gvistau = events.GenVisTau
electrons = events.Electron
photons = events.Photon
lowptelectrons = events.LowPtElectron

staus = gpart[(abs(gpart.pdgId) == 1000015) & (gpart.hasFlags("isLastCopy"))]
staus_taus = staus.distinctChildren[(abs(staus.distinctChildren.pdgId) == 15) & (staus.distinctChildren.hasFlags("isLastCopy"))]
gen_electrons = staus_taus.distinctChildren[(abs(staus_taus.distinctChildren.pdgId) == 11)]
gen_electrons = gen_electrons[(gen_electrons.pt > GenPtMin) & (abs(gen_electrons.eta) < GenEtaMax)]
gen_electrons["dxy"] = abs((gen_electrons.vertexY - gvtx.y) * np.cos(gen_electrons.phi) - (gen_electrons.vertexX - gvtx.x) * np.sin(gen_electrons.phi))
gen_electrons["lxy"] = np.sqrt((gen_electrons.distinctParent.vertexX - ak.firsts(staus.vertexX)) ** 2 + (gen_electrons.distinctParent.vertexY - ak.firsts(staus.vertexY)) ** 2)  

#electrons = electrons[(electrons.hoe < 0.15)]
electrons = electrons[(abs(electrons.matched_gen.pdgId) == 11) & (abs(electrons.matched_gen.distinctParent.distinctParent.pdgId) == 1000015)]
RecoElectrons = electrons[(electrons.matched_gen.pt > GenPtMin) & (abs(electrons.matched_gen.eta) < GenEtaMax)]
RecoElectronsFromGen = electrons.matched_gen[(electrons.matched_gen.pt > GenPtMin) & (abs(electrons.matched_gen.eta) < GenEtaMax)]
RecoElectronsFromGen["dxy"] = (RecoElectronsFromGen.vertexY - gvtx.y) * np.cos(RecoElectronsFromGen.phi) - (RecoElectronsFromGen.vertexX - gvtx.x) * np.sin(RecoElectronsFromGen.phi)
RecoElectronsFromGen["lxy"] = np.sqrt((RecoElectronsFromGen.vertexX - ak.firsts(staus.vertexX)) ** 2 + (RecoElectronsFromGen.vertexY - ak.firsts(staus.vertexY)) ** 2)

PFElectrons = electrons[(electrons.isPFcand == 1)]
RecoPFElectronsFromGen = PFElectrons.matched_gen[(PFElectrons.matched_gen.pt > GenPtMin) & (abs(PFElectrons.matched_gen.eta) < GenEtaMax)]
RecoPFElectronsFromGen["dxy"] = (RecoPFElectronsFromGen.vertexY - gvtx.y) * np.cos(RecoPFElectronsFromGen.phi) - (RecoPFElectronsFromGen.vertexX - gvtx.x) * np.sin(RecoPFElectronsFromGen.phi)
RecoPFElectronsFromGen["lxy"] = np.sqrt((RecoPFElectronsFromGen.vertexX - ak.firsts(staus.vertexX)) ** 2 + (RecoPFElectronsFromGen.vertexY - ak.firsts(staus.vertexY)) ** 2)


#makeResPlot("e", ["Electrons matched to dis gen electrons"], "pt", "dxy", ptRANGE, 20, 100, -0.1, 0.1, 5, [RecoElectronsFromGen.pt.compute()], [(RecoElectrons.dxy - RecoElectronsFromGen.dxy).compute()], "[GeV]", "[cm]", file) 
#makeResPlot("e", ["Electrons matched to dis gen electrons"], "eta", "dxy", etaRANGE, -2.4, 2.4, -0.1, 0.1, 0.2, [RecoElectronsFromGen.eta.compute()], [(RecoElectrons.dxy - RecoElectronsFromGen.dxy).compute()], "", "[cm]", file) 
#makeResPlot_varBin("e", ["Electrons matched to dis gen electrons"], "dxy", "dxy", dxyRANGE, dxyBins, -0.1, 0.1, [abs(RecoElectronsFromGen.dxy.compute())], [(RecoElectrons.dxy - RecoElectronsFromGen.dxy).compute()], "[cm]", "[cm]", file) 
#makeResPlot("e", ["Electrons matched to dis gen electrons"], "lxy", "dxy", lxyRANGE, 0, 15, -0.1, 0.1, 0.5, [RecoElectronsFromGen.lxy.compute()], [(RecoElectrons.dxy - RecoElectronsFromGen.dxy).compute()], "[cm]", "[cm]", file) 

#photons = photons[photons.pixelSeed == True]
photons = photons[(photons.electronIdx > -1) & (photons.electronIdx < ak.num(electrons))]
phelectrons = events.Electron[photons.electronIdx.compute()]
#phelectrons = phelectrons[(phelectrons.photonIdx != -1) & (phelectrons.photonIdx < ak.num(photons))]
nonphelectrons = phelectrons[(phelectrons.photonIdx == -1) | (phelectrons.photonIdx >= ak.num(photons))]
phelphotons = photons[phelectrons.photonIdx.compute()]
nonphelphotons = photons[(electrons[photons.electronIdx.compute()].photonIdx == -1) | (electrons[photons.electronIdx.compute()].photonIdx>= ak.num(photons))]


#for evt in range(30):
#  print("For event", evt)
#  print("Photon electron Idx", events.Photon.electronIdx.compute()[evt])
#  print("Photon pixelSeed", events.Photon.pixelSeed.compute()[evt])
#  print("Photon hasConversionTracks", events.Photon.hasConversionTracks.compute()[evt])
#  print("Photon x calo coordinate is", events.Photon.x_calo.compute()[evt])
#  print("Photon y calo coordinate is", events.Photon.y_calo.compute()[evt])
#  print("Photon r calo coordinate is", np.sqrt(events.Photon.x_calo ** 2 + events.Photon.y_calo ** 2).compute()[evt])

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


#l_ele_pt = ROOT.TLegend()
#l_ele_pt.SetBorderSize(0)
#l_ele_pt.SetFillStyle(0)
#l_ele_pt.SetTextSize(0.025)
#h_ele_pt = ROOT.TH1F('h_ele_pt', ';pt [GeV]; Number of electrons', 40, 20, 100)
#h_pho_pt = ROOT.TH1F('h_pho_pt', ';pt [GeV]; Number of electrons', 40, 20, 100)
#for i in ak.flatten(events.Electron[(events.Electron.pt > GenPtMin) & (abs(events.Electron.eta) < GenEtaMax)].pt.compute(), axis = None):
#  h_ele_pt.Fill(i)
#for i in ak.flatten(events.Photon[(events.Photon.electronIdx > -1) & (events.Photon.pt > GenPtMin) & (abs(events.Photon.eta) < GenEtaMax)].pt.compute(), axis = None):
#  h_pho_pt.Fill(i)
#h_ele_pt.Scale(1/len(ak.flatten(events.Electron[(events.Electron.pt > GenPtMin) & (abs(events.Electron.eta) < GenEtaMax)].pt.compute(), axis = None)))
#h_pho_pt.Scale(1/len(ak.flatten(events.Photon[(events.Photon.electronIdx > -1) & (events.Photon.pt > GenPtMin) & (abs(events.Photon.eta) < GenEtaMax)].pt.compute(), axis = None)))
#h_pho_pt.SetLineColor(3)
#l_ele_pt.AddEntry(h_ele_pt, "Electron[Electron.pt > 20 GeV & abs(Electron.eta) < 2.4]")
#l_ele_pt.AddEntry(h_pho_pt, "Photon[Photon.electronIdx > -1 & Photon.pt > 20 GeV & abs(Photon.eta) < 2.4]")
#h_ele_pt.Draw("HISTE")
#h_pho_pt.Draw("SAMEHISTE")
#l_ele_pt.Draw()
#can.SaveAs("h_ele_pt.png")
#
#h2_ele_pt = ROOT.TH2F('h2_ele_pt', ':Photon[Photon.electronIdx > -1].pt [GeV];Electron[Photon.electronIdx].pt [GeV]', 16, 20, 100,  16, 20, 100)
#h2_ele_eta = ROOT.TH2F('h2_ele_pt', ':Photon[Photon.electronIdx > -1].eta; Electron[Photon.electronIdx].eta', 24, -2.4, 2.4, 24, -2.4, 2.4)
#  
#
#l_ele_eta = ROOT.TLegend()
#l_ele_eta.SetBorderSize(0)
#l_ele_eta.SetFillStyle(0)
#l_ele_eta.SetTextSize(0.025)
#h_ele_eta = ROOT.TH1F('h_ele_eta', ';#eta; Number of electrons', 48, -2.4, 2.4)
#h_pho_eta = ROOT.TH1F('h_pho_eta', ';#eta; Number of electrons', 48, -2.4, 2.4)
#for i in ak.flatten(events.Electron[(events.Electron.pt > GenPtMin) & (abs(events.Electron.eta) < GenEtaMax)].eta.compute(), axis = None):
#  h_ele_eta.Fill(i)
#for i in ak.flatten(events.Photon[(events.Photon.electronIdx > -1) & (events.Photon.pt > GenPtMin) & (abs(events.Photon.eta) < GenEtaMax)].eta.compute(), axis = None):
#  h_pho_eta.Fill(i)
#h_ele_eta.Scale(1/len(ak.flatten(events.Electron[(events.Electron.pt > GenPtMin) & (abs(events.Electron.eta) < GenEtaMax)].eta.compute(), axis = None)))
#h_pho_eta.Scale(1/len(ak.flatten(events.Photon[(events.Photon.electronIdx > -1) & (events.Photon.pt > GenPtMin) & (abs(events.Photon.eta) < GenEtaMax)].eta.compute(), axis = None)))
#h_pho_eta.SetLineColor(3)
#l_ele_eta.AddEntry(h_ele_eta, "Electron[Electron.pt > 20 GeV & abs(Electron.eta) < 2.4]")                    
#l_ele_eta.AddEntry(h_pho_eta, "Photon[Photon.electronIdx > -1 & Photon.pt > 20 GeV & abs(Photon.eta) < 2.4]")
#h_pho_eta.Draw("HISTE")
#h_ele_eta.Draw("SAMEHISTE")
#l_ele_eta.Draw()
#can.SaveAs("h_ele_eta.png")
#
#l_ele_hoe = ROOT.TLegend()
#l_ele_hoe.SetBorderSize(0)
#l_ele_hoe.SetFillStyle(0)
#l_ele_hoe.SetTextSize(0.025)
#h_ele_hoe = ROOT.TH1F('h_ele_hoe', ';H/E; Number of electrons', 25, 0, 1)
#h_pho_hoe = ROOT.TH1F('h_pho_hoe', ';H/E; Number of electrons', 25, 0, 1)
#for i in ak.flatten(events.Electron[(events.Electron.pt > GenPtMin) & (abs(events.Electron.eta) < GenEtaMax)].hoe.compute(), axis = None):
#  h_ele_hoe.Fill(i)
#for i in ak.flatten(events.Photon[(events.Photon.electronIdx > -1) & (events.Photon.pt > GenPtMin) & (abs(events.Photon.eta) < GenEtaMax)].hoe.compute(), axis = None):
#  h_pho_hoe.Fill(i)
#h_ele_hoe.Scale(1/len(ak.flatten(events.Electron[(events.Electron.pt > GenPtMin) & (abs(events.Electron.eta) < GenEtaMax)].hoe.compute(), axis = None)))
#h_pho_hoe.Scale(1/len(ak.flatten(events.Photon[(events.Photon.electronIdx > -1) & (events.Photon.pt > GenPtMin) & (abs(events.Photon.eta) < GenEtaMax)].hoe.compute(), axis = None)))
#h_pho_hoe.SetLineColor(3)
#l_ele_hoe.AddEntry(h_ele_hoe, "Electron[Electron.pt > 20 GeV & abs(Electron.eta) < 2.4]")
#l_ele_hoe.AddEntry(h_pho_hoe, "Photon[Photon.electronIdx > -1 & Photon.pt > 20 GeV & abs(Photon.eta) < 2.4]")
#h_ele_hoe.Draw("HISTE")
#h_pho_hoe.Draw("SAMEHISTE")
#l_ele_hoe.Draw()
#can.SaveAs("h_ele_hoe.png")
#
##l_ele_dxy = ROOT.TLegend()
##l_ele_dxy.SetBorderSize(0)
##l_ele_dxy.SetFillStyle(0)
##h_ele_dxy = ROOT.TH1F('h_ele_dxy', ';d_{xy} [cm]; Number of electrons', 30, 0, 15)
##h_pho_dxy = ROOT.TH1F('h_pho_dxy', ';d_{xy} [cm]; Number of electrons', 30, 0, 15)
##for i in ak.flatten(events.Electron[(events.Electron.pt > GenPtMin) & (abs(events.Electron.eta) < GenEtaMax)].dxy.compute(), axis = None):
##  h_ele_dxy.Fill(i)
##for i in ak.flatten(pho_dxy.compute(), axis = None):
##  h_pho_dxy.Fill(i)
##h_ele_dxy.Scale(1/len(ak.flatten(events.Electron[(events.Electron.pt > GenPtMin) & (abs(events.Electron.eta) < GenEtaMax)].dxy.compute(), axis = None)))
##h_pho_dxy.Scale(1/len(ak.flatten(events.Photon[(events.Photon.electronIdx > -1) & (events.Photon.pt > GenPtMin) & (abs(events.Photon.eta) < GenEtaMax)].dxy.compute(), axis = None)))
##h_pho_dxy.SetLineColor(3)
##l_ele_dxy.AddEntry(h_ele_dxy, "Electron[Photon_electronIdx]")
##l_ele_dxy.AddEntry(h_pho_dxy, "Electrons - Electron[Photon_electronIdx]")
##h_ele_dxy.Draw("HISTE")
##h_pho_dxy.Draw("SAMEHISTE")
##l_ele_dxy.Draw()
##can.SaveAs("h_ele_dxy.png")
#
##l_ele_dz = ROOT.TLegend()
##l_ele_dz.SetBorderSize(0)
##l_ele_dz.SetFillStyle(0)
##h_ele_dz = ROOT.TH1F('h_ele_dz', ';d_{z} [cm]; Number of electrons', 30, 0, 15)
##h_pho_dz = ROOT.TH1F('h_pho_dz', ';d_{z} [cm]; Number of electrons', 30, 0, 15)
##for i in ak.flatten(events.Electron[(events.Electron.pt > GenPtMin) & (abs(events.Electron.eta) < GenEtaMax)].dz.compute(), axis = None):
##  h_ele_dz.Fill(i)
##for i in ak.flatten(pho_dz.compute(), axis = None):
##  h_pho_dz.Fill(i)
##h_ele_dz.Scale(1/len(ak.flatten(events.Electron[(events.Electron.pt > GenPtMin) & (abs(events.Electron.eta) < GenEtaMax)].dz.compute(), axis = None)))
##h_pho_dz.Scale(1/len(ak.flatten(events.Photon[(events.Photon.electronIdx > -1) & (events.Photon.pt > GenPtMin) & (abs(events.Photon.eta) < GenEtaMax)].dz.compute(), axis = None)))
##h_pho_dz.SetLineColor(3)
##l_ele_dz.AddEntry(h_ele_dz, "Electron[Photon_electronIdx]")
##l_ele_dz.AddEntry(h_pho_dz, "Electrons - Electron[Photon_electronIdx]")
##h_ele_dz.Draw("HISTE")
##h_pho_dz.Draw("SAMEHISTE")
##l_ele_dz.Draw()
##can.SaveAs("h_ele_dz.png")
#
#
##l_ele_convVeto = ROOT.TLegend()
##l_ele_convVeto.SetBorderSize(0)
##l_ele_convVeto.SetFillStyle(0)
##h_ele_convVeto = ROOT.TH1F('h_ele_convVeto', ';convVeto; Number of electrons', 2, 0, 2)
##h_pho_convVeto = ROOT.TH1F('h_pho_convVeto', ';convVeto; Number of electrons', 2, 0, 2)
##for i in ak.flatten(events.Electron[(events.Electron.pt > GenPtMin) & (abs(events.Electron.eta) < GenEtaMax)].convVeto.compute(), axis = None):
##  h_ele_convVeto.Fill(i)
##for i in ak.flatten(pho_convVeto.compute(), axis = None):
##  h_pho_convVeto.Fill(i)
##h_ele_convVeto.Scale(1/len(ak.flatten(events.Electron[(events.Electron.pt > GenPtMin) & (abs(events.Electron.eta) < GenEtaMax)].convVeto.compute(), axis = None)))
##h_pho_convVeto.Scale(1/len(ak.flatten(events.Photon[(events.Photon.electronIdx > -1) & (events.Photon.pt > GenPtMin) & (abs(events.Photon.eta) < GenEtaMax)].convVeto.compute(), axis = None)))
##h_pho_convVeto.SetLineColor(3)
##l_ele_convVeto.AddEntry(h_ele_convVeto, "Electron[Photon_electronIdx]")
##l_ele_convVeto.AddEntry(h_pho_convVeto, "Electrons - Electron[Photon_electronIdx]")
##h_pho_convVeto.Draw("HISTE")
##h_ele_convVeto.Draw("SAMEHISTE")
##l_ele_convVeto.Draw()
##can.SaveAs("h_ele_convVeto.png")
#
##l_ele_eInvMinusPInv = ROOT.TLegend()
##l_ele_eInvMinusPInv.SetBorderSize(0)
##l_ele_eInvMinusPInv.SetFillStyle(0)
##h_ele_eInvMinusPInv = ROOT.TH1F('h_ele_eInvMinusPInv', ';eInvMinusPInv; Number of electrons', 30, 0, 0.3)
##h_pho_eInvMinusPInv = ROOT.TH1F('h_pho_eInvMinusPInv', ';eInvMinusPInv; Number of electrons', 30, 0, 0.3)
##for i in ak.flatten(events.Electron[(events.Electron.pt > GenPtMin) & (abs(events.Electron.eta) < GenEtaMax)].eInvMinusPInv.compute(), axis = None):
##  h_ele_eInvMinusPInv.Fill(abs(i))
##for i in ak.flatten(pho_eInvMinusPInv.compute(), axis = None):
##  h_pho_eInvMinusPInv.Fill(abs(i))
##h_ele_eInvMinusPInv.Scale(1/len(ak.flatten(events.Electron[(events.Electron.pt > GenPtMin) & (abs(events.Electron.eta) < GenEtaMax)].eInvMinusPInv.compute(), axis = None)))
##h_pho_eInvMinusPInv.Scale(1/len(ak.flatten(events.Photon[(events.Photon.electronIdx > -1) & (events.Photon.pt > GenPtMin) & (abs(events.Photon.eta) < GenEtaMax)].eInvMinusPInv.compute(), axis = None)))
##h_pho_eInvMinusPInv.SetLineColor(3)
##l_ele_eInvMinusPInv.AddEntry(h_ele_eInvMinusPInv, "Electron[Photon_electronIdx]")
##l_ele_eInvMinusPInv.AddEntry(h_pho_eInvMinusPInv, "Electrons - Electron[Photon_electronIdx]")
##h_pho_eInvMinusPInv.Draw("HISTE")
##h_ele_eInvMinusPInv.Draw("SAMEHISTE")
##l_ele_eInvMinusPInv.Draw()
##can.SaveAs("h_ele_eInvMinusPInv.png")
#
#l_ele_r9 = ROOT.TLegend()
#l_ele_r9.SetBorderSize(0)
#l_ele_r9.SetFillStyle(0)
#l_ele_r9.SetTextSize(0.025)
#h_ele_r9 = ROOT.TH1F('h_ele_r9', ';r9; Number of electrons', 50, 0, 2)
#h_pho_r9 = ROOT.TH1F('h_pho_r9', ';r9; Number of electrons', 50, 0, 2)
#for i in ak.flatten(events.Electron[(events.Electron.pt > GenPtMin) & (abs(events.Electron.eta) < GenEtaMax)].r9.compute(), axis = None):
#  h_ele_r9.Fill(i)
#for i in ak.flatten(events.Photon[(events.Photon.electronIdx > -1) & (events.Photon.pt > GenPtMin) & (abs(events.Photon.eta) < GenEtaMax)].r9.compute(), axis = None):
#  h_pho_r9.Fill(i)
#h_ele_r9.Scale(1/len(ak.flatten(events.Electron[(events.Electron.pt > GenPtMin) & (abs(events.Electron.eta) < GenEtaMax)].r9.compute(), axis = None)))
#h_pho_r9.Scale(1/len(ak.flatten(events.Photon[(events.Photon.electronIdx > -1) & (events.Photon.pt > GenPtMin) & (abs(events.Photon.eta) < GenEtaMax)].r9.compute(), axis = None)))
#h_pho_r9.SetLineColor(3)
#l_ele_r9.AddEntry(h_ele_r9, "Electron[Electron.pt > 20 GeV & abs(Electron.eta) < 2.4]")
#l_ele_r9.AddEntry(h_pho_r9, "Photon[Photon.electronIdx > -1 & Photon.pt > 20 GeV & abs(Photon.eta) < 2.4]")
#h_pho_r9.Draw("HISTE")
#h_ele_r9.Draw("SAMEHISTE")
#l_ele_r9.Draw()
#can.SaveAs("h_ele_r9.png")
#
#l_ele_sieie = ROOT.TLegend()
#l_ele_sieie.SetBorderSize(0)
#l_ele_sieie.SetFillStyle(0)
#l_ele_sieie.SetTextSize(0.025)
#h_ele_sieie = ROOT.TH1F('h_ele_sieie', ';sieie; Number of electrons', 25, 0, 0.1)
#h_pho_sieie = ROOT.TH1F('h_pho_sieie', ';sieie; Number of electrons', 25, 0, 0.1)
#for i in ak.flatten(events.Electron[(events.Electron.pt > GenPtMin) & (abs(events.Electron.eta) < GenEtaMax)].sieie.compute(), axis = None):
#  h_ele_sieie.Fill(i)
#for i in ak.flatten(events.Photon[(events.Photon.electronIdx > -1) & (events.Photon.pt > GenPtMin) & (abs(events.Photon.eta) < GenEtaMax)].sieie.compute(), axis = None):
#  h_pho_sieie.Fill(i)
#h_ele_sieie.Scale(1/len(ak.flatten(events.Electron[(events.Electron.pt > GenPtMin) & (abs(events.Electron.eta) < GenEtaMax)].sieie.compute(), axis = None)))
#h_pho_sieie.Scale(1/len(ak.flatten(events.Photon[(events.Photon.electronIdx > -1) & (events.Photon.pt > GenPtMin) & (abs(events.Photon.eta) < GenEtaMax)].sieie.compute(), axis = None)))
#h_pho_sieie.SetLineColor(3)
#l_ele_sieie.AddEntry(h_ele_sieie, "Electron[Electron.pt > 20 GeV & abs(Electron.eta) < 2.4]")
#l_ele_sieie.AddEntry(h_pho_sieie, "Photon[Photon.electronIdx > -1 & Photon.pt > 20 GeV & abs(Photon.eta) < 2.4]")
#h_pho_sieie.Draw("HISTE")
#h_ele_sieie.Draw("SAMEHISTE")
#l_ele_sieie.Draw()
#can.SaveAs("h_ele_sieie.png")

convTrackPhotons = events.Photon[events.Photon.hasConversionTracks == True]
convTrackPhotons = convTrackPhotons[convTrackPhotons.genPartIdx > -1]
pixelSeedPhotons = events.Photon[events.Photon.pixelSeed == True]
pixelSeedPhotons = pixelSeedPhotons[pixelSeedPhotons.genPartIdx > -1]
RecoPhotonsFromGen   = pixelSeedPhotons.matched_gen
RecoPhotonsFromGen   = RecoPhotonsFromGen[abs(RecoPhotonsFromGen.pdgId) == 11]
RecoPhotonsFromGen   = RecoPhotonsFromGen[abs(RecoPhotonsFromGen.distinctParent.distinctParent.pdgId) == 1000015]
RecoPhotonsFromGen   = RecoPhotonsFromGen[(RecoPhotonsFromGen.pt > GenPtMin) & (abs(RecoPhotonsFromGen.eta) < GenEtaMax)]
dpt_pt = abs(ak.flatten(RecoPhotonsFromGen.pt, axis = None) - ak.flatten(RecoPhotons.pt, axis = None))/ak.flatten(RecoPhotonsFromGen.pt, axis = None)
dpt_pt_mask = dpt_pt < 0.3
RecoPhotonsFromGen = RecoPhotonsFromGen[dpt_pt_mask]

RecoPhotons_pt = ak.flatten(RecoPhotonsFromGen.pt.compute(), axis = None)
photoelectrons_pt = ak.flatten(photoelectrons.pt.compute(), axis = None)



#for i in RecoPhotons_pt:
#  if i not in photoelectrons_pt:
#    print("RecoPhotons is not a subset of photoelectrons")

lowptelectrons = lowptelectrons[lowptelectrons.genPartIdx > -1]
RecoLowPtElecFromGen = gpart[lowptelectrons.genPartIdx.compute()]
RecoLowPtElecFromGen = RecoLowPtElecFromGen[abs(RecoLowPtElecFromGen.distinctParent.distinctParent.pdgId) == 1000015]
RecoLowPtElecFromGen = RecoLowPtElecFromGen[(RecoLowPtElecFromGen.pt > GenPtMin) & (abs(RecoLowPtElecFromGen.eta) < GenEtaMax)]

makeEffPlot("e", "gammacomp_pixelseed_energyassoc", ["Electrons", "LowPtElectrons", "Photons"], "pt", 16, 20, 100, 5, "[GeV]", [gen_electrons.pt.compute(),] * 3, [RecoElectronsFromGen.pt.compute(), RecoLowPtElecFromGen.pt.compute(), RecoPhotonsFromGen.pt.compute()], 0, file)
RecoLowPtElec = lowptelectrons[(abs(gpart[lowptelectrons.genPartIdx.compute()].pdgId) == 11)
                              & (abs(gpart[lowptelectrons.genPartIdx.compute()].distinctParent.distinctParent.pdgId) == 1000015)
                              & (gpart[lowptelectrons.genPartIdx.compute()].pt > GenPtMin) & (abs(gpart[lowptelectrons.genPartIdx.compute()].eta) < GenEtaMax)]

h_lowptelec_score = ROOT.TH1F("h_lowptelec_score", file.split(".")[0]+"; BDT score; Number of LowPtElectrons matched to gen #tau_{e}", 20, 0, 10)
for score in ak.flatten(RecoLowPtElec.ID.compute(), axis = None):
  h_lowptelec_score.Fill(score)
h_lowptelec_score.Draw("HISTE")
can.SaveAs("h_lowptelec_score.pdf")


print("Number of gen electrons:", len(ak.flatten(gen_electrons.pt.compute(), axis = None)))
print("Number of reco electrons from Electron collection:", len(ak.flatten(RecoElectronsFromGen.pt.compute(), axis = None)))
print("Number of reco electrons from LowPtElectron collection:", len(ak.flatten(RecoLowPtElecFromGen.pt.compute(), axis = None)))
print("Number of reco electrons from Photon collection:", len(ak.flatten(RecoPhotonsFromGen.pt.compute(), axis = None)))

#makeEffPlot("e", "gammacomp_gammaMatchedToGenE", ["Electrons matched to dis gen electrons", "Photon[Photon_electronIdx] which are matched to a dis gen electron"], "pt", 16, 20, 100, 5, "[GeV]", [gen_electrons.pt.compute(),] * 2, [RecoElectronsFromGen.pt.compute(), RecoPhotonsFromGen.pt.compute()], 0, file)
#makeEffPlot("e", "gammacomp_gammaMatchedToEMatchedToGenE", ["Electrons matched to dis gen electrons", "Electron[Photon_electronIdx] which are matched to a dis gen electron"], "pt", 16, 20, 100, 5, "[GeV]", [gen_electrons.pt.compute(),] * 2, [RecoElectronsFromGen.pt.compute(), phelectrons.pt.compute()], 0, file)
#makeEffPlot("photons", "gammaConversionTrack", ["#gamma associated to e associated to same #gamma",  "#gamma that weren't associated to e"] , "pt", 20, 0, 100, 5, "[GeV]", [photons_pt,] * 2, [phelphotons_pt, nonphelphotons_pt], 0, file)
#makeEffPlot("e", "gammacomp_matchedgam", ["Electrons", "LowPtElectrons", "Photons"], "hoe", 0, 1, 20, 0.05, "", [gen_electrons.hoe.compute(),] * 3, [RecoElectronsFromGen.eta.compute(), RecoLowPtElecFromGen.pt.compute(), RecoPhotonsFromGen.pt.compute()], 0, file)
#makeEffPlot("photons", "e_not_photon", ["#gamma associated to e associated to same #gamma",  "#gamma that weren't associated to e"] , "pt", 20, 0, 100, 5, "[GeV]", [photons_pt,] * 2, [phelphotons_pt, nonphelphotons_pt], 0, file)
#makeEffPlot("photons", "e_not_photon", ["#gamma associated to e associated to same #gamma",  "#gamma that weren't associated to e"] , "#eta", 24, -2.4, 2.4, 0.2, "", [photons_eta,] * 2, [phelphotons_eta, nonphelphotons_eta], 0, file) 
#makeEffPlot("photons", "e_not_photon", ["#gamma associated to e associated to same #gamma",  "#gamma that weren't associated to e"] , "hoe", 20, 0, 1, 0.05, "", [photons_hoe,] * 2, [phelphotons_hoe, nonphelphotons_hoe], 0, file) 
#makeEffPlot("photons", "gammaComp_pixelSeed", ["Electrons",  "LowPtElectrons", "Photons with pixelSeed"] , "pt", 16, 20, 100, 5, "[GeV]", [gen_electrons.pt.compute(),] * 3, [RecoElectronsFromGen.pt.compute(), RecoLowPtElecFromGen.pt.compute(), RecoPhotonsFromGen.pt.compute()], 0, file)
#makeEffPlot("e", "gammacomp_matchedgam", ["Electrons", "LowPtElectrons", "Photons"], "hoe", 0, 1, 20, 0.05, "", [gen_electrons.hoe.compute(),] * 3, [RecoElectronsFromGen.eta.compute(), RecoLowPtElecFromGen.pt.compute(), RecoPhotonsFromGen.pt.compute()], 0, file)
#makeEffPlot("e", "PFElectrons", ["Electrons matched to dis gen electrons", "PF Electrons matched to dis gen electrons"], "pt", 16, 20, 100, 5, "[GeV]", [gen_electrons.pt.compute(),] * 2, [RecoElectronsFromGen.pt.compute(), RecoPFElectronsFromGen.pt.compute()], 0, file)
makeEffPlot("e", "PFElectrons", ["Electrons matched to dis gen electrons", "PF Electrons matched to dis gen electrons"], "dxy", 30, 0, 15, 0.5, "[cm]", [gen_electrons.dxy.compute(),] * 2, [RecoElectronsFromGen.dxy.compute(), RecoPFElectronsFromGen.dxy.compute()], 0, file)
makeEffPlot("e", "PFElectrons", ["Electrons matched to dis gen electrons", "PF Electrons matched to dis gen electrons"], "lxy", 30, 0, 15, 0.5, "[cm]", [gen_electrons.lxy.compute(),] * 2, [RecoElectronsFromGen.lxy.compute(), RecoPFElectronsFromGen.lxy.compute()], 0, file)
