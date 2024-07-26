import numpy as np
import awkward as ak
import ROOT 
import math
import scipy
import array
import pandas as pd
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from leptonPlot import *
import matplotlib  as mpl
from  matplotlib import pyplot as plt

file = "Staus_M_100_100mm_13p6TeV_Run3Summer22_lpcdisptau_NanoAOD_ExtraDisMuonBranches.root"

events = NanoEventsFactory.from_root({file:"Events"}).events()

gpart = events.GenPart
gvistau = events.GenVisTau
electrons = events.Electron
muons = events.DisMuon
taus = events.Tau
jets = events.Jet

gvtx = events.GenVtx

gpart["dxy"] = (gpart.vertexY - gvtx.y) * np.cos(gpart.phi) - (gpart.vertexX - gvtx.x) * np.sin(gpart.phi)


staus = gpart[(abs(gpart.pdgId) == 1000015) & (gpart.hasFlags("isLastCopy"))]
staus_taus = staus.distinctChildren[(abs(staus.distinctChildren.pdgId) == 15) & (staus.distinctChildren.hasFlags("isLastCopy"))]
gen_mus = staus_taus.distinctChildren[(abs(staus_taus.distinctChildren.pdgId) == 13)]
gen_electrons = staus_taus.distinctChildren[(abs(staus_taus.distinctChildren.pdgId) == 11)]

gen_mus["dxy"] = (gen_mus.vertexY - gvtx.y) * np.cos(gen_mus.phi) - (gen_mus.vertexX - gvtx.x) * np.sin(gen_mus.phi)
gen_electrons["dxy"] = (gen_electrons.vertexY - gvtx.y) * np.cos(gen_electrons.phi) - (gen_electrons.vertexX - gvtx.x) * np.sin(gen_electrons.phi)


gen_mus["lxy"] = np.sqrt((gen_mus.distinctParent.vertexX - ak.firsts(staus.vertexX)) ** 2 + (gen_mus.distinctParent.vertexY - ak.firsts(staus.vertexY)) ** 2)
gen_electrons["lxy"] = np.sqrt((gen_electrons.distinctParent.vertexX - ak.firsts(staus.vertexX)) ** 2 + (gen_electrons.distinctParent.vertexY - ak.firsts(staus.vertexY)) ** 2)  


semiLep_mu_evt = ak.sum((abs(gen_mus.pdgId) == 13), axis = -2) == 1
semiLep_el_evt = ak.sum((abs(gen_mus.pdgId) == 11), axis = -2) == 1
diTau_evt_two = ak.num(gvistau) == 2
diTau_evt_one = (ak.num(gvistau) == 1) & (ak.sum((abs(gen_mus.pdgId) == 13), axis = -2) == 0) & (ak.sum((abs(gen_mus.pdgId) == 11), axis = -2) == 0)   

diTau_evt = (diTau_evt_one) | (diTau_evt_two)

diTauTrig_35 = events.HLT.DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1 == 1
diTauTrig_40 = events.HLT.DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1 == 1

semiLepTrigMu_30 = events.HLT.IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1 == 1
semiLepTrigMu_35 = events.HLT.IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1 == 1

semiLepTrigEl_30 = events.HLT.Ele30_WPTight_Gsf == 1


gen_mus = gen_mus[(gen_mus.pt > GenPtMin) & (abs(gen_mus.eta) < GenEtaMax)]
gen_electrons = gen_electrons[(gen_electrons.pt > GenPtMin) & (abs(gen_electrons.eta) < GenEtaMax)]
gen_taus = gvistau[(gvistau.pt > GenPtMin) & (abs(gvistau.eta) < GenEtaMax)]

gen_taus["dxy"] = (ak.firsts(gpart[gen_taus.genPartIdxMother.compute()].distinctChildren.vertexY, axis = 2) - gvtx.y) * np.cos(gpart[gen_taus.genPartIdxMother.compute()].phi) - (ak.firsts(gpart[gen_taus.genPartIdxMother.compute()].distinctChildren.vertexX, axis = 2) - gvtx.x) * np.sin(gpart[gen_taus.genPartIdxMother.compute()].phi)
gen_taus["lxy"] = np.sqrt((gpart[gen_taus.genPartIdxMother.compute()].vertexX - ak.firsts(staus.vertexX)) ** 2 + (gpart[gen_taus.genPartIdxMother.compute()].vertexY- ak.firsts(staus.vertexY)) ** 2) 

RecoMuonsFromGen = gpart[muons.genPartIdx.compute()]
RecoMuonsFromGen = RecoMuonsFromGen[abs(RecoMuonsFromGen.pdgId) == 13]
RecoMuonsFromGen = RecoMuonsFromGen[(abs(RecoMuonsFromGen.distinctParent.distinctParent.pdgId) == 1000015)]
RecoMuonsFromGen = RecoMuonsFromGen[(RecoMuonsFromGen.pt > GenPtMin) & (abs(RecoMuonsFromGen.eta) < GenEtaMax)]
RecoMuonsFromGen["dxy"] = (RecoMuonsFromGen.vertexY - gvtx.y) * np.cos(RecoMuonsFromGen.phi) - (RecoMuonsFromGen.vertexX - gvtx.x) * np.sin(RecoMuonsFromGen.phi)
RecoMuonsFromGen["lxy"] = np.sqrt((RecoMuonsFromGen.distinctParent.vertexY - gvtx.y) ** 2 + (RecoMuonsFromGen.distinctParent.vertexX- gvtx.x) ** 2)

RecoElectronsFromGen = electrons[(abs(electrons.matched_gen.distinctParent.distinctParent.pdgId) == 1000015)]
RecoElectronsFromGen = RecoElectronsFromGen.matched_gen[(RecoElectronsFromGen.matched_gen.pt > GenPtMin) & (abs(RecoElectronsFromGen.matched_gen.eta) < GenEtaMax)]
RecoElectronsFromGen["dxy"] = (RecoElectronsFromGen.vertexY - gvtx.y) * np.cos(RecoElectronsFromGen.phi) - (RecoElectronsFromGen.vertexX - gvtx.x) * np.sin(RecoElectronsFromGen.phi)
RecoElectronsFromGen["lxy"] = np.sqrt((RecoElectronsFromGen.distinctParent.vertexY - gvtx.y) ** 2 + (RecoElectronsFromGen.distinctParent.vertexX- gvtx.x) ** 2)

jets = jets[(jets.pt > 20) & (abs(jets.eta) < 2.4) & (jets.genJetIdx >= 0) & (jets.genJetIdx < ak.num(events.GenJet.pt))]
RecoTausFromGen = jets.nearest(gen_taus, threshold = 0.3)
RecoTausFromGen = RecoTausFromGen[(RecoTausFromGen.pt > GenPtMin) & (abs(RecoTausFromGen.eta) < GenEtaMax)]

genMu  = gen_mus[semiLep_mu_evt]
genEl  = gen_electrons[semiLep_el_evt]
genTauMu = gen_taus[semiLep_mu_evt]
genTauEl = gen_taus[semiLep_el_evt]

genMu30 = gen_mus[semiLep_mu_evt & semiLepTrigMu_30]
genMu35 = gen_mus[semiLep_mu_evt & semiLepTrigMu_35]

recoMu30 = RecoMuonsFromGen[semiLep_mu_evt & semiLepTrigMu_30]
recoMu35 = RecoMuonsFromGen[semiLep_mu_evt & semiLepTrigMu_35]

genTauMu30  = gen_taus[semiLep_mu_evt & semiLepTrigMu_30]
recoTauMu30 = RecoTausFromGen[semiLep_mu_evt & semiLepTrigMu_30]


genTauMu35  = gen_taus[semiLep_mu_evt & semiLepTrigMu_35]
recoTauMu35 = RecoTausFromGen[semiLep_mu_evt & semiLepTrigMu_35]

genEl30  = gen_electrons[semiLep_el_evt & semiLepTrigEl_30]
recoEl30 = RecoElectronsFromGen[semiLep_el_evt & semiLepTrigEl_30]

genTauEl30  = gen_taus[semiLep_el_evt & semiLepTrigEl_30]
recoTauEl30 = RecoTausFromGen[semiLep_el_evt & semiLepTrigEl_30]

genDiTau = gen_taus[diTau_evt]
genDiTau35  = gen_taus[diTau_evt & diTauTrig_35]
recoDiTau35 = RecoTausFromGen[diTau_evt & diTauTrig_35] 

genDiTau40  = gen_taus[diTau_evt & diTauTrig_40]
recoDiTau40 = RecoTausFromGen[diTau_evt & diTauTrig_40] 


#makeEffPlot("mutau", "crosstaumu_30", ["mu reco eff", "tau reco eff"], "pt", 16, 20, 100, 5, "[GeV]", [genMu30.pt.compute(), genTauMu30.pt.compute()], [recoMu30.pt.compute(), recoTauMu30.pt.compute()], 0, "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1")
#makeEffPlot("mutau", "crosstaumu_35", ["mu reco eff", "tau reco eff"], "pt", 16, 20, 100, 5, "[GeV]", [genMu35.pt.compute(), genTauMu35.pt.compute()], [recoMu35.pt.compute(), recoTauMu35.pt.compute()], 0, "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1")
#makeEffPlot("etau", "crosstaue_30", ["e reco eff", "tau reco eff"], "pt", 16, 20, 100, 5, "[GeV]", [genEl30.pt.compute(), genTauEl30.pt.compute()], [recoEl30.pt.compute(), recoTauEl30.pt.compute()], 0, "HLT_Ele30_WPTight_Gsf")
#makeEffPlot("ditau", "ditau_35", ["tau reco eff"], "pt", 16, 20, 100, 5, "[GeV]", [genDiTau35.pt.compute()], [recoDiTau35.pt.compute()], 0, "HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1") 
#makeEffPlot("ditau", "ditau_40", ["tau reco eff"], "pt", 16, 20, 100, 5, "[GeV]", [genDiTau40.pt.compute()], [recoDiTau40.pt.compute()], 0, "HLT_DoubleMediumDeepTauPFTauHPS40_L2NN_eta2p1") 

makeEffPlot("mu", "triggeff", ["HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1"], "pt", 16, 20, 100, 5, "[GeV]", [genMu.pt.compute(), genMu.pt.compute()], [recoMu30.pt.compute(), recoMu35.pt.compute()], 0, file)
makeEffPlot("e", "triggeff", ["HLT_Ele30_WPTight_Gsf"], "pt", 16, 20, 100, 5, "[GeV]", [genEl.pt.compute()], [recoEl30.pt.compute()], 0, file)
#makeEffPlot("tau", "triggeff", ["HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1", "HLT_Ele30_WPTight_Gsf", "HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1", "HLT_DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1"], "pt", 16, 20, 100, 5, "[GeV]", [genTauMu.pt.compute(), genTauMu.pt.compute(), genTauEl.pt.compute(), genDiTau.pt.compute(), genDiTau.pt.compute()], [recoTauMu30.pt.compute(), recoTauMu35.pt.compute(), recoTauEl30.pt.compute(), recoDiTau35.pt.compute(), recoDiTau40.pt.compute()], 0, file] 
makeEffPlot("tau", "triggeff", ["HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1", "HLT_Ele30_WPTight_Gsf", "HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1", "HLT_DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1"], "pt", 16, 20, 100, 5, "[GeV]", [genTauMu.pt.compute(), genTauMu.pt.compute(), genTauEl.pt.compute(), genDiTau.pt.compute(), genDiTau.pt.compute()], [recoTauMu30.pt.compute(), recoTauMu35.pt.compute(), recoTauEl30.pt.compute(), recoDiTau35.pt.compute(), recoDiTau40.pt.compute()], 0, file)

makeEffPlot("mu", "triggeff", ["HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1"], "eta", 24, -2.4, 2.4, 0.2, "", [genMu.eta.compute(), genMu.eta.compute()], [recoMu30.eta.compute(), recoMu35.eta.compute()], 0, file)
makeEffPlot("e", "triggeff", ["HLT_Ele30_WPTight_Gsf"], "eta", 24, -2.4, 2.4, 0.2, "", [genEl.eta.compute()], [recoEl30.eta.compute()], 0, file)
makeEffPlot("tau", "triggeff", ["HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1", "HLT_Ele30_WPTight_Gsf", "HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1", "HLT_DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1"], "eta", 24, -2.4, 2.4, 0.2, "", [genTauMu.eta.compute(), genTauMu.eta.compute(), genTauEl.eta.compute(), genDiTau.eta.compute(), genDiTau.eta.compute()], [recoTauMu30.eta.compute(), recoTauMu35.eta.compute(), recoTauEl30.eta.compute(), recoDiTau35.eta.compute(), recoDiTau40.eta.compute()], 0, file) 
#
makeEffPlot("mu", "triggeff", ["HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1"], "dxy", 20, 0, 2, 0.1, "[cm]", [genMu30.dxy.compute(), genMu35.dxy.compute()], [recoMu30.dxy.compute(), recoMu35.dxy.compute()], 0, file)
#makeEffPlot("e", "triggeff", ["HLT_Ele30_WPTight_Gsf"], "dxy", 15, 0, 15, 1, "[cm]", [genEl30.dxy.compute()], [recoEl30.dxy.compute()], 0, file)
makeEffPlot("tau", "triggeff", ["HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1", "HLT_Ele30_WPTight_Gsf", "HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1", "HLT_DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1"], "dxy", 15, 0, 15, 1, "", [genTauMu.dxy.compute(), genTauMu.dxy.compute(), genTauEl.dxy.compute(), genDiTau.dxy.compute(), genDiTau.dxy.compute()], [recoTauMu30.dxy.compute(), recoTauMu35.dxy.compute(), recoTauEl30.dxy.compute(), recoDiTau35.dxy.compute(), recoDiTau40.dxy.compute()], 0, file) 
#
#makeEffPlot("mu", "triggeff", ["HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1"], "lxy", 16, 20, 100, 5, "[cm]", [genMu30.lxy.compute(), genMu35.lxy.compute()], [recoMu30.lxy.compute(), recoMu35.lxy.compute()], 0, file)
#makeEffPlot("e", "triggeff", ["HLT_Ele30_WPTight_Gsf"], "lxy", 15, 0, 15, 1, "[GeV]", [genEl30.lxy.compute()], [recoEl30.lxy.compute()], 0, file)
makeEffPlot("tau", "triggeff", ["HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1", "HLT_Ele30_WPTight_Gsf", "HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1", "HLT_DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1"], "lxy", 15, 0, 15, 1, "", [genTauMu.lxy.compute(), genTauMu.lxy.compute(), genTauEl.lxy.compute(), genDiTau.lxy.compute(), genDiTau.lxy.compute()], [recoTauMu30.lxy.compute(), recoTauMu35.lxy.compute(), recoTauEl30.lxy.compute(), recoDiTau35.lxy.compute(), recoDiTau40.lxy.compute()], 0, file) 
