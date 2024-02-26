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
muons = events.DisMuon
taus = events.Tau
jets = events.Jet

gvtx = events.GenVtx

gpart["dxy"] = (gpart.vertexY - gvtx.y) * np.cos(gpart.phi) - (gpart.vertexX - gvtx.x) * np.sin(gpart.phi)

semiLep_evt = ak.num(gvistau) == 1
diTau_evt = ak.num(gvistau) == 2

print(muons.genPartIdx.compute())
print(gpart[muons.genPartIdx.compute()].pt.compute())

staus = gpart[(abs(gpart.pdgId) == 1000015) & (gpart.hasFlags("isLastCopy"))]
staus_taus = staus.distinctChildren[(abs(staus.distinctChildren.pdgId) == 15) & (staus.distinctChildren.hasFlags("isLastCopy"))]

gen_mus = staus_taus.distinctChildren[(abs(staus_taus.distinctChildren.pdgId) == 13)]
gen_electrons = staus_taus.distinctChildren[(abs(staus_taus.distinctChildren.pdgId) == 11)]

gen_mus["dxy"] = (gen_mus.vertexY - gvtx.y) * np.cos(gen_mus.phi) - (gen_mus.vertexX - gvtx.x) * np.sin(gen_mus.phi)
gen_electrons["dxy"] = (gen_electrons.vertexY - gvtx.y) * np.cos(gen_electrons.phi) - (gen_electrons.vertexX - gvtx.x) * np.sin(gen_electrons.phi)


gen_mus["lxy"] = np.sqrt((gen_mus.distinctParent.vertexX - staus.vertexX) ** 2 + (gen_mus.distinctParent.vertexY - staus.vertexY) ** 2)
gen_electrons["lxy"] = np.sqrt((gen_electrons.distinctParent.vertexX - staus.vertexX) ** 2 + (gen_electrons.distinctParent.vertexY - staus.vertexY) ** 2)  


semiLep_mu_evt = ak.num(ak.flatten(gen_mus, axis = 2)) == 1
semiLep_el_evt = ak.num(gen_electrons) == 1

diTauTrig_35 = events.HLT.DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1 == 1
diTauTrig_40 = events.HLT.DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1 == 1

semiLepTrigMu_30 = events.HLT.IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1 == 1
semiLepTrigMu_35 = events.HLT.IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1 == 1

semiLepTrigEl_30 = events.HLT.Ele30_WPTight_Gsf == 1


gen_mus = gen_mus[(gen_mus.pt > GenPtMin) & (abs(gen_mus.eta) < GenEtaMax)]
gen_electrons = gen_electrons[(gen_electrons.pt > GenPtMin) & (abs(gen_electrons.eta) < GenEtaMax)]
gen_taus = gvistau[(gvistau.pt > GenPtMin) & (abs(gvistau.eta) < GenEtaMax)]

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

RecoTausFromGen = jets.nearest(gvistau, threshold = VisTauDR)
RecoTausFromGen = RecoTausFromGen[(RecoTausFromGen.pt > GenPtMin) & (abs(RecoTausFromGen.eta) < GenEtaMax)]

genMu30  = gen_mus[semiLep_evt & semiLepTrigMu_30]
recoMu30 = RecoMuonsFromGen[semiLep_evt & semiLepTrigMu_30]

genTauMu30  = gen_taus[semiLep_evt & semiLepTrigMu_30]
recoTauMu30 = RecoTausFromGen[semiLep_evt & semiLepTrigMu_30]

genMu35  = gen_mus[semiLep_evt & semiLepTrigMu_35]
recoMu35 = RecoMuonsFromGen[semiLep_evt & semiLepTrigMu_35]

genTauMu35  = gen_taus[semiLep_evt & semiLepTrigMu_35]
recoTauMu35 = RecoTausFromGen[semiLep_evt & semiLepTrigMu_35]

genEl30  = gen_electrons[semiLep_evt & semiLepTrigEl_30]
recoEl30 = RecoElectronsFromGen[semiLep_evt & semiLepTrigEl_30]

genTauEl30  = gen_taus[semiLep_evt & semiLepTrigEl_30]
recoTauEl30 = RecoTausFromGen[semiLep_evt & semiLepTrigEl_30]

genDiTau35  = gen_taus[diTau_evt & diTauTrig_35]
recoDiTau35 = RecoTausFromGen[diTau_evt & diTauTrig_35] 

genDiTau40  = gen_taus[diTau_evt & diTauTrig_40]
recoDiTau40 = RecoTausFromGen[diTau_evt & diTauTrig_40] 


#makeEffPlot("mutau", "crosstaumu_30", ["mu reco eff", "tau reco eff"], "pt", 16, 20, 100, 5, "[GeV]", [genMu30.pt.compute(), genTauMu30.pt.compute()], [recoMu30.pt.compute(), recoTauMu30.pt.compute()], 0, "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1")
#makeEffPlot("mutau", "crosstaumu_35", ["mu reco eff", "tau reco eff"], "pt", 16, 20, 100, 5, "[GeV]", [genMu35.pt.compute(), genTauMu35.pt.compute()], [recoMu35.pt.compute(), recoTauMu35.pt.compute()], 0, "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1")
#makeEffPlot("etau", "crosstaue_30", ["e reco eff", "tau reco eff"], "pt", 16, 20, 100, 5, "[GeV]", [genEl30.pt.compute(), genTauEl30.pt.compute()], [recoEl30.pt.compute(), recoTauEl30.pt.compute()], 0, "HLT_Ele30_WPTight_Gsf")
#makeEffPlot("ditau", "ditau_35", ["tau reco eff"], "pt", 16, 20, 100, 5, "[GeV]", [genDiTau35.pt.compute()], [recoDiTau35.pt.compute()], 0, "HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1") 
#makeEffPlot("ditau", "ditau_40", ["tau reco eff"], "pt", 16, 20, 100, 5, "[GeV]", [genDiTau40.pt.compute()], [recoDiTau40.pt.compute()], 0, "HLT_DoubleMediumDeepTauPFTauHPS40_L2NN_eta2p1") 

makeEffPlot("mu", "triggeff", ["HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1"], "pt", 16, 20, 100, 5, "[GeV]", [genMu30.pt.compute(), genMu35.pt.compute()], [recoMu30.pt.compute(), recoMu35.pt.compute()], 0, file)
makeEffPlot("e", "triggeff", ["HLT_Ele30_WPTight_Gsf"], "pt", 16, 20, 100, 5, "[GeV]", [genEl30.pt.compute()], [recoEl30.pt.compute()], 0, file)

makeEffPlot("mu", "triggeff", ["HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1"], "eta", 24, -2.4, 2.4, 0.2, "", [genMu30.eta.compute(), genMu35.eta.compute()], [recoMu30.eta.compute(), recoMu35.eta.compute()], 0, file)
makeEffPlot("e", "triggeff", ["HLT_Ele30_WPTight_Gsf"], "eta", 16, 20, 100, 5, "", [genEl30.eta.compute()], [recoEl30.eta.compute()], 0, file)

makeEffPlot("mu", "triggeff", ["HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1"], "dxy", 15, 0, 15, 1, "[cm]", [genMu30.dxy.compute(), genMu35.dxy.compute()], [recoMu30.dxy.compute(), recoMu35.dxy.compute()], 0, file)
makeEffPlot("e", "triggeff", ["HLT_Ele30_WPTight_Gsf"], "dxy", 15, 0, 15, 1, "[cm]", [genEl30.dxy.compute()], [recoEl30.dxy.compute()], 0, file)

makeEffPlot("mu", "triggeff", ["HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1"], "lxy", 16, 20, 100, 5, "[cm]", [genMu30.lxy.compute(), genMu35.lxy.compute()], [recoMu30.lxy.compute(), recoMu35.lxy.compute()], 0, file)
makeEffPlot("e", "triggeff", ["HLT_Ele30_WPTight_Gsf"], "lxy", 15, 0, 15, 1, "[GeV]", [genEl30.lxy.compute()], [recoEl30.lxy.compute()], 0, file)
