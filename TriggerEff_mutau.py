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

MET_bins = [  0, 80, 120, 150, 200, 300, 400, 800 ] 
MET_bins = array.array('d', MET_bins)

gpart = events.GenPart
gvistau = events.GenVisTau
muons = events.DisMuon
taus = events.Tau
jets = events.Jet
met = events.MET
gmet = events.GenMET
gvtx = events.GenVtx

gpart["dxy"] = (gpart.vertexY - gvtx.y) * np.cos(gpart.phi) - (gpart.vertexX - gvtx.x) * np.sin(gpart.phi)


staus = gpart[(abs(gpart.pdgId) == 1000015) & (gpart.hasFlags("isLastCopy"))]
staus_taus = staus.distinctChildren[(abs(staus.distinctChildren.pdgId) == 15) & (staus.distinctChildren.hasFlags("isLastCopy"))]
gen_mus = staus_taus.distinctChildren[(abs(staus_taus.distinctChildren.pdgId) == 13)]

gen_mus = gen_mus[(gen_mus.pt > GenPtMin) & (abs(gen_mus.eta) < GenEtaMax)]
gen_taus = gvistau[(gvistau.pt > GenPtMin) & (abs(gvistau.eta) < GenEtaMax)]

gen_mus["dxy"] = (gen_mus.vertexY - gvtx.y) * np.cos(gen_mus.phi) - (gen_mus.vertexX - gvtx.x) * np.sin(gen_mus.phi)
gen_taus["dxy"] = (ak.firsts(gpart[gen_taus.genPartIdxMother.compute()].distinctChildren.vertexY, axis = 2) - gvtx.y) * np.cos(gpart[gen_taus.genPartIdxMother.compute()].phi) - (ak.firsts(gpart[gen_taus.genPartIdxMother.compute()].distinctChildren.vertexX, axis = 2) - gvtx.x) * np.sin(gpart[gen_taus.genPartIdxMother.compute()].phi)

gen_mus["lxy"] = np.sqrt((gen_mus.distinctParent.vertexX - ak.firsts(staus.vertexX)) ** 2 + (gen_mus.distinctParent.vertexY - ak.firsts(staus.vertexY)) ** 2)
gen_taus["lxy"] = np.sqrt((gpart[gen_taus.genPartIdxMother.compute()].vertexX - ak.firsts(staus.vertexX)) ** 2 + (gpart[gen_taus.genPartIdxMother.compute()].vertexY- ak.firsts(staus.vertexY)) ** 2) 

semiLep_mu_evt = ak.sum((abs(ak.flatten(gen_mus.pdgId.compute(), axis = 3)) == 13), axis = -2) == 1
semiLep_mu_evt = (ak.where(ak.num(semiLep_mu_evt) > 0, True, False) & (ak.num(gen_taus) == 1).compute())

genMET_mu    = gmet[semiLep_mu_evt]


### Cross Flavor Tau-Muons Triggers (Prompt)
semiLepTrigMu_30 = events.HLT.IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1 == 1
semiLepTrigMu_35 = events.HLT.IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1 == 1

### MET Triggers
PFMET_120 = events.HLT.PFMET120_PFMHT120_IDTight == 1
PFMET_130 = events.HLT.PFMET130_PFMHT130_IDTight == 1
PFMET_140 = events.HLT.PFMET140_PFMHT140_IDTight == 1

MET_105 = events.HLT.MET105_IsoTrk50 == 1
MET_120 = events.HLT.MET120_IsoTrk50 == 1

RecoMuonsFromGen = gpart[muons.genPartIdx.compute()]
RecoMuonsFromGen = RecoMuonsFromGen[abs(RecoMuonsFromGen.pdgId) == 13]
RecoMuonsFromGen = RecoMuonsFromGen[(abs(RecoMuonsFromGen.distinctParent.distinctParent.pdgId) == 1000015)]
RecoMuonsFromGen = RecoMuonsFromGen[(RecoMuonsFromGen.pt > GenPtMin) & (abs(RecoMuonsFromGen.eta) < GenEtaMax)]
RecoMuonsFromGen["dxy"] = (RecoMuonsFromGen.vertexY - gvtx.y) * np.cos(RecoMuonsFromGen.phi) - (RecoMuonsFromGen.vertexX - gvtx.x) * np.sin(RecoMuonsFromGen.phi)
RecoMuonsFromGen["lxy"] = np.sqrt((RecoMuonsFromGen.distinctParent.vertexY - gvtx.y) ** 2 + (RecoMuonsFromGen.distinctParent.vertexX- gvtx.x) ** 2)

jets = jets[(jets.pt > 20) & (abs(jets.eta) < 2.4) & (jets.genJetIdx >= 0) & (jets.genJetIdx < ak.num(events.GenJet.pt))]
RecoTausFromGen = jets.nearest(gen_taus, threshold = 0.3)
RecoTausFromGen = RecoTausFromGen[(RecoTausFromGen.pt > GenPtMin) & (abs(RecoTausFromGen.eta) < GenEtaMax)]

recoSemiLep_mu_evt = (ak.num(RecoMuonsFromGen) == 1) & (ak.num(ak.drop_none(RecoTausFromGen)) == 1)

genMu  = gen_mus[semiLep_mu_evt]
recoMu  = gen_mus[semiLep_mu_evt & recoSemiLep_mu_evt.compute()]


### Semilep mu events which pass triggers

#### Gen Events
genMu_PFMET_120        = genMu[PFMET_120]
genMu_PFMET_130        = genMu[PFMET_130]
genMu_PFMET_140        = genMu[PFMET_140]

genMu_MET_105          = genMu[MET_105]
genMu_MET_120          = genMu[MET_120]

genMu_semiLepTrigMu_30 = genMu[semiLepTrigMu_30]
genMu_semiLepTrigMu_35 = genMu[semiLepTrigMu_35]

#### Gen Events that have reco'd mu and tau for mu
recoMu_PFMET_120            = gen_mus[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & PFMET_120.compute()]
recoMu_PFMET_130            = gen_mus[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & PFMET_130.compute()]
recoMu_PFMET_140            = gen_mus[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & PFMET_140.compute()]

recoMET_mu_PFMET_120        = gmet[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & PFMET_120.compute()]
recoMET_mu_PFMET_130        = gmet[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & PFMET_130.compute()]
recoMET_mu_PFMET_140        = gmet[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & PFMET_140.compute()]

recoMu_MET_105              = gen_mus[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & MET_105.compute()]
recoMu_MET_120              = gen_mus[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & MET_120.compute()]

recoMET_mu_MET_105          = gmet[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & MET_105.compute()]
recoMET_mu_MET_120          = gmet[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & MET_120.compute()]

recoMu_semiLepTrigMu_30     = gen_mus[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & semiLepTrigMu_30.compute()]
recoMu_semiLepTrigMu_35     = gen_mus[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & semiLepTrigMu_35.compute()]

recoMET_mu_semiLepTrigMu_30 = gmet[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & semiLepTrigMu_30.compute()] 
recoMET_mu_semiLepTrigMu_35 = gmet[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & semiLepTrigMu_35.compute()]

### Trigger Eff vs Pt
#makeEffPlot("mu", "PFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "#mu pt", 16, 20, 100, 5, "[GeV]", [recoMu.pt.compute(), recoMu.pt.compute(), recoMu.pt.compute()], [recoMu_PFMET_120.pt.compute(), recoMu_PFMET_130.pt.compute(), recoMu_PFMET_140.pt.compute()], 0, file)
#makeEffPlot("mu", "MET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "#mu pt", 16, 20, 100, 5, "[GeV]", [recoMu.pt.compute(), recoMu.pt.compute()], [recoMu_MET_105.pt.compute(), recoMu_MET_120.pt.compute()], 0, file)
#makeEffPlot("mu", "XFLAV", ["HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1"], "#mu pt", 16, 20, 100, 5, "[GeV]", [recoMu.pt.compute(), recoMu.pt.compute()], [recoMu_semiLepTrigMu_30.pt.compute(), recoMu_semiLepTrigMu_35.pt.compute()], 0, file) 
#makeEffPlot("mu", "ALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1"],  "#mu pt", 16, 20, 100, 5, "[GeV]", [recoMu.pt.compute(),]*3, [recoMu_PFMET_120.pt.compute(), recoMu_MET_105.pt.compute(), recoMu_semiLepTrigMu_30.pt.compute()], 0, file)

#### Trigger Eff vs Eta
#makeEffPlot("mu", "PFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "#mu #eta", 24, -2.4, 2.4, 0.2, "", [recoMu.eta.compute(), recoMu.eta.compute(), recoMu.eta.compute()], [recoMu_PFMET_120.eta.compute(), recoMu_PFMET_130.eta.compute(), recoMu_PFMET_140.eta.compute()], 0, file)
#makeEffPlot("mu", "MET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "#mu #eta", 24, -2.4, 2.4, 0.2, "", [recoMu.eta.compute(), recoMu.eta.compute()], [recoMu_MET_105.eta.compute(), recoMu_MET_120.eta.compute()], 0, file)
#makeEffPlot("mu", "XFLAV", ["HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1"], "#mu #eta", 24, -2.4, 2.4, 0.2, "", [recoMu.eta.compute(), recoMu.eta.compute()], [recoMu_semiLepTrigMu_30.eta.compute(), recoMu_semiLepTrigMu_35.eta.compute()], 0, file) 
#makeEffPlot("mu", "ALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1"],  "#mu #eta", 24, -2.4, 2.4, 0.2, "", [recoMu.eta.compute(),]*3, [recoMu_PFMET_120.eta.compute(), recoMu_MET_105.eta.compute(), recoMu_semiLepTrigMu_30.eta.compute()], 0, file)

#### Trigger Eff vs dxy
#makeEffPlot("mu", "PFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "#mu d_{xy}", 15, 0, 15, 1, "[cm]", [recoMu.dxy.compute(), recoMu.dxy.compute(), recoMu.dxy.compute()], [recoMu_PFMET_120.dxy.compute(), recoMu_PFMET_130.dxy.compute(), recoMu_PFMET_140.dxy.compute()], 0, file)
#makeEffPlot("mu", "MET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "#mu d_{xy}", 15, 0, 15, 1, "[cm]", [recoMu.dxy.compute(), recoMu.dxy.compute()], [recoMu_MET_105.dxy.compute(), recoMu_MET_120.dxy.compute()], 0, file)
#makeEffPlot("mu", "XFLAV", ["HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1"], "#mu d_{xy}", 15, 0, 15, 1, "[cm]", [recoMu.dxy.compute(), recoMu.dxy.compute()], [recoMu_semiLepTrigMu_30.dxy.compute(), recoMu_semiLepTrigMu_35.dxy.compute()], 0, file) 
#makeEffPlot("mu", "ALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1"],  "#mu d_{xy}", 15, 0, 15, 1, "[cm]", [recoMu.dxy.compute(),]*3, [recoMu_PFMET_120.dxy.compute(), recoMu_MET_105.dxy.compute(), recoMu_semiLepTrigMu_30.dxy.compute()], 0, file)

#### Trigger Eff vs lxy
#makeEffPlot("mu", "PFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "#mu l_{xy}", 15, 0, 15, 1, "[cm]", [recoMu.lxy.compute(), recoMu.lxy.compute(), recoMu.lxy.compute()], [recoMu_PFMET_120.lxy.compute(), recoMu_PFMET_130.lxy.compute(), recoMu_PFMET_140.lxy.compute()], 0, file)
#makeEffPlot("mu", "MET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "#mu l_{xy}", 15, 0, 15, 1, "[cm]", [recoMu.lxy.compute(), recoMu.lxy.compute()], [recoMu_MET_105.lxy.compute(), recoMu_MET_120.lxy.compute()], 0, file)
#makeEffPlot("mu", "XFLAV", ["HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1"], "#mu l_{xy}", 15, 0, 15, 1, "[cm]", [recoMu.lxy.compute(), recoMu.lxy.compute()], [recoMu_semiLepTrigMu_30.lxy.compute(), recoMu_semiLepTrigMu_35.lxy.compute()], 0, file) 
#makeEffPlot("mu", "ALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1"],  "#mu l_{xy}", 15, 0, 15, 1, "[cm]", [recoMu.lxy.compute(),]*3, [recoMu_PFMET_120.lxy.compute(), recoMu_MET_105.lxy.compute(), recoMu_semiLepTrigMu_30.lxy.compute()], 0, file)

## Trigger Eff vs MET pt
makeEffPlot_varBin("mu", "PFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [genMET_mu.pt.compute(),]*3, [recoMET_mu_PFMET_120.pt.compute(), recoMET_mu_PFMET_130.pt.compute(), recoMET_mu_PFMET_140.pt.compute()], 0, file)
makeEffPlot_varBin("mu", "MET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [genMET_mu.pt.compute(),]*2, [recoMET_mu_MET_105.pt.compute(), recoMET_mu_MET_120.pt.compute()], 0, file)
makeEffPlot_varBin("mu", "XFLAV", ["HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1"], "MET pT", len(MET_bins) - 1, MET_bins,"[GeV]", [genMET_mu.pt.compute(),]*2, [recoMET_mu_semiLepTrigMu_30.pt.compute(), recoMET_mu_semiLepTrigMu_35.pt.compute()], 0, file)
makeEffPlot_varBin("mu", "ALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1"], "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [genMET_mu.pt.compute(),]*3, [recoMET_mu_PFMET_120.pt.compute(), recoMET_mu_MET_105.pt.compute(), recoMET_mu_semiLepTrigMu_30.pt.compute()], 0, file)
