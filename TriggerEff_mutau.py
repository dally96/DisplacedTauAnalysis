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
dxy_bins = [*np.linspace(0, 5, 6)] + [*np.linspace(5, 30, 6)] 
MET_bins = array.array('d', MET_bins)
dxy_bins = array.array('d', dxy_bins)

gpart = events.GenPart
gvistau = events.GenVisTau
muons = events.DisMuon
taus = events.Tau
jets = events.Jet
met = events.MET
gmet = events.GenMET
gvtx = events.GenVtx

gpart["dxy"] = abs((gpart.vertexY - gvtx.y) * np.cos(gpart.phi) - (gpart.vertexX - gvtx.x) * np.sin(gpart.phi))


staus = gpart[(abs(gpart.pdgId) == 1000015) & (gpart.hasFlags("isLastCopy"))]
staus_taus = staus.distinctChildren[(abs(staus.distinctChildren.pdgId) == 15) & (staus.distinctChildren.hasFlags("isLastCopy"))]
gen_mus = staus_taus.distinctChildren[(abs(staus_taus.distinctChildren.pdgId) == 13)]

gen_mus = gen_mus[(gen_mus.pt > GenPtMin) & (abs(gen_mus.eta) < GenEtaMax)]
gen_taus = gvistau[(gvistau.pt > GenPtMin) & (abs(gvistau.eta) < GenEtaMax)]

gen_mus["dxy"] = abs((gen_mus.vertexY - gvtx.y) * np.cos(gen_mus.phi) - (gen_mus.vertexX - gvtx.x) * np.sin(gen_mus.phi))
gen_taus["dxy"] = abs((ak.firsts(gpart[gen_taus.genPartIdxMother.compute()].distinctChildren.vertexY, axis = 2) - gvtx.y) * np.cos(gpart[gen_taus.genPartIdxMother.compute()].phi) - (ak.firsts(gpart[gen_taus.genPartIdxMother.compute()].distinctChildren.vertexX, axis = 2) - gvtx.x) * np.sin(gpart[gen_taus.genPartIdxMother.compute()].phi))

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

PFMETNoMu_110_FilterHF = events.HLT.PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF == 1
PFMETNoMu_120 = events.HLT.PFMETNoMu120_PFMHTNoMu120_IDTight == 1 
PFMETNoMu_130 = events.HLT.PFMETNoMu130_PFMHTNoMu130_IDTight == 1
PFMETNoMu_140 = events.HLT.PFMETNoMu140_PFMHTNoMu140_IDTight == 1

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

recoMET_mu = gmet[semiLep_mu_evt & recoSemiLep_mu_evt.compute()]

### Semilep mu events which pass triggers

#### Gen Events
genMu_PFMET_120            = gen_mus[semiLep_mu_evt & PFMET_120.compute()]
genMu_PFMET_130            = gen_mus[semiLep_mu_evt & PFMET_130.compute()]
genMu_PFMET_140            = gen_mus[semiLep_mu_evt & PFMET_140.compute()]

genMET_mu_PFMET_120        = gmet[semiLep_mu_evt & PFMET_120.compute()]
genMET_mu_PFMET_130        = gmet[semiLep_mu_evt & PFMET_130.compute()]
genMET_mu_PFMET_140        = gmet[semiLep_mu_evt & PFMET_140.compute()]

genMu_PFMETNoMu_110_FilterHF   = gen_mus[semiLep_mu_evt & PFMETNoMu_110_FilterHF.compute()]

genMu_PFMETNoMu_120            = gen_mus[semiLep_mu_evt & PFMETNoMu_120.compute()]
genMu_PFMETNoMu_130            = gen_mus[semiLep_mu_evt & PFMETNoMu_130.compute()]
genMu_PFMETNoMu_140            = gen_mus[semiLep_mu_evt & PFMETNoMu_140.compute()]

genMET_mu_PFMETNoMu_110_FilterHF  = gmet[semiLep_mu_evt & PFMETNoMu_110_FilterHF.compute()]

genMET_mu_PFMETNoMu_120        = gmet[semiLep_mu_evt & PFMETNoMu_120.compute()]
genMET_mu_PFMETNoMu_130        = gmet[semiLep_mu_evt & PFMETNoMu_130.compute()]
genMET_mu_PFMETNoMu_140        = gmet[semiLep_mu_evt & PFMETNoMu_140.compute()]

genMu_MET_105              = gen_mus[semiLep_mu_evt & MET_105.compute()]
genMu_MET_120              = gen_mus[semiLep_mu_evt & MET_120.compute()]

genMET_mu_MET_105          = gmet[semiLep_mu_evt & MET_105.compute()]
genMET_mu_MET_120          = gmet[semiLep_mu_evt & MET_120.compute()]

genMu_semiLepTrigMu_30     = gen_mus[semiLep_mu_evt & semiLepTrigMu_30.compute()]
genMu_semiLepTrigMu_35     = gen_mus[semiLep_mu_evt & semiLepTrigMu_35.compute()]

genMET_mu_semiLepTrigMu_30 = gmet[semiLep_mu_evt & semiLepTrigMu_30.compute()] 
genMET_mu_semiLepTrigMu_35 = gmet[semiLep_mu_evt & semiLepTrigMu_35.compute()]

#### Gen Events that have reco'd mu and tau for mu
recoMu_PFMET_120            = gen_mus[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & PFMET_120.compute()]
recoMu_PFMET_130            = gen_mus[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & PFMET_130.compute()]
recoMu_PFMET_140            = gen_mus[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & PFMET_140.compute()]

recoMET_mu_PFMET_120        = gmet[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & PFMET_120.compute()]
recoMET_mu_PFMET_130        = gmet[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & PFMET_130.compute()]
recoMET_mu_PFMET_140        = gmet[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & PFMET_140.compute()]

recoMu_PFMETNoMu_110_FilterHF   = gen_mus[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & PFMETNoMu_110_FilterHF.compute()]

recoMu_PFMETNoMu_120            = gen_mus[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & PFMETNoMu_120.compute()]
recoMu_PFMETNoMu_130            = gen_mus[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & PFMETNoMu_130.compute()]
recoMu_PFMETNoMu_140            = gen_mus[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & PFMETNoMu_140.compute()]

recoMET_mu_PFMETNoMu_110_FilterHF  = gmet[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & PFMETNoMu_110_FilterHF.compute()]

recoMET_mu_PFMETNoMu_120        = gmet[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & PFMETNoMu_120.compute()]
recoMET_mu_PFMETNoMu_130        = gmet[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & PFMETNoMu_130.compute()]
recoMET_mu_PFMETNoMu_140        = gmet[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & PFMETNoMu_140.compute()]

recoMu_MET_105              = gen_mus[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & MET_105.compute()]
recoMu_MET_120              = gen_mus[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & MET_120.compute()]

recoMET_mu_MET_105          = gmet[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & MET_105.compute()]
recoMET_mu_MET_120          = gmet[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & MET_120.compute()]

recoMu_semiLepTrigMu_30     = gen_mus[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & semiLepTrigMu_30.compute()]
recoMu_semiLepTrigMu_35     = gen_mus[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & semiLepTrigMu_35.compute()]

recoMET_mu_semiLepTrigMu_30 = gmet[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & semiLepTrigMu_30.compute()] 
recoMET_mu_semiLepTrigMu_35 = gmet[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & semiLepTrigMu_35.compute()]

triggers = ["PFMET_120",
            "PFMET_130",
            "PFMET_140",
            "MET_105",
            "MET_120",
            "semiLepTrigMu_30",
            "semiLepTrigMu_35",
]

### Fraction of Total Events
#e_trigfrac = ROOT.TEfficiency("e_trigfrac", file.split(".")[0]+";;#varepsilon", len(triggers)+1, -0.5, len(triggers)+0.5)
#shared_trig = int(ak.sum(semiLep_mu_evt & recoSemiLep_mu_evt.compute() & PFMET_120.compute() & PFMET_130.compute() & PFMET_140.compute() & MET_105.compute() & MET_120.compute() & semiLepTrigMu_30.compute() & semiLepTrigMu_35.compute()))
#for i in range(len(triggers) + 1):
#  e_trigfrac.SetTotalEvents(i + 1, int(ak.sum(semiLep_mu_evt & recoSemiLep_mu_evt.compute())))
#e_trigfrac.SetPassedEvents(1, shared_trig - int(ak.sum(semiLep_mu_evt & recoSemiLep_mu_evt.compute() & PFMET_120.compute())))
#e_trigfrac.SetPassedEvents(2, shared_trig - int(ak.sum(semiLep_mu_evt & recoSemiLep_mu_evt.compute() & PFMET_120.compute())))
#e_trigfrac.SetPassedEvents(3, shared_trig - int(ak.sum(semiLep_mu_evt & recoSemiLep_mu_evt.compute() & PFMET_130.compute()))) 
#e_trigfrac.SetPassedEvents(4, shared_trig - int(ak.sum(semiLep_mu_evt & recoSemiLep_mu_evt.compute() & PFMET_140.compute()))) 
#e_trigfrac.SetPassedEvents(5, shared_trig - int(ak.sum(semiLep_mu_evt & recoSemiLep_mu_evt.compute() & MET_105.compute())))  
#e_trigfrac.SetPassedEvents(6, shared_trig - int(ak.sum(semiLep_mu_evt & recoSemiLep_mu_evt.compute() & MET_120.compute())))  
#e_trigfrac.SetPassedEvents(7, shared_trig - int(ak.sum(semiLep_mu_evt & recoSemiLep_mu_evt.compute() & semiLepTrigMu_30.compute()))) 
#e_trigfrac.SetPassedEvents(8, shared_trig - int(ak.sum(semiLep_mu_evt & recoSemiLep_mu_evt.compute() & semiLepTrigMu_35.compute())))  
#                                
#e_trigfrac.Draw()
#can.Update()
#
#for i in range(len(triggers)):
#  e_trigfrac.GetPaintedGraph().GetXaxis().SetBinLabel(i+1, triggers[i])  
#
#e_trigfrac.GetPaintedGraph().GetXaxis().LabelsOption("v")
#e_trigfrac.GetPaintedGraph().GetXaxis().SetLabelOffset(0.1)
#
#can.SaveAs("triggerEff_muTau.pdf")
#can.SaveAs("PNG_plots/triggerEff_muTau.png")



### Trigger Eff vs Pt
#### Reco
makeEffPlot_varBin("mu", "PFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "#mu pt",  len(MET_bins) - 1, MET_bins, "[GeV]", [recoMu.pt.compute(), recoMu.pt.compute(), recoMu.pt.compute()], [recoMu_PFMET_120.pt.compute(), recoMu_PFMET_130.pt.compute(), recoMu_PFMET_140.pt.compute()], 0, file)
makeEffPlot_varBin("mu", "PFMETNoMu", ["HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight", "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight", "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight"], "#mu pt",  len(MET_bins) - 1, MET_bins, "[GeV]", [recoMu.pt.compute(), recoMu.pt.compute(), recoMu.pt.compute(), recoMu.pt.compute()], [recoMu_PFMETNoMu_110_FilterHF.pt.compute(), recoMu_PFMETNoMu_120.pt.compute(), recoMu_PFMETNoMu_130.pt.compute(), recoMu_PFMETNoMu_140.pt.compute()], 0, file)
makeEffPlot_varBin("mu", "MET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "#mu pt", len(MET_bins) - 1, MET_bins, "[GeV]", [recoMu.pt.compute(), recoMu.pt.compute()], [recoMu_MET_105.pt.compute(), recoMu_MET_120.pt.compute()], 0, file)
makeEffPlot_varBin("mu", "XFLAV", ["HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1"], "#mu pt", len(MET_bins) - 1, MET_bins, "[GeV]", [recoMu.pt.compute(), recoMu.pt.compute()], [recoMu_semiLepTrigMu_30.pt.compute(), recoMu_semiLepTrigMu_35.pt.compute()], 0, file) 
makeEffPlot_varBin("mu", "ALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_MET105_IsoTrk50", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1"],  "#mu pt", len(MET_bins) - 1, MET_bins, "[GeV]", [recoMu.pt.compute(),]*4, [recoMu_PFMET_120.pt.compute(), recoMu_PFMETNoMu_110_FilterHF.pt.compute(), recoMu_MET_105.pt.compute(), recoMu_semiLepTrigMu_30.pt.compute()], 0, file)

#### Gen 
#makeEffPlot("mu", "genPFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "#mu pt", 16, 20, 100, 5, "[GeV]", [genMu.pt.compute(), genMu.pt.compute(), genMu.pt.compute()], [genMu_PFMET_120.pt.compute(), genMu_PFMET_130.pt.compute(), genMu_PFMET_140.pt.compute()], 0, file)
#makeEffPlot("mu", "genMET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "#mu pt", 16, 20, 100, 5, "[GeV]", [genMu.pt.compute(), genMu.pt.compute()], [genMu_MET_105.pt.compute(), genMu_MET_120.pt.compute()], 0, file)
#makeEffPlot("mu", "genXFLAV", ["HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1"], "#mu pt", 16, 20, 100, 5, "[GeV]", [genMu.pt.compute(), genMu.pt.compute()], [genMu_semiLepTrigMu_30.pt.compute(), genMu_semiLepTrigMu_35.pt.compute()], 0, file) 
#makeEffPlot("mu", "genALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1"],  "#mu pt", 16, 20, 100, 5, "[GeV]", [genMu.pt.compute(),]*3, [genMu_PFMET_120.pt.compute(), genMu_MET_105.pt.compute(), genMu_semiLepTrigMu_30.pt.compute()], 0, file)

### Trigger Eff vs Eta
#### Reco
makeEffPlot("mu", "PFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "#mu #eta", 24, -2.4, 2.4, 0.2, "", [recoMu.eta.compute(), recoMu.eta.compute(), recoMu.eta.compute()], [recoMu_PFMET_120.eta.compute(), recoMu_PFMET_130.eta.compute(), recoMu_PFMET_140.eta.compute()], 0, file)
makeEffPlot("mu", "PFMETNoMu", ["HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight", "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight", "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight"], "#mu #eta", 24, -2.4, 2.4, 0.2, "", [recoMu.eta.compute(), recoMu.eta.compute(), recoMu.eta.compute(), recoMu.eta.compute()], [recoMu_PFMETNoMu_110_FilterHF.eta.compute(), recoMu_PFMETNoMu_120.eta.compute(), recoMu_PFMETNoMu_130.eta.compute(), recoMu_PFMETNoMu_140.eta.compute()], 0, file)
makeEffPlot("mu", "MET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "#mu #eta", 24, -2.4, 2.4, 0.2, "", [recoMu.eta.compute(), recoMu.eta.compute()], [recoMu_MET_105.eta.compute(), recoMu_MET_120.eta.compute()], 0, file)
makeEffPlot("mu", "XFLAV", ["HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1"], "#mu #eta", 24, -2.4, 2.4, 0.2, "", [recoMu.eta.compute(), recoMu.eta.compute()], [recoMu_semiLepTrigMu_30.eta.compute(), recoMu_semiLepTrigMu_35.eta.compute()], 0, file) 
makeEffPlot("mu", "ALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_MET105_IsoTrk50", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1"],  "#mu #eta", 24, -2.4, 2.4, 0.2, "", [recoMu.eta.compute(),]*4, [recoMu_PFMET_120.eta.compute(), recoMu_PFMETNoMu_110_FilterHF.eta.compute(), recoMu_MET_105.eta.compute(), recoMu_semiLepTrigMu_30.eta.compute()], 0, file)

#### Gen
#makeEffPlot("mu", "genPFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "#mu #eta", 24, -2.4, 2.4, 0.2, "", [genMu.eta.compute(), genMu.eta.compute(), genMu.eta.compute()], [genMu_PFMET_120.eta.compute(), genMu_PFMET_130.eta.compute(), genMu_PFMET_140.eta.compute()], 0, file)
#makeEffPlot("mu", "genMET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "#mu #eta", 24, -2.4, 2.4, 0.2, "", [genMu.eta.compute(), genMu.eta.compute()], [genMu_MET_105.eta.compute(), genMu_MET_120.eta.compute()], 0, file)
#makeEffPlot("mu", "genXFLAV", ["HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1"], "#mu #eta", 24, -2.4, 2.4, 0.2, "", [genMu.eta.compute(), genMu.eta.compute()], [genMu_semiLepTrigMu_30.eta.compute(), genMu_semiLepTrigMu_35.eta.compute()], 0, file) 
#makeEffPlot("mu", "genALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1"],  "#mu #eta", 24, -2.4, 2.4, 0.2, "", [genMu.eta.compute(),]*3, [genMu_PFMET_120.eta.compute(), genMu_MET_105.eta.compute(), genMu_semiLepTrigMu_30.eta.compute()], 0, file)

### Trigger Eff vs dxy
#### Reco
makeEffPlot_varBin("mu", "PFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "#mu d_{xy}", len(dxy_bins) - 1, dxy_bins, "[cm]", [recoMu.dxy.compute(), recoMu.dxy.compute(), recoMu.dxy.compute()], [recoMu_PFMET_120.dxy.compute(), recoMu_PFMET_130.dxy.compute(), recoMu_PFMET_140.dxy.compute()], 0, file)
makeEffPlot_varBin("mu", "PFMETNoMu", ["HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight", "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight", "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight"], "#mu d_{xy}", len(dxy_bins) - 1, dxy_bins, "[cm]", [recoMu.dxy.compute(), recoMu.dxy.compute(), recoMu.dxy.compute(), recoMu.dxy.compute()], [recoMu_PFMETNoMu_110_FilterHF.dxy.compute(), recoMu_PFMETNoMu_120.dxy.compute(), recoMu_PFMETNoMu_130.dxy.compute(), recoMu_PFMETNoMu_140.dxy.compute()], 0, file)
makeEffPlot_varBin("mu", "MET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "#mu d_{xy}", len(dxy_bins) - 1, dxy_bins, "[cm]", [recoMu.dxy.compute(), recoMu.dxy.compute()], [recoMu_MET_105.dxy.compute(), recoMu_MET_120.dxy.compute()], 0, file)
makeEffPlot_varBin("mu", "XFLAV", ["HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1"], "#mu d_{xy}", len(dxy_bins) - 1, dxy_bins, "[cm]", [recoMu.dxy.compute(), recoMu.dxy.compute()], [recoMu_semiLepTrigMu_30.dxy.compute(), recoMu_semiLepTrigMu_35.dxy.compute()], 0, file) 
makeEffPlot_varBin("mu", "ALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_MET105_IsoTrk50", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1"],  "#mu d_{xy}", len(dxy_bins) - 1, dxy_bins, "[cm]", [recoMu.dxy.compute(),]*4, [recoMu_PFMET_120.dxy.compute(), recoMu_PFMETNoMu_110_FilterHF.dxy.compute(), recoMu_MET_105.dxy.compute(), recoMu_semiLepTrigMu_30.dxy.compute()], 0, file)

#### Gen
#makeEffPlot_varBin("mu", "genPFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "#mu d_{xy}", 20, 0, 100, 5, "[cm]", [genMu.dxy.compute(), genMu.dxy.compute(), genMu.dxy.compute()], [genMu_PFMET_120.dxy.compute(), genMu_PFMET_130.dxy.compute(), genMu_PFMET_140.dxy.compute()], 0, file)
#makeEffPlot_varBin("mu", "genMET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "#mu d_{xy}", 20, 0, 100, 5, "[cm]", [genMu.dxy.compute(), genMu.dxy.compute()], [genMu_MET_105.dxy.compute(), genMu_MET_120.dxy.compute()], 0, file)
#makeEffPlot_varBin("mu", "genXFLAV", ["HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1"], "#mu d_{xy}", 20, 0, 100, 5, "[cm]", [genMu.dxy.compute(), genMu.dxy.compute()], [genMu_semiLepTrigMu_30.dxy.compute(), genMu_semiLepTrigMu_35.dxy.compute()], 0, file) 
#makeEffPlot_varBin("mu", "genALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1"],  "#mu d_{xy}", 20, 0, 100, 5, "[cm]", [genMu.dxy.compute(),]*3, [genMu_PFMET_120.dxy.compute(), genMu_MET_105.dxy.compute(), genMu_semiLepTrigMu_30.dxy.compute()], 0, file)

### Trigger_varBin Eff vs lxy
#### Reco
makeEffPlot_varBin("mu", "PFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "#mu l_{xy}", len(dxy_bins) - 1, dxy_bins, "[cm]", [recoMu.lxy.compute(), recoMu.lxy.compute(), recoMu.lxy.compute()], [recoMu_PFMET_120.lxy.compute(), recoMu_PFMET_130.lxy.compute(), recoMu_PFMET_140.lxy.compute()], 0, file)
makeEffPlot_varBin("mu", "PFMETNoMu", ["HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight", "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight", "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight"], "#mu l_{xy}", len(dxy_bins) - 1, dxy_bins, "[cm]", [recoMu.lxy.compute(), recoMu.lxy.compute(), recoMu.lxy.compute(), recoMu.lxy.compute()], [recoMu_PFMETNoMu_110_FilterHF.lxy.compute(), recoMu_PFMETNoMu_120.lxy.compute(), recoMu_PFMETNoMu_130.lxy.compute(), recoMu_PFMETNoMu_140.lxy.compute()], 0, file)
makeEffPlot_varBin("mu", "MET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "#mu l_{xy}", len(dxy_bins) - 1, dxy_bins, "[cm]", [recoMu.lxy.compute(), recoMu.lxy.compute()], [recoMu_MET_105.lxy.compute(), recoMu_MET_120.lxy.compute()], 0, file)
makeEffPlot_varBin("mu", "XFLAV", ["HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1"], "#mu l_{xy}", len(dxy_bins) - 1, dxy_bins, "[cm]", [recoMu.lxy.compute(), recoMu.lxy.compute()], [recoMu_semiLepTrigMu_30.lxy.compute(), recoMu_semiLepTrigMu_35.lxy.compute()], 0, file) 
makeEffPlot_varBin("mu", "ALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_MET105_IsoTrk50", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1"],  "#mu l_{xy}", len(dxy_bins) - 1, dxy_bins, "[cm]", [recoMu.lxy.compute(),]*4, [recoMu_PFMET_120.lxy.compute(), recoMu_PFMETNoMu_110_FilterHF.lxy.compute(), recoMu_MET_105.lxy.compute(), recoMu_semiLepTrigMu_30.lxy.compute()], 0, file)

#### Gen
#makeEffPlot_var("mu", "genPFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "#mu l_{xy}", 20, 0, 100, 5, "[cm]", [genMu.lxy.compute(), genMu.lxy.compute(), genMu.lxy.compute()], [genMu_PFMET_120.lxy.compute(), genMu_PFMET_130.lxy.compute(), genMu_PFMET_140.lxy.compute()], 0, file)
#makeEffPlot_var("mu", "genMET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "#mu l_{xy}", 20, 0, 100, 5, "[cm]", [genMu.lxy.compute(), genMu.lxy.compute()], [genMu_MET_105.lxy.compute(), genMu_MET_120.lxy.compute()], 0, file)
#makeEffPlot_var("mu", "genXFLAV", ["HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1"], "#mu l_{xy}", 20, 0, 100, 5, "[cm]", [genMu.lxy.compute(), genMu.lxy.compute()], [genMu_semiLepTrigMu_30.lxy.compute(), genMu_semiLepTrigMu_35.lxy.compute()], 0, file) 
#makeEffPlot_var("mu", "genALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1"],  "#mu l_{xy}", 20, 0, 100, 5, "[cm]", [genMu.lxy.compute(),]*3, [genMu_PFMET_120.lxy.compute(), genMu_MET_105.lxy.compute(), genMu_semiLepTrigMu_30.lxy.compute()], 0, file)

### Trigger Eff vs MET pt
#### Reco
makeEffPlot_varBin("mu", "PFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [recoMET_mu.pt.compute(),]*3, [recoMET_mu_PFMET_120.pt.compute(), recoMET_mu_PFMET_130.pt.compute(), recoMET_mu_PFMET_140.pt.compute()], 0, file)
makeEffPlot_varBin("mu", "PFMETNoMu", ["HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight", "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight", "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight"], "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [recoMET_mu.pt.compute(),]*4, [recoMET_mu_PFMETNoMu_110_FilterHF.pt.compute(), recoMET_mu_PFMETNoMu_120.pt.compute(), recoMET_mu_PFMETNoMu_130.pt.compute(), recoMET_mu_PFMETNoMu_140.pt.compute()], 0, file)
makeEffPlot_varBin("mu", "MET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [recoMET_mu.pt.compute(),]*2, [recoMET_mu_MET_105.pt.compute(), recoMET_mu_MET_120.pt.compute()], 0, file)
makeEffPlot_varBin("mu", "XFLAV", ["HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1"], "MET pT", len(MET_bins) - 1, MET_bins,"[GeV]", [recoMET_mu.pt.compute(),]*2, [recoMET_mu_semiLepTrigMu_30.pt.compute(), recoMET_mu_semiLepTrigMu_35.pt.compute()], 0, file)
makeEffPlot_varBin("mu", "ALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_MET105_IsoTrk50", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1"], "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [recoMET_mu.pt.compute(),]*4, [recoMET_mu_PFMET_120.pt.compute(), recoMET_mu_PFMETNoMu_110_FilterHF.pt.compute(), recoMET_mu_MET_105.pt.compute(), recoMET_mu_semiLepTrigMu_30.pt.compute()], 0, file)

#### Gen
#makeEffPlot_varBin("mu", "genPFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [genMET_mu.pt.compute(),]*3, [genMET_mu_PFMET_120.pt.compute(), genMET_mu_PFMET_130.pt.compute(), genMET_mu_PFMET_140.pt.compute()], 0, file)
#makeEffPlot_varBin("mu", "genMET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [genMET_mu.pt.compute(),]*2, [genMET_mu_MET_105.pt.compute(), genMET_mu_MET_120.pt.compute()], 0, file)
#makeEffPlot_varBin("mu", "genXFLAV", ["HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1"], "MET pT", len(MET_bins) - 1, MET_bins,"[GeV]", [genMET_mu.pt.compute(),]*2, [genMET_mu_semiLepTrigMu_30.pt.compute(), genMET_mu_semiLepTrigMu_35.pt.compute()], 0, file)
#makeEffPlot_varBin("mu", "genALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1"], "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [genMET_mu.pt.compute(),]*3, [genMET_mu_PFMET_120.pt.compute(), genMET_mu_MET_105.pt.compute(), genMET_mu_semiLepTrigMu_30.pt.compute()], 0, file)

