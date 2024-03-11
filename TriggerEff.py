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
electrons = events.Electron
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
gen_electrons = staus_taus.distinctChildren[(abs(staus_taus.distinctChildren.pdgId) == 11)]

gen_mus = gen_mus[(gen_mus.pt > GenPtMin) & (abs(gen_mus.eta) < GenEtaMax)]
gen_electrons = gen_electrons[(gen_electrons.pt > GenPtMin) & (abs(gen_electrons.eta) < GenEtaMax)]
gen_taus = gvistau[(gvistau.pt > GenPtMin) & (abs(gvistau.eta) < GenEtaMax)]

gen_mus["dxy"] = (gen_mus.vertexY - gvtx.y) * np.cos(gen_mus.phi) - (gen_mus.vertexX - gvtx.x) * np.sin(gen_mus.phi)
gen_electrons["dxy"] = (gen_electrons.vertexY - gvtx.y) * np.cos(gen_electrons.phi) - (gen_electrons.vertexX - gvtx.x) * np.sin(gen_electrons.phi)
gen_taus["dxy"] = (ak.firsts(gpart[gen_taus.genPartIdxMother.compute()].distinctChildren.vertexY, axis = 2) - gvtx.y) * np.cos(gpart[gen_taus.genPartIdxMother.compute()].phi) - (ak.firsts(gpart[gen_taus.genPartIdxMother.compute()].distinctChildren.vertexX, axis = 2) - gvtx.x) * np.sin(gpart[gen_taus.genPartIdxMother.compute()].phi)

gen_mus["lxy"] = np.sqrt((gen_mus.distinctParent.vertexX - ak.firsts(staus.vertexX)) ** 2 + (gen_mus.distinctParent.vertexY - ak.firsts(staus.vertexY)) ** 2)
gen_electrons["lxy"] = np.sqrt((gen_electrons.distinctParent.vertexX - ak.firsts(staus.vertexX)) ** 2 + (gen_electrons.distinctParent.vertexY - ak.firsts(staus.vertexY)) ** 2)  
gen_taus["lxy"] = np.sqrt((gpart[gen_taus.genPartIdxMother.compute()].vertexX - ak.firsts(staus.vertexX)) ** 2 + (gpart[gen_taus.genPartIdxMother.compute()].vertexY- ak.firsts(staus.vertexY)) ** 2) 


semiLep_mu_evt = ak.sum((abs(ak.flatten(gen_mus.pdgId.compute(), axis = 3)) == 13), axis = -2) == 1
semiLep_mu_evt = (ak.where(ak.num(semiLep_mu_evt) > 0, True, False) & (ak.num(gen_taus) == 1).compute())

semiLep_el_evt = ak.sum((abs(ak.flatten(gen_electrons.pdgId.compute(), axis = 3)) == 11), axis = -2) == 1
semiLep_el_evt = (ak.where(ak.num(semiLep_el_evt) > 0, True, False) & (ak.num(gen_taus) == 1).compute())
diTau_evt = ak.num(gvistau) == 2

genMET_mu    = gmet[semiLep_mu_evt]
genMET_el    = gmet[semiLep_el_evt]
genMET_diTau = gmet[diTau_evt]

### Di Tau Triggers (Prompt)
diTauTrig_35 = events.HLT.DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1 == 1
diTauTrig_40 = events.HLT.DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1 == 1
diTauTrig_displaced_32 = events.HLT.DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1 == 1


### Cross Flavor Tau-Muons Triggers (Prompt)
semiLepTrigMu_30 = events.HLT.IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1 == 1
semiLepTrigMu_35 = events.HLT.IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1 == 1

### Cross Flavor Electron-Tau Triggers (Prompt)
semiLepTrigEl_30 = events.HLT.Ele30_WPTight_Gsf == 1

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

RecoElectronsFromGen = electrons[(abs(electrons.matched_gen.distinctParent.distinctParent.pdgId) == 1000015)]
RecoElectronsFromGen = RecoElectronsFromGen.matched_gen[(RecoElectronsFromGen.matched_gen.pt > GenPtMin) & (abs(RecoElectronsFromGen.matched_gen.eta) < GenEtaMax)]
RecoElectronsFromGen["dxy"] = (RecoElectronsFromGen.vertexY - gvtx.y) * np.cos(RecoElectronsFromGen.phi) - (RecoElectronsFromGen.vertexX - gvtx.x) * np.sin(RecoElectronsFromGen.phi)
RecoElectronsFromGen["lxy"] = np.sqrt((RecoElectronsFromGen.distinctParent.vertexY - gvtx.y) ** 2 + (RecoElectronsFromGen.distinctParent.vertexX- gvtx.x) ** 2)

jets = jets[(jets.pt > 20) & (abs(jets.eta) < 2.4) & (jets.genJetIdx >= 0) & (jets.genJetIdx < ak.num(events.GenJet.pt))]
RecoTausFromGen = jets.nearest(gen_taus, threshold = 0.3)
RecoTausFromGen = RecoTausFromGen[(RecoTausFromGen.pt > GenPtMin) & (abs(RecoTausFromGen.eta) < GenEtaMax)]

recoSemiLep_mu_evt = (ak.num(RecoMuonsFromGen) == 1) & (ak.num(ak.drop_none(RecoTausFromGen)) == 1)
recoSemiLep_el_evt = (ak.num(RecoElectronsFromGen) == 1) & (ak.num(ak.drop_none(RecoTausFromGen)) == 1) 
recoDiTau_evt      = (ak.num(ak.drop_none(RecoTausFromGen)) == 2)

genMu  = gen_mus[semiLep_mu_evt]
genEl  = gen_electrons[semiLep_el_evt]
genTauMu = gen_taus[semiLep_mu_evt]
genTauEl = gen_taus[semiLep_el_evt]
genDiTau = gen_taus[diTau_evt]

recoMu  = gen_mus[semiLep_mu_evt & recoSemiLep_mu_evt.compute()]
recoEl  = gen_electrons[semiLep_el_evt & recoSemiLep_el_evt.compute()]
recoTauMu = gen_taus[semiLep_mu_evt & recoSemiLep_mu_evt.compute()]
recoTauEl = gen_taus[semiLep_el_evt & recoSemiLep_el_evt.compute()]
recoDiTau = gen_taus[diTau_evt & recoDiTau_evt.compute()]

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

### Semilep tau-mu events which pass triggers
#### Gen Events
genTauMu_PFMET_120        = genTauMu[PFMET_120]
genTauMu_PFMET_130        = genTauMu[PFMET_130]
genTauMu_PFMET_140        = genTauMu[PFMET_140]

genTauMu_MET_105          = genTauMu[MET_105]
genTauMu_MET_120          = genTauMu[MET_120]

genTauMu_semiLepTrigMu_30 = genTauMu[semiLepTrigMu_30]
genTauMu_semiLepTrigMu_35 = genTauMu[semiLepTrigMu_35]

#### Gen Events that have reco'd mu and taus for tau
recoTauMu_PFMET_120        = gen_taus[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & PFMET_120.compute()]
recoTauMu_PFMET_130        = gen_taus[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & PFMET_130.compute()]
recoTauMu_PFMET_140        = gen_taus[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & PFMET_140.compute()]
                                                                                                                  
recoTauMu_MET_105          = gen_taus[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & MET_105.compute()]
recoTauMu_MET_120          = gen_taus[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & MET_120.compute()]
                                                                                                                  
recoTauMu_semiLepTrigMu_30 = gen_taus[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & semiLepTrigMu_30.compute()]
recoTauMu_semiLepTrigMu_35 = gen_taus[semiLep_mu_evt & recoSemiLep_mu_evt.compute() & semiLepTrigMu_35.compute()]

### Semilep e events which pass triggers 
#### Gen Events
genEl_PFMET_120              = genEl[PFMET_120]
genEl_PFMET_130              = genEl[PFMET_130]
genEl_PFMET_140              = genEl[PFMET_140]

genEl_MET_105                = genEl[MET_105]
genEl_MET_120                = genEl[MET_120]

genEl_PFMETNoMu_110_FilterHF = genEl[PFMETNoMu_110_FilterHF]

genEl_PFMETNoMu_120          = genEl[PFMETNoMu_120]
genEl_PFMETNoMu_130          = genEl[PFMETNoMu_130]
genEl_PFMETNoMu_140          = genEl[PFMETNoMu_140]

genEl_semiLepTrigEl_30       = genEl[semiLepTrigEl_30]

#### Gen Events that have reco'd e and tau for e
recoEl_PFMET_120                  = gen_electrons[semiLep_el_evt & recoSemiLep_el_evt.compute() & PFMET_120.compute()]
recoEl_PFMET_130                  = gen_electrons[semiLep_el_evt & recoSemiLep_el_evt.compute() & PFMET_130.compute()]
recoEl_PFMET_140                  = gen_electrons[semiLep_el_evt & recoSemiLep_el_evt.compute() & PFMET_140.compute()]

recoMET_el_PFMET_120              = gmet[semiLep_el_evt & recoSemiLep_el_evt.compute() & PFMET_120.compute()] 
recoMET_el_PFMET_130              = gmet[semiLep_el_evt & recoSemiLep_el_evt.compute() & PFMET_130.compute()] 
recoMET_el_PFMET_140              = gmet[semiLep_el_evt & recoSemiLep_el_evt.compute() & PFMET_140.compute()] 

recoEl_MET_105                    = gen_electrons[semiLep_el_evt & recoSemiLep_el_evt.compute() & MET_105.compute()]
recoEl_MET_120                    = gen_electrons[semiLep_el_evt & recoSemiLep_el_evt.compute() & MET_120.compute()]

recoMET_el_MET_105                = gmet[semiLep_el_evt & recoSemiLep_el_evt.compute() & MET_105.compute()]
recoMET_el_MET_120                = gmet[semiLep_el_evt & recoSemiLep_el_evt.compute() & MET_120.compute()]

recoEl_PFMETNoMu_110_FilterHF     = gen_electrons[semiLep_el_evt & recoSemiLep_el_evt.compute() & PFMETNoMu_110_FilterHF.compute()]

recoMET_el_PFMETNoMu_110_FilterHF = gmet[semiLep_el_evt & recoSemiLep_el_evt.compute() & PFMETNoMu_110_FilterHF.compute()]

recoEl_PFMETNoMu_120              = gen_electrons[semiLep_el_evt & recoSemiLep_el_evt.compute() & PFMETNoMu_120.compute()]
recoEl_PFMETNoMu_130              = gen_electrons[semiLep_el_evt & recoSemiLep_el_evt.compute() & PFMETNoMu_130.compute()]
recoEl_PFMETNoMu_140              = gen_electrons[semiLep_el_evt & recoSemiLep_el_evt.compute() & PFMETNoMu_140.compute()]

recoMET_el_PFMETNoMu_120          = gmet[semiLep_el_evt & recoSemiLep_el_evt.compute() & PFMETNoMu_120.compute()] 
recoMET_el_PFMETNoMu_130          = gmet[semiLep_el_evt & recoSemiLep_el_evt.compute() & PFMETNoMu_130.compute()] 
recoMET_el_PFMETNoMu_140          = gmet[semiLep_el_evt & recoSemiLep_el_evt.compute() & PFMETNoMu_140.compute()] 

recoEl_semiLepTrigEl_30           = gen_electrons[semiLep_el_evt & recoSemiLep_el_evt.compute() & semiLepTrigEl_30.compute()]

recoMET_el_semiLepTrigEl_30       = gmet[semiLep_el_evt & recoSemiLep_el_evt.compute() & semiLepTrigEl_30.compute()]

### Semilep tau-e events which pass triggers 
#### Gen Events
genTauEl_PFMET_120              = genTauEl[PFMET_120]
genTauEl_PFMET_130              = genTauEl[PFMET_130]
genTauEl_PFMET_140              = genTauEl[PFMET_140]

genTauEl_MET_105                = genTauEl[MET_105]
genTauEl_MET_120                = genTauEl[MET_120]

genTauEl_PFMETNoMu_110_FilterHF = genTauEl[PFMETNoMu_110_FilterHF]

genTauEl_PFMETNoMu_120          = genTauEl[PFMETNoMu_120]
genTauEl_PFMETNoMu_130          = genTauEl[PFMETNoMu_130]
genTauEl_PFMETNoMu_140          = genTauEl[PFMETNoMu_140]

genTauEl_semiLepTrigEl_30       = genTauEl[semiLepTrigEl_30]

#### Gen Events that have reco'd e and tau for tau
recoTauEl_PFMET_120              = gen_taus[semiLep_el_evt & recoSemiLep_el_evt.compute() & PFMET_120.compute()]
recoTauEl_PFMET_130              = gen_taus[semiLep_el_evt & recoSemiLep_el_evt.compute() & PFMET_130.compute()]
recoTauEl_PFMET_140              = gen_taus[semiLep_el_evt & recoSemiLep_el_evt.compute() & PFMET_140.compute()]

recoTauEl_MET_105                = gen_taus[semiLep_el_evt & recoSemiLep_el_evt.compute() & MET_105.compute()]
recoTauEl_MET_120                = gen_taus[semiLep_el_evt & recoSemiLep_el_evt.compute() & MET_120.compute()]

recoTauEl_PFMETNoMu_110_FilterHF = gen_taus[semiLep_el_evt & recoSemiLep_el_evt.compute() & PFMETNoMu_110_FilterHF.compute()]

recoTauEl_PFMETNoMu_120          = gen_taus[semiLep_el_evt & recoSemiLep_el_evt.compute() & PFMETNoMu_120.compute()]
recoTauEl_PFMETNoMu_130          = gen_taus[semiLep_el_evt & recoSemiLep_el_evt.compute() & PFMETNoMu_130.compute()]
recoTauEl_PFMETNoMu_140          = gen_taus[semiLep_el_evt & recoSemiLep_el_evt.compute() & PFMETNoMu_140.compute()]

recoTauEl_semiLepTrigEl_30       = gen_taus[semiLep_el_evt & recoSemiLep_el_evt.compute() & semiLepTrigEl_30.compute()]

### Di-Tau events that pass triggers
#### Gen Events
genDiTau_PFMET_120              = genDiTau[PFMET_120]              
genDiTau_PFMET_130              = genDiTau[PFMET_130]
genDiTau_PFMET_140              = genDiTau[PFMET_140]

genDiTau_MET_105                = genDiTau[MET_105]
genDiTau_MET_120                = genDiTau[MET_120]

genDiTau_PFMETNoMu_110_FilterHF = genDiTau[PFMETNoMu_110_FilterHF]

genDiTau_PFMETNoMu_120          = genDiTau[PFMETNoMu_120]
genDiTau_PFMETNoMu_130          = genDiTau[PFMETNoMu_130]
genDiTau_PFMETNoMu_140          = genDiTau[PFMETNoMu_140]

genDiTau_diTauTrig_35           = genDiTau[diTauTrig_35] 
genDiTau_diTauTrig_40           = genDiTau[diTauTrig_40] 
genDiTau_diTauTrig_displaced_32 = genDiTau[diTauTrig_displaced_32]

#### Gen Events that have reco's di tau
recoDiTau_PFMET_120                  = gen_taus[diTau_evt & recoDiTau_evt.compute() & PFMET_120.compute()]              
recoDiTau_PFMET_130                  = gen_taus[diTau_evt & recoDiTau_evt.compute() & PFMET_130.compute()]
recoDiTau_PFMET_140                  = gen_taus[diTau_evt & recoDiTau_evt.compute() & PFMET_140.compute()]

recoMET_diTau_PFMET_120              = gmet[diTau_evt & recoDiTau_evt.compute() & PFMET_120.compute()]               
recoMET_diTau_PFMET_130              = gmet[diTau_evt & recoDiTau_evt.compute() & PFMET_130.compute()]               
recoMET_diTau_PFMET_140              = gmet[diTau_evt & recoDiTau_evt.compute() & PFMET_140.compute()]               

recoDiTau_MET_105                    = gen_taus[diTau_evt & recoDiTau_evt.compute() & MET_105.compute()]
recoDiTau_MET_120                    = gen_taus[diTau_evt & recoDiTau_evt.compute() & MET_120.compute()]

recoMET_diTau_MET_105                = gmet[diTau_evt & recoDiTau_evt.compute() & MET_105.compute()] 
recoMET_diTau_MET_120                = gmet[diTau_evt & recoDiTau_evt.compute() & MET_120.compute()] 

recoDiTau_PFMETNoMu_110_FilterHF     = gen_taus[diTau_evt & recoDiTau_evt.compute() & PFMETNoMu_110_FilterHF.compute()]

recoMET_diTau_PFMETNoMu_110_FilterHF = gmet[diTau_evt & recoDiTau_evt.compute() & PFMETNoMu_110_FilterHF.compute()]

recoDiTau_PFMETNoMu_120              = gen_taus[diTau_evt & recoDiTau_evt.compute() & PFMETNoMu_120.compute()]
recoDiTau_PFMETNoMu_130              = gen_taus[diTau_evt & recoDiTau_evt.compute() & PFMETNoMu_130.compute()]
recoDiTau_PFMETNoMu_140              = gen_taus[diTau_evt & recoDiTau_evt.compute() & PFMETNoMu_140.compute()]

recoMET_diTau_PFMETNoMu_120          = gmet[diTau_evt & recoDiTau_evt.compute() & PFMETNoMu_120.compute()] 
recoMET_diTau_PFMETNoMu_130          = gmet[diTau_evt & recoDiTau_evt.compute() & PFMETNoMu_130.compute()] 
recoMET_diTau_PFMETNoMu_140          = gmet[diTau_evt & recoDiTau_evt.compute() & PFMETNoMu_140.compute()] 

recoDiTau_diTauTrig_35               = gen_taus[diTau_evt & recoDiTau_evt.compute() & diTauTrig_35.compute()] 
recoDiTau_diTauTrig_40               = gen_taus[diTau_evt & recoDiTau_evt.compute() & diTauTrig_40.compute()] 
recoDiTau_diTauTrig_displaced_32     = gen_taus[diTau_evt & recoDiTau_evt.compute() & diTauTrig_displaced_32.compute()]

recoMET_diTau_diTauTrig_35           = gmet[diTau_evt & recoDiTau_evt.compute() & diTauTrig_35.compute()]           
recoMET_diTau_diTauTrig_40           = gmet[diTau_evt & recoDiTau_evt.compute() & diTauTrig_40.compute()]           
recoMET_diTau_diTauTrig_displaced_32 = gmet[diTau_evt & recoDiTau_evt.compute() & diTauTrig_displaced_32.compute()] 

### Trigger Eff vs Pt
#makeEffPlot("mu", "PFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "#mu pt", 16, 20, 100, 5, "[GeV]", [recoMu.pt.compute(), recoMu.pt.compute(), recoMu.pt.compute()], [recoMu_PFMET_120.pt.compute(), recoMu_PFMET_130.pt.compute(), recoMu_PFMET_140.pt.compute()], 0, file)
#makeEffPlot("mu", "MET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "#mu pt", 16, 20, 100, 5, "[GeV]", [recoMu.pt.compute(), recoMu.pt.compute()], [recoMu_MET_105.pt.compute(), recoMu_MET_120.pt.compute()], 0, file)
#makeEffPlot("mu", "XFLAV", ["HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1"], "#mu pt", 16, 20, 100, 5, "[GeV]", [recoMu.pt.compute(), recoMu.pt.compute()], [recoMu_semiLepTrigMu_30.pt.compute(), recoMu_semiLepTrigMu_35.pt.compute()], 0, file) 
#makeEffPlot("mu", "ALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1"],  "#mu pt", 16, 20, 100, 5, "[GeV]", [recoMu.pt.compute(),]*3, [recoMu_PFMET_120.pt.compute(), recoMu_MET_105.pt.compute(), recoMu_semiLepTrigMu_30.pt.compute()], 0, file)
#
#makeEffPlot("e", "PFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "e pt", 16, 20, 100, 5, "[GeV]", [recoEl.pt.compute(), recoEl.pt.compute(), recoEl.pt.compute()], [recoEl_PFMET_120.pt.compute(), recoEl_PFMET_130.pt.compute(), recoEl_PFMET_140.pt.compute()], 0, file)
#makeEffPlot("e", "MET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "e pt", 16, 20, 100, 5, "[GeV]", [recoEl.pt.compute(), recoEl.pt.compute()], [recoEl_MET_105.pt.compute(), recoEl_MET_120.pt.compute()], 0, file)
#makeEffPlot("e", "PFMETNoMu", ["HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight", "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight", "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight"], "e pt", 16, 20, 100, 5, "[GeV]", [recoEl.pt.compute(), recoEl.pt.compute(), recoEl.pt.compute(), recoEl.pt.compute()], [recoEl_PFMETNoMu_110_FilterHF.pt.compute(), recoEl_PFMETNoMu_120.pt.compute(), recoEl_PFMETNoMu_130.pt.compute(), recoEl_PFMETNoMu_140.pt.compute()], 0, file)
#makeEffPlot("e", "XFLAV", ["HLT_Ele30_WPTight_Gsf"], "e pt", 16, 20, 100, 5, "[GeV]", [recoEl.pt.compute()], [recoEl_semiLepTrigEl_30.pt.compute()], 0, file) 
#makeEffPlot("e", "ALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_Ele30_WPTight_Gsf"], "e pt", 16, 20, 100, 5, "[GeV]", [recoEl.pt.compute(),]*4, [recoEl_PFMET_120.pt.compute(), recoEl_MET_105.pt.compute(), recoEl_PFMETNoMu_110_FilterHF.pt.compute(), recoEl_semiLepTrigEl_30.pt.compute()], 0, file) 
#
#makeEffPlot("ditau", "PFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "#tau pt", 16, 20, 100, 5, "[GeV]", [recoDiTau.pt.compute(), recoDiTau.pt.compute(), recoDiTau.pt.compute()], [recoDiTau_PFMET_120.pt.compute(), recoDiTau_PFMET_130.pt.compute(), recoDiTau_PFMET_140.pt.compute()], 0, file)
#makeEffPlot("ditau", "MET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "#tau pt", 16, 20, 100, 5, "[GeV]", [recoDiTau.pt.compute(), recoDiTau.pt.compute()], [recoDiTau_MET_105.pt.compute(), recoDiTau_MET_120.pt.compute()], 0, file)
#makeEffPlot("ditau", "PFMETNoMu", ["HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight", "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight", "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight"], "#tau pt", 16, 20, 100, 5, "[GeV]", [recoDiTau.pt.compute(), recoDiTau.pt.compute(), recoDiTau.pt.compute(), recoDiTau.pt.compute()], [recoDiTau_PFMETNoMu_110_FilterHF.pt.compute(), recoDiTau_PFMETNoMu_120.pt.compute(), recoDiTau_PFMETNoMu_130.pt.compute(), recoDiTau_PFMETNoMu_140.pt.compute()], 0, file)
#makeEffPlot("ditau", "XFLAV", ["HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1", "HLT_DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1", "HLT_DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1"], "#tau pt", 16, 20, 100, 5, "[GeV]", [recoDiTau.pt.compute(), recoDiTau.pt.compute(), recoDiTau.pt.compute()], [recoDiTau_diTauTrig_35.pt.compute(), recoDiTau_diTauTrig_40.pt.compute(), recoDiTau_diTauTrig_displaced_32.pt.compute()], 0, file) 
#makeEffPlot("ditau", "ALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1"], "#tau pt", 16, 20, 100, 5, "[GeV]", [recoDiTau.pt.compute(),]*4, [recoDiTau_PFMET_120.pt.compute(), recoDiTau_MET_105.pt.compute(), recoDiTau_PFMETNoMu_110_FilterHF.pt.compute(), recoDiTau_diTauTrig_displaced_32.pt.compute()], 0, file)
#
#### Trigger Eff vs Eta
#makeEffPlot("mu", "PFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "#mu #eta", 24, -2.4, 2.4, 0.2, "", [recoMu.eta.compute(), recoMu.eta.compute(), recoMu.eta.compute()], [recoMu_PFMET_120.eta.compute(), recoMu_PFMET_130.eta.compute(), recoMu_PFMET_140.eta.compute()], 0, file)
#makeEffPlot("mu", "MET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "#mu #eta", 24, -2.4, 2.4, 0.2, "", [recoMu.eta.compute(), recoMu.eta.compute()], [recoMu_MET_105.eta.compute(), recoMu_MET_120.eta.compute()], 0, file)
#makeEffPlot("mu", "XFLAV", ["HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1"], "#mu #eta", 24, -2.4, 2.4, 0.2, "", [recoMu.eta.compute(), recoMu.eta.compute()], [recoMu_semiLepTrigMu_30.eta.compute(), recoMu_semiLepTrigMu_35.eta.compute()], 0, file) 
#makeEffPlot("mu", "ALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1"],  "#mu #eta", 24, -2.4, 2.4, 0.2, "", [recoMu.eta.compute(),]*3, [recoMu_PFMET_120.eta.compute(), recoMu_MET_105.eta.compute(), recoMu_semiLepTrigMu_30.eta.compute()], 0, file)
#
#makeEffPlot("e", "PFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "e #eta", 24, -2.4, 2.4, 0.2, "", [recoEl.eta.compute(), recoEl.eta.compute(), recoEl.eta.compute()], [recoEl_PFMET_120.eta.compute(), recoEl_PFMET_130.eta.compute(), recoEl_PFMET_140.eta.compute()], 0, file)
#makeEffPlot("e", "MET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "e #eta", 24, -2.4, 2.4, 0.2, "", [recoEl.eta.compute(), recoEl.eta.compute()], [recoEl_MET_105.eta.compute(), recoEl_MET_120.eta.compute()], 0, file)
#makeEffPlot("e", "PFMETNoMu", ["HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight", "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight", "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight"], "e #eta", 24, -2.4, 2.4, 0.2, "", [recoEl.eta.compute(), recoEl.eta.compute(), recoEl.eta.compute(), recoEl.eta.compute()], [recoEl_PFMETNoMu_110_FilterHF.eta.compute(), recoEl_PFMETNoMu_120.eta.compute(), recoEl_PFMETNoMu_130.eta.compute(), recoEl_PFMETNoMu_140.eta.compute()], 0, file)
#makeEffPlot("e", "XFLAV", ["HLT_Ele30_WPTight_Gsf"], "e #eta", 24, 2.4, 2.4, 0.2, "", [recoEl.eta.compute()], [recoEl_semiLepTrigEl_30.eta.compute()], 0, file) 
#makeEffPlot("e", "ALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_Ele30_WPTight_Gsf"], "e #eta", 24, -2.4, 2.4, 0.2, "", [recoEl.eta.compute(),]*4, [recoEl_PFMET_120.eta.compute(), recoEl_MET_105.eta.compute(), recoEl_PFMETNoMu_110_FilterHF.eta.compute(), recoEl_semiLepTrigEl_30.eta.compute()], 0, file)
#
#makeEffPlot("ditau", "PFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "#tau #eta", 24, -2.4, 2.4, 0.2, "", [recoDiTau.eta.compute(), recoDiTau.eta.compute(), recoDiTau.eta.compute()], [recoDiTau_PFMET_120.eta.compute(), recoDiTau_PFMET_130.eta.compute(), recoDiTau_PFMET_140.eta.compute()], 0, file)
#makeEffPlot("ditau", "MET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "#tau #eta", 24, -2.4, 2.4, 0.2, "", [recoDiTau.eta.compute(), recoDiTau.eta.compute()], [recoDiTau_MET_105.eta.compute(), recoDiTau_MET_120.eta.compute()], 0, file)
#makeEffPlot("ditau", "PFMETNoMu", ["HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight", "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight", "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight"], "#tau #eta", 24, -2.4, 2.4, 0.2, "", [recoDiTau.eta.compute(), recoDiTau.eta.compute(), recoDiTau.eta.compute(), recoDiTau.eta.compute()], [recoDiTau_PFMETNoMu_110_FilterHF.eta.compute(), recoDiTau_PFMETNoMu_120.eta.compute(), recoDiTau_PFMETNoMu_130.eta.compute(), recoDiTau_PFMETNoMu_140.eta.compute()], 0, file)
#makeEffPlot("ditau", "XFLAV", ["HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1", "HLT_DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1", "HLT_DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1"], "#tau #eta", 24, -2.4, 2.4, 0.2, "", [recoDiTau.eta.compute(), recoDiTau.eta.compute(), recoDiTau.eta.compute()], [recoDiTau_diTauTrig_35.eta.compute(), recoDiTau_diTauTrig_40.eta.compute(), recoDiTau_diTauTrig_displaced_32.eta.compute()], 0, file) 
#makeEffPlot("ditau", "ALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1"], "#tau #eta", 24, -2.4, 2.4, 0.2, "", [recoDiTau.eta.compute(),]*4, [recoDiTau_PFMET_120.eta.compute(), recoDiTau_MET_105.eta.compute(), recoDiTau_PFMETNoMu_110_FilterHF.eta.compute(), recoDiTau_diTauTrig_displaced_32.eta.compute()], 0, file)
#
#### Trigger Eff vs dxy
#makeEffPlot("mu", "PFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "#mu d_{xy}", 15, 0, 15, 1, "[cm]", [recoMu.dxy.compute(), recoMu.dxy.compute(), recoMu.dxy.compute()], [recoMu_PFMET_120.dxy.compute(), recoMu_PFMET_130.dxy.compute(), recoMu_PFMET_140.dxy.compute()], 0, file)
#makeEffPlot("mu", "MET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "#mu d_{xy}", 15, 0, 15, 1, "[cm]", [recoMu.dxy.compute(), recoMu.dxy.compute()], [recoMu_MET_105.dxy.compute(), recoMu_MET_120.dxy.compute()], 0, file)
#makeEffPlot("mu", "XFLAV", ["HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1"], "#mu d_{xy}", 15, 0, 15, 1, "[cm]", [recoMu.dxy.compute(), recoMu.dxy.compute()], [recoMu_semiLepTrigMu_30.dxy.compute(), recoMu_semiLepTrigMu_35.dxy.compute()], 0, file) 
#makeEffPlot("mu", "ALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1"],  "#mu d_{xy}", 15, 0, 15, 1, "[cm]", [recoMu.dxy.compute(),]*3, [recoMu_PFMET_120.dxy.compute(), recoMu_MET_105.dxy.compute(), recoMu_semiLepTrigMu_30.dxy.compute()], 0, file)
#
#makeEffPlot("e", "PFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "e d_{xy}", 15, 0, 15, 1, "[cm]", [recoEl.dxy.compute(), recoEl.dxy.compute(), recoEl.dxy.compute()], [recoEl_PFMET_120.dxy.compute(), recoEl_PFMET_130.dxy.compute(), recoEl_PFMET_140.dxy.compute()], 0, file)
#makeEffPlot("e", "MET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "e d_{xy}", 15, 0, 15, 1, "[cm]", [recoEl.dxy.compute(), recoEl.dxy.compute()], [recoEl_MET_105.dxy.compute(), recoEl_MET_120.dxy.compute()], 0, file)
#makeEffPlot("e", "PFMETNoMu", ["HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight", "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight", "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight"], "e d_{xy}", 15, 0, 15, 1, "[cm]", [recoEl.dxy.compute(), recoEl.dxy.compute(), recoEl.dxy.compute(), recoEl.dxy.compute()], [recoEl_PFMETNoMu_110_FilterHF.dxy.compute(), recoEl_PFMETNoMu_120.dxy.compute(), recoEl_PFMETNoMu_130.dxy.compute(), recoEl_PFMETNoMu_140.dxy.compute()], 0, file)
#makeEffPlot("e", "XFLAV", ["HLT_Ele30_WPTight_Gsf"], "e d_{xy}", 15, 0, 15, 1, "[cm]", [recoEl.dxy.compute()], [recoEl_semiLepTrigEl_30.dxy.compute()], 0, file) 
#makeEffPlot("e", "ALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_Ele30_WPTight_Gsf"], "e d_{xy}", 15, 0, 15, 1, "[cm]", [recoEl.dxy.compute(),]*4, [recoEl_PFMET_120.dxy.compute(), recoEl_MET_105.dxy.compute(), recoEl_PFMETNoMu_110_FilterHF.dxy.compute(), recoEl_semiLepTrigEl_30.dxy.compute()], 0, file)
#
#makeEffPlot("ditau", "PFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "#tau d_{xy}", 15, 0, 15, 1, "[cm]", [recoDiTau.dxy.compute(), recoDiTau.dxy.compute(), recoDiTau.dxy.compute()], [recoDiTau_PFMET_120.dxy.compute(), recoDiTau_PFMET_130.dxy.compute(), recoDiTau_PFMET_140.dxy.compute()], 0, file)
#makeEffPlot("ditau", "MET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "#tau d_{xy}", 15, 0, 15, 1, "[cm]", [recoDiTau.dxy.compute(), recoDiTau.dxy.compute()], [recoDiTau_MET_105.dxy.compute(), recoDiTau_MET_120.dxy.compute()], 0, file)
#makeEffPlot("ditau", "PFMETNoMu", ["HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight", "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight", "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight"], "#tau d_{xy}", 15, 0, 15, 1, "[cm]", [recoDiTau.dxy.compute(), recoDiTau.dxy.compute(), recoDiTau.dxy.compute(), recoDiTau.dxy.compute()], [recoDiTau_PFMETNoMu_110_FilterHF.dxy.compute(), recoDiTau_PFMETNoMu_120.dxy.compute(), recoDiTau_PFMETNoMu_130.dxy.compute(), recoDiTau_PFMETNoMu_140.dxy.compute()], 0, file)
#makeEffPlot("ditau", "XFLAV", ["HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1", "HLT_DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1", "HLT_DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1"], "#tau d_{xy}", 15, 0, 15, 1, "[cm]", [recoDiTau.dxy.compute(), recoDiTau.dxy.compute(), recoDiTau.dxy.compute()], [recoDiTau_diTauTrig_35.dxy.compute(), recoDiTau_diTauTrig_40.dxy.compute(), recoDiTau_diTauTrig_displaced_32.dxy.compute()], 0, file) 
#makeEffPlot("ditau", "ALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1"], "#tau d_{xy}", 15, 0, 15, 1, "[cm]", [recoDiTau.dxy.compute(),]*4, [recoDiTau_PFMET_120.dxy.compute(), recoDiTau_MET_105.dxy.compute(), recoDiTau_PFMETNoMu_110_FilterHF.dxy.compute(), recoDiTau_diTauTrig_displaced_32.dxy.compute()], 0, file)
#
#### Trigger Eff vs lxy
#makeEffPlot("mu", "PFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "#mu l_{xy}", 15, 0, 15, 1, "[cm]", [recoMu.lxy.compute(), recoMu.lxy.compute(), recoMu.lxy.compute()], [recoMu_PFMET_120.lxy.compute(), recoMu_PFMET_130.lxy.compute(), recoMu_PFMET_140.lxy.compute()], 0, file)
#makeEffPlot("mu", "MET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "#mu l_{xy}", 15, 0, 15, 1, "[cm]", [recoMu.lxy.compute(), recoMu.lxy.compute()], [recoMu_MET_105.lxy.compute(), recoMu_MET_120.lxy.compute()], 0, file)
#makeEffPlot("mu", "XFLAV", ["HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1"], "#mu l_{xy}", 15, 0, 15, 1, "[cm]", [recoMu.lxy.compute(), recoMu.lxy.compute()], [recoMu_semiLepTrigMu_30.lxy.compute(), recoMu_semiLepTrigMu_35.lxy.compute()], 0, file) 
#makeEffPlot("mu", "ALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1"],  "#mu l_{xy}", 15, 0, 15, 1, "[cm]", [recoMu.lxy.compute(),]*3, [recoMu_PFMET_120.lxy.compute(), recoMu_MET_105.lxy.compute(), recoMu_semiLepTrigMu_30.lxy.compute()], 0, file)
#
#makeEffPlot("e", "PFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "e l_{xy}", 15, 0, 15, 1, "[cm]", [recoEl.lxy.compute(), recoEl.lxy.compute(), recoEl.lxy.compute()], [recoEl_PFMET_120.lxy.compute(), recoEl_PFMET_130.lxy.compute(), recoEl_PFMET_140.lxy.compute()], 0, file)
#makeEffPlot("e", "MET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "e l_{xy}", 15, 0, 15, 1, "[cm]", [recoEl.lxy.compute(), recoEl.lxy.compute()], [recoEl_MET_105.lxy.compute(), recoEl_MET_120.lxy.compute()], 0, file)
#makeEffPlot("e", "PFMETNoMu", ["HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_PFMETNoMu130_PFMHTNoMu120_IDTight", "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight", "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight"], "e l_{xy}", 15, 0, 15, 1, "[cm]", [recoEl.lxy.compute(), recoEl.lxy.compute(), recoEl.lxy.compute(), recoEl.lxy.compute()], [recoEl_PFMETNoMu_110_FilterHF.lxy.compute(), recoEl_PFMETNoMu_120.lxy.compute(), recoEl_PFMETNoMu_130.lxy.compute(), recoEl_PFMETNoMu_140.lxy.compute()], 0, file)
#makeEffPlot("e", "XFLAV", ["HLT_Ele30_WPTight_Gsf"], "e l_{xy}", 15, 0, 15, 1, "[cm]", [recoEl.lxy.compute()], [recoEl_semiLepTrigEl_30.lxy.compute()], 0, file) 
#makeEffPlot("e", "ALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_Ele30_WPTight_Gsf"], "e l_{xy}", 15, 0, 15, 1, "[cm]", [recoEl.lxy.compute(),]*4, [recoEl_PFMET_120.lxy.compute(), recoEl_MET_105.lxy.compute(), recoEl_PFMETNoMu_110_FilterHF.lxy.compute(), recoEl_semiLepTrigEl_30.lxy.compute()], 0, file)
#
#makeEffPlot("ditau", "PFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "#tau l_{xy}", 15, 0, 15, 1, "[cm]", [recoDiTau.lxy.compute(), recoDiTau.lxy.compute(), recoDiTau.lxy.compute()], [recoDiTau_PFMET_120.lxy.compute(), recoDiTau_PFMET_130.lxy.compute(), recoDiTau_PFMET_140.lxy.compute()], 0, file)
#makeEffPlot("ditau", "MET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "#tau l_{xy}", 15, 0, 15, 1, "[cm]", [recoDiTau.lxy.compute(), recoDiTau.lxy.compute()], [recoDiTau_MET_105.lxy.compute(), recoDiTau_MET_120.lxy.compute()], 0, file)
#makeEffPlot("ditau", "PFMETNoMu", ["HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight", "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight", "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight"], "#tau l_{xy}", 15, 0, 15, 1, "[cm]", [recoDiTau.lxy.compute(), recoDiTau.lxy.compute(), recoDiTau.lxy.compute(), recoDiTau.lxy.compute()], [recoDiTau_PFMETNoMu_110_FilterHF.lxy.compute(), recoDiTau_PFMETNoMu_120.lxy.compute(), recoDiTau_PFMETNoMu_130.lxy.compute(), recoDiTau_PFMETNoMu_140.lxy.compute()], 0, file)
#makeEffPlot("ditau", "XFLAV", ["HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1", "HLT_DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1", "HLT_DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1"], "#tau l_{xy}", 15, 0, 15, 1, "[cm]", [recoDiTau.lxy.compute(), recoDiTau.lxy.compute(), recoDiTau.lxy.compute()], [recoDiTau_diTauTrig_35.lxy.compute(), recoDiTau_diTauTrig_40.lxy.compute(), recoDiTau_diTauTrig_displaced_32.lxy.compute()], 0, file) 
#makeEffPlot("ditau", "ALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1"], "#tau l_{xy}", 15, 0, 15, 1, "[cm]", [recoDiTau.lxy.compute(),]*4, [recoDiTau_PFMET_120.lxy.compute(), recoDiTau_MET_105.lxy.compute(), recoDiTau_PFMETNoMu_110_FilterHF.lxy.compute(), recoDiTau_diTauTrig_displaced_32.lxy.compute()], 0, file)

## Trigger Eff vs MET pt
makeEffPlot_varBin("mu", "PFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [genMET_mu.pt.compute(),]*3, [recoMET_mu_PFMET_120.pt.compute(), recoMET_mu_PFMET_130.pt.compute(), recoMET_mu_PFMET_140.pt.compute()], 0, file)
makeEffPlot_varBin("mu", "MET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [genMET_mu.pt.compute(),]*2, [recoMET_mu_MET_105.pt.compute(), recoMET_mu_MET_120.pt.compute()], 0, file)
makeEffPlot_varBin("mu", "XFLAV", ["HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1"], "MET pT", len(MET_bins) - 1, MET_bins,"[GeV]", [genMET_mu.pt.compute(),]*2, [recoMET_mu_semiLepTrigMu_30.pt.compute(), recoMET_mu_semiLepTrigMu_35.pt.compute()], 0, file)
makeEffPlot_varBin("mu", "ALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1"], "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [genMET_mu.pt.compute(),]*3, [recoMET_mu_PFMET_120.pt.compute(), recoMET_mu_MET_105.pt.compute(), recoMET_mu_semiLepTrigMu_30.pt.compute()], 0, file)

makeEffPlot_varBin("e", "PFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [genMET_el.pt.compute(),]*3, [recoMET_el_PFMET_120.pt.compute(), recoMET_el_PFMET_130.pt.compute(), recoMET_el_PFMET_140.pt.compute()], 0, file)
makeEffPlot_varBin("e", "MET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [genMET_el.pt.compute(),]*2, [recoMET_el_MET_105.pt.compute(), recoMET_el_MET_120.pt.compute()], 0, file)
makeEffPlot_varBin("e", "PFMETNoMu", ["HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight", "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight", "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight"], "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [genMET_el.pt.compute(),]*4, [recoMET_el_PFMETNoMu_110_FilterHF.pt.compute(), recoMET_el_PFMETNoMu_120.pt.compute(), recoMET_el_PFMETNoMu_130.pt.compute(), recoMET_el_PFMETNoMu_140.pt.compute()], 0, file)
makeEffPlot_varBin("e", "XFLAV", ["HLT_Ele30_WPTight_Gsf"], "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [genMET_el.pt.compute()], [recoMET_el_semiLepTrigEl_30.pt.compute()], 0, file)
makeEffPlot_varBin("e", "ALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_Ele30_WPTight_Gsf"],  "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [genMET_el.pt.compute(),]*4, [recoMET_el_PFMET_120.pt.compute(), recoMET_el_MET_105.pt.compute(), recoMET_el_PFMETNoMu_110_FilterHF.pt.compute(), recoMET_el_semiLepTrigEl_30.pt.compute()], 0, file)

makeEffPlot_varBin("ditau", "PFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [genMET_diTau.pt.compute(),]*3, [recoMET_diTau_PFMET_120.pt.compute(), recoMET_diTau_PFMET_130.pt.compute(), recoMET_diTau_PFMET_140.pt.compute()], 0, file)
makeEffPlo_varBint("ditau", "MET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [genMET_diTau.pt.compute(),]*2, [recoMET_diTau_MET_105.pt.compute(), recoMET_diTau_MET_120.pt.compute()], 0, file)
makeEffPlot_varBin("ditau", "PFMETNoMu", ["HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight", "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight", "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight"], "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [genMET_diTau.pt.compute(),]*4, [recoMET_diTau_PFMETNoMu_110_FilterHF.pt.compute(), recoMET_diTau_PFMETNoMu_120.pt.compute(), recoMET_diTau_PFMETNoMu_120.pt.compute(), recoMET_diTau_PFMETNoMu_140.pt.compute()], 0, file)
makeEffPlot_varBin("ditau", "XFLAV", ["HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1", "HLT_DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1", "HLT_DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1"], "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [genMET_diTau.pt.compute(),]*3, [recoMET_diTau_diTauTrig_35.pt.compute(), recoMET_diTau_diTauTrig_40.pt.compute(), recoMET_diTau_diTauTrig_displaced_32.pt.compute()], 0, file)
makeEffPlot_varBin("ditau", "ALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1"], "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [genMET_diTau.pt.compute(),]*4, [recoMET_diTau_PFMET_120.pt.compute(), recoMET_diTau_MET_105.pt.compute(), recoMET_diTau_PFMETNoMu_110_FilterHF.pt.compute(), recoMET_diTau_diTauTrig_displaced_32.pt.compute()], 0, file)
