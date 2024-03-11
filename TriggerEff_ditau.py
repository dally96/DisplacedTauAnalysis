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
taus = events.Tau
jets = events.Jet
met = events.MET
gmet = events.GenMET

gvtx = events.GenVtx

gpart["dxy"] = (gpart.vertexY - gvtx.y) * np.cos(gpart.phi) - (gpart.vertexX - gvtx.x) * np.sin(gpart.phi)


staus = gpart[(abs(gpart.pdgId) == 1000015) & (gpart.hasFlags("isLastCopy"))]
staus_taus = staus.distinctChildren[(abs(staus.distinctChildren.pdgId) == 15) & (staus.distinctChildren.hasFlags("isLastCopy"))]

gen_taus = gvistau[(gvistau.pt > GenPtMin) & (abs(gvistau.eta) < GenEtaMax)]

gen_taus["dxy"] = (ak.firsts(gpart[gen_taus.genPartIdxMother.compute()].distinctChildren.vertexY, axis = 2) - gvtx.y) * np.cos(gpart[gen_taus.genPartIdxMother.compute()].phi) - (ak.firsts(gpart[gen_taus.genPartIdxMother.compute()].distinctChildren.vertexX, axis = 2) - gvtx.x) * np.sin(gpart[gen_taus.genPartIdxMother.compute()].phi)

gen_taus["lxy"] = np.sqrt((gpart[gen_taus.genPartIdxMother.compute()].vertexX - ak.firsts(staus.vertexX)) ** 2 + (gpart[gen_taus.genPartIdxMother.compute()].vertexY- ak.firsts(staus.vertexY)) ** 2) 

diTau_evt = ak.num(gvistau) == 2

genMET_diTau = gmet[diTau_evt]

### Di Tau Triggers (Prompt)
diTauTrig_35 = events.HLT.DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1 == 1
diTauTrig_40 = events.HLT.DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1 == 1
diTauTrig_displaced_32 = events.HLT.DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1 == 1

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

jets = jets[(jets.pt > 20) & (abs(jets.eta) < 2.4) & (jets.genJetIdx >= 0) & (jets.genJetIdx < ak.num(events.GenJet.pt))]
RecoTausFromGen = jets.nearest(gen_taus, threshold = 0.3)
RecoTausFromGen = RecoTausFromGen[(RecoTausFromGen.pt > GenPtMin) & (abs(RecoTausFromGen.eta) < GenEtaMax)]

recoDiTau_evt = (ak.num(ak.drop_none(RecoTausFromGen)) == 2)

genDiTau = gen_taus[diTau_evt]
recoDiTau = gen_taus[diTau_evt & recoDiTau_evt.compute()]


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
#makeEffPlot("ditau", "PFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "#tau pt", 16, 20, 100, 5, "[GeV]", [recoDiTau.pt.compute(), recoDiTau.pt.compute(), recoDiTau.pt.compute()], [recoDiTau_PFMET_120.pt.compute(), recoDiTau_PFMET_130.pt.compute(), recoDiTau_PFMET_140.pt.compute()], 0, file)
#makeEffPlot("ditau", "MET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "#tau pt", 16, 20, 100, 5, "[GeV]", [recoDiTau.pt.compute(), recoDiTau.pt.compute()], [recoDiTau_MET_105.pt.compute(), recoDiTau_MET_120.pt.compute()], 0, file)
#makeEffPlot("ditau", "PFMETNoMu", ["HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight", "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight", "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight"], "#tau pt", 16, 20, 100, 5, "[GeV]", [recoDiTau.pt.compute(), recoDiTau.pt.compute(), recoDiTau.pt.compute(), recoDiTau.pt.compute()], [recoDiTau_PFMETNoMu_110_FilterHF.pt.compute(), recoDiTau_PFMETNoMu_120.pt.compute(), recoDiTau_PFMETNoMu_130.pt.compute(), recoDiTau_PFMETNoMu_140.pt.compute()], 0, file)
#makeEffPlot("ditau", "XFLAV", ["HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1", "HLT_DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1", "HLT_DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1"], "#tau pt", 16, 20, 100, 5, "[GeV]", [recoDiTau.pt.compute(), recoDiTau.pt.compute(), recoDiTau.pt.compute()], [recoDiTau_diTauTrig_35.pt.compute(), recoDiTau_diTauTrig_40.pt.compute(), recoDiTau_diTauTrig_displaced_32.pt.compute()], 0, file) 
#makeEffPlot("ditau", "ALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1"], "#tau pt", 16, 20, 100, 5, "[GeV]", [recoDiTau.pt.compute(),]*4, [recoDiTau_PFMET_120.pt.compute(), recoDiTau_MET_105.pt.compute(), recoDiTau_PFMETNoMu_110_FilterHF.pt.compute(), recoDiTau_diTauTrig_displaced_32.pt.compute()], 0, file)

#### Trigger Eff vs Eta
#makeEffPlot("ditau", "PFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "#tau #eta", 24, -2.4, 2.4, 0.2, "", [recoDiTau.eta.compute(), recoDiTau.eta.compute(), recoDiTau.eta.compute()], [recoDiTau_PFMET_120.eta.compute(), recoDiTau_PFMET_130.eta.compute(), recoDiTau_PFMET_140.eta.compute()], 0, file)
#makeEffPlot("ditau", "MET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "#tau #eta", 24, -2.4, 2.4, 0.2, "", [recoDiTau.eta.compute(), recoDiTau.eta.compute()], [recoDiTau_MET_105.eta.compute(), recoDiTau_MET_120.eta.compute()], 0, file)
#makeEffPlot("ditau", "PFMETNoMu", ["HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight", "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight", "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight"], "#tau #eta", 24, -2.4, 2.4, 0.2, "", [recoDiTau.eta.compute(), recoDiTau.eta.compute(), recoDiTau.eta.compute(), recoDiTau.eta.compute()], [recoDiTau_PFMETNoMu_110_FilterHF.eta.compute(), recoDiTau_PFMETNoMu_120.eta.compute(), recoDiTau_PFMETNoMu_130.eta.compute(), recoDiTau_PFMETNoMu_140.eta.compute()], 0, file)
#makeEffPlot("ditau", "XFLAV", ["HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1", "HLT_DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1", "HLT_DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1"], "#tau #eta", 24, -2.4, 2.4, 0.2, "", [recoDiTau.eta.compute(), recoDiTau.eta.compute(), recoDiTau.eta.compute()], [recoDiTau_diTauTrig_35.eta.compute(), recoDiTau_diTauTrig_40.eta.compute(), recoDiTau_diTauTrig_displaced_32.eta.compute()], 0, file) 
#makeEffPlot("ditau", "ALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1"], "#tau #eta", 24, -2.4, 2.4, 0.2, "", [recoDiTau.eta.compute(),]*4, [recoDiTau_PFMET_120.eta.compute(), recoDiTau_MET_105.eta.compute(), recoDiTau_PFMETNoMu_110_FilterHF.eta.compute(), recoDiTau_diTauTrig_displaced_32.eta.compute()], 0, file)

#### Trigger Eff vs dxy
#makeEffPlot("ditau", "PFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "#tau d_{xy}", 15, 0, 15, 1, "[cm]", [recoDiTau.dxy.compute(), recoDiTau.dxy.compute(), recoDiTau.dxy.compute()], [recoDiTau_PFMET_120.dxy.compute(), recoDiTau_PFMET_130.dxy.compute(), recoDiTau_PFMET_140.dxy.compute()], 0, file)
#makeEffPlot("ditau", "MET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "#tau d_{xy}", 15, 0, 15, 1, "[cm]", [recoDiTau.dxy.compute(), recoDiTau.dxy.compute()], [recoDiTau_MET_105.dxy.compute(), recoDiTau_MET_120.dxy.compute()], 0, file)
#makeEffPlot("ditau", "PFMETNoMu", ["HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight", "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight", "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight"], "#tau d_{xy}", 15, 0, 15, 1, "[cm]", [recoDiTau.dxy.compute(), recoDiTau.dxy.compute(), recoDiTau.dxy.compute(), recoDiTau.dxy.compute()], [recoDiTau_PFMETNoMu_110_FilterHF.dxy.compute(), recoDiTau_PFMETNoMu_120.dxy.compute(), recoDiTau_PFMETNoMu_130.dxy.compute(), recoDiTau_PFMETNoMu_140.dxy.compute()], 0, file)
#makeEffPlot("ditau", "XFLAV", ["HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1", "HLT_DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1", "HLT_DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1"], "#tau d_{xy}", 15, 0, 15, 1, "[cm]", [recoDiTau.dxy.compute(), recoDiTau.dxy.compute(), recoDiTau.dxy.compute()], [recoDiTau_diTauTrig_35.dxy.compute(), recoDiTau_diTauTrig_40.dxy.compute(), recoDiTau_diTauTrig_displaced_32.dxy.compute()], 0, file) 
#makeEffPlot("ditau", "ALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1"], "#tau d_{xy}", 15, 0, 15, 1, "[cm]", [recoDiTau.dxy.compute(),]*4, [recoDiTau_PFMET_120.dxy.compute(), recoDiTau_MET_105.dxy.compute(), recoDiTau_PFMETNoMu_110_FilterHF.dxy.compute(), recoDiTau_diTauTrig_displaced_32.dxy.compute()], 0, file)

#### Trigger Eff vs lxy
#makeEffPlot("ditau", "PFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "#tau l_{xy}", 15, 0, 15, 1, "[cm]", [recoDiTau.lxy.compute(), recoDiTau.lxy.compute(), recoDiTau.lxy.compute()], [recoDiTau_PFMET_120.lxy.compute(), recoDiTau_PFMET_130.lxy.compute(), recoDiTau_PFMET_140.lxy.compute()], 0, file)
#makeEffPlot("ditau", "MET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "#tau l_{xy}", 15, 0, 15, 1, "[cm]", [recoDiTau.lxy.compute(), recoDiTau.lxy.compute()], [recoDiTau_MET_105.lxy.compute(), recoDiTau_MET_120.lxy.compute()], 0, file)
#makeEffPlot("ditau", "PFMETNoMu", ["HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight", "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight", "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight"], "#tau l_{xy}", 15, 0, 15, 1, "[cm]", [recoDiTau.lxy.compute(), recoDiTau.lxy.compute(), recoDiTau.lxy.compute(), recoDiTau.lxy.compute()], [recoDiTau_PFMETNoMu_110_FilterHF.lxy.compute(), recoDiTau_PFMETNoMu_120.lxy.compute(), recoDiTau_PFMETNoMu_130.lxy.compute(), recoDiTau_PFMETNoMu_140.lxy.compute()], 0, file)
#makeEffPlot("ditau", "XFLAV", ["HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1", "HLT_DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1", "HLT_DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1"], "#tau l_{xy}", 15, 0, 15, 1, "[cm]", [recoDiTau.lxy.compute(), recoDiTau.lxy.compute(), recoDiTau.lxy.compute()], [recoDiTau_diTauTrig_35.lxy.compute(), recoDiTau_diTauTrig_40.lxy.compute(), recoDiTau_diTauTrig_displaced_32.lxy.compute()], 0, file) 
#makeEffPlot("ditau", "ALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1"], "#tau l_{xy}", 15, 0, 15, 1, "[cm]", [recoDiTau.lxy.compute(),]*4, [recoDiTau_PFMET_120.lxy.compute(), recoDiTau_MET_105.lxy.compute(), recoDiTau_PFMETNoMu_110_FilterHF.lxy.compute(), recoDiTau_diTauTrig_displaced_32.lxy.compute()], 0, file)

## Trigger Eff vs MET pt
makeEffPlot_varBin("ditau", "PFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [genMET_diTau.pt.compute(),]*3, [recoMET_diTau_PFMET_120.pt.compute(), recoMET_diTau_PFMET_130.pt.compute(), recoMET_diTau_PFMET_140.pt.compute()], 0, file)
makeEffPlot_varBin("ditau", "MET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [genMET_diTau.pt.compute(),]*2, [recoMET_diTau_MET_105.pt.compute(), recoMET_diTau_MET_120.pt.compute()], 0, file)
makeEffPlot_varBin("ditau", "PFMETNoMu", ["HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight", "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight", "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight"], "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [genMET_diTau.pt.compute(),]*4, [recoMET_diTau_PFMETNoMu_110_FilterHF.pt.compute(), recoMET_diTau_PFMETNoMu_120.pt.compute(), recoMET_diTau_PFMETNoMu_120.pt.compute(), recoMET_diTau_PFMETNoMu_140.pt.compute()], 0, file)
makeEffPlot_varBin("ditau", "XFLAV", ["HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1", "HLT_DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1", "HLT_DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1"], "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [genMET_diTau.pt.compute(),]*3, [recoMET_diTau_diTauTrig_35.pt.compute(), recoMET_diTau_diTauTrig_40.pt.compute(), recoMET_diTau_diTauTrig_displaced_32.pt.compute()], 0, file)
makeEffPlot_varBin("ditau", "ALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1"], "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [genMET_diTau.pt.compute(),]*4, [recoMET_diTau_PFMET_120.pt.compute(), recoMET_diTau_MET_105.pt.compute(), recoMET_diTau_PFMETNoMu_110_FilterHF.pt.compute(), recoMET_diTau_diTauTrig_displaced_32.pt.compute()], 0, file)
