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
dxy_bins = [*np.linspace(0,5,6)] + [*np.linspace(5, 30, 6)]
MET_bins = array.array('d', MET_bins)
dxy_bins = array.array('d', dxy_bins)

triggers = ["PFMET_120",
            "PFMET_130",
            "PFMET_140",
            "PFMETNoMu_110_FilterHF",
            "PFMETNoMu_120",
            "PFMETNoMu_130",
            "PFMETNoMu_140",
            "MET_105",
            "MET_120",
            "semiLepTrigEl_30",
        ]

gpart = events.GenPart
gvistau = events.GenVisTau
electrons = events.Electron
taus = events.Tau
jets = events.Jet
met = events.MET
gmet = events.GenMET

gvtx = events.GenVtx

gpart["dxy"] = abs((gpart.vertexY - gvtx.y) * np.cos(gpart.phi) - (gpart.vertexX - gvtx.x) * np.sin(gpart.phi))


staus = gpart[(abs(gpart.pdgId) == 1000015) & (gpart.hasFlags("isLastCopy"))]
staus_taus = staus.distinctChildren[(abs(staus.distinctChildren.pdgId) == 15) & (staus.distinctChildren.hasFlags("isLastCopy"))]
gen_electrons = staus_taus.distinctChildren[(abs(staus_taus.distinctChildren.pdgId) == 11)]

gen_electrons = gen_electrons[(gen_electrons.pt > GenPtMin) & (abs(gen_electrons.eta) < GenEtaMax)]
gen_taus = gvistau[(gvistau.pt > GenPtMin) & (abs(gvistau.eta) < GenEtaMax)]

gen_electrons["dxy"] = abs((gen_electrons.vertexY - gvtx.y) * np.cos(gen_electrons.phi) - (gen_electrons.vertexX - gvtx.x) * np.sin(gen_electrons.phi))
gen_taus["dxy"] = abs((ak.firsts(gpart[gen_taus.genPartIdxMother.compute()].distinctChildren.vertexY, axis = 2) - gvtx.y) * np.cos(gpart[gen_taus.genPartIdxMother.compute()].phi) - (ak.firsts(gpart[gen_taus.genPartIdxMother.compute()].distinctChildren.vertexX, axis = 2) - gvtx.x) * np.sin(gpart[gen_taus.genPartIdxMother.compute()].phi))

gen_electrons["lxy"] = np.sqrt((gen_electrons.distinctParent.vertexX - ak.firsts(staus.vertexX)) ** 2 + (gen_electrons.distinctParent.vertexY - ak.firsts(staus.vertexY)) ** 2)  
gen_taus["lxy"] = np.sqrt((gpart[gen_taus.genPartIdxMother.compute()].vertexX - ak.firsts(staus.vertexX)) ** 2 + (gpart[gen_taus.genPartIdxMother.compute()].vertexY- ak.firsts(staus.vertexY)) ** 2) 

semiLep_el_evt = ak.sum((abs(ak.flatten(gen_electrons.pdgId.compute(), axis = 3)) == 11), axis = -2) == 1
semiLep_el_evt = (ak.where(ak.num(semiLep_el_evt) > 0, True, False) & (ak.num(gen_taus) == 1).compute())
diTau_evt = ak.num(gvistau) == 2

genMET_el    = gmet[semiLep_el_evt]

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

RecoElectronsFromGen = electrons[(abs(electrons.matched_gen.distinctParent.distinctParent.pdgId) == 1000015)]
RecoElectronsFromGen = RecoElectronsFromGen.matched_gen[(RecoElectronsFromGen.matched_gen.pt > GenPtMin) & (abs(RecoElectronsFromGen.matched_gen.eta) < GenEtaMax)]
RecoElectronsFromGen["dxy"] = (RecoElectronsFromGen.vertexY - gvtx.y) * np.cos(RecoElectronsFromGen.phi) - (RecoElectronsFromGen.vertexX - gvtx.x) * np.sin(RecoElectronsFromGen.phi)
RecoElectronsFromGen["lxy"] = np.sqrt((RecoElectronsFromGen.distinctParent.vertexY - gvtx.y) ** 2 + (RecoElectronsFromGen.distinctParent.vertexX- gvtx.x) ** 2)

jets = jets[(jets.pt > 20) & (abs(jets.eta) < 2.4) & (jets.genJetIdx >= 0) & (jets.genJetIdx < ak.num(events.GenJet.pt))]
RecoTausFromGen = jets.nearest(gen_taus, threshold = 0.3)
RecoTausFromGen = RecoTausFromGen[(RecoTausFromGen.pt > GenPtMin) & (abs(RecoTausFromGen.eta) < GenEtaMax)]

recoSemiLep_el_evt = (ak.num(RecoElectronsFromGen) == 1) & (ak.num(ak.drop_none(RecoTausFromGen)) == 1) 

genEl  = gen_electrons[semiLep_el_evt]
recoEl  = gen_electrons[semiLep_el_evt & recoSemiLep_el_evt.compute()]

recoMET_el = gmet[semiLep_el_evt & recoSemiLep_el_evt.compute()]

### Semilep e events which pass triggers 
#### Gen Events
genEl_PFMET_120                  = gen_electrons[semiLep_el_evt & PFMET_120.compute()]
genEl_PFMET_130                  = gen_electrons[semiLep_el_evt & PFMET_130.compute()]
genEl_PFMET_140                  = gen_electrons[semiLep_el_evt & PFMET_140.compute()]

genMET_el_PFMET_120              = gmet[semiLep_el_evt & PFMET_120.compute()] 
genMET_el_PFMET_130              = gmet[semiLep_el_evt & PFMET_130.compute()] 
genMET_el_PFMET_140              = gmet[semiLep_el_evt & PFMET_140.compute()] 

genEl_MET_105                    = gen_electrons[semiLep_el_evt & MET_105.compute()]
genEl_MET_120                    = gen_electrons[semiLep_el_evt & MET_120.compute()]

genMET_el_MET_105                = gmet[semiLep_el_evt & MET_105.compute()]
genMET_el_MET_120                = gmet[semiLep_el_evt & MET_120.compute()]

genEl_PFMETNoMu_110_FilterHF     = gen_electrons[semiLep_el_evt & PFMETNoMu_110_FilterHF.compute()]

genMET_el_PFMETNoMu_110_FilterHF = gmet[semiLep_el_evt & PFMETNoMu_110_FilterHF.compute()]

genEl_PFMETNoMu_120              = gen_electrons[semiLep_el_evt & PFMETNoMu_120.compute()]
genEl_PFMETNoMu_130              = gen_electrons[semiLep_el_evt & PFMETNoMu_130.compute()]
genEl_PFMETNoMu_140              = gen_electrons[semiLep_el_evt & PFMETNoMu_140.compute()]

genMET_el_PFMETNoMu_120          = gmet[semiLep_el_evt & PFMETNoMu_120.compute()] 
genMET_el_PFMETNoMu_130          = gmet[semiLep_el_evt & PFMETNoMu_130.compute()] 
genMET_el_PFMETNoMu_140          = gmet[semiLep_el_evt & PFMETNoMu_140.compute()] 

genEl_semiLepTrigEl_30           = gen_electrons[semiLep_el_evt & semiLepTrigEl_30.compute()]

genMET_el_semiLepTrigEl_30       = gmet[semiLep_el_evt & semiLepTrigEl_30.compute()]

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


### Fraction of Total Events
#e_trigfrac = ROOT.TEfficiency("e_trigfrac", file.split(".")[0]+";;#varepsilon", len(triggers) + 1, -0.5, len(triggers)+0.5)
#
#for i in range(len(triggers) + 1):
#  e_trigfrac.SetTotalEvents(i + 1, int(ak.sum(semiLep_el_evt & recoSemiLep_el_evt.compute())))
#e_trigfrac.SetPassedEvents(1, int(ak.sum(semiLep_el_evt & recoSemiLep_el_evt.compute() & PFMET_120.compute())))
#e_trigfrac.SetPassedEvents(2, int(ak.sum(semiLep_el_evt & recoSemiLep_el_evt.compute() & PFMET_130.compute()))) 
#e_trigfrac.SetPassedEvents(3, int(ak.sum(semiLep_el_evt & recoSemiLep_el_evt.compute() & PFMET_140.compute()))) 
#e_trigfrac.SetPassedEvents(4, int(ak.sum(semiLep_el_evt & recoSemiLep_el_evt.compute() & MET_105.compute())))  
#e_trigfrac.SetPassedEvents(5, int(ak.sum(semiLep_el_evt & recoSemiLep_el_evt.compute() & MET_120.compute())))  
#e_trigfrac.SetPassedEvents(6, int(ak.sum(semiLep_el_evt & recoSemiLep_el_evt.compute() & PFMETNoMu_110_FilterHF.compute()))) 
#e_trigfrac.SetPassedEvents(7, int(ak.sum(semiLep_el_evt & recoSemiLep_el_evt.compute() & PFMETNoMu_120.compute()))) 
#e_trigfrac.SetPassedEvents(8, int(ak.sum(semiLep_el_evt & recoSemiLep_el_evt.compute() & PFMETNoMu_130.compute()))) 
#e_trigfrac.SetPassedEvents(9, int(ak.sum(semiLep_el_evt & recoSemiLep_el_evt.compute() & PFMETNoMu_140.compute()))) 
#e_trigfrac.SetPassedEvents(10,int(ak.sum(semiLep_el_evt & recoSemiLep_el_evt.compute() & semiLepTrigEl_30.compute()))) 
#                                
#e_trigfrac.Draw()
#can.Update()
#
#for i in range(len(triggers)):
#  e_trigfrac.GetPaintedGraph().GetXaxis().SetBinLabel(i+1, triggers[i])  
#
#e_trigfrac.GetPaintedGraph().GetXaxis().LabelsOption("v")
#e_trigfrac.GetPaintedGraph().GetXaxis().SetLabelOffset(0.02)
#
#can.SaveAs("triggerEff_elTau.pdf")
#can.SaveAs("PNG_plots/triggerEff_elTau.png")

### Trigger Eff vs Pt
makeEffPlot_varBin("e", "PFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "e pt", len(MET_bins) - 1, MET_bins, "[GeV]", [recoEl.pt.compute(), recoEl.pt.compute(), recoEl.pt.compute()], [recoEl_PFMET_120.pt.compute(), recoEl_PFMET_130.pt.compute(), recoEl_PFMET_140.pt.compute()], 0, file)
makeEffPlot_varBin("e", "MET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "e pt", len(MET_bins) - 1, MET_bins, "[GeV]", [recoEl.pt.compute(), recoEl.pt.compute()], [recoEl_MET_105.pt.compute(), recoEl_MET_120.pt.compute()], 0, file)
makeEffPlot_varBin("e", "PFMETNoMu", ["HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight", "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight", "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight"], "e pt", len(MET_bins) - 1, MET_bins, "[GeV]", [recoEl.pt.compute(), recoEl.pt.compute(), recoEl.pt.compute(), recoEl.pt.compute()], [recoEl_PFMETNoMu_110_FilterHF.pt.compute(), recoEl_PFMETNoMu_120.pt.compute(), recoEl_PFMETNoMu_130.pt.compute(), recoEl_PFMETNoMu_140.pt.compute()], 0, file)
makeEffPlot_varBin("e", "XFLAV", ["HLT_Ele30_WPTight_Gsf"], "e pt", len(MET_bins) - 1, MET_bins, "[GeV]", [recoEl.pt.compute()], [recoEl_semiLepTrigEl_30.pt.compute()], 0, file) 
makeEffPlot_varBin("e", "ALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_Ele30_WPTight_Gsf"], "e pt", len(MET_bins) - 1, MET_bins, "[GeV]", [recoEl.pt.compute(),]*4, [recoEl_PFMET_120.pt.compute(), recoEl_MET_105.pt.compute(), recoEl_PFMETNoMu_110_FilterHF.pt.compute(), recoEl_semiLepTrigEl_30.pt.compute()], 0, file) 

#### Gen
#makeEffPlot("e", "genPFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "e pt", 16, 20, 100, 5, "[GeV]", [genEl.pt.compute(), genEl.pt.compute(), genEl.pt.compute()], [genEl_PFMET_120.pt.compute(), genEl_PFMET_130.pt.compute(), genEl_PFMET_140.pt.compute()], 0, file)
#makeEffPlot("e", "genMET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "e pt", 16, 20, 100, 5, "[GeV]", [genEl.pt.compute(), genEl.pt.compute()], [genEl_MET_105.pt.compute(), genEl_MET_120.pt.compute()], 0, file)
#makeEffPlot("e", "genPFMETNoMu", ["HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight", "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight", "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight"], "e pt", 16, 20, 100, 5, "[GeV]", [genEl.pt.compute(), genEl.pt.compute(), genEl.pt.compute(), genEl.pt.compute()], [genEl_PFMETNoMu_110_FilterHF.pt.compute(), genEl_PFMETNoMu_120.pt.compute(), genEl_PFMETNoMu_130.pt.compute(), genEl_PFMETNoMu_140.pt.compute()], 0, file)
#makeEffPlot("e", "genXFLAV", ["HLT_Ele30_WPTight_Gsf"], "e pt", 16, 20, 100, 5, "[GeV]", [genEl.pt.compute()], [genEl_semiLepTrigEl_30.pt.compute()], 0, file) 
#makeEffPlot("e", "genALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_Ele30_WPTight_Gsf"], "e pt", 16, 20, 100, 5, "[GeV]", [genEl.pt.compute(),]*4, [genEl_PFMET_120.pt.compute(), genEl_MET_105.pt.compute(), genEl_PFMETNoMu_110_FilterHF.pt.compute(), genEl_semiLepTrigEl_30.pt.compute()], 0, file) 

### Trigger Eff vs Eta
#### Reco
makeEffPlot("e", "PFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "e #eta", 24, -2.4, 2.4, 0.2, "", [recoEl.eta.compute(), recoEl.eta.compute(), recoEl.eta.compute()], [recoEl_PFMET_120.eta.compute(), recoEl_PFMET_130.eta.compute(), recoEl_PFMET_140.eta.compute()], 0, file)
makeEffPlot("e", "MET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "e #eta", 24, -2.4, 2.4, 0.2, "", [recoEl.eta.compute(), recoEl.eta.compute()], [recoEl_MET_105.eta.compute(), recoEl_MET_120.eta.compute()], 0, file)
makeEffPlot("e", "PFMETNoMu", ["HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight", "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight", "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight"], "e #eta", 24, -2.4, 2.4, 0.2, "", [recoEl.eta.compute(), recoEl.eta.compute(), recoEl.eta.compute(), recoEl.eta.compute()], [recoEl_PFMETNoMu_110_FilterHF.eta.compute(), recoEl_PFMETNoMu_120.eta.compute(), recoEl_PFMETNoMu_130.eta.compute(), recoEl_PFMETNoMu_140.eta.compute()], 0, file)
makeEffPlot("e", "XFLAV", ["HLT_Ele30_WPTight_Gsf"], "e #eta", 24, 2.4, 2.4, 0.2, "", [recoEl.eta.compute()], [recoEl_semiLepTrigEl_30.eta.compute()], 0, file) 
makeEffPlot("e", "ALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_Ele30_WPTight_Gsf"], "e #eta", 24, -2.4, 2.4, 0.2, "", [recoEl.eta.compute(),]*4, [recoEl_PFMET_120.eta.compute(), recoEl_MET_105.eta.compute(), recoEl_PFMETNoMu_110_FilterHF.eta.compute(), recoEl_semiLepTrigEl_30.eta.compute()], 0, file)

### Gen
#makeEffPlot("e", "genPFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "e #eta", 24, -2.4, 2.4, 0.2, "", [genEl.eta.compute(), genEl.eta.compute(), genEl.eta.compute()], [genEl_PFMET_120.eta.compute(), genEl_PFMET_130.eta.compute(), genEl_PFMET_140.eta.compute()], 0, file)
#makeEffPlot("e", "genMET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "e #eta", 24, -2.4, 2.4, 0.2, "", [genEl.eta.compute(), genEl.eta.compute()], [genEl_MET_105.eta.compute(), genEl_MET_120.eta.compute()], 0, file)
#makeEffPlot("e", "genPFMETNoMu", ["HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight", "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight", "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight"], "e #eta", 24, -2.4, 2.4, 0.2, "", [genEl.eta.compute(), genEl.eta.compute(), genEl.eta.compute(), genEl.eta.compute()], [genEl_PFMETNoMu_110_FilterHF.eta.compute(), genEl_PFMETNoMu_120.eta.compute(), genEl_PFMETNoMu_130.eta.compute(), genEl_PFMETNoMu_140.eta.compute()], 0, file)
#makeEffPlot("e", "genXFLAV", ["HLT_Ele30_WPTight_Gsf"], "e #eta", 24, 2.4, 2.4, 0.2, "", [genEl.eta.compute()], [genEl_semiLepTrigEl_30.eta.compute()], 0, file) 
#makeEffPlot("e", "genALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_Ele30_WPTight_Gsf"], "e #eta", 24, -2.4, 2.4, 0.2, "", [genEl.eta.compute(),]*4, [genEl_PFMET_120.eta.compute(), genEl_MET_105.eta.compute(), genEl_PFMETNoMu_110_FilterHF.eta.compute(), genEl_semiLepTrigEl_30.eta.compute()], 0, file)

### Trigger Eff vs dxy
#### Reco
makeEffPlot_varBin("e", "PFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "e d_{xy}", len(dxy_bins) - 1, dxy_bins, "[cm]", [recoEl.dxy.compute(), recoEl.dxy.compute(), recoEl.dxy.compute()], [recoEl_PFMET_120.dxy.compute(), recoEl_PFMET_130.dxy.compute(), recoEl_PFMET_140.dxy.compute()], 0, file)
makeEffPlot_varBin("e", "MET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "e d_{xy}", len(dxy_bins) - 1, dxy_bins, "[cm]", [recoEl.dxy.compute(), recoEl.dxy.compute()], [recoEl_MET_105.dxy.compute(), recoEl_MET_120.dxy.compute()], 0, file)
makeEffPlot_varBin("e", "PFMETNoMu", ["HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight", "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight", "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight"], "e d_{xy}", len(dxy_bins) - 1, dxy_bins, "[cm]", [recoEl.dxy.compute(), recoEl.dxy.compute(), recoEl.dxy.compute(), recoEl.dxy.compute()], [recoEl_PFMETNoMu_110_FilterHF.dxy.compute(), recoEl_PFMETNoMu_120.dxy.compute(), recoEl_PFMETNoMu_130.dxy.compute(), recoEl_PFMETNoMu_140.dxy.compute()], 0, file)
makeEffPlot_varBin("e", "XFLAV", ["HLT_Ele30_WPTight_Gsf"], "e d_{xy}", len(dxy_bins) - 1, dxy_bins, "[cm]", [recoEl.dxy.compute()], [recoEl_semiLepTrigEl_30.dxy.compute()], 0, file) 
makeEffPlot_varBin("e", "ALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_Ele30_WPTight_Gsf"], "e d_{xy}", len(dxy_bins) - 1, dxy_bins, "[cm]", [recoEl.dxy.compute(),]*4, [recoEl_PFMET_120.dxy.compute(), recoEl_MET_105.dxy.compute(), recoEl_PFMETNoMu_110_FilterHF.dxy.compute(), recoEl_semiLepTrigEl_30.dxy.compute()], 0, file)

#### Gen
#makeEffPlot("e", "genPFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "e d_{xy}", 15, 0, 15, 1, "[cm]", [genEl.dxy.compute(), genEl.dxy.compute(), genEl.dxy.compute()], [genEl_PFMET_120.dxy.compute(), genEl_PFMET_130.dxy.compute(), genEl_PFMET_140.dxy.compute()], 0, file)
#makeEffPlot("e", "genMET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "e d_{xy}", 15, 0, 15, 1, "[cm]", [genEl.dxy.compute(), genEl.dxy.compute()], [genEl_MET_105.dxy.compute(), genEl_MET_120.dxy.compute()], 0, file)
#makeEffPlot("e", "genPFMETNoMu", ["HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight", "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight", "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight"], "e d_{xy}", 15, 0, 15, 1, "[cm]", [genEl.dxy.compute(), genEl.dxy.compute(), genEl.dxy.compute(), genEl.dxy.compute()], [genEl_PFMETNoMu_110_FilterHF.dxy.compute(), genEl_PFMETNoMu_120.dxy.compute(), genEl_PFMETNoMu_130.dxy.compute(), genEl_PFMETNoMu_140.dxy.compute()], 0, file)
#makeEffPlot("e", "genXFLAV", ["HLT_Ele30_WPTight_Gsf"], "e d_{xy}", 15, 0, 15, 1, "[cm]", [genEl.dxy.compute()], [genEl_semiLepTrigEl_30.dxy.compute()], 0, file) 
#makeEffPlot("e", "genALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_Ele30_WPTight_Gsf"], "e d_{xy}", 15, 0, 15, 1, "[cm]", [genEl.dxy.compute(),]*4, [genEl_PFMET_120.dxy.compute(), genEl_MET_105.dxy.compute(), genEl_PFMETNoMu_110_FilterHF.dxy.compute(), genEl_semiLepTrigEl_30.dxy.compute()], 0, file)

### Trigger Eff vs lxy
#### Reco
makeEffPlot_varBin("e", "PFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "e l_{xy}", len(dxy_bins) - 1, dxy_bins, "[cm]", [recoEl.lxy.compute(), recoEl.lxy.compute(), recoEl.lxy.compute()], [recoEl_PFMET_120.lxy.compute(), recoEl_PFMET_130.lxy.compute(), recoEl_PFMET_140.lxy.compute()], 0, file)
makeEffPlot_varBin("e", "MET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "e l_{xy}", len(dxy_bins) - 1, dxy_bins, "[cm]", [recoEl.lxy.compute(), recoEl.lxy.compute()], [recoEl_MET_105.lxy.compute(), recoEl_MET_120.lxy.compute()], 0, file)
makeEffPlot_varBin("e", "PFMETNoMu", ["HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_PFMETNoMu130_PFMHTNoMu120_IDTight", "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight", "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight"], "e l_{xy}", len(dxy_bins) - 1, dxy_bins, "[cm]", [recoEl.lxy.compute(), recoEl.lxy.compute(), recoEl.lxy.compute(), recoEl.lxy.compute()], [recoEl_PFMETNoMu_110_FilterHF.lxy.compute(), recoEl_PFMETNoMu_120.lxy.compute(), recoEl_PFMETNoMu_130.lxy.compute(), recoEl_PFMETNoMu_140.lxy.compute()], 0, file)
makeEffPlot_varBin("e", "XFLAV", ["HLT_Ele30_WPTight_Gsf"], "e l_{xy}", len(dxy_bins) - 1, dxy_bins, "[cm]", [recoEl.lxy.compute()], [recoEl_semiLepTrigEl_30.lxy.compute()], 0, file) 
makeEffPlot_varBin("e", "ALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_Ele30_WPTight_Gsf"], "e l_{xy}", len(dxy_bins) - 1, dxy_bins, "[cm]", [recoEl.lxy.compute(),]*4, [recoEl_PFMET_120.lxy.compute(), recoEl_MET_105.lxy.compute(), recoEl_PFMETNoMu_110_FilterHF.lxy.compute(), recoEl_semiLepTrigEl_30.lxy.compute()], 0, file)

#### Gen
#makeEffPlot("e", "genPFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "e l_{xy}", 15, 0, 15, 1, "[cm]", [genEl.lxy.compute(), genEl.lxy.compute(), genEl.lxy.compute()], [genEl_PFMET_120.lxy.compute(), genEl_PFMET_130.lxy.compute(), genEl_PFMET_140.lxy.compute()], 0, file)
#makeEffPlot("e", "genMET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "e l_{xy}", 15, 0, 15, 1, "[cm]", [genEl.lxy.compute(), genEl.lxy.compute()], [genEl_MET_105.lxy.compute(), genEl_MET_120.lxy.compute()], 0, file)
#makeEffPlot("e", "genPFMETNoMu", ["HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_PFMETNoMu130_PFMHTNoMu120_IDTight", "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight", "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight"], "e l_{xy}", 15, 0, 15, 1, "[cm]", [genEl.lxy.compute(), genEl.lxy.compute(), genEl.lxy.compute(), genEl.lxy.compute()], [genEl_PFMETNoMu_110_FilterHF.lxy.compute(), genEl_PFMETNoMu_120.lxy.compute(), genEl_PFMETNoMu_130.lxy.compute(), genEl_PFMETNoMu_140.lxy.compute()], 0, file)
#makeEffPlot("e", "genXFLAV", ["HLT_Ele30_WPTight_Gsf"], "e l_{xy}", 15, 0, 15, 1, "[cm]", [genEl.lxy.compute()], [genEl_semiLepTrigEl_30.lxy.compute()], 0, file) 
#makeEffPlot("e", "genALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_Ele30_WPTight_Gsf"], "e l_{xy}", 15, 0, 15, 1, "[cm]", [genEl.lxy.compute(),]*4, [genEl_PFMET_120.lxy.compute(), genEl_MET_105.lxy.compute(), genEl_PFMETNoMu_110_FilterHF.lxy.compute(), genEl_semiLepTrigEl_30.lxy.compute()], 0, file)

### Trigger Eff vs MET pt
#### Reco
makeEffPlot_varBin("e", "PFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [recoMET_el.pt.compute(),]*3, [recoMET_el_PFMET_120.pt.compute(), recoMET_el_PFMET_130.pt.compute(), recoMET_el_PFMET_140.pt.compute()], 0, file)
makeEffPlot_varBin("e", "MET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [recoMET_el.pt.compute(),]*2, [recoMET_el_MET_105.pt.compute(), recoMET_el_MET_120.pt.compute()], 0, file)
makeEffPlot_varBin("e", "PFMETNoMu", ["HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight", "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight", "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight"], "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [recoMET_el.pt.compute(),]*4, [recoMET_el_PFMETNoMu_110_FilterHF.pt.compute(), recoMET_el_PFMETNoMu_120.pt.compute(), recoMET_el_PFMETNoMu_130.pt.compute(), recoMET_el_PFMETNoMu_140.pt.compute()], 0, file)
makeEffPlot_varBin("e", "XFLAV", ["HLT_Ele30_WPTight_Gsf"], "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [recoMET_el.pt.compute()], [recoMET_el_semiLepTrigEl_30.pt.compute()], 0, file)
makeEffPlot_varBin("e", "ALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_Ele30_WPTight_Gsf"],  "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [recoMET_el.pt.compute(),]*4, [recoMET_el_PFMET_120.pt.compute(), recoMET_el_MET_105.pt.compute(), recoMET_el_PFMETNoMu_110_FilterHF.pt.compute(), recoMET_el_semiLepTrigEl_30.pt.compute()], 0, file)

#### Gen
#makeEffPlot_varBin("e", "genPFMET", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET130_PFMHT130_IDTight", "HLT_PFMET140_PFMHT140_IDTight"], "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [genMET_el.pt.compute(),]*3, [genMET_el_PFMET_120.pt.compute(), genMET_el_PFMET_130.pt.compute(), genMET_el_PFMET_140.pt.compute()], 0, file)
#makeEffPlot_varBin("e", "genMET", ["HLT_MET105_IsoTrk50", "HLT_MET120_IsoTrk50"], "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [genMET_el.pt.compute(),]*2, [genMET_el_MET_105.pt.compute(), genMET_el_MET_120.pt.compute()], 0, file)
#makeEffPlot_varBin("e", "genPFMETNoMu", ["HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight", "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight", "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight"], "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [genMET_el.pt.compute(),]*4, [genMET_el_PFMETNoMu_110_FilterHF.pt.compute(), genMET_el_PFMETNoMu_120.pt.compute(), genMET_el_PFMETNoMu_130.pt.compute(), genMET_el_PFMETNoMu_140.pt.compute()], 0, file)
#makeEffPlot_varBin("e", "genXFLAV", ["HLT_Ele30_WPTight_Gsf"], "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [genMET_el.pt.compute()], [genMET_el_semiLepTrigEl_30.pt.compute()], 0, file)
#makeEffPlot_varBin("e", "genALL", ["HLT_PFMET120_PFMHT120_IDTight", "HLT_MET105_IsoTrk50", "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF", "HLT_Ele30_WPTight_Gsf"],  "MET pT", len(MET_bins) - 1, MET_bins, "[GeV]", [genMET_el.pt.compute(),]*4, [genMET_el_PFMET_120.pt.compute(), genMET_el_MET_105.pt.compute(), genMET_el_PFMETNoMu_110_FilterHF.pt.compute(), genMET_el_semiLepTrigEl_30.pt.compute()], 0, file)

