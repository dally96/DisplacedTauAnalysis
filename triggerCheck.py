import ROOT 
import numpy as np 
import awkward as ak 
import matplotlib as mpl
from matplotlib import pyplot as plt
import scipy 
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import math 

With_file = "With_trigselec.root"
Without_file = "Without_trigselec.root"

With_events = NanoEventsFactory.from_root({With_file:"Events"}).events()
Without_events = NanoEventsFactory.from_root({Without_file:"Events"}).events()

With_triggers = [
            With_events.HLT.PFMET120_PFMHT120_IDTight,
            With_events.HLT.PFMET130_PFMHT130_IDTight,
            With_events.HLT.PFMET140_PFMHT140_IDTight,
            With_events.HLT.PFMETNoMu120_PFMHTNoMu120_IDTight,
            With_events.HLT.PFMETNoMu130_PFMHTNoMu130_IDTight,
            With_events.HLT.PFMETNoMu140_PFMHTNoMu140_IDTight,
            With_events.HLT.MET105_IsoTrk50,
            With_events.HLT.MET120_IsoTrk50,
            With_events.HLT.PFMET120_PFMHT120_IDTight_PFHT60,
            With_events.HLT.MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight,
            With_events.HLT.PFMETTypeOne140_PFMHT140_IDTight,

            With_events.HLT.DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1,
            With_events.HLT.DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1,
            With_events.HLT.DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1,
            With_events.HLT.Ele30_WPTight_Gsf,
            With_events.HLT.IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1,
            With_events.HLT.IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1,
]

Without_triggers = [
            Without_events.HLT.PFMET120_PFMHT120_IDTight,
            Without_events.HLT.PFMET130_PFMHT130_IDTight,
            Without_events.HLT.PFMET140_PFMHT140_IDTight,
            Without_events.HLT.PFMETNoMu120_PFMHTNoMu120_IDTight,
            Without_events.HLT.PFMETNoMu130_PFMHTNoMu130_IDTight,
            Without_events.HLT.PFMETNoMu140_PFMHTNoMu140_IDTight,
            Without_events.HLT.MET105_IsoTrk50,
            Without_events.HLT.MET120_IsoTrk50,
            Without_events.HLT.PFMET120_PFMHT120_IDTight_PFHT60,
            Without_events.HLT.MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight,
            Without_events.HLT.PFMETTypeOne140_PFMHT140_IDTight,

            Without_events.HLT.DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1,
            Without_events.HLT.DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1,
            Without_events.HLT.DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1,
            Without_events.HLT.Ele30_WPTight_Gsf,
            Without_events.HLT.IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1,
            Without_events.HLT.IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1,
]

With_trig_or = ak.zeros_like(With_triggers[0])
Without_trig_or = ak.zeros_like(Without_triggers[0])

for trig in range(len(With_triggers)):
  With_trig_or = With_trig_or | With_triggers[trig]

for trig in range(len(Without_triggers)):
  Without_trig_or = Without_trig_or | Without_triggers[trig]


With_events = With_events[With_trig_or]
Without_events = Without_events[Without_trig_or]

With_lead_genPart    = ak.firsts(With_events.GenPart.pt[ak.argsort(With_events.GenPart.pt, ascending=False)])
Without_lead_genPart = ak.firsts(Without_events.GenPart.pt[ak.argsort(Without_events.GenPart.pt, ascending=False)])


pt_diff = With_lead_genPart.compute() - Without_lead_genPart.compute()

for pt in range(len(pt_diff)):
  if abs(pt_diff[pt]) > 0:
    print("Not the same event, pt diff is", pt_diff[pt])
  else:
    print("Event is the same")



