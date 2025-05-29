import uproot
import scipy
import matplotlib as mpl
import awkward as ak
import dask_awkward as dak
import numpy as np
import array
import matplotlib as mpl
from matplotlib import pyplot as plt
import ROOT

import dask
from dask import config as cfg
cfg.set({'distributed.scheduler.worker-ttl': None}) # Check if this solves some dask issues
from dask.distributed import Client, wait, progress, LocalCluster 
from dask_jobqueue import HTCondorCluster

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea import processor
from coffea.dataset_tools import (
    apply_to_fileset,
    max_chunks,
    preprocess,
)

import warnings
warnings.filterwarnings("ignore", module="coffea") # Suppress annoying deprecation warnings for coffea vector, c.f. https://github.com/CoffeaTeam/coffea/blob/master/src/coffea/nanoevents/methods/candidate.py
import logging

from leptonPlot import *

NanoAODSchema.warn_missing_crossrefs = False

GenPtMin = 20
GenEtaMax = 2.4


ptRANGE = ["20-25", "25-30", "30-35", "35-40", "40-45", "45-50", "50-55", "55-60", 
           "60-65", "65-70", "70-75", "75-80", "80-85", "85-90", "90-95", "95-100"]

etaRANGE = ["-2.4--2.2", "-2.2--2.0", "-2.0--1.8", "-1.8--1.6", "-1.6--1.4", "-1.4--1.2", "-1.2--1.0", "-1.0--0.8", "-0.8--0.6", "-0.6--0.4", "-0.4--0.2", "-0.2-0",
            "0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0", "1.0-1.2", "1.2-1.4", "1.4-1.6", "1.6-1.8", "1.8-2.0", "2.0-2.2", "2.2-2.4"]

dxyRANGE = ["0-0.5", "0.5-1", "1-1.5", "1.5-2", "2-2.5","2.5-3", "3-3.5", "3.5-4", "4-4.5", "4.5-5",
            "5-5.5", "5.5-6", "6-6.5", "6.5-7", "7-7.5","7.5-8", "8-8.5", "8.5-9", "9-9.5", "9.5-10",
            "10-10.5", "10.5-11", "11-11.5", "11.5-12", "12-12.5","12.5-13", "13-13.5", "13.5-14", "14-14.5", "14.5-15"]
            
lxyRANGE = ["0-0.5", "0.5-1", "1-1.5", "1.5-2", "2-2.5","2.5-3", "3-3.5", "3.5-4", "4-4.5", "4.5-5",
            "5-5.5", "5.5-6", "6-6.5", "6.5-7", "7-7.5","7.5-8", "8-8.5", "8.5-9", "9-9.5", "9.5-10",
            "10-10.5", "10.5-11", "11-11.5", "11.5-12", "12-12.5","12.5-13", "13-13.5", "13.5-14", "14-14.5", "14.5-15"]

SAMP = [
      ['Stau_100_1000mm', 'SIG'],
      ['Stau_100_100mm', 'SIG'],
      ['Stau_100_10mm', 'SIG'],
      ['Stau_100_1mm', 'SIG'],
      ['Stau_100_0p1mm', 'SIG'],
      ['Stau_100_0p01mm', 'SIG'],
      ['Stau_200_1000mm', 'SIG'],
      ['Stau_200_100mm', 'SIG'],
      ['Stau_200_10mm', 'SIG'],
      ['Stau_200_1mm', 'SIG'],
      ['Stau_200_0p1mm', 'SIG'],
      ['Stau_200_0p01mm', 'SIG'],
      ['Stau_300_1000mm', 'SIG'],
      ['Stau_300_100mm', 'SIG'],
      ['Stau_300_10mm', 'SIG'],
      ['Stau_300_1mm', 'SIG'],
      ['Stau_300_0p1mm', 'SIG'],
      ['Stau_300_0p01mm', 'SIG'],
      ['Stau_500_1000mm', 'SIG'],
      ['Stau_500_100mm', 'SIG'],
      ['Stau_500_10mm', 'SIG'],
      ['Stau_500_1mm', 'SIG'],
      ['Stau_500_0p1mm', 'SIG'],
      ['Stau_500_0p01mm', 'SIG'],
      ['QCD50_80', 'QCD'],
      ['QCD80_120','QCD'],
      ['QCD120_170','QCD'],
      ['QCD170_300','QCD'],
      ['QCD300_470','QCD'],
      ['QCD470_600','QCD'],
      ['QCD600_800','QCD'],
      ['QCD800_1000','QCD'],
      ['QCD1000_1400','QCD'],
      ['QCD1400_1800','QCD'],
      ['QCD1800_2400','QCD'],
      ['QCD2400_3200','QCD'],
      ['QCD3200','QCD'],
      ["DYJetsToLL", 'EWK'],  
      #["WtoLNu2Jets", 'EWK'],
      ["TTtoLNu2Q",  'TT'],
      ["TTto4Q", 'TT'],
      ["TTto2L2Nu", 'TT'],
      ]

ROOT.gStyle.SetOptStat(0)

#file = "Staus_M_100_100mm_13p6TeV_Run3Summer22_DisMuon_GenPartMatch.root" 
#file = "SMS-TStauStau_MStau-100_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_NanoAOD.root"
 
#events = NanoEventsFactory.from_root({file:"Events"}, schemaclass=NanoAODSchema).events()
NanoAODSchema.mixins["DisMuon"] = "Muon"
class IDProcessor(processor.ProcessorABC):
    def __init__(self):
        self._accumulator = {}
        for sample in SAMP:
            self._accumulator[sample[0]] = {}        
            self._accumulator[sample[0]]["tight"] = {}
            self._accumulator[sample[0]]["medium"] = {}
            self._accumulator[sample[0]]["gen"] = {}
 
    def process(self, events)    
        id_dict = {}
            
        id_dict[events.metadata['dataset']] = {}
        id_dict[events.metadata['dataset']]["tight"] = {}
        id_dict[events.metadata['dataset']]["medium"] = {}
        id_dict[events.metadata['dataset']]["gen"] = {}

        gpart = events.GenPart
        #dxy = (events.GenPart.vertexY - events.GenVtx.y) * np.cos(events.GenPart.phi) - \
        #      (events.GenPart.vertexX - events.GenVtx.x) * np.sin(events.GenPart.phi)
        #lxy = np.sqrt( (events.GenPart.vertexX - events.GenVtx.x) ** 2 + (events.GenPart.vertexY - events.GenVtx.y) ** 2)
        #events['GenPart'] = ak.with_field(events.GenPart, dxy, where="dxy")
        #events['GenPart'] = ak.with_field(events.GenPart, lxy, where="lxy")
        #print(gpart.fields)
        staus = gpart[(abs(gpart.pdgId) == 1000015) & (gpart.hasFlags("isLastCopy"))]
        #print(staus.fields)
        staus_taus = staus.distinctChildren[(abs(staus.distinctChildren.pdgId) == 15) & (staus.distinctChildren.hasFlags("isLastCopy")) & (staus.distinctChildren.hasFlags("fromHardProcess"))]
        #print(staus_taus.fields)
        staus_taus = ak.firsts(staus_taus[ak.argsort(staus_taus.pt, ascending=False)], axis = 2)
        gen_mu = staus_taus.distinctChildren[(abs(staus_taus.distinctChildren.pdgId) == 13) & (staus_taus.distinctChildren.hasFlags("isLastCopy"))]
        
        gen_mu = gen_mu[(gen_mu.pt > GenPtMin) & (abs(gen_mu.eta) < GenEtaMax)]
        
        ### Make sure that the reco muons can be traced back to a gen particle 
        dis_mu = events.DisMuon[(events.DisMuon.genPartIdx > 0)]
        
        ### Make sure the sample of reco muons we're looking at have a gen particle that is the grandchild of a stau
        reco_mu = dis_mu[(abs(events.GenPart[dis_mu.genPartIdx].distinctParent.distinctParent.pdgId) == 1000015)]

        ### Separate the reco muons into different IDs
        loosereco_mu  = reco_mu[reco_mu.looseId == 1]
        mediumreco_mu = reco_mu[reco_mu.mediumId == 1]
        tightreco_mu  = reco_mu[reco_mu.tightId == 1]
        
        ### Now choose the gen particles those reco muons trace back to
        rfg_mu       = events.GenPart[reco_mu.genPartIdx]

        ### Choose the gen muons based on the reco muon ID
        looserfg_mu  = events.GenPart[loosereco_mu.genPartIdx] 
        mediumrfg_mu = events.GenPart[mediumreco_mu.genPartIdx] 
        tightrfg_mu  = events.GenPart[tightreco_mu.genPartIdx] 
        
        ### Apply fiducial cuts
        rfg_mu = rfg_mu[(rfg_mu.pt > GenPtMin) & (abs(rfg_mu.eta) < GenEtaMax)]
        looserfg_mu = looserfg_mu[(looserfg_mu.pt > GenPtMin) & (abs(looserfg_mu.eta) < GenEtaMax)]
        mediumrfg_mu = mediumrfg_mu[(mediumrfg_mu.pt > GenPtMin) & (abs(mediumrfg_mu.eta) < GenEtaMax)]
        tightrfg_mu = tightrfg_mu[(tightrfg_mu.pt > GenPtMin) & (abs(tightrfg_mu.eta) < GenEtaMax)]
        
        
        id_dict[events.metadata['dataset']]["gen"]["pt"]  = gen_mu.pt
        id_dict[events.metadata['dataset']]["gen"]["eta"] = gen_mu.eta
        #GenMu_dxy = gen_mu.dxy
        #GenMu_lxy = gen_mu.lxy
        
        #RecoMuonsFromGen_pt  = rfg_mu.pt
        #RecoMuonsFromGen_eta = rfg_mu.eta
        #RecoMuonsFromGen_dxy = rfg_mu.dxy
        #RecoMuonsFromGen_lxy = rfg_mu.lxy
        
        LooseRecoMuonsFromGen_pt   = looserfg_mu.pt
        LooseRecoMuonsFromGen_eta  = looserfg_mu.eta
        #LooseRecoMuonsFromGen_dxy  = looserfg_mu.dxy
        #LooseRecoMuonsFromGen_lxy  = looserfg_mu.lxy
        
        id_dict[events.metadata['dataset']]["medium"]["pt"]  = mediumrfg_mu.pt
        id_dict[events.metadata['dataset']]["medium"]["eta"]  = mediumrfg_mu.eta
        #MediumRecoMuonsFromGen_dxy = mediumrfg_mu.dxy
        #MediumRecoMuonsFromGen_lxy = mediumrfg_mu.lxy
        
        id_dict[events.metadata['dataset']]["tight"]["pt"]   = tightrfg_mu.pt
        id_dict[events.metadata['dataset']]["tight"]["eta"]  = tightrfg_mu.eta
        #TightRecoMuonsFromGen_dxy  = tightrfg_mu.dxy
        #TightRecoMuonsFromGen_lxy  = tightrfg_mu.lxy
        
        return id_dict 

    def postprocess(self):
        pass


background_samples = {} 
background_samples["QCD"] = []
background_samples["TT"] = []
background_samples["W"] = []
background_samples["DY"] = []

for samples in SAMP:
    if "QCD" in samples[0]:
        background_samples["QCD"].append( ("/eos/uscms/store/user/dally/second_skim_muon_root/merged/merged_SRcuts_noID_noJetDxy" + samples[0] + "/*.root", xsecs[samples[0]] * lumi * 1000 * 1/num_events[samples[0]]))
    if "TT" in samples[0]:
        background_samples["TT"].append(  ("/eos/uscms/store/user/dally/second_skim_muon_root/merged/merged_SRcuts_noID_noJetDxy" + samples[0] + "/*.root", xsecs[samples[0]] * lumi * 1000 * 1/num_events[samples[0]]))
    if "W" in samples[0]:
        background_samples["W"].append(   ("/eos/uscms/store/user/dally/second_skim_muon_root/merged/merged_SRcuts_noID_noJetDxy" + samples[0] + "/*.root", xsecs[samples[0]] * lumi * 1000 * 1/num_events[samples[0]]))
    if "DY" in samples[0]:
        background_samples["DY"].append(  ("/eos/uscms/store/user/dally/second_skim_muon_root/merged/merged_SRcuts_noID_noJetDxy" + samples[0] + "/*.root", xsecs[samples[0]] * lumi * 1000 * 1/num_events[samples[0]]))
    if "Stau" in samples[0]:
        background_samples[samples[0]] = [("/eos/uscms/store/user/dally/second_skim_muon_root/merged/merged_SRcuts_noID_noJetDxy" + samples[0] + "/*.root", xsecs[samples[0]] * lumi * 1000 * 1/num_events[samples[0]])]

### Efficiency Plots
#makeEffPlot("mu", "ID", ["No ID", "Tight", "Med", "Loose"], "pt", 16, 20, 100, 5, "[GeV]",  [GenMu_pt.compute(),]*4,   [RecoMuonsFromGen_pt.compute(),  TightRecoMuonsFromGen_pt.compute(),  MediumRecoMuonsFromGen_pt.compute(),  LooseRecoMuonsFromGen_pt.compute()], 0, file)
#makeEffPlot("mu", "ID", ["No ID", "Tight", "Med", "Loose"], "eta", 24, -2.4, 2.4, 0.2, " ", [GenMu_eta.compute(),]*4, [RecoMuonsFromGen_eta.compute(), TightRecoMuonsFromGen_eta.compute(), MediumRecoMuonsFromGen_eta.compute(), LooseRecoMuonsFromGen_eta.compute()], 0, file)
#makeEffPlot("mu", "ID", ["No ID", "Tight", "Med", "Loose"], "dxy", 30, 0, 15, 0.5, "[cm]",  [GenMu_dxy.compute(),]*4,  [RecoMuonsFromGen_dxy.compute(), TightRecoMuonsFromGen_dxy.compute(), MediumRecoMuonsFromGen_dxy.compute(), LooseRecoMuonsFromGen_dxy.compute()], 0, file)
#makeEffPlot("mu", "ID", ["No ID", "Tight", "Med", "Loose"], "lxy", 30, 0, 15, 0.5, "[cm]",  [GenMu_lxy.compute(),]*4,  [RecoMuonsFromGen_lxy.compute(), TightRecoMuonsFromGen_lxy.compute(), MediumRecoMuonsFromGen_lxy.compute(), LooseRecoMuonsFromGen_lxy.compute()], 0, file) 
#
#makeEffPlot("e", "ID", ["No ID", "Tight", "Med", "Loose"], "pt", 16, 20, 100, 5, "[GeV]", GenE_pt, [RecoElectronsFromGen_pt, TightRecoElectronsFromGen_pt, MediumRecoElectronsFromGen_pt, LooseRecoElectronsFromGen_pt], 0, file)
#makeEffPlot("e", "ID", ["No ID", "Tight", "Med", "Loose"], "eta", 24, -2.4, 2.4, 0.2, " ", GenE_eta, [RecoElectronsFromGen_eta, TightRecoElectronsFromGen_eta, MediumRecoElectronsFromGen_eta, LooseRecoElectronsFromGen_eta], 0, file)
#makeEffPlot("e", "ID", ["No ID", "Tight", "Med", "Loose", "Veto"], "dxy", 30, 0, 15, 0.5, "[cm]", GenE_dxy, [RecoElectronsFromGen_dxy, TightRecoElectronsFromGen_dxy, MediumRecoElectronsFromGen_dxy, LooseRecoElectronsFromGen_dxy, VetoRecoElectronsFromGen_dxy], 0, file)
#makeEffPlot("e", "ID", ["No ID", "Tight", "Med", "Loose"], "lxy", 30, 0, 15, 0.5, "[cm]", GenE_lxy, [RecoElectronsFromGen_lxy, TightRecoElectronsFromGen_lxy, MediumRecoElectronsFromGen_lxy, LooseRecoElectronsFromGen_lxy], 0, file) 

#makeEffPlotEta("e", ["no ID", "convVeto & lostHits","Veto", "Loose ID", "Medium ID", "Tight ID"], "pt", "[GeV]", GenE_pt, GenE_eta,[RecoElectronsFromGen_pt, Hand2RecoElectronsFromGen_pt, VetoRecoElectronsFromGen_pt, LooseRecoElectronsFromGen_pt, MediumRecoElectronsFromGen_pt, TightRecoElectronsFromGen_pt], [RecoElectronsFromGen_eta, Hand2RecoElectronsFromGen_eta, VetoRecoElectronsFromGen_eta, LooseRecoElectronsFromGen_eta, MediumRecoElectronsFromGen_eta, TightRecoElectronsFromGen_eta], [RecoElectronsFromGen_pt, Hand3RecoElectronsFromGen_pt, VetoRecoElectronsFromGen_pt, LooseRecoElectronsFromGen_pt, MediumRecoElectronsFromGen_pt, TightRecoElectronsFromGen_pt], [RecoElectronsFromGen_eta, Hand3RecoElectronsFromGen_eta, VetoRecoElectronsFromGen_eta, LooseRecoElectronsFromGen_eta, MediumRecoElectronsFromGen_eta, TightRecoElectronsFromGen_eta], 0, file)
#makeEffPlotEta("e", ["no ID", "convVeto & lostHits","Veto", "Loose ID", "Medium ID", "Tight ID"], "dxy", "[cm]", GenE_dxy, GenE_eta,[RecoElectronsFromGen_dxy, Hand2RecoElectronsFromGen_dxy, VetoRecoElectronsFromGen_dxy, LooseRecoElectronsFromGen_dxy, MediumRecoElectronsFromGen_dxy, TightRecoElectronsFromGen_dxy], [RecoElectronsFromGen_eta, Hand2RecoElectronsFromGen_eta, VetoRecoElectronsFromGen_eta, LooseRecoElectronsFromGen_eta, MediumRecoElectronsFromGen_eta, TightRecoElectronsFromGen_eta], [RecoElectronsFromGen_dxy, Hand3RecoElectronsFromGen_dxy, VetoRecoElectronsFromGen_dxy, LooseRecoElectronsFromGen_dxy, MediumRecoElectronsFromGen_dxy, TightRecoElectronsFromGen_dxy], [RecoElectronsFromGen_eta, Hand3RecoElectronsFromGen_eta, VetoRecoElectronsFromGen_eta, LooseRecoElectronsFromGen_eta, MediumRecoElectronsFromGen_eta, TightRecoElectronsFromGen_eta], 0, file, np.linspace(0,15,16))
