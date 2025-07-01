import uproot
import scipy
import matplotlib as mpl
import awkward as ak
import dask_awkward as dak
import numpy as np
import array
import matplotlib as mpl
from matplotlib import pyplot as plt
from glob import glob
from xsec import *

lumi = 38.01 ##fb-1

import dask
import hist
import hist.dask as hda
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

colors  = {'QCD': '#56CBF9', 'TT': '#FDCA40', 'DY': '#D3C0CD'}
markers = {'medium': 'o', 'tight': '^'}
ids     = ['medium', 'tight']
var     = ['pt', 'eta']
#file = "Staus_M_100_100mm_13p6TeV_Run3Summer22_DisMuon_GenPartMatch.root" 
#file = "SMS-TStauStau_MStau-100_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_NanoAOD.root"
 
NanoAODSchema.mixins["DisMuon"] = "Muon"
#events = NanoEventsFactory.from_root({file:"Events"}, schemaclass=NanoAODSchema).events()
class IDProcessor(processor.ProcessorABC):
    def __init__(self):
        NanoAODSchema.mixins["DisMuon"] = "Muon"
        self._accumulator = {}
        for sample in SAMP:
            self._accumulator["tight"] = {}
            self._accumulator["medium"] = {}
            self._accumulator["gen"] = {}
 
    def process(self, events):    
        id_dict = {}
        id_dict["tight"] = {}
        id_dict["medium"] = {}
        id_dict["gen"] = {}

        gpart = events.GenPart
        #dxy = (events.GenPart.vertexY - events.GenVtx.y) * np.cos(events.GenPart.phi) - \
        #      (events.GenPart.vertexX - events.GenVtx.x) * np.sin(events.GenPart.phi)
        #lxy = np.sqrt( (events.GenPart.vertexX - events.GenVtx.x) ** 2 + (events.GenPart.vertexY - events.GenVtx.y) ** 2)
        #events['GenPart'] = ak.with_field(events.GenPart, dxy, where="dxy")
        #events['GenPart'] = ak.with_field(events.GenPart, lxy, where="lxy")
        #print(gpart.fields)
        #staus = gpart[(abs(gpart.pdgId) == 1000015) & (gpart.hasFlags("isLastCopy"))]
        #print(staus.fields)
        #staus_taus = staus.distinctChildren[(abs(staus.distinctChildren.pdgId) == 15) & (staus.distinctChildren.hasFlags("isLastCopy")) & (staus.distinctChildren.hasFlags("fromHardProcess"))]
        #print(staus_taus.fields)
        #staus_taus = ak.firsts(staus_taus[ak.argsort(staus_taus.pt, ascending=False)], axis = 2)
        #gen_mu = staus_taus.distinctChildren[(abs(staus_taus.distinctChildren.pdgId) == 13) & (staus_taus.distinctChildren.hasFlags("isLastCopy"))]  
        gen_mu = events.GenPart[(abs(events.GenPart.pdgId) == 13) & (events.GenPart.hasFlags("isLastCopy"))]
        ### Make sure that the reco muons can be traced back to a gen particle 
        dis_mu = events.DisMuon[(events.DisMuon.genPartIdx > 0)]
        
        ### Make sure the sample of reco muons we're looking at have a gen particle that is the grandchild of a stau
        reco_mu = dis_mu[(abs(events.GenPart.pdgId[dis_mu.genPartIdx]) == 13)]
        #reco_mu = gen_mu.nearest(events.DisMuon)

        ### Separate the reco muons into different IDs
        loosereco_mu  = reco_mu[reco_mu.looseId == 1]
        mediumreco_mu = reco_mu[reco_mu.mediumId == 1]
        tightreco_mu  = reco_mu[reco_mu.tightId == 1]
        
        ### Now choose the gen particles those reco muons trace back to
        rfg_mu       = events.GenPart[reco_mu.genPartIdx]
        #rfg_mu       = reco_mu.nearest(gen_mu)        

        ### Choose the gen muons based on the reco muon ID
        looserfg_mu  = events.GenPart[loosereco_mu.genPartIdx] 
        mediumrfg_mu = events.GenPart[mediumreco_mu.genPartIdx] 
        tightrfg_mu  = events.GenPart[tightreco_mu.genPartIdx] 
        
        #looserfg_mu  = loosereco_mu.nearest(gen_mu) 
        #mediumrfg_mu = mediumreco_mu.nearest(gen_mu) 
        #tightrfg_mu  = tightreco_mu.nearest(gen_mu) 
        
        ### Apply fiducial cuts
        rfg_mu = rfg_mu[(rfg_mu.pt > GenPtMin) & (abs(rfg_mu.eta) < GenEtaMax)]
        looserfg_mu = looserfg_mu[(looserfg_mu.pt > GenPtMin) & (abs(looserfg_mu.eta) < GenEtaMax)]
        mediumrfg_mu = mediumrfg_mu[(mediumrfg_mu.pt > GenPtMin) & (abs(mediumrfg_mu.eta) < GenEtaMax)]
        tightrfg_mu = tightrfg_mu[(tightrfg_mu.pt > GenPtMin) & (abs(tightrfg_mu.eta) < GenEtaMax)]
        gen_mu = gen_mu[(gen_mu.pt > GenPtMin) & (abs(gen_mu.eta) < GenEtaMax)]
        
        
        id_dict["gen"]["pt"]  = gen_mu.pt
        id_dict["gen"]["eta"] = gen_mu.eta
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
        
        id_dict["medium"]["pt"]  = mediumrfg_mu.pt
        id_dict["medium"]["eta"]  = mediumrfg_mu.eta
        #MediumRecoMuonsFromGen_dxy = mediumrfg_mu.dxy
        #MediumRecoMuonsFromGen_lxy = mediumrfg_mu.lxy
        
        id_dict["tight"]["pt"]   = tightrfg_mu.pt
        id_dict["tight"]["eta"]  = tightrfg_mu.eta
        #TightRecoMuonsFromGen_dxy  = tightrfg_mu.dxy
        #TightRecoMuonsFromGen_lxy  = tightrfg_mu.lxy
        
        return id_dict 

    def postprocess(self):
        pass


background_samples = {} 
background_samples["QCD"] = []
background_samples["TT"] = []
#background_samples["W"] = []
background_samples["DY"] = []

for samples in SAMP:
    if "QCD" in samples[0]:
        background_samples["QCD"].extend( glob("/eos/uscms/store/user/dally/second_jet_dxy/merged/merged_loosened_cuts_noIso_" + samples[0] + "/*.root"))
    if "TT" in samples[0]:
        background_samples["TT"].extend(  glob("/eos/uscms/store/user/dally/second_jet_dxy/merged/merged_loosened_cuts_noIso_" + samples[0] + "/*.root"))
#    if "W" in samples[0]:
#        background_samples["W"].append(   ("/eos/uscms/store/user/dally/second_jet_dxy/merged/merged_SRcuts_noIso_noID_noJetDxy_" + samples[0] + "/*.root", xsecs[samples[0]] * lumi * 1000 * 1/num_events[samples[0]]))
    if "DY" in samples[0]:
        background_samples["DY"].extend(  glob("/eos/uscms/store/user/dally/second_jet_dxy/merged/merged_loosened_cuts_noIso_" + samples[0] + "/*.root"))
#    if "Stau" in samples[0]:
#        background_samples[samples[0]] = [("/eos/uscms/store/user/dally/second_skim_muon_root/merged/merged_SRcuts_noIso_noID_noJetDxy_" + samples[0] + "/*.root", xsecs[samples[0]] * lumi * 1000 * 1/num_events[samples[0]])]

#for samples in SAMP: 
   # background_samples["QCD"]

background_histograms = {}
for background, samples in background_samples.items():
    # Initialize a dictionary to hold ROOT histograms for the current background
    background_histograms[background] = {}
    
    background_histograms[background]["medium"] = {}
    background_histograms[background]["tight"] = {}
    background_histograms[background]["gen"] = {}

    background_histograms[background]["medium"]["pt"]  = hda.hist.Hist(hist.axis.Regular(16, 20, 100, name="medium_pt", label = 'pt [GeV]'))
    background_histograms[background]["medium"]["eta"] = hda.hist.Hist(hist.axis.Regular(16, -3.2, 3.2, name="medium_eta", label = r'$\eta$'))

    background_histograms[background]["tight"]["pt"]   = hda.hist.Hist(hist.axis.Regular(16, 20, 100, name="tight_pt", label = 'pt [GeV]'))
    background_histograms[background]["tight"]["eta"]  = hda.hist.Hist(hist.axis.Regular(16, -3.2, 3.2, name="tight_eta", label = r'$\eta$'))

    background_histograms[background]["gen"]["pt"]     = hda.hist.Hist(hist.axis.Regular(16, 20, 100, name="gen_pt", label = 'pt [GeV]'))
    background_histograms[background]["gen"]["eta"]    = hda.hist.Hist(hist.axis.Regular(16, -3.2, 3.2, name="gen_eta", label = r'$\eta$'))

    print(f"For {background} here are samples {samples}") 
    for sample_file in samples:
        try:
            # Step 1: Load events for the sample using dask-awkward
            events = NanoEventsFactory.from_root({sample_file:"Events"}, schemaclass= NanoAODSchema).events()
            #events = uproot.dask(sample_file)
            print(f'Starting {sample_file} histogram')         

            processor_instance = IDProcessor()
            output = processor_instance.process(events)
            print(f'{sample_file} finished successfully')

            background_histograms[background]["medium"]["pt"].fill(dak.flatten(output["medium"]["pt"], axis=None)) 
            background_histograms[background]["medium"]["eta"].fill(dak.flatten(output["medium"]["eta"], axis=None))
                                                              
            background_histograms[background]["tight"]["pt"].fill(dak.flatten(output["tight"]["pt"], axis=None))  
            background_histograms[background]["tight"]["eta"].fill(dak.flatten(output["tight"]["eta"], axis=None)) 

            background_histograms[background]["gen"]["pt"].fill(dak.flatten(output["gen"]["pt"], axis=None))  
            background_histograms[background]["gen"]["eta"].fill(dak.flatten(output["gen"]["eta"], axis=None)) 

        except Exception as e:
            print(f"Error processing {sample_file}: {e}")

for background in background_histograms.keys():
    print(f"Starting on {background}")
    for variable in var:
        print(f"Starting on {variable}")
        fig, ax = plt.subplots(2, 1, figsize=(10, 15))
        for cut in ids:
            print(f"Starting on {cut} ID")
            background_histograms[background][cut][variable].compute().show()
            background_histograms[background]["gen"][variable].compute().show()
            background_histograms[background][cut][variable].compute().plot_ratio(
                            background_histograms[background]["gen"][variable].compute(),
                            rp_num_label        = cut + " " + variable,
                            rp_denom_label      = "gen " + variable,
                            rp_uncert_draw_type = "line",
                            rp_uncertainty_type = "efficiency",
                            ax_dict = {'main_ax': ax[0], f"ratio_ax": ax[1]}
                            )
            print(f"Finished plotting {cut}")
        cut_counter = 0
        for artist in ax[1].containers:
            artist[0].set_color(colors[background])
            artist[0].set_label(ids[cut_counter])
            artist[0].set_marker(markers[ids[cut_counter]])
            artist[0].set_markeredgecolor("black")
            cut_counter += 1
        ax[1].set_title(background)
        ax[0].legend()
        ax[1].legend()
        fig.savefig(f"MuonID_{background}_{variable}.pdf")
        print(f"MuonID_{background}_{variable}.pdf saved!")

#background_histograms["QCD"]["medium"]["pt"].compute().plot_ratio(
#                            background_histograms["QCD"]["gen"]["pt"].compute(),
#                            rp_num_label="medium pt",
#                            rp_denom_label="gen pt",
#                            rp_uncert_draw_type="line",
#                            rp_uncertainty_type="efficiency",
#                            ax_dict = {'main_ax': ax[0], f"ratio_ax": ax[1]}
#                            )
#background_histograms["QCD"]["tight"]["pt"].compute().plot_ratio(
#                            background_histograms["QCD"]["gen"]["pt"].compute(),
#                            rp_num_label="tight pt",
#                            rp_denom_label="gen pt",
#                            rp_uncert_draw_type="line",
#                            rp_uncertainty_type="efficiency",
#                            ax_dict = {'main_ax': ax[0], f"ratio_ax": ax[1]}
#                            )
#ax[0].remove()
#for artist in ax[1].containers:
#    artist[0].set_color("red")
#    artist[0].set_label("red")
#ax[1].legend()
#fig.savefig("example.pdf")



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
