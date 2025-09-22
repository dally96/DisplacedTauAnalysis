import sys, argparse, os
import uproot
import uproot.exceptions
from uproot.exceptions import KeyInFileError
import dill as pickle
import json, gzip, correctionlib
import fsspec_xrootd
from  fsspec_xrootd import XRootDFileSystem

import argparse
import numpy as np
from coffea import processor
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema, NanoAODSchema
from coffea.dataset_tools import (
    apply_to_fileset,
    max_chunks,
    preprocess,
)
from coffea.lumi_tools import LumiData, LumiList, LumiMask
import awkward as ak
import dask
from dask import config as cfg
cfg.set({'distributed.scheduler.worker-ttl': None}) # Check if this solves some dask issues
import dask_awkward as dak
from dask_jobqueue import HTCondorCluster
from dask.distributed import Client, wait, progress, LocalCluster

import time
from distributed import Client
from lpcjobqueue import LPCCondorCluster
import hist
import hist.dask as hda
import matplotlib as mpl
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", module="coffea") # Suppress annoying deprecation warnings for coffea vector, c.f. https://github.com/CoffeaTeam/coffea/blob/master/src/coffea/nanoevents/methods/candidate.py
import logging

from skimmed_fileset import *
from xsec import *
from selection_function import SR_selections, loose_SR_selections, loose_noIso_SR_selections, event_selection, Zpeak_selection

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-m"    , "--muon"    , dest = "muon"   , help = "Leading muon variable"    , default = "pt")
parser.add_argument("-j"    , "--jet"     , dest = "jet"    , help = "Leading jet variable"     , default = "pt")         

leading_var = parser.parse_args()
lumi = 26.3

samp_folder = "/eos/uscms/store/group/lpcdisptau/dally/first_skim/noLepVeto/merged/"
redirector = "root://cmsxrootd.fnal.gov/"
samp_folder_red = "/store/group/lpcdisptau/dally/first_skim/noLepVeto/merged/"
SAMP = os.listdir(samp_folder)

second_skim_dir = 'data_MC_TT_CR'

colors = ['#56CBF9', '#FDCA40', '#5DFDCB', '#D3C0CD', '#3A5683', '#FF773D']
selections = {
              "muon_pt":                    30., ##GeV
              "muon_ID":                    "DisMuon_mediumId",
              "muon_dxy_prompt_max":        50E-4, ##cm
              "muon_dxy_prompt_min":        0E-4, ##cm
              "muon_dxy_displaced_min":     0.1, ##cm
              "muon_dxy_displaced_max":     10.,  ##cm
              "muon_iso_max":               0.18,

              "jet_score":                  0.9, 
              "jet_pt":                     32.,  ##GeV
              "jet_dxy_displaced_min":      0.02, ##cm

              "MET_pt":                     105., ##GeV
             }

class MCProcessor(processor.ProcessorABC):
    def __init__(self, samp, leading_muon_var, leading_jet_var):

        self.samp = samp
        self.leading_muon_var = leading_muon_var
        self.leading_jet_var  = leading_jet_var

        # Load pileup weights evaluators 
        jsonpog = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration"
        pileup_file = jsonpog + "/POG/LUM/2022_Summer22EE/puWeights.json.gz"

        with gzip.open(pileup_file, 'rt') as f:
            self.pileup_set = correctionlib.CorrectionSet.from_string(f.read().strip())

    def get_pileup_weights(self, events, also_syst=False):
        # Apply pileup weights
        evaluator = self.pileup_set["Collisions2022_359022_362760_eraEFG_GoldenJson"]
        sf = evaluator.evaluate(events.Pileup.nTrueInt, "nominal")
        return {'nominal': sf}


    def process_weight_corrs_and_systs(self, events, weights):
        pileup_weights = self.get_pileup_weights(events)
        weight_dict = {
            'weight': weights * pileup_weights['nominal'] #* muon_weights['muon_trigger_SF'],
        }

        return weight_dict

    def process(self, events):

        #Trigger Selection
        trigger_mask = (
                        events.HLT.PFMET120_PFMHT120_IDTight                                    |\
                        events.HLT.PFMET130_PFMHT130_IDTight                                    |\
                        events.HLT.PFMET140_PFMHT140_IDTight                                    |\
                        events.HLT.PFMETNoMu120_PFMHTNoMu120_IDTight                            |\
                        events.HLT.PFMETNoMu130_PFMHTNoMu130_IDTight                            |\
                        events.HLT.PFMETNoMu140_PFMHTNoMu140_IDTight                            |\
                        events.HLT.PFMET120_PFMHT120_IDTight_PFHT60                             |\
                        events.HLT.PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF                   |\
                        events.HLT.PFMETTypeOne140_PFMHT140_IDTight                             |\
                        events.HLT.MET105_IsoTrk50                                              |\
                        events.HLT.MET120_IsoTrk50                                              #|\
                        #events.HLT.IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1   |\
                        #events.HLT.IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1   |\
                        #events.HLT.Ele30_WPTight_Gsf                                            |\
                        #events.HLT.DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1                    |\
                        #events.HLT.DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1        |\
                        #events.HLT.DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1                 #|\
                        #events.HLT.MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight        |\ #Trigger not included in current Nanos
        )
        
        events = events[trigger_mask]
        logger.info(f"Applied trigger mask")
        # Determine if dataset is MC or Data
        sumWeights = num_events[self.samp]
        
        #events["DisMuon"] = events.DisMuon[ak.argsort(events["DisMuon"][self.leading_muon_var], ascending=False, axis = 1)]
        #events["Jet"] = events.Jet[ak.argsort(events["Jet"][self.leading_jet_var], ascending=False, axis = 1)]

        #events["DisMuon"] = ak.singletons(ak.firsts(events.DisMuon))
        #events["Jet"] = ak.singletons(ak.firsts(events.Jet))

        #events = event_selection(events, SR_selections, "TT_CR")
        good_muons  = ((events.DisMuon.pt > selections["muon_pt"])           &\
                       (events.DisMuon.mediumId == 1)                                    &\
                       (abs(events.DisMuon.dxy) > selections["muon_dxy_prompt_min"]) &\
                       (abs(events.DisMuon.dxy) < selections["muon_dxy_prompt_max"]) &\
                       (events.DisMuon.pfRelIso03_all < selections["muon_iso_max"])
                      )
        events['DisMuon'] = events.DisMuon[good_muons]
        num_good_muons = ak.count_nonzero(good_muons, axis=1)
        events = events[num_good_muons >= 1]


        good_jets   = ((events.Jet.disTauTag_score1 < selections["jet_score"])   &\
                       (events.Jet.pt > selections["jet_pt"])                               &\
                       (abs(events.Jet.dxy) > selections["jet_dxy_displaced_min"])          #&\
                       #(abs(events.Jet.dxy) < selections["muon_dxy_prompt_max"])
                      )
        events['Jet'] = events.Jet[good_jets]
        num_good_jets = ak.count_nonzero(good_jets, axis=1)
        events = events[num_good_jets >= 1]

        good_events = (events.PFMET.pt > selections["MET_pt"])
        #events = events[good_muons & good_jets & good_events]
        events = events[good_events]
        #print(f"For {self.samp}, the number of events is {ak.num(events, axis = 0).compute()}")
        weights = events.event/events.event
        weights = weights / sumWeights

        weight_branches = self.process_weight_corrs_and_systs(events, weights)

        events = ak.with_field(events, weight_branches["weight"], "weight") 
        
        out_dict = {}
        out_dict['n_muon'] = hda.hist.Hist(hist.axis.Regular(20, 0, 20, name = 'n_muons', label = 'n_muons'))
        out_dict['n_jet']  = hda.hist.Hist(hist.axis.Regular(20, 0, 20, name = 'n_jets', label = 'n_jets'))
        #out_dict['jet_pt']  = hda.hist.Hist(hist.axis.Regular(49, 20, 1000, name = 'jet_pt', label = 'jet_pt [GeV]'))

        out_dict['n_muon'].fill(dak.flatten(ak.num(events.DisMuon), axis = None), weight = events.weight * lumi * 1000 * xsecs[self.samp])  
        out_dict['n_jet'] .fill(dak.flatten(ak.num(events.Jet), axis = None), weight = events.weight * lumi * 1000 * xsecs[self.samp])  
        #out_dict['jet_pt'] .fill(dak.flatten(events.Jet.pt, axis = None), weight = events.weight * lumi * 1000 * xsecs[self.samp])  

        output = {"histograms": out_dict}
        return output

    def postprocess(self):
        pass

class DataProcessor(processor.ProcessorABC):
    def __init__(self, samp, leading_muon_var, leading_jet_var):
        self.samp = samp
        self.leading_muon_var = leading_muon_var
        self.leading_jet_var  = leading_jet_var
        
    def process(self, events):

        #Trigger Selection
        trigger_mask = (
                        events.HLT.PFMET120_PFMHT120_IDTight                                    |\
                        events.HLT.PFMET130_PFMHT130_IDTight                                    |\
                        events.HLT.PFMET140_PFMHT140_IDTight                                    |\
                        events.HLT.PFMETNoMu120_PFMHTNoMu120_IDTight                            |\
                        events.HLT.PFMETNoMu130_PFMHTNoMu130_IDTight                            |\
                        events.HLT.PFMETNoMu140_PFMHTNoMu140_IDTight                            |\
                        events.HLT.PFMET120_PFMHT120_IDTight_PFHT60                             |\
                        events.HLT.PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF                   |\
                        events.HLT.PFMETTypeOne140_PFMHT140_IDTight                             |\
                        events.HLT.MET105_IsoTrk50                                              |\
                        events.HLT.MET120_IsoTrk50                                              #|\
                        #events.HLT.IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1   |\
                        #events.HLT.IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1   |\
                        #events.HLT.Ele30_WPTight_Gsf                                            |\
                        #events.HLT.DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1                    |\
                        #events.HLT.DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1        |\
                        #events.HLT.DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1                 #|\
                        #events.HLT.MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight        |\ #Trigger not included in current Nanos
        )
        
        events = events[trigger_mask]
        logger.info(f"Applied trigger mask")

        #events["DisMuon"] = events.DisMuon[ak.argsort(events["DisMuon"][self.leading_muon_var], ascending=False, axis = 1)]
        #events["Jet"] = events.Jet[ak.argsort(events["Jet"][self.leading_jet_var], ascending=False, axis = 1)]

        #events["DisMuon"] = ak.singletons(ak.firsts(events.DisMuon))
        #events["Jet"] = ak.singletons(ak.firsts(events.Jet))

        #logger.info(f"Chose leading muon and jet")
        #events = event_selection(events, SR_selections, "TT_CR")

        good_muons  = ((events.DisMuon.pt > selections["muon_pt"])           &\
                       (events.DisMuon.mediumId == 1)                                    &\
                       (abs(events.DisMuon.dxy) > selections["muon_dxy_prompt_min"]) &\
                       (abs(events.DisMuon.dxy) < selections["muon_dxy_prompt_max"]) &\
                       (events.DisMuon.pfRelIso03_all < selections["muon_iso_max"])
                      )
        events['DisMuon'] = events.DisMuon[good_muons]
        num_good_muons = ak.count_nonzero(good_muons, axis=1)
        events = events[num_good_muons >= 1]


        good_jets   = ((events.Jet.disTauTag_score1 < selections["jet_score"])   &\
                       (events.Jet.pt > selections["jet_pt"])                               &\
                       (abs(events.Jet.dxy) > selections["jet_dxy_displaced_min"])          #&\
                       #(abs(events.Jet.dxy) < selections["muon_dxy_prompt_max"])
                      )
        events['Jet'] = events.Jet[good_jets]
        num_good_jets = ak.count_nonzero(good_jets, axis=1)
        events = events[num_good_jets >= 1]

        good_events = (events.PFMET.pt > selections["MET_pt"])
        events = events[good_events]    
        #events = events[good_muons & good_jets & good_events]
        
        out_dict = {}
        out_dict['n_muon'] = hda.hist.Hist(hist.axis.Regular(20, 0, 20, name = 'n_muons', label = 'n_muons'))
        out_dict['n_jet']  = hda.hist.Hist(hist.axis.Regular(20, 0, 20, name = 'n_jets', label = 'n_jets'))
        out_dict['jet_pt']  = hda.hist.Hist(hist.axis.Regular(49, 20, 1000, name = 'jet_pt', label = 'jet_pt [GeV]'))

        out_dict['n_muon'].fill(dak.flatten(ak.num(events.DisMuon), axis = None))  
        out_dict['n_jet'] .fill(dak.flatten(ak.num(events.Jet), axis = None))  
        out_dict['jet_pt'] .fill(dak.flatten(events.Jet.pt, axis = None))  

        output = {"histograms": out_dict}
        return output

    def postprocess(self):
        pass


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    tic = time.time()

    #XRootDFileSystem(hostid = "root://cmsxrootd.fnal.gov/", filehandle_cache_size = 250)
    #tic = time.time()
    #cluster = LPCCondorCluster(ship_env=True, transfer_input_files='/uscms/home/dally/x509up_u57864')
     #minimum > 0: https://github.com/CoffeaTeam/coffea/issues/465
    #cluster.adapt(minimum=1, maximum=10000)
    #client = Client(cluster)

    background_samples = {} 
    data_samples = {}
    background_samples["QCD"] = []
    background_samples["TT"] = []
    background_samples["W"] = []
    background_samples["DY"] = []
    data_samples["JetMET"] = []

    for samples in SAMP:
        print(samples)
        #if "QCD50_80" not in samples and "JetMET" not in samples: continue
        samp = '_'.join(samples.split("_")[1:])
        samp_ext = '_'.join(samples.split("_")[1:-1])
        if "QCD" in samp:
            print(samp)
            background_samples["QCD"].append(    (samp_folder + samples + "/*.root", samp ))
        if "TT" in samp:
            print(samp)
            if "ext" in samp:
                background_samples["TT"].append(     (samp_folder + samples + "/*.root", samp_ext ))
            else:
                background_samples["TT"].append(     (samp_folder + samples + "/*.root", samp ))
        if "W" in samp:
            print(samp)
            if "ext" in samp:
                background_samples["W"].append(      (samp_folder + samples + "/*.root", samp_ext ))
            else:
                background_samples["W"].append(      (samp_folder + samples + "/*.root", samp ))
        if "DY" in samp:
            print(samp)
            if "ext" in samp:
                background_samples["DY"].append(      (samp_folder + samples + "/*.root", samp_ext ))
            else:
                background_samples["DY"].append(     (samp_folder + samples + "/*.root", samp ))
        if "JetMET" in samp: 
            print(samp)
            if "ext" in samp:
                data_samples["JetMET"].append(   (samp_folder + samples + "/*.root", samp_ext ))
            else:
                data_samples["JetMET"].append(   (samp_folder + samples + "/*.root", samp     ))

    background_histograms = {}

    for background, samples in background_samples.items():
        print(background)
        background_histograms[background] = {}
        background_histograms[background]['n_muon'] = hda.hist.Hist(hist.axis.Regular(20, 0, 20, name = 'n_muons', label = 'n_muons')).compute()
        background_histograms[background]['n_jet'] = hda.hist.Hist(hist.axis.Regular(20, 0, 20, name = 'n_jets', label = 'n_jets')).compute()
        background_histograms[background]['jet_pt'] = hda.hist.Hist(hist.axis.Regular(49, 20, 1000, name = 'jet_pt', label = 'jet_pt [GeV]')).compute()
        
        for sample_file, sample_weight in samples:
            try:
                # Step 1: Load events for the sample using dask-awkward
                events = NanoEventsFactory.from_root({sample_file:"Events"}, schemaclass= PFNanoAODSchema).events()
                #print(f"Number of {sample_file} events is {ak.num(events, axis = 0).compute()}")
                #events = uproot.dask(sample_file)
                print(f'Starting {sample_file} histogram')         

                processor_instance = MCProcessor(sample_weight, leading_var.muon, leading_var.jet)
                output = processor_instance.process(events)
                print(f'{sample_file} finished successfully')

                # Loop through each variable's histogram in the output
                for var, dask_histo in output["histograms"].items():
                    background_histograms[background][var] += dask_histo.compute()

            except Exception as e:
                print(f"Error processing {sample_file}: {e}")

    data_histograms = {}

    for data, samples in data_samples.items():
        data_histograms[data] = {}
        data_histograms[data]['n_muon'] = hda.hist.Hist(hist.axis.Regular(20, 0, 20, name = 'n_muons', label = 'n_muons')).compute()
        data_histograms[data]['n_jet'] = hda.hist.Hist(hist.axis.Regular(20, 0, 20, name = 'n_jets', label = 'n_jets')).compute()
        data_histograms[data]['jet_pt'] = hda.hist.Hist(hist.axis.Regular(49, 20, 1000, name = 'jet_pt', label = 'jet_pt [GeV]')).compute()

        for sample_file, sample_weight in samples:
            try:
                # Step 1: Load events for the sample using dask-awkward
                events = NanoEventsFactory.from_root({sample_file:"Events"}, schemaclass= PFNanoAODSchema).events()
                #events = uproot.dask(sample_file)
                print(f'Starting {sample_file} histogram')         

                processor_instance = DataProcessor(sample_weight, leading_var.muon, leading_var.jet)
                output = processor_instance.process(events)
                print(f'{sample_file} finished successfully')

                # Loop through each variable's histogram in the output
                for var, dask_histo in output["histograms"].items():
                    data_histograms[data][var]  += dask_histo.compute()

            except Exception as e:
                print(f"Error processing {sample_file}: {e}")

    for var in background_histograms["QCD"].keys():
        QCD_event_num = background_histograms["QCD"][var].sum()
        TT_event_num = background_histograms["TT"][var].sum()
        DY_event_num = background_histograms["DY"][var].sum()
        W_event_num = background_histograms["W"][var].sum()
    
        total_event_number = QCD_event_num + \
                             TT_event_num  + \
                             DY_event_num  + \
                             W_event_num
        plt.cla()
        plt.clf()
    
        QCD_frac = 0
        TT_frac  = 0
        DY_frac  = 0
        W_frac   = 0
    
        if total_event_number > 0:
            QCD_frac = QCD_event_num/total_event_number
            TT_frac  = TT_event_num/total_event_number
            DY_frac  = DY_event_num/total_event_number
            W_frac  = W_event_num/total_event_number

        s = hist.Stack.from_dict({f"QCD " + "%.2f"%(QCD_frac): background_histograms["QCD"][var],
                                f"TT " + "%.2f"%(TT_frac) : background_histograms["TT"][var],
                                "DY " + "%.2f"%(DY_frac): background_histograms["DY"][var],       
                                "W " + "%.2f"%(W_frac): background_histograms["W"][var],
                                  })
        s.plot(stack = True, histtype= "fill", color = [colors[0], colors[1], colors[3], colors[2]])
        data_histograms["JetMET"][var].plot(color = 'black', label = 'data')
        box = plt.subplot().get_position()
        plt.subplot().set_position([box.x0, box.y0, box.width * 0.8, box.height])   
    
        plt.xlabel(var)
        plt.ylabel("A.U.")
        plt.yscale('log')
        plt.title(r"$\mathcal{L}_{int}$ = 26.3 fb$^{-1}$")
        plt.ylim([1E-2,1E6])
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = {"size": 8})
        if second_skim_dir not in os.listdir("../www/"):
            os.mkdir(f"../www/{second_skim_dir}")
        plt.savefig(f"../www/{second_skim_dir}/data_histogram_{var}.png")
    
        plt.cla()
        plt.clf()
    
        elapsed = time.time() - tic 
        print(f"Finished in {elapsed:.1f}s at {time.time()}")
