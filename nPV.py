
import sys, argparse, os
import uproot
import uproot.exceptions
from uproot.exceptions import KeyInFileError
import dill as pickle
import json, gzip, correctionlib
import fsspec_xrootd
from  fsspec_xrootd import XRootDFileSystem
import matplotlib as mpl
from matplotlib import pyplot as plt

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
import hist
import hist.dask as hda
from dask import config as cfg
cfg.set({'distributed.scheduler.worker-ttl': None}) # Check if this solves some dask issues
import dask_awkward as dak
from dask_jobqueue import HTCondorCluster
from dask.distributed import Client, wait, progress, LocalCluster

import time
from distributed import Client
from lpcjobqueue import LPCCondorCluster

import warnings
warnings.filterwarnings("ignore", module="coffea") # Suppress annoying deprecation warnings for coffea vector, c.f. https://github.com/CoffeaTeam/coffea/blob/master/src/coffea/nanoevents/methods/candidate.py
import logging

from skimmed_fileset import *
from xsec import *
from selection_function import SR_selections, loose_SR_selections, loose_noIso_SR_selections, event_selection, Zpeak_selection

lumi = 26.3 ##fb-1
colors = ['#56CBF9', '#FDCA40', '#5DFDCB', '#D3C0CD', '#3A5683', '#FF773D']
samples = ['QCD', 'TT', 'W', 'JetMET']

hist_vars = {'nPV': [70, 0, 70, ' '],
            }

hist_dict = {}
for samp  in samples: 
    hist_dict[samp] = {}
    for var in hist_vars.keys():
        hist_dict[samp][var] = hda.hist.Hist(hist.axis.Regular(hist_vars[var][0], hist_vars[var][1], hist_vars[var][2], name = var, label =var + ' ' + hist_vars[var][3])).compute()

def get_stack_maximum(stack):
    max_value = 0 

    for hist in stack:
        max_value = max(max_value, hist.view().max())

    return max_value 
 
class skimProcessor(processor.ProcessorABC):
    def __init__(self, hist_vars, samp):
        # Load pileup weights evaluators 
        self.hist_vars = hist_vars
        self.samp = samp
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
        histogram_dict = {}
        for var in self.hist_vars.keys():
            histogram_dict[var] = hda.hist.Hist(hist.axis.Regular(self.hist_vars[var][0], self.hist_vars[var][1], self.hist_vars[var][2], name = var, label = var + ' ' + self.hist_vars[var][3]))
 
        good_muon_mask = (
            #((events.DisMuon.isGlobal == 1) | (events.DisMuon.isTracker == 1)) & # Equivalent to loose ID cut without isPFcand requirement. Reduce background from non-prompt muons
             (events.DisMuon.pt > 20)
            & (abs(events.DisMuon.eta) < 2.4) # Acceptance of the CMS muon system
        )
        print(f"Defined good muons")
        events['DisMuon'] = events.DisMuon[good_muon_mask]
        print(f"Applied mask to DisMuon")
        num_evts = ak.num(events, axis=0)
        print("Counted the number of original events")
        num_good_muons = ak.count_nonzero(good_muon_mask, axis=1)
        print("Counted the number of events with good muons")
        events = events[num_good_muons >= 1]
        print("Counted the number of events with one or more good muons")
        print(f"Cut muons")

        good_jet_mask = (
            (events.Jet.pt > 20)
            & (abs(events.Jet.eta) < 2.4)
        )
        print("Defined good jets")
        
        events['Jet'] = events.Jet[good_jet_mask]
        num_good_jets = ak.count_nonzero(good_jet_mask, axis=1)
        events = events[num_good_jets >= 1]
        print(f"Cut jets")

        #Noise filter
        noise_mask = (
                     (events.Flag.goodVertices == 1) 
                     & (events.Flag.globalSuperTightHalo2016Filter == 1)
                     & (events.Flag.EcalDeadCellTriggerPrimitiveFilter == 1)
                     & (events.Flag.BadPFMuonFilter == 1)
                     & (events.Flag.BadPFMuonDzFilter == 1)
                     & (events.Flag.hfNoisyHitsFilter == 1)
                     & (events.Flag.eeBadScFilter == 1)
                     & (events.Flag.ecalBadCalibFilter == 1)
                         )

        events = events[noise_mask] 
        print(f"Filtered noise")

        charged_sel = events.Jet.constituents.pf.charge != 0
        dxy = ak.where(ak.all(events.Jet.constituents.pf.charge == 0, axis = -1), -999, ak.flatten(events.Jet.constituents.pf[ak.argmax(events.Jet.constituents.pf[charged_sel].pt, axis=2, keepdims=True)].d0, axis = -1))

        events["Jet"] = ak.with_field(events.Jet, dxy, where = "dxy")

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
        print(f"Applied  TT trigger mask")

        weights = events.genWeight
        if "ext" in self.samp:
            weights = weights / num_events['_'.join(self.samp.split('_')[:-1])]
        else:
            weights = weights / num_events[self.samp] 

        weight_branches = self.process_weight_corrs_and_systs(events, weights)

        events = ak.with_field(events, weight_branches["weight"], "weight") 


        true_samp = self.samp
        if "ext"  in self.samp:
            true_samp = '_'.join(self.samp.split('_')[:-1])
        if "JetMET" in samp:
            histogram_dict['nPV'].fill(dak.flatten(events.PV.npvs, axis = None)) 
        else:
            histogram_dict['nPV'].fill(dak.flatten(events.PV.npvs, axis = None), weight = dak.flatten(events.weight, axis = None) * lumi * 1000 * xsecs[true_samp])

        return histogram_dict

    def postprocess(self):
        pass

if __name__ == "__main__":  

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
        
    tic = time.time()

    #XRootDFileSystem(hostid = "root://cmsxrootd.fnal.gov/", filehandle_cache_size = 250)
    #tic = time.time()
    cluster = LPCCondorCluster(ship_env=True, transfer_input_files='/uscms/home/dally/x509up_u57864')
    #minimum > 0: https://github.com/CoffeaTeam/coffea/issues/465
    cluster.adapt(minimum=1, maximum=10000)
    client = Client(cluster)
    
    with open("preprocessed_fileset.pkl", "rb") as  f:
        Stau_QCD_DY_dataset_runnable = pickle.load(f)    
    del Stau_QCD_DY_dataset_runnable["TTtoLNu2Q"]
    del Stau_QCD_DY_dataset_runnable["TTto2L2Nu"]
    del Stau_QCD_DY_dataset_runnable["TTto4Q"]
    del Stau_QCD_DY_dataset_runnable["DYJetsToLL"]
    with open("TT_preprocessed_fileset.pkl", "rb") as  f:  
        TT_dataset_runnable = pickle.load(f)        
    with open("W_preprocessed_fileset.pkl", "rb") as  f:  
        W_dataset_runnable = pickle.load(f)        
    with open("data_preprocessed_fileset.pkl", "rb") as f:
        JetMET_dataset_runnable = pickle.load(f)
    dataset_runnable = Stau_QCD_DY_dataset_runnable | TT_dataset_runnable | W_dataset_runnable | JetMET_dataset_runnable

    start = time.time()

    for samp in dataset_runnable.keys():
        if "Stau" in samp: continue
        print(samp)
        samp_runnable = {}
        samp_runnable[samp] = dataset_runnable[samp]
        if f"{samp}.pkl" in os.listdir("/eos/uscms/store/user/dally/DisplacedTauAnalysis/prelimSelections/"): continue
        to_compute = apply_to_fileset(
            skimProcessor(hist_vars, samp),
            max_chunks(samp_runnable, 100000000), # add 10 to run over 10
            schemaclass=PFNanoAODSchema,
        )
        (out, ) = dask.compute(to_compute)
        print(out) 
        with open(f"/eos/uscms/store/user/dally/DisplacedTauAnalysis/prelimSelections/{samp}.pkl", "wb")  as f:
            pickle.dump(out, f)
        end = time.time()
        print(f"{samp} took { (end - start) / 60 } minutes to run")
        
        for var in hist_vars.keys():
            if "QCD" in samp:
                hist_dict["QCD"][var] += out[samp][var]
            if "TT" in samp:
                hist_dict["TT"][var] += out[samp][var]
            if "W" in samp:
                hist_dict["W"][var] += out[samp][var]
            if "JetMET" in samp:
                hist_dict["JetMET"][var] += out[samp][var]

    for var in hist_vars.keys():    
        s = hist.Stack.from_dict( {"QCD": hist_dict["QCD"][var],
                                   "TT":  hist_dict["TT"][var],
                                   "W":   hist_dict["W"][var],
                                   } )
        s.plot( stack = True, histtype = "fill", color = [colors[0], colors[1], colors[2]] )
        hist_dict["JetMET"][var].plot( color = black, label = 'data' )
        box = plt.subplot().get_position()
        plt.subplot().set_position([box.x0, box.y0, box.width * 0.8, box.height])   

        plt.xlabel(var + ' ' + hist_vars[var][3])
        plt.ylabel("A.U.")
        plt.yscale('log')
        plt.title(r"$\mathcal{L}_{int}$ = 26.3 fb$^{-1}$")
        plt.ylim( top = max( get_stack_maximum(s), data_histograms["JetMET"][var].view().max() ) * 10 )
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = {"size": 8})
        if "nPV" not in os.listdir("/eos/uscms/store/user/dally/www/"):
            os.mkdir(f"/eos/uscms/store/user/dally/www/nPV")
        plt.savefig(f"/eos/uscms/store/user/dally/www/nPV/data_histogram_{var}.png")
        plt.savefig(f"/eos/uscms/store/user/dally/www/nPV/data_histogram_{var}.pdf")

        plt.cla()
        plt.clf()
        
