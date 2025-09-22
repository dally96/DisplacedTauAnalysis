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

hist_vars = {'n_jets':         [20, 0, 20, ''],
             'HT':             [100, 0, 2000, '[GeV]'],
             'n_muons':        [10, 0, 10, ''],
             'n_bjets_loose':  [10, 0, 10, ''],
             'n_bjets_medium': [10, 0, 10, ''],
             'n_bjets_tight':  [10, 0, 10, ''],
            }

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

folder = "prompt_muon_SR_METTrig_njets"

lumi = 26.3 ##fb-1
colors = ['#56CBF9', '#FDCA40', '#5DFDCB', '#D3C0CD', '#3A5683', '#FF773D']
Stau_colors = ['#EA7AF4', '#B43E8F', '#6200B3', '#218380']

hist_dict = {}
hist_dict['TT'] = {}
hist_dict['W'] = {}

for var in hist_vars.keys():
    hist_dict['TT'][var] = hda.hist.Hist(hist.axis.Regular(hist_vars[var][0], hist_vars[var][1], hist_vars[var][2], name = var, label = var + ' ' + hist_vars[var][3])).compute()
    hist_dict['W'][var]  = hda.hist.Hist(hist.axis.Regular(hist_vars[var][0], hist_vars[var][1], hist_vars[var][2], name = var, label = var + ' ' + hist_vars[var][3])).compute()
#'''
#TT_events = {}
#
#for f in os.listdir("/eos/uscms/store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/"):
#    if "TT" in f:
#        TT_events[f.split('_')[0]] = NanoEventsFactory.from_root({"/eos/uscms/store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/" + f + "/*.root": "Events",}, 
#                                                                                  schemaclass = PFNanoAODSchema, metadata = {"dataset": f.split('_')[0]},).events()
#
#W_events = {}
#
#for f in os.listdir("/eos/uscms/store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/"):
#    if "W" in f:
#        if "4Jets_1J" in f: continue
#        if "Q" in f:
#            sample = '_'.join([f.split('-')[0], f.split('-')[1].split('_')[0], f.split('-')[2].split('_')[1], f.split('-')[2].split('_')[0]])
#            W_events[sample] = NanoEventsFactory.from_root({"/eos/uscms/store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/" + f + "/*.root": "Events",},
#                                                                                  schemaclass = PFNanoAODSchema, metadata = {"dataset": sample},).events()
#        if "LNu" in f: 
#            sample = '_'.join([f.split('-')[0], f.split('-')[1].split('_')[0], f.split('-')[1].split('_')[1]])
#            W_events[sample] = NanoEventsFactory.from_root({"/eos/uscms/store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/" + f + "/*.root": "Events",},
#                                                                                  schemaclass = PFNanoAODSchema, metadata = {"dataset": sample},).events()
#'''             

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

        good_muons  = ((events.DisMuon.pt > selections["muon_pt"])           &\
                       (events.DisMuon.mediumId == 1)                                    &\
                       (abs(events.DisMuon.dxy) > selections["muon_dxy_prompt_min"]) &\
                       (abs(events.DisMuon.dxy) < selections["muon_dxy_prompt_max"]) &\
                       (events.DisMuon.pfRelIso03_all < selections["muon_iso_max"])
                      )
        events['DisMuon'] = events.DisMuon[good_muons]
        num_good_muons = ak.count_nonzero(good_muons, axis=1)
        events = events[num_good_muons >= 1]
        print("Selected good muons")


        good_jets   = ((events.Jet.disTauTag_score1 > selections["jet_score"])   &\
                       (events.Jet.pt > selections["jet_pt"])                               &\
                       (abs(events.Jet.dxy) > selections["jet_dxy_displaced_min"])          #&\
                       #(abs(events.Jet.dxy) < selections["muon_dxy_prompt_max"])
                      )
        events['Jet'] = events.Jet[good_jets]
        num_good_jets = ak.count_nonzero(good_jets, axis=1)
        events = events[num_good_jets >= 1]
        print("Selected good jets")
        print(dak.flatten(ak.num(events.Jet.pt), axis = None))

        good_events = (events.PFMET.pt > selections["MET_pt"])
        events = events[good_events]
        print("Selected good events")

        true_samp = self.samp
        if "ext"  in self.samp:
            true_samp = '_'.join(self.samp.split('_')[:-1])

        histogram_dict['n_jets'].fill(dak.flatten(ak.num(events.Jet.pt), axis = None), weight = dak.flatten(events.weight * lumi * 1000 * xsecs[true_samp], axis = None) )
        print("Filled njets")
        histogram_dict['n_muons'].fill(dak.flatten(ak.num(events.DisMuon.pt) - 1, axis = None), weight = events.weight * lumi * 1000 * xsecs[true_samp] )
        print("Filled nmuons")
        histogram_dict['HT'].fill(dak.flatten(ak.sum(events.Jet.pt, axis = -1), axis = None), weight = events.weight * lumi * 1000 * xsecs[true_samp] )
        print("Filled HT")
        histogram_dict['n_bjets_loose'].fill(dak.flatten(ak.num(events.Jet.pt[events.Jet.btagPNetB > 0.0499]), axis = None), weight = events.weight * lumi * 1000 * xsecs[true_samp] )
        print("Filled n_bjets_loose")
        histogram_dict['n_bjets_medium'].fill(dak.flatten(ak.num(events.Jet.pt[events.Jet.btagPNetB > 0.2605]), axis = None), weight = events.weight * lumi * 1000 * xsecs[true_samp] )
        print("Filled n_bjets_medium")
        histogram_dict['n_bjets_tight'].fill(dak.flatten(ak.num(events.Jet.pt[events.Jet.btagPNetB > 0.6915]), axis = None), weight = events.weight * lumi * 1000 * xsecs[true_samp] )
        print("Filled n_bjets_tight")

        return histogram_dict
        
    def postprocess(self):
        pass

'''
for samp in W_events.keys():
    print(f"Processing {samp}")
    good_muon_mask = (
        #((W_events[samp].DisMuon.isGlobal == 1) | (W_events[samp].DisMuon.isTracker == 1)) & # Equivalent to loose ID cut without isPFcand requirement. Reduce background from non-prompt muons
         (W_events[samp].DisMuon.pt > 20)
        & (abs(W_events[samp].DisMuon.eta) < 2.4) # Acceptance of the CMS muon system
    )
    print(f"Defined good muons")
    W_events[samp]['DisMuon'] = W_events[samp].DisMuon[good_muon_mask]
    print(f"Applied mask to DisMuon")
    num_evts = ak.num(W_events[samp], axis=0)
    print("Counted the number of original W_events[samp]")
    num_good_muons = ak.count_nonzero(good_muon_mask, axis=1)
    print("Counted the number of W_events[samp] with good muons")
    W_events[samp] = W_events[samp][num_good_muons >= 1]
    print("Counted the number of W_events[samp] with one or more good muons")
    print(f"Cut muons")

    good_jet_mask = (
        (W_events[samp].Jet.pt > 20)
        & (abs(W_events[samp].Jet.eta) < 2.4)
    )
    print("Defined good jets")
    
    W_events[samp]['Jet'] = W_events[samp].Jet[good_jet_mask]
    num_good_jets = ak.count_nonzero(good_jet_mask, axis=1)
    W_events[samp] = W_events[samp][num_good_jets >= 1]
    print(f"Cut jets")

    #Noise filter
    noise_mask = (
                 (W_events[samp].Flag.goodVertices == 1) 
                 & (W_events[samp].Flag.globalSuperTightHalo2016Filter == 1)
                 & (W_events[samp].Flag.EcalDeadCellTriggerPrimitiveFilter == 1)
                 & (W_events[samp].Flag.BadPFMuonFilter == 1)
                 & (W_events[samp].Flag.BadPFMuonDzFilter == 1)
                 & (W_events[samp].Flag.hfNoisyHitsFilter == 1)
                 & (W_events[samp].Flag.eeBadScFilter == 1)
                 & (W_events[samp].Flag.ecalBadCalibFilter == 1)
                     )

    W_events[samp] = W_events[samp][noise_mask] 
    print(f"Filtered noise")

    charged_sel = W_events[samp].Jet.constituents.pf.charge != 0
    dxy = ak.where(ak.all(W_events[samp].Jet.constituents.pf.charge == 0, axis = -1), -999, ak.flatten(W_events[samp].Jet.constituents.pf[ak.argmax(W_events[samp].Jet.constituents.pf[charged_sel].pt, axis=2, keepdims=True)].d0, axis = -1))

    W_events[samp]["Jet"] = ak.with_field(W_events[samp].Jet, dxy, where = "dxy")

    trigger_mask = (
                    W_events[samp].HLT.PFMET120_PFMHT120_IDTight                                    |\
                    W_events[samp].HLT.PFMET130_PFMHT130_IDTight                                    |\
                    W_events[samp].HLT.PFMET140_PFMHT140_IDTight                                    |\
                    W_events[samp].HLT.PFMETNoMu120_PFMHTNoMu120_IDTight                            |\
                    W_events[samp].HLT.PFMETNoMu130_PFMHTNoMu130_IDTight                            |\
                    W_events[samp].HLT.PFMETNoMu140_PFMHTNoMu140_IDTight                            |\
                    W_events[samp].HLT.PFMET120_PFMHT120_IDTight_PFHT60                             |\
                    W_events[samp].HLT.PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF                   |\
                    W_events[samp].HLT.PFMETTypeOne140_PFMHT140_IDTight                             |\
                    W_events[samp].HLT.MET105_IsoTrk50                                              |\
                    W_events[samp].HLT.MET120_IsoTrk50                                              #|\
                    #W_events[samp].HLT.IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1   |\
                    #W_events[samp].HLT.IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1   |\
                    #W_events[samp].HLT.Ele30_WPTight_Gsf                                            |\
                    #W_events[samp].HLT.DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1                    |\
                    #W_events[samp].HLT.DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1        |\
                    #W_events[samp].HLT.DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1                 #|\
                    #W_events[samp].HLT.MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight        |\ #Trigger not included in current Nanos
    )
    
    W_events[samp] = W_events[samp][trigger_mask]
    print(f"Applied  W trigger mask")

    weights = W_events[samp].genWeight
    if "ext" in samp:
        weights = weights / num_events['_'.join(W_events[samp].metadata["dataset"].split('_')[:-1])]
    else:
        weights = weights / num_events[W_events[samp].metadata["dataset"]] 

    weight_branches = process_weight_corrs_and_systs(W_events[samp], weights)

    W_events[samp] = ak.with_field(W_events[samp], weight_branches["weight"], "weight") 

    good_muons  = ((W_events[samp].DisMuon.pt > selections["muon_pt"])           &\
                   (W_events[samp].DisMuon.mediumId == 1)                                    &\
                   (abs(W_events[samp].DisMuon.dxy) > selections["muon_dxy_prompt_min"]) &\
                   (abs(W_events[samp].DisMuon.dxy) < selections["muon_dxy_prompt_max"]) &\
                   (W_events[samp].DisMuon.pfRelIso03_all < selections["muon_iso_max"])
                  )
    W_events[samp]['DisMuon'] = W_events[samp].DisMuon[good_muons]
    num_good_muons = ak.count_nonzero(good_muons, axis=1)
    W_events[samp] = W_events[samp][num_good_muons >= 1]
    print("Selected good muons")


    good_jets   = ((W_events[samp].Jet.disTauTag_score1 > selections["jet_score"])   &\
                   (W_events[samp].Jet.pt > selections["jet_pt"])                               &\
                   (abs(W_events[samp].Jet.dxy) > selections["jet_dxy_displaced_min"])          #&\
                   #(abs(W_events[samp].Jet.dxy) < selections["muon_dxy_prompt_max"])
                  )
    W_events[samp]['Jet'] = W_events[samp].Jet[good_jets]
    num_good_jets = ak.count_nonzero(good_jets, axis=1)
    W_events[samp] = W_events[samp][num_good_jets >= 1]
    print("Selected good jets")

    good_events = (W_events[samp].PFMET.pt > selections["MET_pt"])
    W_events[samp] = W_events[samp][good_events]
    print("Selected good events")    


    W_hist = {}
    W_hist['njets'] = hda.hist.Hist(hist.axis.Regular(20, 0, 20, name="njets", label = "njets"))
    W_hist['nmuons'] = hda.hist.Hist(hist.axis.Regular(10, 0, 10, name="nmuons", label = "nmuons"))
    W_hist['HT'] = hda.hist.Hist(hist.axis.Regular(100, 0, 2000, name="H_T", label = "H_T"))
    if "ext"  in samp:
        W_hist['njets'].fill(dak.flatten(ak.num(W_events[samp].Jet.pt), axis = None), weight = W_events[samp].weight * lumi * 1000 * xsecs['_'.join(samp.split('_')[:-1])] )
        print("Filled njets")
        W_hist['nmuons'].fill(dak.flatten(ak.num(W_events[samp].DisMuon.pt), axis = None), weight = W_events[samp].weight * lumi * 1000 * xsecs['_'.join(samp.split('_')[:-1])] )
        print("Filled nmuons")
        W_hist['HT'].fill(dak.flatten(ak.sum(W_events[samp].Jet.pt, axis = -1), axis = None), weight = W_events[samp].weight * lumi * 1000 * xsecs['_'.join(samp.split('_')[:-1])] )
        print("Filled HT")
    else:
        W_hist['njets'].fill(dak.flatten(ak.num(W_events[samp].Jet.pt), axis = None), weight = W_events[samp].weight * lumi * 1000 * xsecs[samp] )
        print("Filled njets")
        W_hist['nmuons'].fill(dak.flatten(ak.num(W_events[samp].DisMuon.pt), axis = None), weight = W_events[samp].weight * lumi * 1000 * xsecs[samp] )
        print("Filled nmuons")
        W_hist['HT'].fill(dak.flatten(ak.sum(W_events[samp].Jet.pt, axis = -1), axis = None), weight = W_events[samp].weight * lumi * 1000 * xsecs[samp] )
        print("Filled HT")

    hist_dict['W']['njets'] += W_hist['njets'].compute()
    print("Add to central njets histogram")
    hist_dict['W']['nmuons'] += W_hist['nmuons'].compute()
    print("Add to central nmuons histogram")
    hist_dict['W']['HT']    += W_hist['HT'].compute()
    print("Add to central HT histogram")
'''
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
    with open("TT_preprocessed_fileset.pkl", "rb") as  f:  
        TT_dataset_runnable = pickle.load(f)        
    with open("W_preprocessed_fileset.pkl", "rb") as  f:  
        W_dataset_runnable = pickle.load(f)        
    dataset_runnable = TT_dataset_runnable | W_dataset_runnable

    for samp in dataset_runnable.keys():
        print(samp)
        samp_runnable = {}
        samp_runnable[samp] = dataset_runnable[samp]
        if f"{samp}.pkl" in os.listdir("/eos/uscms/store/user/dally/DisplacedTauAnalysis/W_CR_hists/"): continue
        to_compute = apply_to_fileset(
            skimProcessor(hist_vars, samp),
            max_chunks(samp_runnable, 100000000), # add 10 to run over 10
            schemaclass=PFNanoAODSchema,
        )
        start = time.time()        
        (out, ) = dask.compute(to_compute)
        print(out) 
        with open("/eos/uscms/store/user/dally/DisplacedTauAnalysis/W_CR_hists/{samp}.pkl", "wb")  as f:
            pickle.dump(out, f)

        start = time.time()

        if "TT" in samp:
            for var in hist_dict['TT'].keys():
                hist_dict['TT'][var] += out[samp][var]
        if "W" in samp:
            for var in hist_dict['W'].keys():
                hist_dict['W'][var] += out[samp][var]
                        

    for var in hist_dict['TT'].keys():
        hist_dict['TT'][var].plot(histtype= "step", color = colors[3], label = "TT")
        hist_dict['W'][var].plot(histtype= "step", color = colors[2], label = "W")
        
        
        box = plt.subplot().get_position()
        plt.subplot().set_position([box.x0, box.y0, box.width * 0.8, box.height])   
        
        plt.xlabel(var)
        plt.ylabel("A.U.")
        plt.yscale('log')
        plt.ylim(top=max(hist_dict['TT'][var].view().max(), hist_dict['W'][var].view().max())*10)
        plt.title(r"TT vs W L$_{int}$ = 26.3 fb$^{-1}$")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if folder not in os.listdir("/eos/uscms/store/user/dally/www/"):
            os.mkdir(f"/eos/uscms/store/user/dally/www/{folder}")
        plt.savefig(f"/eos/uscms/store/user/dally/www/{folder}/{var}.png")
        plt.savefig(f"/eos/uscms/store/user/dally/www/{folder}/{var}.pdf")
        
        plt.cla()
        plt.clf()
        end = time.time()
        print(f"{var} histogram took {(end-start)/60} minutes to complete")
