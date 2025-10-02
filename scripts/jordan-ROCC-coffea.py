import coffea
import sys
import os
import pickle
import uproot
import scipy
import dask
import hist 
import hist.dask as hda
import time
import json, gzip, correctionlib
import dask_awkward as dak
import warnings
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from coffea import processor
from collections import defaultdict
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.dataset_tools import (
    apply_to_fileset,
    max_chunks,
    preprocess,
)
from dask import config as cfg
cfg.set({'distributed.scheduler.worker-ttl': None}) # Check if this solves some dask issues
import dask_awkward as dak
from dask_jobqueue import HTCondorCluster
from dask.distributed import Client, wait, progress, LocalCluster
from xsec import *

import time
from distributed import Client
from lpcjobqueue import LPCCondorCluster

import warnings
warnings.filterwarnings("ignore", module="coffea") # Suppress annoying deprecation warnings for coffea vector, c.f. https://github.com/CoffeaTeam/coffea/blob/master/src/coffea/nanoevents/methods/candidate.py
import logging

# Prevent branch definition problems
NanoAODSchema.mixins["DisMuon"] = "Muon"
NanoAODSchema.mixins["StauTau"] = "Tau"

# Silence obnoxious warnings
NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore", module="coffea.nanoevents.methods")

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-s"    , "--samp"    , dest = "samp"   , help = "Which samples to run over"    , default = "QCD")
bkg = parser.parse_args()

# --- DEFINITIONS --- #
max_dr = 0.4
score_granularity = 100

lifetimes = ['1mm', '10mm', '100mm', '1000mm']
masses    = ['100', '200', '300', '500']

stau_colors = {'100': '#EA7AF4', '200': '#B43E8F', '300': '#6200B3', '500': '#218380'}
stau_lifetime_colors = {'1mm': '#EA7AF4', '10mm': '#B43E8F', '100mm': '#6200B3', '1000mm': '#218380'}
jetId_colors = {'tight': '#EA7AF4', 'tightLV': '#B43E8F', 'tight_chHEF': '#6200B3', 'tightLV_chHEF': '#218380'}

lumi = 26.3

###From Brandi's code###
pt_bins_low    = np.arange(20, 100, 20)
pt_bins_med    = np.arange(100, 400, 30)
pt_bins_high   = np.arange(400, 600, 40)
pt_bins_higher = np.arange(600, 1000, 50)
pt_bins_eff = np.array(np.concatenate([pt_bins_low, pt_bins_med, pt_bins_high, pt_bins_higher]))
print(pt_bins_eff)
###

def passing_mask(jets, score):
    return jets['score'] >= score

def get_passing_jets(jets, score):
    return jets[passing_mask(jets, score)]

def delta_r_mask(first: ak.highlevel.Array, second: ak.highlevel.Array, threshold: float) -> ak.highlevel.Array:
    mval = first.metric_table(second)
    return ak.all(mval > threshold, axis=-1)

class BGProcessor(processor.ProcessorABC):
    def __init__(self):
        #if "ext" in samp: 
        #    self.samp = '_'.join(samp.split('_')[:-1])
        #else:
        #    self.samp = samp
        # Load pileup weights evaluators 
        jsonpog = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration"
        pileup_file = jsonpog + "/POG/LUM/2022_Summer22EE/puWeights.json.gz"

        with gzip.open(pileup_file, 'rt') as f:
            self.pileup_set = correctionlib.CorrectionSet.from_string(f.read().strip())

    def get_pileup_weights(self, events, also_syst=False):
        # Apply pileup weights
        evaluator = self.pileup_set["Collisions2022_359022_362760_eraEFG_GoldenJson"]
        sf = evaluator.evaluate(events.Pileup.nTrueInt, "nominal")
#         if also_syst:
#             sf_up = evaluator.evaluate(events.Pileup.nTrueInt, "up")
#             sf_down = evaluator.evaluate(events.Pileup.nTrueInt, "down")
#         return {'nominal': sf, 'up': sf_up, 'down': sf_down}
        return {'nominal': sf}


    def process_weight_corrs_and_systs(self, events, weights):
        pileup_weights = self.get_pileup_weights(events)
        # Compute nominal weight and systematic variations by multiplying relevant factors
        # For pileup, do not multiply nominal correction factor, as it is already included in the up/down variations
        # To see this, one can reproduce the ratio in
        # https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/blob/master/misc/LUM/2018_UL/puWeights.png?ref_type=heads
        # from the plain correctionset
        weight_dict = {
            'weight': weights * pileup_weights['nominal'] #* muon_weights['muon_trigger_SF'],
#             'weight_pileup_up': weights * pileup_weights['up'] * muon_weights['muon_trigger_SF'],
#             'weight_pileup_down': weights * pileup_weights['down'] * muon_weights['muon_trigger_SF'],
#             'weight_muon_trigger_up': weights * pileup_weights['nominal'] * (muon_weights['muon_trigger_SF'] + muon_weights['muon_trigger_SF_syst']),
#             'weight_muon_trigger_down': weights * pileup_weights['nominal'] * (muon_weights['muon_trigger_SF'] - muon_weights['muon_trigger_SF_syst']),
        }

        return weight_dict

    def process(self,events):
        if len(events) == 0:
            return {}
        
        if "ext" in events.metadata['dataset']:
            dataset = '_'.join(events.metadata['dataset'].split('_')[:-1])
        else:
            dataset        = events.metadata['dataset']
        sumWeights = num_events[dataset]
        weights = events.genWeight
        weights = weights / sumWeights 
        weight_branches = self.process_weight_corrs_and_systs(events, weights)
        events = ak.with_field(events, weight_branches["weight"], "weight")

        # Define the "good muon" condition for each muon per event
        good_muon_mask = (
            #((events.DisMuon.isGlobal == 1) | (events.DisMuon.isTracker == 1)) & # Equivalent to loose ID cut without isPFcand requirement. Reduce background from non-prompt muons
             (events.DisMuon.pt > 20)
            & (abs(events.DisMuon.eta) < 2.4) # Acceptance of the CMS muon system
        )
        logger.info(f"Defined good muons")
        events['DisMuon'] = events.DisMuon[good_muon_mask]
        num_good_muons = ak.count_nonzero(good_muon_mask, axis=1)
        events = events[num_good_muons >= 1]
        logger.info(f"Cut muons")

        good_jet_mask = (
            (events.Jet.pt > 20)
            & (abs(events.Jet.eta) < 2.4)
        )
        logger.info("Defined good jets")
        
        events['Jet'] = events.Jet[good_jet_mask]
        num_good_jets = ak.count_nonzero(good_jet_mask, axis=1)
        events = events[num_good_jets >= 1]
        logger.info(f"Cut jets")

        

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
        print(f"Starting {dataset}")
        # Determine if dataset is MC or Data
        is_MC = True if "Stau" in dataset else False

        if is_MC:
            vx = events.GenVisTau.parent.vx - events.GenVisTau.parent.parent.vx
            vy = events.GenVisTau.parent.vy - events.GenVisTau.parent.parent.vy
            Lxy = np.sqrt(vx**2 + vy**2)
            events["GenVisTau"] = ak.with_field(events.GenVisTau, Lxy, where="lxy")            

        events["GenVisStauTau"] = events.GenVisTau[(abs(events.GenVisTau.parent.distinctParent.pdgId) == 1000015)   &\
                                               (events.GenVisTau.pt > 20)                                       &\
                                               (abs(events.GenVisTau.eta) < 2.4)                                &\
                                               (events.GenVisTau.parent.distinctParent.hasFlags("isLastCopy"))  &\
                                               (events.GenVisTau.parent.hasFlags("fromHardProcess"))
                                              ]

        if is_MC: 
            events["GenVisStauTau"] = events.GenVisStauTau[(abs(events.GenVisStauTau.lxy) < 100)]

            num_gvt_mask = ak.count_nonzero(events.GenVisStauTau.pt, axis = 1)
            events = events[num_gvt_mask < 2]

        tight_jet_mask         = (events.Jet.isTight)
        tightLV_jet_mask       = (events.Jet.isTightLeptonVeto)
        tight_chHEF_jet_mask   = ((events.Jet.isTight) & (events.Jet.chHEF > 0.01))
        tightLV_chHEF_jet_mask = ((events.Jet.isTightLeptonVeto) & (events.Jet.chHEF > 0.01))

        num_tight_jet = ak.count_nonzero(tight_jet_mask, axis = 1)
        num_tightLV_jet = ak.count_nonzero(tightLV_jet_mask, axis = 1)
        num_tight_chHEF_jet = ak.count_nonzero(tight_chHEF_jet_mask, axis = 1)
        num_tightLV_chHEF_jet = ak.count_nonzero(tightLV_chHEF_jet_mask, axis = 1)

        tight_events = events[num_tight_jet >= 1]
        tightLV_events = events[num_tightLV_jet >= 1]
        tight_chHEF_events = events[num_tight_chHEF_jet >= 1]
        tightLV_chHEF_events = events[num_tightLV_chHEF_jet >= 1]

        tight_events["Jet"] = tight_events.Jet[tight_jet_mask[num_tight_jet >= 1]]
        tightLV_events["Jet"] = tightLV_events.Jet[tightLV_jet_mask[num_tightLV_jet >= 1]]
        tight_chHEF_events["Jet"] = tight_chHEF_events.Jet[tight_chHEF_jet_mask[num_tight_chHEF_jet >= 1]]
        tightLV_chHEF_events["Jet"] = tightLV_chHEF_events.Jet[tightLV_chHEF_jet_mask[num_tightLV_chHEF_jet >= 1]]
        
        tight_events["Jet"] = tight_events.Jet[ak.argsort(tight_events["Jet"]["disTauTag_score1"], ascending=False, axis = 1)]
        tightLV_events["Jet"] = tightLV_events.Jet[ak.argsort(tightLV_events["Jet"]["disTauTag_score1"], ascending=False, axis = 1)]
        tight_chHEF_events["Jet"] = tight_chHEF_events.Jet[ak.argsort(tight_chHEF_events["Jet"]["disTauTag_score1"], ascending=False, axis = 1)]
        tightLV_chHEF_events["Jet"] = tightLV_chHEF_events.Jet[ak.argsort(tightLV_chHEF_events["Jet"]["disTauTag_score1"], ascending=False, axis = 1)]

        tight_events["Jet"] = ak.singletons(ak.firsts(tight_events.Jet))
        tightLV_events["Jet"] = ak.singletons(ak.firsts(tightLV_events.Jet))
        tight_chHEF_events["Jet"] = ak.singletons(ak.firsts(tight_chHEF_events.Jet))
        tightLV_chHEF_events["Jet"] = ak.singletons(ak.firsts(tightLV_chHEF_events.Jet))

        jetid_dict = {}

        jetid_dict["tight"] = {}
        jetid_dict["tightLV"] = {}
        jetid_dict["tight_chHEF"] = {}
        jetid_dict["tightLV_chHEF"] = {}

        jetid_dict["tight"]["events"] = tight_events
        jetid_dict["tightLV"]["events"] = tightLV_events
        jetid_dict["tight_chHEF"]["events"] = tight_chHEF_events
        jetid_dict["tightLV_chHEF"]["events"] = tightLV_chHEF_events
        
        output_dict = {}
        gvt_hist = hda.hist.Hist(hist.axis.Regular(10, 0, 2000, name = 'gvt_pt', label = 'gvt_pt'))
        gvt_hist.fill( dak.flatten(events.GenVisStauTau.pt, axis = None), weight = dak.flatten(ak.broadcast_arrays(events.GenVisStauTau.pt, events.weight)[1] * lumi * 1000 * xsecs[dataset], axis = None)) 
        nGenVisStauTau_weighted = ak.num(ak.drop_none(events.GenVisStauTau.pt), axis = 1) * ak.drop_none(events.weight) * lumi * 1000 * xsecs[dataset]
        #nGenVisStauTau_weighted_total = ak.sum(nGenVisStauTau_weighted)
        nGenVisStauTau_weighted_total = ak.sum( gvt_hist.compute().values(flow=True) )
        output_dict["nGenVisStauTau"] = nGenVisStauTau_weighted_total
        output_dict["all_jet_histo"] = hda.hist.Hist(hist.axis.Regular(score_granularity, 0, 1, name = 'all_jet_score', label = 'score', overflow = True))
        output_dict["all_jet_histo"].fill(dak.flatten(events.Jet.disTauTag_score1, axis = None), weight = dak.flatten(ak.broadcast_arrays(events.Jet.disTauTag_score1, events.weight)[1] * lumi * 1000 * xsecs[dataset], axis = None))
        for cut in jetid_dict.keys():
            output_dict[cut] = {}
            tau_jets = jetid_dict[cut]["events"].GenVisStauTau.nearest(jetid_dict[cut]["events"].Jet, threshold = max_dr)
            match_mask = ak.count_nonzero(tau_jets.pt, axis = 1) == 1
            tau_jets = tau_jets[match_mask]
            fake_jets = jetid_dict[cut]["events"].Jet[delta_r_mask(jetid_dict[cut]["events"].Jet, jetid_dict[cut]["events"].GenVisStauTau, max_dr)]
            fake_mask = ak.count_nonzero(fake_jets.pt, axis = 1) > 0
            #matched_jets   = ak.zip({
            #                "jets": tau_jets,
            #                "score": tau_jets.disTauTag_score1
            #                })
            #unmatched_jets = ak.zip({
            #                "jets": jetid_dict[cut]["events"].Jet[delta_r_mask(jetid_dict[cut]["events"].Jet, jetid_dict[cut]["events"].GenVisStauTau, max_dr)],
            #                "score": jetid_dict[cut]["events"].Jet[delta_r_mask(jetid_dict[cut]["events"].Jet, jetid_dict[cut]["events"].GenVisStauTau, max_dr)].disTauTag_score1
            #                })
            output_dict[cut]["matched_jet_histo"]   = hda.hist.Hist(hist.axis.Regular(score_granularity , 0, 1, name = 'matched_jet_score', label = 'score', overflow = True))
            output_dict[cut]["unmatched_jet_histo"] = hda.hist.Hist(hist.axis.Regular(score_granularity, 0, 1, name = 'unmatched_jet_score', label = 'score', overflow = True))

            output_dict[cut]["matched_jet_histo"].fill(dak.flatten(tau_jets.disTauTag_score1, axis = None), weight = dak.flatten(jetid_dict[cut]["events"].weight[match_mask], axis = None) * lumi * 1000 * xsecs[dataset]) 
            output_dict[cut]["unmatched_jet_histo"].fill(dak.flatten(fake_jets.disTauTag_score1, axis = None),  weight = dak.flatten(jetid_dict[cut]["events"].weight[fake_mask], axis = None) * lumi * 1000 * xsecs[dataset])
            if 'tightLV' in cut and 'chHEF' not in cut:
                tau_jets = tau_jets[tau_jets.disTauTag_score1 > 0.9]
                tau_jets_tau = tau_jets.nearest(jetid_dict[cut]["events"].GenVisStauTau[match_mask], threshold = max_dr)
                #print(tau_jets_tau.pt.compute())
                #print(jetid_dict[cut]["events"]["GenVisStauTau"].pt.compute())
                
                num_tjt = ak.count_nonzero(tau_jets_tau.pt, axis = 1)
                #tau_jets_tau = tau_jets_tau[num_tjt > 0]

                #pt_num =  hda.hist.Hist(hist.axis.Regular(pt_bins_eff, name = "pt_num", label = 'pt_num'))
                #pt_num_w_weights =  hda.hist.Hist(hist.axis.Regular(pt_bins_eff, name = "pt_num", label = 'pt_num'))
                #pt_den = hda.hist.Hist(hist.axis.Regular(pt_bins_eff, name = "pt_den", label = 'pt_den'))
                #pt_den_w_weights =  hda.hist.Hist(hist.axis.Regular(pt_bins_eff, name = "pt_den", label = 'pt_den')) 
                
                pt_num =  hda.hist.Hist(hist.axis.Variable(pt_bins_eff, name = "pt_num", label = 'pt_num'))
                pt_num_w_weights =  hda.hist.Hist(hist.axis.Variable(pt_bins_eff, name = "pt_num", label = 'pt_num'))
                pt_den = hda.hist.Hist(hist.axis.Variable(pt_bins_eff, name = "pt_den", label = 'pt_den'))
                pt_den_w_weights =  hda.hist.Hist(hist.axis.Variable(pt_bins_eff, name = "pt_den", label = 'pt_den')) 
                
                pt_num.fill(ak.flatten(tau_jets_tau.pt, axis = None))
                #pt_num_w_weights.fill(ak.flatten(tau_jets_tau.pt, axis = None), weight = ak.flatten(jetid_dict[cut]["events"].weight[match_mask][num_tjt > 0], axis = None) * lumi * 1000 * xsecs[dataset])

                pt_den.fill(ak.flatten(jetid_dict[cut]["events"]["GenVisStauTau"].pt, axis = None))
                #pt_den_w_weights.fill(ak.flatten(jetid_dict[cut]["events"]["GenVisStauTau"][num_tjt > 0], axis = None), weight = ak.flatten(jetid_dict[cut]["events"].weight[match_mask][num_tjt > 0], axis = None) * lumi * 1000 * xsecs[dataset])

                output_dict["pt_num"] = pt_num
                #output_dict["pt_numw_weights"] = pt_num_w_weights
                output_dict["pt_den"] = pt_den
                #output_dict["pt_den_w_weights"] = pt_den_w_weights


        #print(jetid_dict["tight"]["total_jets"].compute())
        #print(jetid_dict["tight"]["matched_jet_histo"].compute())
        #print(jetid_dict[cut]['unmatched_jet_histo'].compute())
        #total_jets     = ak.sum( ak.num(events.Jet) )
        #tau_jets       = events.GenVisTau.nearest(events.Jet, threshold = max_dr)
        #match_mask     = ak.num(tau_jets) > 0 
        #matched_jets   = ak.zip({
        #                "jets": tau_jets,
        #                "score": tau_jets.disTauTag_score1
        #                })
        #unmatched_jets = ak.zip({
        #                "jets": events.Jet[delta_r_mask(events.Jet, events.GenVisTau, max_dr)],
        #                "score": events.Jet[delta_r_mask(events.Jet, events.GenVisTau, max_dr)].disTauTag_score1
        #                })
    
        #matched_jet_histo = hda.hist.Hist(hist.axis.Regular(score_granularity , 0, 1, name = 'matched_jet_score', label = 'score', overflow = True))
        #unmatched_jet_histo = hda.hist.Hist(hist.axis.Regular(score_granularity, 0, 1, name = 'unmatched_jet_score', label = 'score', overflow = True))

        #matched_jet_histo.fill(dak.flatten(tau_jets.disTauTag_score1, axis = None))
        #unmatched_jet_histo.fill(dak.flatten(events.Jet[delta_r_mask(events.Jet, events.GenVisTau, 0.3)].disTauTag_score1, axis = None))

        #results = []
        #scores = np.linspace(0, 1, score_granularity)
        #print(f"scores is {scores}")
        #for s in scores:
        #    pmj = get_passing_jets(matched_jets, s) # passing matched jets
        #    pfj = get_passing_jets(unmatched_jets, s) # passing fake jets
        #    num_pmj = ak.sum( ak.num(pmj) )
        #    #print(f"num_pmj is {num_pmj.compute()}")
        #    num_pfj = ak.sum( ak.num(pfj) )
        #    #print(f"num_pfj is {num_pfj.compute()}")
        #    results.append( (dataset, s, num_pmj, num_pfj) )


        #return {
        #    'matched_jet_score': matched_jet_histo,
        #    'unmatched_jet_score': unmatched_jet_histo,
        #    'total_jets':  total_jets,
        #    }

        return output_dict

    def postprocess(self,accumulator):
        pass

# --- IMPORT DATASETS --- #
#with open("root_files.txt") as f:
#    lines = [line.strip() for line in f if line.strip()]
#
#xrootd_prefix    = 'root://cmseos.fnal.gov/'
#base_prefix_full = '/eos/uscms/store/user/dally/DisplacedTauAnalysis/skimmed_muon_'
#base_prefix      = '/eos/uscms'
#base_suffix      = '_root:'
#
#paths = []
#sets  = []
#
#i = 0 
#while i < len(lines):
#    if '.root' in lines[i]:
#        i += 1
#        continue
#
#    base_path = lines[i]
#    dataset_name = base_path.removeprefix(base_prefix_full).removesuffix(base_suffix)
#    sets.append(dataset_name)
#    xrootd_path = base_path.removeprefix(base_prefix).removesuffix(':')
#
#    # Look ahead for .root file
#    if i + 1 < len(lines) and '.root' in lines[i + 1]: 
#        root_file = lines[i + 1]
#        paths.append(xrootd_prefix + xrootd_path + '/' + root_file)
#        i += 2  # Move past both lines
#    else:
#        i += 1  # Only move past base path
#
#fileset = {}
#
#for data in sets:
#    matched_paths = [p for p in paths if data in p]
#    fileset[data] = {
#        "files": {p: "Events" for p in matched_paths}
#    }
#
tstart = time.time()
#
#dataset_runnable, dataset_updated = preprocess(
#    fileset,
#    align_clusters=False,
#    step_size=100_000,
#    files_per_batch=1,
#    skip_bad_files=False,
#    save_form=False,
#)
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    cluster = LPCCondorCluster(ship_env=True, transfer_input_files='/uscms/home/dally/x509up_u57864')
     #minimum > 0: https://github.com/CoffeaTeam/coffea/issues/465
    cluster.adapt(minimum=1, maximum=10000)
    client = Client(cluster)

    all_matched = 0
    
    
    # Aggregation dict: s → [sum of 2nd elements, sum of 3rd elements]
    
    s_sums = defaultdict(lambda: [0])
    
    dataset_runnable = {}
    stau_dict = {}
    
    unmatched_histo =  {}
    unmatched_histo['tight'] = hda.hist.Hist(hist.axis.Regular(score_granularity, 0, 1, name = 'unmatched_jet_score', label = 'score', overflow = True)).compute()
    unmatched_histo['tightLV'] = hda.hist.Hist(hist.axis.Regular(score_granularity, 0, 1, name = 'unmatched_jet_score', label = 'score', overflow = True)).compute()
    unmatched_histo['tight_chHEF'] = hda.hist.Hist(hist.axis.Regular(score_granularity, 0, 1, name = 'unmatched_jet_score', label = 'score', overflow = True)).compute()
    unmatched_histo['tightLV_chHEF'] = hda.hist.Hist(hist.axis.Regular(score_granularity, 0, 1, name = 'unmatched_jet_score', label = 'score', overflow = True)).compute()
    
    
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
    dataset_runnable = Stau_QCD_DY_dataset_runnable | TT_dataset_runnable | W_dataset_runnable 
    #dataset_runnable = {"QCD1000_1400": dataset_runnable["QCD1000_1400"], "QCD50_80": dataset_runnable["QCD50_80"]}

    #with open("merged_preprocessed_fileset.pkl", "rb") as f:
        #sub_dataset_runnable = pickle.load(f)
    #for dataset in sub_dataset_runnable.keys():
    #    if "JetMET" in dataset or "DY" in dataset: continue
    #    #if "Stau" not in dataset and "QCD50_80" not in dataset: continue
    #    print(dataset)
    #    dataset_runnable[dataset] = sub_dataset_runnable[dataset] 
    
    for samp in dataset_runnable.keys():
        samp_runnable = {}
        samp_runnable[samp] = dataset_runnable[samp]
        if bkg.samp not in samp: continue        
        #if f"{samp}.pkl" in os.listdir("/eos/uscms/store/user/dally/DisplacedTauAnalysis/ROC_hists/"): continue
        to_compute = apply_to_fileset(
            BGProcessor(),
            max_chunks(samp_runnable, 100000000), # add 10 to run over 10
            schemaclass=NanoAODSchema,
        )
        
        (out, ) = dask.compute(to_compute)
        print(out) 
        
        with open(f"/eos/uscms/store/user/dally/DisplacedTauAnalysis/ROC_hists/{samp}.pkl", "wb") as f:
            pickle.dump(out, f)
        '''
        for samp in out.keys():
            if "Stau" in samp:
                lifetime = samp.split('_')[-1]
                mass     = samp.split('_')[-2]
                if lifetime in stau_dict.keys():
                    stau_dict[lifetime][mass] = {}
                    for cut in out[samp].keys():
                        stau_dict[lifetime][mass][cut] = {}
                else:
                    stau_dict[lifetime] = {}
                    stau_dict[lifetime][mass] = {}
                    for cut in out[samp].keys():
                        stau_dict[lifetime][mass][cut] = {}
                for cut in out[samp].keys():
                    out[samp][cut]['matched_jet_histo'].values(flow = True)[-2] = out[samp][cut]['matched_jet_histo'].values(flow = True)[-2] + out[samp][cut]['matched_jet_histo'].values(flow = True)[-1]
        
                    matched_binned_scores = out[samp][cut]['matched_jet_histo'].values()
        
                    matched_score_series = np.cumsum(matched_binned_scores[::-1])[::-1]
        
                    #for i in range(score_granularity):
                    #    matched_score_series.append(np.sum(matched_binned_scores[i:]))
        
                    matched_score_series = ak.Array(matched_score_series)
        
                    passed_matched_jets =  {}
                    stau_dict[lifetime][mass][cut]['matched'] = matched_score_series/matched_score_series[0]
                #passed_matched_jets =  {}
                #for score_list in out[samp]['set_s_pmj_pfj']:
                #    passed_matched_jets[score_list[1]] = score_list[2]
                #    stau_dict[lifetime][mass][score_list[1]] = passed_matched_jets[score_list[1]]/out[samp]["total_matched_jets"]  
            else:
                for cut in out[samp].keys(): 
                    unmatched_histo[cut] += out[samp][cut]['unmatched_jet_histo']
        #
        #    #if 'set_s_pmj_pfj' not in out[samp]: continue
        #    #for score_list in out[samp]['set_s_pmj_pfj']:
        #    #    s_sums[score_list[1]][0] += score_list[3]
        #    
        #  
        '''  
        tprocessor = time.time() - tstart
        print(f"{tprocessor/60} minutes for processor to finish")
    #
    #'''
    ## --- ROC Calculations --- #
    ## Totals
    #all_matched = sum(
    #    val["total_matched_jets"]
    #    for val in out.values()
    #    if "total_matched_jets" in val
    #)
    #print(f"all_matched is {all_matched}")
    #all_jets = sum(
    #    val["total_number_jets"]
    #    for val in out.values()
    #    if "total_number_jets" in val
    #)
    #print(f"all_jets is {all_jets}")
    #print(f"{all_jets} total jets, with {all_matched} matched")
    #'''
    #
    '''
    thresholds   = []
    fake_rate = {}
    for cut in unmatched_histo.keys():
        unmatched_histo[cut].values(flow = True)[-2] = unmatched_histo[cut].values(flow = True)[-2] + unmatched_histo[cut].values(flow = True)[-1]
        unmatched_binned_scores = unmatched_histo[cut].values()
        unmatched_score_series = np.cumsum(unmatched_binned_scores[::-1])[::-1]
        
        #for i in range(score_granularity): 
        #    unmatched_score_series.append(np.sum(unmatched_binned_scores[i:]))
        unmatched_score_series = ak.Array(unmatched_score_series)
        fake_rate[cut] = unmatched_score_series / unmatched_score_series[0]
        score = np.arange(0, 1, 1/score_granularity)
        fake_rate_standard = (fake_rate[cut] >= 5E-3)
        if ak.any(fake_rate_standard):
            print(f"For {cut}, the score that corresponds to 5E-3 fake rate is {score[fake_rate_standard][-1]}")
    ##for lifetime in lifetimes:
    ##    for mass in masses:
    ##        stau_dict[lifetime][mass][cut]['eff'] = []
    #
    ##for s, vals in s_sums.items():
    ##    thresholds.append(s)
    ##    fake_rate = vals[0] / all_jets
    ##    fake_rates.append(fake_rate)
    ##    for lifetime in lifetimes:
    ##        for mass in masses:
    ##            stau_dict[lifetime][mass]['eff'].append(stau_dict[lifetime][mass][s])
    #
    tcalc = time.time() - tstart - tprocessor
    #
    ##print("sets")
    ##print(sets)
    ##print("Score thresholds:")
    ##print(thresholds)
    ##print("Fake rates:")
    ##print(fake_rates)
    ##print("Efficiencies:")
    ##print(efficiencies)
    ##print(f"{tcalc} seconds for calculations to finish")
    #
    # Plot stuff
    #for lifetime in lifetimes:
    #    fig, ax = plt.subplots()
    #    roc = {}
    #    for mass in masses:
    #        roc[mass] = ax.plot(fake_rate, stau_dict[lifetime][mass]['matched'], color = stau_colors[mass], label = mass + ' GeV')
    #
    #    plt.xlabel(r"Fake rate $\left(\frac{fake\_passing\_jets}{total\_jets}\right)$")
    #    plt.ylabel(r"Tau tagger efficiency $\left(\frac{matched\_passing\_jets}{total\_matched\_jets}\right)$")
    #    plt.title(f"{lifetime}")
    #    plt.xscale('log')
    #    plt.legend(loc='lower right')
    #    
    #    #plt.grid()
    #    plt.savefig(f'limited-log-skimmed-bg-tau-tagger-roc-scatter-{lifetime}_test.pdf')
        
    #cbar = fig.colorbar(roc, ax=ax, label='Score threshold')
    
    #ax.set_xscale("log")
    #ax.set_xlim(-1e-1, 6e-3)
    #ax.set_ylim(0.85, 1.05)
    roc = {}
    for mass in masses:
        roc[mass] = {}
        for lifetime in lifetimes:
            print(lifetime)
            roc[mass][lifetime] = {}
            fig, ax = plt.subplots()
            for cut in unmatched_histo.keys():
                roc[mass][lifetime][cut]= ax.plot(fake_rate[cut], stau_dict[lifetime][mass][cut]['matched'], color = jetId_colors[cut], label = cut)
            plt.xlabel(r"Fake rate $\left(\frac{fake\_passing\_jets}{total\_jets}\right)$")
            plt.ylabel(r"Tau tagger efficiency $\left(\frac{matched\_passing\_jets}{total\_matched\_jets}\right)$")
            plt.title(f"Stau {mass}GeV {lifetime} + TT")
            plt.xscale('log')
            plt.legend(loc='lower right')
            
            #plt.grid()
            plt.savefig(f'limited-log-skimmed-bg-tau-tagger-roc-scatter-{mass}-{lifetime}-jetId_log_TT.pdf')
            plt.savefig(f'limited-log-skimmed-bg-tau-tagger-roc-scatter-{mass}-{lifetime}-jetId_log_TT.png')
    
    roc = {}
    for mass in masses:
        roc[mass] = {}
        for lifetime in lifetimes:
            print(lifetime)
            roc[mass][lifetime] = {}
            fig, ax = plt.subplots()
            for cut in unmatched_histo.keys():
                roc[mass][lifetime][cut]= ax.plot(fake_rate[cut], stau_dict[lifetime][mass][cut]['matched'], color = jetId_colors[cut], label = cut)
            plt.xlabel(r"Fake rate $\left(\frac{fake\_passing\_jets}{total\_jets}\right)$")
            plt.ylabel(r"Tau tagger efficiency $\left(\frac{matched\_passing\_jets}{total\_matched\_jets}\right)$")
            plt.title(f"Stau {mass}GeV {lifetime} + TT")
            plt.xscale('log')
            plt.ylim([0.8, 1])
            plt.legend(loc='lower right')
            
            #plt.grid()
            plt.savefig(f'limited-log-skimmed-bg-tau-tagger-roc-scatter-{mass}-{lifetime}-jetId_log_samescale_TT.pdf')
            plt.savefig(f'limited-log-skimmed-bg-tau-tagger-roc-scatter-{mass}-{lifetime}-jetId_log_samescale_TT.png')
    
    
    tplotting = time.time() - tstart - tprocessor - tcalc
    print(f"{tplotting/3600} hours for plotting to finish")
    '''

