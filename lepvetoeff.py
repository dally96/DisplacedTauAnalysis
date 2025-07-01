import uproot
import matplotlib as mpl
import awkward as ak
import dask_awkward as dak
import numpy as np
import array
import matplotlib as mpl
from matplotlib import pyplot as plt
import pickle
import time
from datetime import datetime
import json, gzip, correctionlib
from xsec import *

lumi = 38.01 ##fb-1

import dask
import hist
import hist.dask as hda
from dask import config as cfg
cfg.set({'distributed.scheduler.worker-ttl': None}) # Check if this solves some dask issues
from dask.distributed import Client, wait, progress, LocalCluster 
from dask_jobqueue import HTCondorCluster
from distributed import Client
from lpcjobqueue import LPCCondorCluster

from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema
from coffea import processor
from coffea.dataset_tools import (
    apply_to_fileset,
    max_chunks,
    preprocess,
)

import warnings
warnings.filterwarnings("ignore", module="coffea") # Suppress annoying deprecation warnings for coffea vector, c.f. https://github.com/CoffeaTeam/coffea/blob/master/src/coffea/nanoevents/methods/candidate.py
import logging


PFNanoAODSchema.warn_missing_crossrefs = False

SAMP = [
        "user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-120to170_TuneCP5_13p6TeV_pythia8",
        #"user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-120to170_TuneCP5_13p6TeV_pythia8_ext",
        #"user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-170to300_TuneCP5_13p6TeV_pythia8",
        #"user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-170to300_TuneCP5_13p6TeV_pythia8_ext",
        #"user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-300to470_TuneCP5_13p6TeV_pythia8",
        #"user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-300to470_TuneCP5_13p6TeV_pythia8_ext",
        #"user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-470to600_TuneCP5_13p6TeV_pythia8_ext",
        #"user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-50to80_TuneCP5_13p6TeV_pythia8",
        #"user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-600to800_TuneCP5_13p6TeV_pythia8_ext",
        #"user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-800to1000_TuneCP5_13p6TeV_pythia8_ext",
        #"user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-80to120_TuneCP5_13p6TeV_pythia8",
        #"group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-1000to1400_TuneCP5_13p6TeV_pythia8_ext",
        #"group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-1400to1800_TuneCP5_13p6TeV_pythia8_ext",
        #"group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-1800to2400_TuneCP5_13p6TeV_pythia8_ext",
        #"group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-2400to3200_TuneCP5_13p6TeV_pythia8_ext",
        #"group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-3200_TuneCP5_13p6TeV_pythia8_ext",
        #"user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8",
        #"user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8",
        #"user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8",
        #"user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/DYJetsToLL_M-50_TuneCP5_13p6TeV-madgraphMLM-pythia8", 
       ]

selections = {
              "muon_pt":                    30., ##GeV
              "muon_ID":                    "muon_tightId",
              "muon_dxy_prompt_max":        50E-4, ##cm
              "muon_dxy_prompt_min":        0E-4, ##cm
              "muon_dxy_displaced_min":     0.1, ##cm
              "muon_dxy_displaced_max":     10.,  ##cm
              "muon_iso_max":               0.19,

              "jet_score":                  0.9, 
              "jet_pt":                     32.,  ##GeV
              "jet_dxy_displaced_min":      0.02, ##cm

              "MET_pt":                     105., ##GeV
             }

GenPtMin = 20
GenEtaMax = 2.4

colors  = {'QCD': '#56CBF9', 'TT': '#FDCA40', 'DY': '#D3C0CD'}
markers = {'noId': 'o', 'Id': '^'}
ids     = ['noId', 'Id', 'Jets']
var     = ['pt', 'dxy', 'disTauTag_score1']
#var     = ['disTauTag_score1']

var_bins = {'pt':               [(245,  20, 1000), "[GeV]"],
            'dxy':              [(20,  -5,  5),    "[cm]" ],
            'disTauTag_score1': [(10,   0.9,  1),    ""     ]
            }

 
def delta_r_mask(first: ak.highlevel.Array, second: ak.highlevel.Array, threshold: float) -> ak.highlevel.Array:
            mval = first.metric_table(second)
            return ak.all(mval > threshold, axis=-1)

PFNanoAODSchema.mixins["DisMuon"] = "Muon"
#events = NanoEventsFactory.from_root({file:"Events"}, schemaclass=PFNanoAODSchema).events()
class IDProcessor(processor.ProcessorABC):
    def __init__(self):
        PFNanoAODSchema.mixins["DisMuon"] = "Muon"
        self._accumulator = {}
        for cat in ids:
            self._accumulator[cat] = {}
            for v in var:
                self._accumulator[cat][v] = hda.hist.Hist(hist.axis.Regular(*var_bins[v][0], name = cat + "_" + v, label = v + " " + var_bins[v][1])) 

    def process(self, events):    
        id_dict = {}

        for cat in ids:
            id_dict[cat] = {}

        n_jets = ak.num(events.Jet.pt)

        default = ak.full_like(events.Jet.pt, -999)
        charged_sel = events.Jet.constituents.pf.charge != 0            
        dxy = ak.flatten(ak.drop_none(events.Jet.constituents.pf[ak.argmax(events.Jet.constituents.pf[charged_sel].pt, axis=2, keepdims=True)].d0), axis=-1)
        fixed_dxy = ak.where(ak.num(dxy) == n_jets, dxy, default)

        events["Jet"] = ak.with_field(events.Jet, fixed_dxy, where = "dxy")

        # Define the "good muon" condition for each muon per event
        good_muon_mask = (
            #((events.DisMuon.isGlobal == 1) | (events.DisMuon.isTracker == 1)) & # Equivalent to loose ID cut without isPFcand requirement. Reduce background from non-prompt muons
             (events.DisMuon.pt > 20)
            & (abs(events.DisMuon.eta) < 2.4) # Acceptance of the CMS muon system
        )
        logger.info(f"Defined good muons")
        events['DisMuon'] = events.DisMuon[good_muon_mask]
        logger.info(f"Applied mask to DisMuon")
        num_evts = ak.num(events, axis=0)
        logger.info("Counted the number of original events")
        num_good_muons = ak.count_nonzero(good_muon_mask, axis=1)
        logger.info("Counted the number of events with good muons")
        events = events[num_good_muons >= 1]
        logger.info("Counted the number of events with one or more good muons")
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
        print(events.fields)

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
        logger.info(f"Filtered noise")

        #Trigger Selection
        trigger_mask = (
                        events.HLT.PFMET120_PFMHT120_IDTight                                    |\
                        events.HLT.PFMET130_PFMHT130_IDTight                                    |\
                        events.HLT.PFMET140_PFMHT140_IDTight                                    |\
                        events.HLT.PFMETNoMu120_PFMHTNoMu120_IDTight                            |\
                        events.HLT.PFMETNoMu130_PFMHTNoMu130_IDTight                            |\
                        events.HLT.PFMETNoMu140_PFMHTNoMu140_IDTight                            |\
                        events.HLT.PFMET120_PFMHT120_IDTight_PFHT60                             |\
                        #events.HLT.MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight        |\ #Trigger not included in current Nanos
                        events.HLT.PFMETTypeOne140_PFMHT140_IDTight                             |\
                        events.HLT.MET105_IsoTrk50                                              |\
                        events.HLT.PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF                   |\
                        events.HLT.MET120_IsoTrk50                                              |\
                        events.HLT.IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1   |\
                        events.HLT.IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1   |\
                        events.HLT.Ele30_WPTight_Gsf                                            |\
                        events.HLT.DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1                    |\
                        events.HLT.DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1        |\
                        events.HLT.DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1                 
        )
        
        events = events[trigger_mask]
        logger.info(f"Applied trigger mask")
        events["DisMuon"] = events.DisMuon[ak.argsort(events["DisMuon"]["pt"], ascending=False, axis = 1)]
        events["Jet"] = events.Jet[ak.argsort(events["Jet"]["pt"], ascending=False, axis = 1)]

        events["DisMuon"] = ak.singletons(ak.firsts(events.DisMuon))
        events["Jet"] = ak.singletons(ak.firsts(events.Jet))

        logger.info(f"Chose leading muon and jet")

        good_muons  = dak.flatten((events.DisMuon.pt > selections["muon_pt"])           &\
                       (events.DisMuon.tightId == 1)                                    &\
                       (abs(events.DisMuon.dxy) > selections["muon_dxy_displaced_min"]) &\
                       (abs(events.DisMuon.dxy) < selections["muon_dxy_displaced_max"]) &\
                       (events.DisMuon.pfRelIso03_all < selections["muon_iso_max"])
                      )

        good_jets   = dak.flatten((events.Jet.disTauTag_score1 > selections["jet_score"])   &\
                       (events.Jet.pt > selections["jet_pt"])                               &\
                       (abs(events.Jet.dxy) > selections["jet_dxy_displaced_min"])          #&\
                       #(abs(events.Jet.dxy) < selections["muon_dxy_prompt_max"])
                      )

        good_events = (events.PFMET.pt > selections["MET_pt"])
        events = events[good_muons & good_jets & good_events]



        for v in var:
            id_dict['Jets'][v] = hda.hist.Hist(hist.axis.Regular(*var_bins[v][0], name = "Jets_" + v, label = v + " " + var_bins[v][1])) 
            id_dict['Jets'][v].fill(dak.flatten(events["Jet"][v], axis = None))
        print(f"Added hists to id_dict for basic jet denominator")

        # Perform the overlap removal with respect to muons, electrons and photons, dR=0.4
        noId_jets = events.Jet[ delta_r_mask(events.Jet, events.Photon, 0.4)    ]
        noId_jets = noId_jets[  delta_r_mask(noId_jets, events.Electron, 0.4)  ]
        noId_jets = noId_jets[  delta_r_mask(noId_jets, events.Muon, 0.4)      ]
        noId_jets = noId_jets[  delta_r_mask(noId_jets, events.DisMuon, 0.4)   ]
        print(f"Added lepton veto without any cuts")

        for v in var:
            id_dict['noId'][v] = hda.hist.Hist(hist.axis.Regular(*var_bins[v][0], name = "noId_" + v, label = v + " " + var_bins[v][1]))
            id_dict['noId'][v].fill(dak.flatten(noId_jets[v], axis = None))
        print(f"Filled hists to id_dict with no cut jets")

        events['Photon']   = events.Photon[   (events.Photon.pt   > 20) & (abs(events.Photon.eta)   < 2.4) & (events.Photon.electronVeto)  ]
        events['Electron'] = events.Electron[ (events.Electron.pt > 20) & (abs(events.Electron.eta) < 2.4) & (events.Electron.convVeto)      ]
        events['Muon']     = events.Muon[     (events.Muon.pt     > 20) & (abs(events.Muon.eta)     < 2.4) & (events.Muon.looseId    == 1)  ]
        events['DisMuon']  = events.DisMuon[  (events.DisMuon.pt  > 20) & (abs(events.DisMuon.eta)  < 2.4) ]
        print(f"Added lepton cuts")

        # Perform the overlap removal with respect to muons, electrons and photons, dR=0.4
        Id_jets = events.Jet[ delta_r_mask(events.Jet, events.Photon,   0.4)    ]
        Id_jets = Id_jets[    delta_r_mask(Id_jets, events.Electron, 0.4)  ]
        Id_jets = Id_jets[    delta_r_mask(Id_jets, events.Muon,     0.4)      ]
        Id_jets = Id_jets[    delta_r_mask(Id_jets, events.DisMuon,  0.4)   ]
        print(f"Added lepton veto with cuts")

        Id_charged_sel = Id_jets.constituents.pf.charge != 0
        Id_jets_dxy = ak.flatten(ak.drop_none(Id_jets.constituents.pf[ak.argmax(Id_jets.constituents.pf[Id_charged_sel].pt, axis=2, keepdims=True)].d0), axis=-1)

        for v in var:
            id_dict['Id'][v] = hda.hist.Hist(hist.axis.Regular(*var_bins[v][0], name = "Id_" + v, label = v + " " + var_bins[v][1]))
            id_dict['Id'][v].fill(dak.flatten(Id_jets[v], axis = None))
        print(f"Filles hists to id dict with  cut jets")
        return id_dict 

    def postprocess(self, accumulator):
        return accumulator

if __name__ == "__main__": 
    cluster = LPCCondorCluster(ship_env=True, transfer_input_files='/uscms/home/dally/x509up_u57864')
    cluster.adapt(minimum=1, maximum=10000)
    client = Client(cluster)
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    background_samples = {} 
    background_samples["QCD"] = []
    background_samples["TT"] = []
    #background_samples["W"] = []
    background_samples["DY"] = []
    
    background_histograms = {}
 
    for background, samples in background_samples.items():
        # Initialize a dictionary to hold ROOT histograms for the current background
        print(f"Starting {background}")
        background_histograms[background] = {}
        
        for i in ids: 
            background_histograms[background][i] = {}
            for v in var:
                background_histograms[background][i][v] = hda.hist.Hist(hist.axis.Regular(*var_bins[v][0], name = i + "_" + v, label = v + " " + var_bins[v][1])).compute()

    with open("preprocessed_fileset.pkl", "rb") as  f:
        dataset_runnable = pickle.load(f)    

    for samp in dataset_runnable.keys(): 
        if "Stau" in samp: continue
        samp_runnable = {}
        samp_runnable[samp] = dataset_runnable[samp]
        print("Time before comupute:", datetime.now().strftime("%H:%M:%S")) 
        to_compute = apply_to_fileset(
                 IDProcessor(),
                 max_chunks(samp_runnable, 100000),
                 schemaclass=PFNanoAODSchema,
                 uproot_options={"coalesce_config": uproot.source.coalesce.CoalesceConfig(max_request_ranges=10, max_request_bytes=1024*1024),
                                 }
        )
        print(to_compute)
        output = dask.compute(to_compute)
    
        for i in ids:
            for v in var:
                if "QCD" in samp: 
                    background_histograms["QCD"][i][v] += output[0][samp][i][v]            
                if "TT"  in samp:    
                    background_histograms["TT" ][i][v] += output[0][samp][i][v]
                if "DY"  in samp:
                    background_histograms["DY" ][i][v] += output[0][samp][i][v]
        print(f"Successfully finished {samp}")
    
    for background in background_histograms.keys():
        print(f"Starting on {background}")
        for variable in var:
            print(f"Starting on {variable}")
            plt.cla()
            plt.clf()
            fig, ax = plt.subplots(2, 1, figsize=(10, 15))
            for cut in ids:
                if cut == "Jets": continue
                print(f"Starting on {cut} ID")
                background_histograms[background][cut][variable].plot_ratio(
                                background_histograms[background]["Jets"][variable],
                                rp_num_label        = cut + " " + variable,
                                rp_denom_label      = "Jets " + variable,
                                rp_uncert_draw_type = "line",
                                rp_uncertainty_type = "efficiency",
                                ax_dict = {'main_ax': ax[0], f"ratio_ax": ax[1]}
                                )
                print(f"Finished plotting {cut}")
            cut_counter = 0
            for artist in ax[0].containers:
                artist[0].set_color(colors[background])
                artist[0].set_label(ids[cut_counter])
            for artist in ax[1].containers:
                artist[0].set_color(colors[background])
                artist[0].set_label(ids[cut_counter])
                artist[0].set_marker(markers[ids[cut_counter]])
                artist[0].set_markeredgecolor("black")
                cut_counter += 1
            ax[0].set_title(background)
            ax[1].set_title(background)
            ax[0].legend()
            ax[1].legend()
            fig.savefig(f"LepVeto_{background}_{variable}_SR_wjetdxy.pdf")
            print(f"LepVeto_{background}_{variable}_SR_wjetdxy.pdf saved!")

    
    elapsed = time.time() - tic
    print(f"Finished in {elapsed:.1f}s")

    client.shutdown()
    cluster.close()
    
