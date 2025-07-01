import numpy as np  
from datetime import datetime
import logging
import uproot
import hist
import awkward as ak  
import math
import scipy
import array
import dask
import dask_awkward as dak 
from coffea import processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, PFNanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection
import hist.dask as hda 
import matplotlib  as mpl 
from  matplotlib import pyplot as plt 
from xsec import *
import time
import pickle
import os

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
NanoAODSchema.warn_missing_crossrefs = False

SAMP = [ 
      ['QCD50_80',     '/eos/uscms/store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-50to80_TuneCP5_13p6TeV_pythia8'],
      ['QCD80_120',    '/eos/uscms/store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-80to120_TuneCP5_13p6TeV_pythia8'],
      ['QCD120_170',   '/eos/uscms/store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-120to170_TuneCP5_13p6TeV_pythia8', '/eos/uscms/store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-120to170_TuneCP5_13p6TeV_pythia8_ext'],
      ['QCD170_300',   '/eos/uscms/store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-170to300_TuneCP5_13p6TeV_pythia8', '/eos/uscms/store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-170to300_TuneCP5_13p6TeV_pythia8_ext'],
      ['QCD300_470',   '/eos/uscms/store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-300to470_TuneCP5_13p6TeV_pythia8', '/eos/uscms/store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-300to470_TuneCP5_13p6TeV_pythia8_ext'],
      ['QCD470_600',   '/eos/uscms/store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-470to600_TuneCP5_13p6TeV_pythia8_ext'],
      ['QCD600_800',   '/eos/uscms/store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-600to800_TuneCP5_13p6TeV_pythia8_ext'],
      ['QCD800_1000',  '/eos/uscms/store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-800to1000_TuneCP5_13p6TeV_pythia8_ext'],
      ['QCD1000_1400', '/eos/uscms/store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-1000to1400_TuneCP5_13p6TeV_pythia8_ext'],
      ['QCD1400_1800', '/eos/uscms/store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-1400to1800_TuneCP5_13p6TeV_pythia8_ext'],
      ['QCD1800_2400', '/eos/uscms/store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-1800to2400_TuneCP5_13p6TeV_pythia8_ext'],
      ['QCD2400_3200', '/eos/uscms/store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-2400to3200_TuneCP5_13p6TeV_pythia8_ext'],
      ['QCD3200',      '/eos/uscms/store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-3200_TuneCP5_13p6TeV_pythia8_ext'],
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

lumi = 38.01 ##fb-1
colors = ['#56CBF9', '#FDCA40', '#5DFDCB', '#D3C0CD', '#3A5683', '#FF773D', '#EA7AF4', '#B43E8F', '#6200B3', '#218380', '#FF9770', '#E9FF70', '#FF70A6']
variables_with_bins = { 
#    "DisMuon_pt": [(245, 20, 1000), "GeV"],
#     "DisMuon_eta": [(50, -2.5, 2.5), ""],
#     "DisMuon_phi": [(64, -3.2, 3.2), ""],
#     "DisMuon_dxy": [(50, -50E-4, 50E-4), "cm"],
#     "DisMuon_dz" : [(50, -0.1, 0.1), "cm"],
#     "DisMuon_pfRelIso03_all": [(50, 0, 1), ""],
    #"DisMuon_pfRelIso03_chg": [(50, 0, 1), ""],
    #"DisMuon_pfRelIso04_all": [(50, 0, 1), ""],
    #"DisMuon_tkRelIso":       [(50, 0, 1), ""],

    #"Jet_pt" : [(348, 20, 3500), "GeV"],
    "Generator_scalePDF": [(174, 20, 3500), ""],
#     "Jet_eta": [(48, -2.4, 2.4), ""],
#     "Jet_phi": [(64, -3.2, 3.2), ""],
#     "Jet_disTauTag_score1": [(20, 0, 1), ""],
#     "Jet_dxy": [(50, -50E-4, 50E-4), "cm"],
}

def get_histogram_minimum(hist_dict, var):
    """Returns the minimum non-zero value of a ROOT histogram (TH1) by checking each bin."""
    min_value = float('inf')  # Start with infinity to find the minimum

    for name, hist in hist_dict.items():
        if "Stau" in name:
            # Loop over all bins in the histogram
            for bin_idx in range(1, hist[var].GetNbinsX() + 1):  # Bins start at 1
                bin_content = hist[var].GetBinContent(bin_idx)
                # Only consider non-zero values to avoid returning 0 if there are empty bins
                if bin_content > 0 and bin_content < min_value:
                    min_value = bin_content

    # Return min_value, or 0 if all bins were zero
    return min_value if min_value != float('inf') else 1E-8

def get_stack_maximum(stack):
    max_value = 0

    for hist in stack:
        max_value = max(max_value, hist.view().max())

    return max_value


class ExampleProcessor(processor.ProcessorABC):
    def __init__(self, vars_with_bins):
        self.vars_with_bins = vars_with_bins
        print("Initializing ExampleProcessor")

    def initialize_histograms(self):
        histograms = {}
        # Initialize histograms for each variable based on provided binning
        for var, bin_info in self.vars_with_bins.items():
            print(f"Creating histogram for {var} with bin_info {bin_info}")

            histograms[var] = hda.hist.Hist(hist.axis.Regular(*bin_info[0], name=var, label = var + ' ' + bin_info[1]))

            print(f"Successfully created histogram for {var}")
        return histograms

    def process(self, events):

        histograms = self.initialize_histograms()

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
                       (abs(events.DisMuon.dxy) > selections["muon_dxy_prompt_min"]) &\
                       (abs(events.DisMuon.dxy) < selections["muon_dxy_prompt_max"]) &\
                       (events.DisMuon.pfRelIso03_all < selections["muon_iso_max"])
                      )

        good_jets   = dak.flatten((events.Jet.disTauTag_score1 < selections["jet_score"])   &\
                       (events.Jet.pt > selections["jet_pt"])                               &\
                       (abs(events.Jet.dxy) > selections["jet_dxy_displaced_min"])          #&\
                       #(abs(events.Jet.dxy) < selections["muon_dxy_prompt_max"])
                      )

        good_events = (events.PFMET.pt > selections["MET_pt"])
        events = events[good_muons & good_jets & good_events]

        # Loop over variables and fill histograms
        for var in histograms:
            var_name = '_'.join(var.split('_')[1:])

            histograms[var].fill(
                **{var: dak.flatten(events[var.split('_')[0]][var_name], axis = None)},
                weight = xsecs[events.metadata['dataset']] * lumi * 1000 * 1/num_events[events.metadata['dataset']]
            )


        output = {"histograms": histograms}
        print(output)
        return output

    def postprocess(self):
        pass

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    background_samples = {}
    for samp in SAMP:
        if len(samp) == 2:
            background_samples[samp[0]] = [(samp[1] + "/*.root", xsecs[samp[0]] * lumi * 1000 * 1/num_events[samp[0]])]
        if len(samp) == 3:
            background_samples[samp[0]] = []
            background_samples[samp[0]].append( (samp[1] + "/*.root", xsecs[samp[0]] * lumi * 1000 * 1/num_events[samp[0]]) )
            background_samples[samp[0]].append( (samp[2] + "/*.root", xsecs[samp[0]] * lumi * 1000 * 1/num_events[samp[0]]) )        
    
    full_QCD = hda.hist.Hist(hist.axis.Regular(*variables_with_bins["Generator_scalePDF"][0], name="Generator_scalePDF", label = "Generator_scalePDF" + ' ' + variables_with_bins["Generator_scalePDF"][1])).compute()
    # Process each background
    #for background, samples in background_samples.items():
    #    background_histograms[background] = {}
    #    background_histograms[background] = hda.hist.Hist(hist.axis.Regular(*variables_with_bins["Generator_scalePDF"][0], name="Generator_scalePDF", label = "Generator_scalePDF" + ' ' + variables_with_bins["Generator_scalePDF"][1])).compute()
    #
    #    print(f"For {background} here are samples {samples}") 
    #    for sample_file, sample_weight in samples:
    #        try:
    #            # Step 1: Load events for the sample using dask-awkward
    #            events = NanoEventsFactory.from_root({sample_file:"Events"}, schemaclass= PFNanoAODSchema).events()
    #            #events = uproot.dask(sample_file)
    #            print(f'Starting {sample_file} histogram')         
    #
    #            processor_instance = ExampleProcessor(variables_with_bins)
    #            output = processor_instance.process(events, sample_weight)
    #            print(f'{sample_file} finished successfully')
    #            print(output)
    #
    #            # Loop through each variable's histogram in the output
    #            background_histograms[background] += output["histograms"]["Generator_scalePDF"].compute()
    #
    #        except Exception as e:
    #            print(f"Error processing {sample_file}: {e}")
    
    with open("preprocessed_fileset.pkl", "rb") as  f:
        dataset_runnable = pickle.load(f)    

    col = 0
    for samp in dataset_runnable.keys(): 
        if "QCD" in samp:
            samp_runnable = {}
            samp_runnable[samp] = dataset_runnable[samp]
            print("Time before comupute:", datetime.now().strftime("%H:%M:%S")) 
            to_compute = apply_to_fileset(
                     ExampleProcessor(variables_with_bins),
                     max_chunks(samp_runnable, 100000),
                     schemaclass=PFNanoAODSchema,
                     uproot_options={"coalesce_config": uproot.source.coalesce.CoalesceConfig(max_request_ranges=10, max_request_bytes=1024*1024),
                                     }
            )
            print(to_compute)
            output = dask.compute(to_compute)
            print(output[0][samp]["histograms"].keys())            

            full_QCD += output[0][samp]["histograms"]["Generator_scalePDF"]
            output[0][samp]["histograms"]["Generator_scalePDF"].plot(color=colors[col],label = samp)
            print(col)
            col += 1
        
    plt.yscale('log')
    plt.legend(fontsize="5")
    plt.savefig("QCD_slice_saraNano_generator_scalepdf.pdf")
    
    
    plt.cla()
    plt.clf()
    plt.yscale('log')
    full_QCD.plot(histtype="fill")
    plt.savefig("fullQCD_saraNano_generator_scalepdf.pdf")
