import numpy as np  
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

    "Jet_pt" : [(348, 20, 3500), "GeV"],
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

    def process(self, events, weights):

        histograms = self.initialize_histograms()
        # Loop over variables and fill histograms
        for var in histograms:
            var_name = '_'.join(var.split('_')[1:])

            histograms[var].fill(
                **{var: dak.flatten(events[var.split('_')[0]][var_name], axis = None)},
                weight = weights
            )


        output = {"histograms": histograms}
        print(output)
        return output

    def postprocess(self):
        pass


background_samples = {}
for samp in SAMP:
    if len(samp) == 2:
        background_samples[samp[0]] = [(samp[1] + "/*.root", xsecs[samp[0]] * lumi * 1000 * 1/num_events[samp[0]])]
    if len(samp) == 3:
        background_samples[samp[0]] = []
        background_samples[samp[0]].append( (samp[1] + "/*.root", xsecs[samp[0]] * lumi * 1000 * 1/num_events[samp[0]]) )
        background_samples[samp[0]].append( (samp[2] + "/*.root", xsecs[samp[0]] * lumi * 1000 * 1/num_events[samp[0]]) )        

background_histograms = {}
# Process each background
for background, samples in background_samples.items():
    background_histograms[background] = {}
    background_histograms[background] = hda.hist.Hist(hist.axis.Regular(*variables_with_bins["Jet_pt"][0], name="Jet_pt", label = "Jet_pt" + ' ' + variables_with_bins["Jet_pt"][1])).compute()

    print(f"For {background} here are samples {samples}") 
    for sample_file, sample_weight in samples:
        try:
            # Step 1: Load events for the sample using dask-awkward
            events = NanoEventsFactory.from_root({sample_file:"Events"}, schemaclass= PFNanoAODSchema).events()
            #events = uproot.dask(sample_file)
            print(f'Starting {sample_file} histogram')         

            processor_instance = ExampleProcessor(variables_with_bins)
            output = processor_instance.process(events, sample_weight)
            print(f'{sample_file} finished successfully')
            print(output)

            # Loop through each variable's histogram in the output
            background_histograms[background] += output["histograms"]["Jet_pt"].compute()

        except Exception as e:
            print(f"Error processing {sample_file}: {e}")


full_QCD = hda.hist.Hist(hist.axis.Regular(*variables_with_bins["Jet_pt"][0], name="Jet_pt", label = "Jet_pt" + ' ' + variables_with_bins["Jet_pt"][1])).compute()
print(background_histograms)
col = 0
for var, hist in background_histograms.items():
    full_QCD += hist
    hist.plot(color=colors[col],label = var)
    print(col)
    col += 1
    
plt.yscale('log')
plt.legend(fontsize="5")
plt.savefig("QCD_slice_saraNano_jet_pt.pdf")


plt.cla()
plt.clf()
plt.yscale('log')
full_QCD.plot(histtype="fill")
plt.savefig("fullQCD_saraNano_jet_pt.pdf")
