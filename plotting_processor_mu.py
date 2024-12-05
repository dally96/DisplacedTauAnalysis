#FDCA40#5DFDCBimport numpy as np 
import hist
import awkward as ak 
import math
import scipy
import array
import ROOT
import dask
import dask_awkward as dak
from coffea import processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection
import hist.dask as hda
import matplotlib  as mpl
from  matplotlib import pyplot as plt
from xsec import *
import time

start_time = time.time()


SAMP = [
      ['Stau_100_100mm', 'SIG'],
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
      ["WtoLNu2Jets", 'EWK'],
      ["TTtoLNu2Q",  'TT'],
      ["TTto4Q", 'TT'],
      ["TTto2L2Nu", 'TT'],
      ]

lumi = 38.01 ##fb-1
colors = ['#56CBF9', '#FDCA40', '#5DFDCB', '#D3C0CD', '#3A5683', '#FF773D']
selections = {
              "electron_pt":                30,  ##GeV 
              "electron_eta":               1.44, 
              "electron_cutBased":          4, ## 4 = tight
              "electron_dxy_prompt_max":    50E-4, ##cm
              "electron_dxy_prompt_min":    0E-4, ##cm
              "electron_dxy_displaced_min": 100E-4, ##cm
              "electron_dxy_displaced_max": 10, ##cm

              "muon_pt":                    30, ##GeV
              "muon_eta":                   1.5,
              "muon_ID":                    "muon_tightId",
              "muon_dxy_prompt_max":        50E-4, ##cm
              "muon_dxy_prompt_min":        0E-4, ##cm
              "muon_dxy_displaced_min":     100E-4, ##cm
              "muon_dxy_displaced_max":     10, ##cm

              "jet_score":                  0.9, 
              "jet_pt":                     32, ##GeV

              "MET_pT":                     105, ##GeV
             }

variables_with_bins = {
    "muon_pt": [(245, 20, 1000), "GeV"],
    "muon_eta": [(50, -2.5, 2.5), ""],
    "muon_phi": [(64, -3.2, 3.2), ""],
    "muon_dxy": [(200, -1, 1), "cm"],
    "muon_dz" : [(200, -1, 1), "cm"],
    "muon_pfRelIso03_all": [(100, 0, 10), ""],
    "muon_pfRelIso03_chg": [(100, 0, 10), ""],
    "muon_pfRelIso04_all": [(100, 0, 10), ""],

    "jet_pt" : [(245, 20, 1000), "GeV"],
    "jet_eta": [(48, -2.4, 2.4), ""],
    "jet_phi": [(64, -3.2, 3.2), ""],
    "jet_score": [(20, 0, 1), ""],

    "leadingmuon_pt": [(245, 20, 1000), "GeV"],
    "leadingmuon_eta": [(50, -2.5, 2.5), ""],
    "leadingmuon_phi": [(64, -3.2, 3.2), ""],
    "leadingmuon_dxy": [(200, -1, 1), "cm"],
    "leadingmuon_dz" : [(200, -1, 1), "cm"],

    "leadingjet_pt" : [(245, 20, 1000), "GeV"],
    "leadingjet_eta": [(48, -2.4, 2.4), ""],
    "leadingjet_phi": [(64, -3.2, 3.2), ""],
    "leadingjet_score": [(20, 0, 1), ""],

    "dR" : [(20, 0, 1), ""],
    "deta": [(100, -5, 5), ""],
    "dphi": [(64, -3.2, 3.2), ""],
    "MET_pT": [(225, 100, 1000), "GeV"],
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
        # Object selection
        good_muons = ((events["muon_pt"] > selections["muon_pt"]) 
                         &(events["muon_tightId"] ==  1)
                         & (abs(events["muon_dxy"]) > selections["muon_dxy_displaced_min"])
                         & (abs(events["muon_dxy"]) < selections["muon_dxy_displaced_max"])
                         )        
        good_jets = ((events["jet_score"] > selections["jet_score"])
                    & (events["jet_pt"] > selections["jet_pt"])
                    )        

        good_events = (events["MET_pT"] > selections["MET_pT"])

        num_muons = ak.num(events["muon_pt"][good_muons])
        num_jets = ak.num(events["jet_pt"][good_jets])
        muon_event_mask = num_muons > 0
        jet_event_mask = num_jets > 0
        events = events[muon_event_mask & jet_event_mask & good_events]


        for branch in self.vars_with_bins:
            if ("muon_" in branch) and ("leading" not in branch): 
                events[branch] = events[branch][good_muons[muon_event_mask & jet_event_mask & good_events]]
            if ("jet_" in branch) and ("leading" not in branch):
                events[branch] = events[branch][good_jets[jet_event_mask & muon_event_mask & good_events]]
                    
        histograms = self.initialize_histograms()
        # Loop over variables and fill histograms
        for var in histograms:
            histograms[var].fill(
                **{var: dak.flatten(events[var], axis = None)},
                weight = weights
            )
            
            
        output = {"histograms": histograms}
        print(output)
        return output

    def postprocess(self):
        pass
        

background_samples = {} 
background_samples["QCD"] = []
background_samples["TT"] = []
background_samples["W"] = []
background_samples["DY"] = []
#
for samples in SAMP:
    if "QCD" in samples[0]:
        background_samples["QCD"].append(("my_skim_muon_" + samples[0] + "/*.parquet", xsecs[samples[0]] * lumi * 1000 * 1/num_events[samples[0]]))
    if "TT" in samples[0]:
        background_samples["TT"].append(("my_skim_muon_" + samples[0] + "/*.parquet", xsecs[samples[0]] * lumi * 1000 * 1/num_events[samples[0]]))
    if "W" in samples[0]:
        background_samples["W"].append(("my_skim_muon_" + samples[0] + "/*.parquet", xsecs[samples[0]] * lumi * 1000 * 1/num_events[samples[0]]))
    if "DY" in samples[0]:
        background_samples["DY"].append(("my_skim_muon_" + samples[0] + "/*.parquet", xsecs[samples[0]] * lumi * 1000 * 1/num_events[samples[0]]))
    if "Stau" in samples[0]:
        background_samples[samples[0]] = [("my_skim_muon_" + samples[0] + "/*.parquet", xsecs[samples[0]] * lumi * 1000 * 1/num_events[samples[0]])]

# Initialize dictionary to hold accumulated ROOT histograms for each background
background_histograms = {}

# Process each background
for background, samples in background_samples.items():
    # Initialize a dictionary to hold ROOT histograms for the current background
    background_histograms[background] = {}
    for var in variables_with_bins:
        background_histograms[background][var] = hda.hist.Hist(hist.axis.Regular(*variables_with_bins[var][0], name=var, label = var + ' ' + variables_with_bins[var][1])).compute()

    print(f"For {background} here are samples {samples}") 
    for sample_file, sample_weight in samples:
        try:
            # Step 1: Load events for the sample using dask-awkward
            events = dak.from_parquet(sample_file)
            print(f'Starting {sample_file} histogram')         

            processor_instance = ExampleProcessor(variables_with_bins)
            output = processor_instance.process(events, sample_weight)
            print(f'{sample_file} finished successfully')

            # Loop through each variable's histogram in the output
            for var, dask_histo in output["histograms"].items():
                background_histograms[background][var]  = background_histograms[background][var] + dask_histo.compute()

        except Exception as e:
            print(f"Error processing {sample_file}: {e}")
            

for var in variables_with_bins:
    plt.cla()
    plt.clf()
    s = hist.Stack.from_dict({"QCD": background_histograms["QCD"][var],
                              "TT" : background_histograms["TT"][var],
                              "W": background_histograms["W"][var],
                              "DY": background_histograms["DY"][var],       
                                })
    s.plot(stack = True, histtype= "fill", color = [colors[0],colors[1],colors[2]])
    for sample in background_samples:
        if "Stau" in sample: 
            background_histograms[sample][var].plot(color = '#B80C09', label = sample)

    plt.xlabel(var + ' ' + variables_with_bins[var][1])
    plt.ylabel("A.U.")
    plt.yscale('log')
    plt.ylim(top=get_stack_maximum(s)*10)
    plt.legend()
    plt.savefig(f"../www/pt30_tightId_displaced_score90_jetPt32_MET105/mu_stacked_histogram_{var}_111111.png")




end_time = time.time()

elapsed_time = end_time - start_time

print(f"Code started at: {time.ctime(start_time)}")
print(f"Code ended at: {time.ctime(end_time)}")
print(f"Total time taken: {elapsed_time:.2f} seconds")
