import numpy as np 
import hist
import awkward as ak 
import math
import scipy
import array
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
      #'Stau_100_100mm',
      'QCD50_80',
      'QCD80_120',
      'QCD120_170',
      'QCD170_300',
      'QCD300_470',
      'QCD470_600',
      'QCD600_800',
      'QCD800_1000',
      'QCD1000_1400',
      'QCD1400_1800',
      'QCD1800_2400',
      'QCD2400_3200',
      'QCD3200',
      "DYJetsToLL",   
      "WtoLNu2Jets",
      "TTtoLNu2Q", 
      "TTto4Q",   
      "TTto2L2Nu",
      ]

lumi = 38.01 ##fb-1
colors = ['#56CBF9', '#6abecc', '#64b0bf', '#5da2b3', '#5694a8', '#4f879d', '#477a93', '#3f6d89', '#386180', '#2f5477', '#27486e', '#1d3c65', '#13315c', '#BF98A0', '#FDCA40', '#5DFDCB', '#3A5683', '#FF773D']
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
             }

class ExampleProcessor(processor.ProcessorABC):
    def __init__(self, variables_with_bins):
        print("Initializing ExampleProcessor")
        # Create a dictionary to hold histograms for each variable
        self.variables_with_bins = variables_with_bins
    
    def initialize_histograms(self):
        histograms = {}
        # Initialize histograms for each variable based on provided binning
        for var, bin_info in self.variables_with_bins.items():
            print(f"Creating histogram for {var} with bin_info {bin_info}")
            print(f"Processing sample {sample_file} with electron_pt shape: {ak.num(events['electron_pt']).compute()}")
            histograms[var] = hda.hist.Hist(hist.axis.Regular(*bin_info, name=var, label = 'pt [GeV]'),)
            print(f"Successfully created histogram for {var}")
        return histograms

    def process(self, events, weights):
        # Object selection
        good_electrons = ((events["electron_pt"] > selections["electron_pt"]) 
                         #& (abs(electron["electron_eta"]) < selections["electron_eta"])
                         & (events["electron_cutBased"] == selections["electron_cutBased"])
                         #& (abs(electron["electron_dxy"]) > electron_dxy_prompt_min)
                         #& (abs(electron["electron_dxy"]) < electron_dxy_prompt_max)
                         )        

        num_electrons = ak.num(events["electron_pt"][good_electrons])
        electron_event_mask = num_electrons > 0
        #events = events[electron_event_mask]

        #for branch in events.fields:
        #    if ("electron_" in branch) and ("leading" not in branch): 
        #        events[branch] = events[branch][good_electrons[electron_event_mask]]
                
        histograms = self.initialize_histograms()
        # Loop over variables and fill histograms
        for var in histograms:
            print(f'The array that should be filling the histo is {dak.flatten(events[var]).compute()}')
            histograms[var].fill(
                **{var: dak.flatten(events[var])},
                weight = weights
            )
        output = {"histograms": histograms}
        print(output)
        return output

    def postprocess(self, accumulator):
        pass

variables_with_bins = {
    "electron_pt": (245, 20, 1000),
    }

background_samples = {} 
background_samples["QCD"] = []
background_samples["TT"] = []
background_samples["W+DY"] = []

for samples in SAMP:
    if "QCD" in samples:
        background_samples["QCD"].append(("my_skim_electron_" + samples + "/*.parquet", xsecs[samples] * lumi * 1000 * 1/num_events[samples]))
    if "TT" in samples:
        background_samples["TT"].append(("my_skim_electron_" + samples + "/*.parquet", xsecs[samples] * lumi * 1000 * 1/num_events[samples]))
    if "Jets" in samples:
        background_samples["W+DY"].append(("my_skim_electron_" + samples + "/*.parquet", xsecs[samples] * lumi * 1000 * 1/num_events[samples]))
            

# Initialize dictionary to hold accumulated histograms for each background
background_histograms = {}
# Process each background
for background, samples in background_samples.items():
    # Initialize a dictionary to hold histograms for the current background
    background_histograms[background] = {}
    
    for sample_file, sample_weight in samples:
        try:
            # Step 1: Load events for the sample using dask-awkward
            events = dak.from_parquet(sample_file)
            print(f'Starting {sample_file} histogram')         

            processor_instance = ExampleProcessor(variables_with_bins)
            output = processor_instance.process(events, sample_weight)
            print(f'{sample_file} finished successfully')
            for var, histo in output["histograms"].items():
                if var not in background_histograms[background]:
                    background_histograms[background][var] = histo.compute()
                else:
                    background_histograms[background][var] += histo.compute()
        except Exception as e:
            print(f"Error processing {sample_file}: {e}")

# Now, for plotting all variables
for var in variables_with_bins:
    stacked_histograms = []
    labels = []

    # Collect histograms for this variable across all backgrounds
    for background in background_histograms:
        if var in background_histograms[background]:
            stacked_histograms.append(background_histograms[background][var])
            print(background_histograms[background][var])
            labels.append(background)

    if stacked_histograms:
        print(f"Plotting for {var}")
        plt.hist(stacked_histograms, stacked_histograms[0].axes[0].edges, stacked=True, histtype="step", color = [colors[0], colors[14], colors[15]], label=labels)
        plt.legend()
        plt.xlabel(var)  # Set x-axis label to the variable name
        plt.ylabel('A.U.')  # Set y-axis label
        plt.title(f"Stacked Histogram for {var}")
        plt.savefig(f"test_stacked_histogram_{var}.pdf")

#for var, histo in histograms.items():
#    #hist.plot1d(stack=True, histtype="fill", label=["QCD", "TT", "DY+W"])
#    plt.hist(background_histograms
#    #histo.plot()
#        #plt.legend()
#        #plt.savefig("test_electron_pt.pdf")

end_time = time.time()

elapsed_time = end_time - start_time

print(f"Code started at: {time.ctime(start_time)}")
print(f"Code ended at: {time.ctime(end_time)}")
print(f"Total time taken: {elapsed_time:.2f} seconds")
