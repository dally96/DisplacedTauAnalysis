import numpy as np 
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
      #('Stau_100_100mm', 'SIG'),
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
colors = ['#56CBF9', '#FDCA40', '#5DFDCB', '#3A5683', '#FF773D']
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

variables_with_bins = {
    "electron_pt": (245, 20, 1000),
    "electron_eta": (50, -2.5, 2.5)
    }

class ExampleProcessor(processor.ProcessorABC):
    def __init__(self, vars_with_bins):
        self.vars_with_bins = vars_with_bins
        print("Initializing ExampleProcessor")

    def initialize_histograms(self):
        histograms = {}
        # Initialize histograms for each variable based on provided binning
        for var, bin_info in self.vars_with_bins.items():
            print(f"Creating histogram for {var} with bin_info {bin_info}")

            histograms[var] = hda.hist.Hist(hist.axis.Regular(*bin_info, name=var, label = 'pt [GeV]'))

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
        events = events[electron_event_mask]

        for branch in events.fields:
            if ("electron_" in branch) and ("leading" not in branch): 
                events[branch] = events[branch][good_electrons[electron_event_mask]]
                
        histograms = self.initialize_histograms()
        # Loop over variables and fill histograms
        for var in histograms:
            histograms[var].fill(
                **{var: dak.flatten(events[var])},
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
background_samples["W+DY"] = []
#
for samples in SAMP:
    if "QCD" in samples[0]:
        background_samples["QCD"].append(("my_skim_electron_" + samples[0] + "/*.parquet", xsecs[samples[0]] * lumi * 1000 * 1/num_events[samples[0]]))
    if "TT" in samples[0]:
        background_samples["TT"].append(("my_skim_electron_" + samples[0] + "/*.parquet", xsecs[samples[0]] * lumi * 1000 * 1/num_events[samples[0]]))
    if "Jets" in samples[0]:
        background_samples["W+DY"].append(("my_skim_electron_" + samples[0] + "/*.parquet", xsecs[samples[0]] * lumi * 1000 * 1/num_events[samples[0]]))

# Initialize dictionary to hold accumulated ROOT histograms for each background
background_histograms = {}

# Process each background
for background, samples in background_samples.items():
    # Initialize a dictionary to hold ROOT histograms for the current background
    background_histograms[background] = {}
    
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
                # Compute the Dask histogram to get bin counts and edges
                counts, edges = dask_histo.compute().to_numpy()
                n_bins = len(counts)

                # Check if a ROOT histogram already exists for this variable and background
                if var not in background_histograms[background]:
                    # Initialize ROOT histogram if not created yet
                    background_histograms[background][var] = ROOT.TH1F(
                        f"{background}_{var}", f"{background} {var} Histogram", n_bins, edges[0], edges[-1]
                    )

                # Add counts to the ROOT histogram bins
                root_hist = background_histograms[background][var]
                for i in range(n_bins):
                    # Accumulate by adding the new counts to the existing bin content
                    current_content = root_hist.GetBinContent(i + 1)
                    root_hist.SetBinContent(i + 1, current_content + counts[i])
        except Exception as e:
            print(f"Error processing {sample_file}: {e}")


# Set up a color scheme for different backgrounds
color_map = {
    "QCD": ROOT.TColor.GetColor(colors[0]),
    "TT": ROOT.TColor.GetColor(colors[1]),
    "W+DY": ROOT.TColor.GetColor(colors[2])
}

# Create a THStack for each variable and add each background's histogram to it
for var in variables_with_bins:
    stack = ROOT.THStack(f"stack_{var}", f"Stacked Histogram for {var}")

    # Loop over backgrounds and add histograms to the stack
    for background, vars_dict in background_histograms.items():
        if var in vars_dict:
            hist = vars_dict[var]
            hist.SetFillColor(color_map[background])  # Set color
            hist.SetLineColor(color_map[background])  # Optional: add a black border to each histogram
            stack.Add(hist)  # Add to stack

    # Create a canvas to draw the stacked histogram
    canvas = ROOT.TCanvas(f"canvas_{var}", f"{var} Canvas", 800, 600)
    canvas.SetLogy(1)
    
    # Draw the stack and customize axes
    stack.Draw("HIST")  # "HIST" option to draw bars
    stack.GetXaxis().SetTitle(var)
    stack.GetYaxis().SetTitle("Counts")
    stack.SetMinimum(1)
    stack.SetMaximum(1E8)

    # Add a legend
    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    for background in background_histograms:
        if var in background_histograms[background]:
            legend.AddEntry(background_histograms[background][var], background, "f")  # "f" for fill
    legend.Draw()

    # Save the stacked plot
    canvas.SaveAs(f"test_stacked_histogram_{var}.pdf")
    canvas.Close()


end_time = time.time()

elapsed_time = end_time - start_time

print(f"Code started at: {time.ctime(start_time)}")
print(f"Code ended at: {time.ctime(end_time)}")
print(f"Total time taken: {elapsed_time:.2f} seconds")
