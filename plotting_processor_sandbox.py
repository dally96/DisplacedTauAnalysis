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
#histograms = {}
#for var, bin_info in variables_with_bins.items():
#    histograms[var] = hda.hist.Hist(hist.axis.Regular(*bin_info, name = var, label = var), hist.axis.StrCategory(["QCD", "TT", "EWK"], name="category"), hist.storage.Weight())
                        
#print(histograms)
#data_array = []
#weight_array = []
#category_array = []

class ExampleProcessor(processor.ProcessorABC):
    def __init__(self, vars_with_bins):
        self.vars_with_bins = vars_with_bins
    #    self.histos = histos
        print("Initializing ExampleProcessor")
    #    self.variables_with_bins = variables_with_bins
    def initialize_histograms(self):
        histograms = {}
        # Initialize histograms for each variable based on provided binning
        for var, bin_info in self.vars_with_bins.items():
            print(f"Creating histogram for {var} with bin_info {bin_info}")

            histograms[var] = hda.hist.Hist(hist.axis.Regular(*bin_info, name=var, label = 'pt [GeV]'))

    #                            
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
            #print(f'The array that should be filling the histo is {dak.flatten(events[var]).compute()}')
            histograms[var].fill(
                **{var: dak.flatten(events[var])},
                weight = weights
            )
        #delayed_histos = {}
        #for var in self.vars_with_bins:
        #    print(f"histogram before", self.histos[var].compute())
        #    print(f'The array that should be filling the histo is {dak.flatten(events[var]).compute()}')
        #    data = dak.flatten(events[var])
        #    print(f"Category is {categories}")
        #    print(f"Var is {data}")
        #    print(f"Weights is {weights}")
        #    delayed_histos[var] = dask.delayed(self.histos[var].fill)(
        #        category = categories,
        #        **{var: data},
        #        weight = weights
        #    )   
        #print(f"histogram after", self.histos[var].compute())
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


## Initialize dictionary to hold accumulated histograms for each background
#background_histograms = {}
## Process each background
#for background, samples in background_samples.items():
#    # Initialize a dictionary to hold histograms for the current background
#    background_histograms[background] = {}
#    for sample_file, sample_weight in samples:
#        try:
#            # Step 1: Load events for the sample using dask-awkward
#            events = dak.from_parquet(sample_file)
#            print(f'Starting {sample_file} histogram')         
#
#            processor_instance = ExampleProcessor(variables_with_bins)
#            output = processor_instance.process(events, sample_weight)
#            print(f'{sample_file} finished successfully')
#            for var, histo in output["histograms"].items():
#                if var not in background_histograms[background]:
#                    background_histograms[background][var] = histo.compute()
#                else:
#                    background_histograms[background][var] += histo.compute()
#        except Exception as e:
#            print(f"Error processing {sample_file}: {e}")
#processor_instance = ExampleProcessor(variables_with_bins, histograms)
#for samples in SAMP:
#    #try:
#    events = dak.from_parquet("my_skim_electron_" + samples[0] + "/*.parquet")
#    processor_instance.process(events, xsecs[samples[0]] * lumi * 1000 * 1/num_events[samples[0]], samples[1])
    #processor_instance.postprocess()
    #print(f"Postprocess output is", processor_instance.postprocess()output)

# Now, for plotting all variables
# Import ROOT if not already done

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

#for var in variables_with_bins:
#    # Collect histograms for this variable across all backgrounds
#    stacked_counts = []
#    labels = []
#    bottoms = None  # Track the bottom for stacking
#    bin_edges = None  # Store bin edges (assuming all backgrounds share the same binning)
#
#    for i, background in enumerate(background_histograms):
#        if var in background_histograms[background]:
#            # Extract counts and bin edges for the background histogram
#            counts, edges = background_histograms[background][var].to_numpy()
#            if bin_edges is None:
#                bin_edges = edges  # Initialize bin edges (only needed once)
#
#            # Plot stacked bars for each background
#            if bottoms is None:
#                bottoms = np.zeros_like(counts)  # Initialize bottom for the first stack
#
#            # Use `plt.bar` to plot the counts
#            plt.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), bottom=bottoms,
#                    color=colors[i], label=background, align='edge')
#
#            # Update `bottoms` for stacking in the next iteration
#            bottoms += counts
#            labels.append(background)
#
#    # Customize and save the plot
#    plt.xlabel(var)  # Set x-axis label to the variable name
#    plt.ylabel('A.U.')  # Set y-axis label
#    plt.ylim(1, 1E8)
#    plt.title(f"Stacked Histogram for {var}")
#    plt.yscale('log')
#    plt.legend()
#    plt.savefig(f"test_stacked_histogram_{var}.pdf")
#    plt.clf()  # Clear the figure for the next variable
#for var in variables_with_bins:
#    stacked_histograms = []
#    labels = []
#
#    # Collect histograms for this variable across all backgrounds
#    for background in background_histograms:
#        if var in background_histograms[background]:
#            stacked_histograms.append(background_histograms[background][var])
#            print(background_histograms[background][var])
#            labels.append(background)
#
#    print(f"Plotting for {var}")
#    print(f"Stacked histograms", stacked_histograms)
#    print(f"Histogram axes", stacked_histograms[0].axes[0].edges)
#    for hist in stacked_histograms:
#        #plt.hist(stacked_histograms, stacked_histograms[0].axes[0].edges, stacked=True, histtype="step", color = [colors[0], colors[14], colors[15]], label=labels)
#        hist.plot1d(stack = True, histtype='step')
#        plt.yscale('log')
#        plt.ylim([1, 1E8])
#        #plt.legend()
#        plt.xlabel(var)  # Set x-axis label to the variable name
#        plt.ylabel('A.U.')  # Set y-axis label
#        plt.title(f"Stacked Histogram for {var}")
#    plt.savefig(f"test_stacked_histogram_{var}.pdf")
#for var in variables_with_bins:
#    for entry in range(len(data_array)):
#        data = dak.flatten(data_array[entry][var])
#        data = data.repartition(npartitions=200)
#        print(data)
#        weights = [weight_array[entry],] * len(data.compute())
#        categories = [category_array[entry],] * len(data.compute())
#        histograms[var].fill(category = categories, **{var:data}, weight = weights)
        
#for var in variables_with_bins:
#    histos[var] = histograms[var].compute()

#    except Exception as e:
#        print(f"Error processing {samples[0]}: {e}")
#output = processor_instance.postprocess(histograms)
#print(output)
#for var in variables_with_bins.keys():
    #print(f"Postprocess output is", output[var].compute())
    #print(f"histograms is ", histograms[var].compute())

#for var in variables_with_bins:
#    #print(histograms[var].compute())
#    
#    histos[var] = input_histo[var].compute()
#final_histos = {var:dask.compute(*histos[var]) for var in variables_with_bins}
# Initialize dictionary to hold accumulated histograms for each background
#background_histograms = {}
## Process each background
#for background, samples in background_samples.items():
#    # Initialize a dictionary to hold histograms for the current background
#    background_histograms[background] = {}
#    
#    for sample_file, sample_weight, sample_category in samples:
#        try:
#            # Step 1: Load events for the sample using dask-awkward
#            events = dak.from_parquet(sample_file)
#            print(f'Starting {sample_file} histogram')         
#            print(f'Category is {sample_category}')
#
#            processor_instance = ExampleProcessor(variables_with_bins)
#            output = processor_instance.process(events, sample_weight, sample_category)
#            print(f'{sample_file} finished successfully')
#            for var, histo in output["histograms"].items():
#                histos[var] = histo.compute()
#                print(histo.compute())
#                #if var not in background_histograms[background]:
#                #    background_histograms[background][var] = histo.compute()
#                #else:
#                #    background_histograms[background][var] += histo.compute()
#        except Exception as e:
#            print(f"Error processing {sample_file}: {e}")

# Now, for plotting all variables
    #histos[var].stack("category")
#for var in variables_with_bins:
#    histograms[var].compute().plot(stack = True, color = [colors[0], colors[14], colors[15]])
#    plt.xlabel(var)
#    plt.ylabel('A.U.')
#    plt.ylim(1, 1E8)
#    plt.yscale('log')
#    plt.legend()
#    plt.savefig('test_' + var + '.pdf')
#print(var + "plot saved")
    #stacked_histograms = []
    #stacked_histograms_values = []
    #weights = []
    #labels = []

    ## Collect histograms for this variable across all backgrounds
    #for background in background_histograms:
    #    if var in background_histograms[background]:
    #        stacked_histograms_values.append(background_histograms[background][var].view().value)
    #        stacked_histograms.append(background_histograms[background][var])
    #        print(background_histograms[background][var].view().value)
    #        labels.append(background)
    #        weights.append(background_histograms[background][var].sum())            
    #        print(background_histograms[background][var].sum())

    #if stacked_histograms:
    #    print(f"Plotting for {var}")
    #    print(f'Axes for {var} are', stacked_histograms[0].axes[0].edges)
    #    plt.hist(stacked_histograms_values[0], bins = stacked_histograms[0].axes[0].edges, stacked=True, color = colors[0], label=labels[0])
    #    print(f'Hist entries are', hist)
    #    #plt.hist(stacked_histograms_values, bins = stacked_histograms[0].axes[0].edges, stacked=True, color = [colors[0], colors[14], colors[15]], label=labels)
    #    #plt.hist([np.random.normal(100, 20, 1000), np.random.normal(150, 25, 800), np.random.normal(200, 30, 600)], bins = np.linspace(50, 250, 30), histtype = 'step', stacked=True, color = [colors[0], colors[14], colors[15]], label=labels)
    #    plt.legend()
    #    plt.xlabel(var)  # Set x-axis label to the variable name
    #    plt.ylabel('A.U.')  # Set y-axis label
    #    plt.xscale('log')
    #    plt.title(f"Stacked Histogram for {var}")
    #plt.savefig(f"test_stacked_histogram_{var}.pdf")

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
