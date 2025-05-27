import numpy as np 
import uproot
import hist
import awkward as ak 
import math
import scipy
import array
import ROOT
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

start_time = time.time()


SAMP = [
      ['Stau_100_1000mm', 'SIG'],
      ['Stau_100_100mm', 'SIG'],
      ['Stau_100_10mm', 'SIG'],
      ['Stau_100_1mm', 'SIG'],
      ['Stau_100_0p1mm', 'SIG'],
      ['Stau_100_0p01mm', 'SIG'],
      ['Stau_200_1000mm', 'SIG'],
      ['Stau_200_100mm', 'SIG'],
      ['Stau_200_10mm', 'SIG'],
      ['Stau_200_1mm', 'SIG'],
      ['Stau_200_0p1mm', 'SIG'],
      ['Stau_200_0p01mm', 'SIG'],
      ['Stau_300_1000mm', 'SIG'],
      ['Stau_300_100mm', 'SIG'],
      ['Stau_300_10mm', 'SIG'],
      ['Stau_300_1mm', 'SIG'],
      ['Stau_300_0p1mm', 'SIG'],
      ['Stau_300_0p01mm', 'SIG'],
      ['Stau_500_1000mm', 'SIG'],
      ['Stau_500_100mm', 'SIG'],
      ['Stau_500_10mm', 'SIG'],
      ['Stau_500_1mm', 'SIG'],
      ['Stau_500_0p1mm', 'SIG'],
      ['Stau_500_0p01mm', 'SIG'],
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
      #["WtoLNu2Jets", 'EWK'],
      ["TTtoLNu2Q",  'TT'],
      ["TTto4Q", 'TT'],
      ["TTto2L2Nu", 'TT'],
      ]

lumi = 38.01 ##fb-1
colors = ['#56CBF9', '#FDCA40', '#5DFDCB', '#D3C0CD', '#3A5683', '#FF773D']
Stau_colors = ['#EA7AF4', '#B43E8F', '#6200B3', '#218380']
variables_with_bins = {
    #"DisMuon_pt": [(245, 20, 1000), "GeV"],
    #"DisMuon_eta": [(50, -2.5, 2.5), ""],
    #"DisMuon_phi": [(64, -3.2, 3.2), ""],
    #"DisMuon_dxy": [(50, -50E-4, 50E-4), "cm"],
    #"DisMuon_dz" : [(50, -0.1, 0.1), "cm"],
    #"DisMuon_pfRelIso03_all": [(50, 0, 1), ""],
    #"DisMuon_pfRelIso03_chg": [(50, 0, 1), ""],
    #"DisMuon_pfRelIso04_all": [(50, 0, 1), ""],
    #"DisMuon_tkRelIso":       [(50, 0, 1), ""],

    "Jet_pt" : [(49, 20, 1000), "GeV"],
    #"Jet_eta": [(48, -2.4, 2.4), ""],
    #"Jet_phi": [(64, -3.2, 3.2), ""],
    #"Jet_disTauTag_score1": [(20, 0, 1), ""],
    #"Jet_dxy": [(50, -0.5, 0.5), "cm"],

    #"dR" : [(20, 0, 1), ""],
    #"deta": [(100, -5, 5), ""],
    #"dphi": [(64, -3.2, 3.2), ""],
    #"PFMET_pt": [(225, 100, 1000), "GeV"],
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
background_samples["QCD"] = []
background_samples["TT"] = []
#background_samples["LNu2Q"] = []
#background_samples["4Q"] = []
background_samples["W"] = []
background_samples["DY"] = []

for samples in SAMP:
    if "QCD" in samples[0]:
        background_samples["QCD"].append( ("/eos/uscms/store/user/dally/second_skim_muon_root/merged/merged_prompt_score" + samples[0] + "/*.root", xsecs[samples[0]] * lumi * 1000 * 1/num_events[samples[0]]))
    if "TT" in samples[0]:
        background_samples["TT"].append(  ("/eos/uscms/store/user/dally/second_skim_muon_root/merged/merged_prompt_score" + samples[0] + "/*.root", xsecs[samples[0]] * lumi * 1000 * 1/num_events[samples[0]]))
        #background_samples["2L2Nu"].append(  ("/eos/uscms/store/user/dally/second_skim_muon_root/merged/merged_prompt_scoreprompt_isoTTto2L2Nu" + "/*.root", xsecs[samples[0]] * lumi * 1000 * 1/num_events[samples[0]]))
        #background_samples["LNu2Q"].append(  ("/eos/uscms/store/user/dally/second_skim_muon_root/merged/merged_prompt_scoreprompt_isoTTtoLNu2Q" + "/*.root", xsecs[samples[0]] * lumi * 1000 * 1/num_events[samples[0]]))
        #background_samples["4Q"].append(  ("/eos/uscms/store/user/dally/second_skim_muon_root/merged/merged_prompt_scoreprompt_isoTTto4Q" + "/*.root", xsecs[samples[0]] * lumi * 1000 * 1/num_events[samples[0]]))
    if "W" in samples[0]:
        background_samples["W"].append(   ("/eos/uscms/store/user/dally/second_skim_muon_root/merged/merged_prompt_score" + samples[0] + "/*.root", xsecs[samples[0]] * lumi * 1000 * 1/num_events[samples[0]]))
    if "DY" in samples[0]:
        background_samples["DY"].append(  ("/eos/uscms/store/user/dally/second_skim_muon_root/merged/merged_prompt_score" + samples[0] + "/*.root", xsecs[samples[0]] * lumi * 1000 * 1/num_events[samples[0]]))
    if "Stau" in samples[0]:
        background_samples[samples[0]] = [("/eos/uscms/store/user/dally/second_skim_muon_root/merged/merged_prompt_score" + samples[0] + "/*.root", xsecs[samples[0]] * lumi * 1000 * 1/num_events[samples[0]])]

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
            events = NanoEventsFactory.from_root({sample_file:"Events"}, schemaclass= PFNanoAODSchema).events()
            #events = uproot.dask(sample_file)
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
    QCD_event_num = background_histograms["QCD"][var].sum()
    TT_event_num = background_histograms["TT"][var].sum()
    DY_event_num = background_histograms["DY"][var].sum()

    total_event_number = QCD_event_num + \
                         TT_event_num  + \
                         DY_event_num
    plt.cla()
    plt.clf()

    s = hist.Stack.from_dict({f"QCD " + "%.2f"%(QCD_event_num/total_event_number): background_histograms["QCD"][var],
                              #"2L2Nu" : background_histograms["2L2Nu"][var],
                              #"LNu2Q" : background_histograms["LNu2Q"][var],
                              #"4Q" : background_histograms["4Q"][var],
                              f"TT " + "%.2f"%(TT_event_num/total_event_number) : background_histograms["TT"][var],
                              #"W": background_histograms["W"][var],
                              "DY " + "%.2f"%(DY_event_num/total_event_number): background_histograms["DY"][var],       
                                })
    s.plot(stack = True, histtype= "fill", color = [colors[0], colors[1], colors[3]])
    stau_counter = 0
    for sample in background_samples:
        if "Stau" in sample and "1000mm" in sample: 
            background_histograms[sample][var].plot(color = Stau_colors[stau_counter], label = sample)
            stau_counter += 1

    box = plt.subplot().get_position()
    plt.subplot().set_position([box.x0, box.y0, box.width * 0.8, box.height])   

    plt.xlabel(var + ' ' + variables_with_bins[var][1])
    plt.ylabel("A.U.")
    plt.yscale('log')
    plt.ylim(top=get_stack_maximum(s)*10)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = {"size": 8})
    plt.savefig(f"../www/inverted_cuts_prompt_score/mu_stacked_histogram_{var}_lessbins_1000mm.png")

    plt.cla()
    plt.clf()
    s = hist.Stack.from_dict({f"QCD " + "%.2f"%(QCD_event_num/total_event_number): background_histograms["QCD"][var],
                              #"2L2Nu" : background_histograms["2L2Nu"][var],
                              #"LNu2Q" : background_histograms["LNu2Q"][var],
                              #"4Q" : background_histograms["4Q"][var],
                              f"TT " + "%.2f"%(TT_event_num/total_event_number) : background_histograms["TT"][var],
                              #"W": background_histograms["W"][var],
                              "DY " + "%.2f"%(DY_event_num/total_event_number): background_histograms["DY"][var],       
                                })
    s.plot(stack = True, histtype= "fill", color = [colors[0], colors[1], colors[3]])
    stau_counter = 0
    for sample in background_samples:
        if "Stau" in sample and "100mm" in sample: 
            background_histograms[sample][var].plot(color = Stau_colors[stau_counter], label = sample)
            stau_counter += 1

    box = plt.subplot().get_position()
    plt.subplot().set_position([box.x0, box.y0, box.width * 0.8, box.height])   

    plt.xlabel(var + ' ' + variables_with_bins[var][1])
    plt.ylabel("A.U.")
    plt.yscale('log')
    plt.ylim(top=get_stack_maximum(s)*10)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = {"size": 8})
    plt.savefig(f"../www/inverted_cuts_prompt_score/mu_stacked_histogram_{var}_lessbins_100mm.png")

    plt.cla()
    plt.clf()
    s = hist.Stack.from_dict({f"QCD " + "%.2f"%(QCD_event_num/total_event_number): background_histograms["QCD"][var],
                              #"2L2Nu" : background_histograms["2L2Nu"][var],
                              #"LNu2Q" : background_histograms["LNu2Q"][var],
                              #"4Q" : background_histograms["4Q"][var],
                              f"TT " + "%.2f"%(TT_event_num/total_event_number) : background_histograms["TT"][var],
                              #"W": background_histograms["W"][var],
                              "DY " + "%.2f"%(DY_event_num/total_event_number): background_histograms["DY"][var],       
                                })
    s.plot(stack = True, histtype= "fill", color = [colors[0], colors[1], colors[3]])
    stau_counter = 0
    for sample in background_samples:
        if "Stau" in sample and "10mm" in sample: 
            background_histograms[sample][var].plot(color = Stau_colors[stau_counter], label = sample)
            stau_counter += 1

    box = plt.subplot().get_position()
    plt.subplot().set_position([box.x0, box.y0, box.width * 0.8, box.height])   

    plt.xlabel(var + ' ' + variables_with_bins[var][1])
    plt.ylabel("A.U.")
    plt.yscale('log')
    plt.ylim(top=get_stack_maximum(s)*10)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = {"size": 8})
    plt.savefig(f"../www/inverted_cuts_prompt_score/mu_stacked_histogram_{var}_lessbins_10mm.png")

    plt.cla()
    plt.clf()
    s = hist.Stack.from_dict({f"QCD " + "%.2f"%(QCD_event_num/total_event_number): background_histograms["QCD"][var],
                              #"2L2Nu" : background_histograms["2L2Nu"][var],
                              #"LNu2Q" : background_histograms["LNu2Q"][var],
                              #"4Q" : background_histograms["4Q"][var],
                              f"TT " + "%.2f"%(TT_event_num/total_event_number) : background_histograms["TT"][var],
                              #"W": background_histograms["W"][var],
                              "DY " + "%.2f"%(DY_event_num/total_event_number): background_histograms["DY"][var],       
                                })
    s.plot(stack = True, histtype= "fill", color = [colors[0], colors[1], colors[3]])
    stau_counter = 0
    for sample in background_samples:
        if "Stau" in sample and "_1mm" in sample: 
            print(f"Stau_counter {stau_counter}")
            background_histograms[sample][var].plot(color = Stau_colors[stau_counter], label = sample)
            stau_counter += 1

    box = plt.subplot().get_position()
    plt.subplot().set_position([box.x0, box.y0, box.width * 0.8, box.height])   

    plt.xlabel(var + ' ' + variables_with_bins[var][1])
    plt.ylabel("A.U.")
    plt.yscale('log')
    plt.ylim(top=get_stack_maximum(s)*10)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = {"size": 8})
    plt.savefig(f"../www/inverted_cuts_prompt_score/mu_stacked_histogram_{var}_lessbins_1mm.png")

    plt.cla()
    plt.clf()
    s = hist.Stack.from_dict({f"QCD " + "%.2f"%(QCD_event_num/total_event_number): background_histograms["QCD"][var],
                              #"2L2Nu" : background_histograms["2L2Nu"][var],
                              #"LNu2Q" : background_histograms["LNu2Q"][var],
                              #"4Q" : background_histograms["4Q"][var],
                              f"TT " + "%.2f"%(TT_event_num/total_event_number) : background_histograms["TT"][var],
                              #"W": background_histograms["W"][var],
                              "DY " + "%.2f"%(DY_event_num/total_event_number): background_histograms["DY"][var],       
                                })
    s.plot(stack = True, histtype= "fill", color = [colors[0], colors[1], colors[3]])
    stau_counter = 0
    for sample in background_samples:
        if "Stau" in sample and "0p1mm" in sample: 
            background_histograms[sample][var].plot(color = Stau_colors[stau_counter], label = sample)
            stau_counter += 1

    box = plt.subplot().get_position()
    plt.subplot().set_position([box.x0, box.y0, box.width * 0.8, box.height])   

    plt.xlabel(var + ' ' + variables_with_bins[var][1])
    plt.ylabel("A.U.")
    plt.yscale('log')
    plt.ylim(top=get_stack_maximum(s)*10)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = {"size": 8})
    plt.savefig(f"../www/inverted_cuts_prompt_score/mu_stacked_histogram_{var}_lessbins_0p1mm.png")

    plt.cla()
    plt.clf()
    s = hist.Stack.from_dict({f"QCD " + "%.2f"%(QCD_event_num/total_event_number): background_histograms["QCD"][var],
                              #"2L2Nu" : background_histograms["2L2Nu"][var],
                              #"LNu2Q" : background_histograms["LNu2Q"][var],
                              #"4Q" : background_histograms["4Q"][var],
                              f"TT " + "%.2f"%(TT_event_num/total_event_number) : background_histograms["TT"][var],
                              #"W": background_histograms["W"][var],
                              "DY " + "%.2f"%(DY_event_num/total_event_number): background_histograms["DY"][var],       
                                })
    s.plot(stack = True, histtype= "fill", color = [colors[0], colors[1], colors[3]])
    stau_counter = 0
    for sample in background_samples:
        if "Stau" in sample and "0p01mm" in sample: 
            background_histograms[sample][var].plot(color = Stau_colors[stau_counter], label = sample)
            stau_counter += 1

    box = plt.subplot().get_position()
    plt.subplot().set_position([box.x0, box.y0, box.width * 0.8, box.height])   

    plt.xlabel(var + ' ' + variables_with_bins[var][1])
    plt.ylabel("A.U.")
    plt.yscale('log')
    plt.ylim(top=get_stack_maximum(s)*10)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = {"size": 8})
    plt.savefig(f"../www/inverted_cuts_prompt_score/mu_stacked_histogram_{var}_lessbins__0p01mm.png")


#with open(f"muon_QCD_hists_Iso_NotDisplaced.pkl", "wb") as f:
#    pickle.dump(background_histograms["QCD"], f)
#print(f"pkl file written")




end_time = time.time()

elapsed_time = end_time - start_time

print(f"Code started at: {time.ctime(start_time)}")
print(f"Code ended at: {time.ctime(end_time)}")
print(f"Total time taken: {elapsed_time:.2f} seconds")
