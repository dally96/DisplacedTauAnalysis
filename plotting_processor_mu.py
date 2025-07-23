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

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-d", "--data", dest = "data", help = "Are we plotting data", default = "MC")
parser.add_argument("-f", "--folder", dest = "folder", help = "Where to put plot. Directory located inside of ../www/", default = "test_dir")   

is_data = parser.parse_args()

#SAMP = [
#      ['Stau_100_1000mm', 'SIG'],
#      ['Stau_100_100mm', 'SIG'],
#      ['Stau_100_10mm', 'SIG'],
#      ['Stau_100_1mm', 'SIG'],
#      #['Stau_100_0p1mm', 'SIG'],
#      #['Stau_100_0p01mm', 'SIG'],
#      ['Stau_200_1000mm', 'SIG'],
#      ['Stau_200_100mm', 'SIG'],
#      ['Stau_200_10mm', 'SIG'],
#      ['Stau_200_1mm', 'SIG'],
#      #['Stau_200_0p1mm', 'SIG'],
#      #['Stau_200_0p01mm', 'SIG'],
#      ['Stau_300_1000mm', 'SIG'],
#      ['Stau_300_100mm', 'SIG'],
#      ['Stau_300_10mm', 'SIG'],
#      ['Stau_300_1mm', 'SIG'],
#      #['Stau_300_0p1mm', 'SIG'],
#      #['Stau_300_0p01mm', 'SIG'],
#      ['Stau_500_1000mm', 'SIG'],
#      ['Stau_500_100mm', 'SIG'],
#      ['Stau_500_10mm', 'SIG'],
#      ['Stau_500_1mm', 'SIG'],
#      #['Stau_500_0p1mm', 'SIG'],
#      #['Stau_500_0p01mm', 'SIG'],
#      ['QCD50_80', 'QCD'],
#      ['QCD80_120','QCD'],
#      ['QCD120_170','QCD'],
#      ['QCD170_300','QCD'],
#      ['QCD300_470','QCD'],
#      ['QCD470_600','QCD'],
#      ['QCD600_800','QCD'],
#      ['QCD800_1000','QCD'],
#      ['QCD1000_1400','QCD'],
#      ['QCD1400_1800','QCD'],
#      ['QCD1800_2400','QCD'],
#      ['QCD2400_3200','QCD'],
#      ['QCD3200','QCD'],
#      ["DYJetsToLL", 'EWK'],  
#      ['WtoLNu_2Jets_0J', 'EWK'],          
#      ['WtoLNu_2Jets_1J', 'EWK'],       
#      ['WtoLNu_2Jets_2J', 'EWK'],       
#      #['WtoLNu_4Jets_1J', 'EWK'],       
#      ['WtoLNu_4Jets_2J', 'EWK'],       
#      ['WtoLNu_4Jets_3J', 'EWK'],       
#      ['WtoLNu_4Jets_4J', 'EWK'],       
#      ['Wto2Q_2Jets_2J_100to200', 'EWK'],
#      ['Wto2Q_2Jets_2J_200to400', 'EWK'],
#      ['Wto2Q_2Jets_2J_400to600', 'EWK'],
#      ['Wto2Q_2Jets_2J_600', 'EWK'],    
#      ['Wto2Q_2Jets_1J_100to200', 'EWK'],
#      ['Wto2Q_2Jets_1J_200to400', 'EWK'],
#      ['Wto2Q_2Jets_1J_400to600', 'EWK'],
#      ['Wto2Q_2Jets_1J_600', 'EWK'],    
#      ["TTtoLNu2Q",  'TT'],
#      ["TTto4Q", 'TT'],
#      ["TTto2L2Nu", 'TT'],
#      ["JetMET_Run2022E", 'JetMET'],
#      ["JetMET_Run2022F", 'JetMET'],
#      ["JetMET_Run2022G", 'JetMET'],
#      ]

with open("merged_preprocessed_fileset.pkl", "rb") as f:
    samples = pickle.load(f)
SAMP = list(samples.keys())
print(SAMP)

lumi = 26.7 ##fb-1
colors = ['#56CBF9', '#FDCA40', '#5DFDCB', '#D3C0CD', '#3A5683', '#FF773D']
Stau_colors = ['#EA7AF4', '#B43E8F', '#6200B3', '#218380']
variables_with_bins = {
    "DisMuon_pt": [(245, 20, 1000), "GeV"],
    "DisMuon_eta": [(50, -2.5, 2.5), ""],
    "DisMuon_phi": [(64, -3.2, 3.2), ""],
    "DisMuon_dxy": [(50, -1, 1), "cm"],
    "DisMuon_dz" : [(50, -0.1, 0.1), "cm"],
    "DisMuon_pfRelIso03_all": [(50, 0, 1), ""],
    "DisMuon_pfRelIso03_chg": [(50, 0, 1), ""],
    "DisMuon_pfRelIso04_all": [(50, 0, 1), ""],
    "DisMuon_tkRelIso":       [(50, 0, 1), ""],

    "Jet_pt" : [(49, 20, 1000), "GeV"],
    "Jet_eta": [(48, -2.4, 2.4), ""],
    "Jet_phi": [(64, -3.2, 3.2), ""],
    "Jet_disTauTag_score1": [(20, 0, 1), ""],
    "Jet_dxy": [(50, -0.5, 0.5), "cm"],

    #"dR" : [(20, 0, 1), ""],
    #"deta": [(100, -5, 5), ""],
    #"dphi": [(64, -3.2, 3.2), ""],
    "PFMET_pt": [(225, 100, 1000), "GeV"],
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


class MCProcessor(processor.ProcessorABC):
    def __init__(self, vars_with_bins, sample):
        self.vars_with_bins = vars_with_bins
        self.sample = sample
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
        # Loop over variables and fill histograms
        for var in histograms:
            var_name = '_'.join(var.split('_')[1:])
            histograms[var].fill(
                **{var: dak.flatten(events[var.split('_')[0]][var_name], axis = None)},
                weight = events.weight * lumi * 1000 * self.sample
            )
            
            
        output = {"histograms": histograms}
        print(output)
        return output

    def postprocess(self):
        pass
        

class DataProcessor(processor.ProcessorABC):
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
        # Loop over variables and fill histograms
        for var in histograms:
            var_name = '_'.join(var.split('_')[1:])
            histograms[var].fill(
                **{var: dak.flatten(events[var.split('_')[0]][var_name], axis = None)},
                weight = events.weight / events.weight
            )
            
            
        output = {"histograms": histograms}
        print(output)
        return output

    def postprocess(self):
        pass
background_samples = {} 
data_samples = {}
background_samples["QCD"] = []
background_samples["TT"] = []
background_samples["W"] = []
background_samples["DY"] = []
data_samples["JetMET"] = []

stau_dict = {}
TT_dict = {}

for samples in SAMP:
    #if "QCD" in samples[0]:
    #    background_samples["QCD"].append(    ("/eos/uscms/store/user/dally/first_skim/merged/merged_" + samples[0] + "/*.root", xsecs[samples[0]] ))
    #if "TT" in samples[0]:
    if "TT" in samples:
        if "ext" in samples:
            background_samples["TT"].append(     ("/eos/uscms/store/group/lpcdisptau/dally/second_skim/first_skim_MET_trig_" + samples + "/*.root", xsecs['_'.join(samples.split('_')[0:-1])]))
            TT_dict[samples] = (     ("/eos/uscms/store/group/lpcdisptau/dally/second_skim/first_skim_all_trig_" + samples + "/*.root", xsecs['_'.join(samples.split('_')[0:-1])]))
        else:
            background_samples["TT"].append(     ("/eos/uscms/store/group/lpcdisptau/dally/second_skim/first_skim_MET_trig_" + samples + "/*.root", xsecs[samples] ))
            TT_dict[samples] = (     ("/eos/uscms/store/group/lpcdisptau/dally/second_skim/first_skim_all_trig_" + samples + "/*.root", xsecs[samples] ))
    #if "W" in samples[0]:
    #    background_samples["W"].append(      ("/eos/uscms/store/user/dally/first_skim/merged/merged_" + samples[0] + "/*.root", xsecs[samples[0]] ))
    #if "DY" in samples[0]:
    #    background_samples["DY"].append(     ("/eos/uscms/store/user/dally/first_skim/merged/merged_" + samples[0] + "/*.root", xsecs[samples[0]] ))
    #if is_data.data == "data":
    #    if "JetMET" in samples[0]: 
    #        data_samples["JetMET"].append(   ("/eos/uscms/store/user/dally/first_skim/merged/merged_" + samples[0] + "/*.root", 1                 ))
    #else:
    #    if "Stau" in samples[0]:
    #        lifetime = samples[0].split("_")[2]
    #        mass =     samples[0].split("_")[1]
    #        if lifetime in stau_dict.keys(): 
    #            stau_dict[lifetime][mass] = [    ("/eos/uscms/store/user/dally/first_skim/merged/merged_" + samples[0] + "/*.root", xsecs[samples[0]] )]
    #        else:
    #            stau_dict[lifetime] = {}
    #            stau_dict[lifetime][mass] = [    ("/eos/uscms/store/user/dally/first_skim/merged/merged_" + samples[0] + "/*.root", xsecs[samples[0]] )]

# Initialize dictionary to hold accumulated ROOT histograms for each background
background_histograms = {}

# Process each background
for background, samples in background_samples.items():
    # Initialize a dictionary to hold ROOT histograms for the current background
    background_histograms[background] = {}
    for var in variables_with_bins:
        background_histograms[background][var] = hda.hist.Hist(hist.axis.Regular(*variables_with_bins[var][0], name=var, label = var + ' ' + variables_with_bins[var][1])).compute()

    for sample_file, sample_weight in samples:
        try:
            # Step 1: Load events for the sample using dask-awkward
            events = NanoEventsFactory.from_root({sample_file:"Events"}, schemaclass= PFNanoAODSchema).events()
            #events = uproot.dask(sample_file)
            print(f'Starting {sample_file} histogram')         

            processor_instance = MCProcessor(variables_with_bins, sample_weight)
            output = processor_instance.process(events)
            print(f'{sample_file} finished successfully')

            # Loop through each variable's histogram in the output
            for var, dask_histo in output["histograms"].items():
                background_histograms[background][var]  = background_histograms[background][var] + dask_histo.compute()

        except Exception as e:
            print(f"Error processing {sample_file}: {e}")

TT_histograms = {}

# Process each background
for background, samples in TT_dict.items():
    # Initialize a dictionary to hold ROOT histograms for the current background
    for var in variables_with_bins:
        TT_histograms[var] = hda.hist.Hist(hist.axis.Regular(*variables_with_bins[var][0], name=var, label = var + ' ' + variables_with_bins[var][1])).compute()

    for sample_file, sample_weight in samples:
        try:
            # Step 1: Load events for the sample using dask-awkward
            events = NanoEventsFactory.from_root({sample_file:"Events"}, schemaclass= PFNanoAODSchema).events()
            #events = uproot.dask(sample_file)
            print(f'Starting {sample_file} histogram')         

            processor_instance = MCProcessor(variables_with_bins, sample_weight)
            output = processor_instance.process(events)
            print(f'{sample_file} finished successfully')

            # Loop through each variable's histogram in the output
            for var, dask_histo in output["histograms"].items():
                TT_histograms[var]  = TT_histograms[var] + dask_histo.compute()

        except Exception as e:
            print(f"Error processing {sample_file}: {e}")

if is_data.data == "data":
    print("Should only be going through this loop if fata flag true")
    data_histograms = {}

    for data, samples in data_samples.items():
        data_histograms[data] = {}
        for var in variables_with_bins:
            data_histograms[data][var] = hda.hist.Hist(hist.axis.Regular(*variables_with_bins[var][0], name=var, label = var + ' ' + variables_with_bins[var][1])).compute()

        for sample_file, sample_weight in samples:
            try:
                # Step 1: Load events for the sample using dask-awkward
                events = NanoEventsFactory.from_root({sample_file:"Events"}, schemaclass= PFNanoAODSchema).events()
                #events = uproot.dask(sample_file)
                print(f'Starting {sample_file} histogram')         

                processor_instance = DataProcessor(variables_with_bins)
                output = processor_instance.process(events)
                print(f'{sample_file} finished successfully')

                # Loop through each variable's histogram in the output
                for var, dask_histo in output["histograms"].items():
                    data_histograms[data][var]  = data_histograms[data][var] + dask_histo.compute()

            except Exception as e:
                print(f"Error processing {sample_file}: {e}")


# Process each stau sample
else:
    stau_histograms = {}
               
    for lifetime in stau_dict.keys():
        stau_histograms[lifetime] = {}
        for mass in stau_dict[lifetime].keys():
            stau_histograms[lifetime][mass] = {}
    
            for var in variables_with_bins:
                stau_histograms[lifetime][mass][var] = hda.hist.Hist(hist.axis.Regular(*variables_with_bins[var][0], name=var, label = var + ' ' + variables_with_bins[var][1])).compute()
            for sample_file, sample_weight in stau_dict[lifetime][mass]:
                try:
                    events = NanoEventsFactory.from_root({sample_file:"Events"}, schemaclass= PFNanoAODSchema).events()
                    print(f'Starting {sample_file} histogram')
                    
                    processor_instance = MCProcessor(variables_with_bins, sample_weight)
                    output = processor_instance.process(events)
                    print(f'{sample_file} finished successfully')
    
                    for var, dask_histo in output["histograms"].items():
                        stau_histograms[lifetime][mass][var] = stau_histograms[lifetime][mass][var] + dask_histo.compute()
                except Exception as e:
                    print(f"Error processing {sample_file}: {e}")

with open(f"muon_QCD_hists_Iso_NotDisplaced.pkl", "wb") as f:
    pickle.dump(background_histograms["QCD"], f)
print(f"pkl file written")

for var in variables_with_bins:
    #QCD_event_num = background_histograms["QCD"][var].sum()
    #TT_event_num = background_histograms["TT"][var].sum()
    #DY_event_num = background_histograms["DY"][var].sum()
    #W_event_num = background_histograms["W"][var].sum()

    #total_event_number = QCD_event_num + \
    #                     TT_event_num  + \
    #                     DY_event_num  + \
    #                     W_event_num
    plt.cla()
    plt.clf()

    #QCD_frac = 0
    #TT_frac  = 0
    #DY_frac  = 0
    #W_frac   = 0

    #if total_event_number > 0:
    #    QCD_frac = QCD_event_num/total_event_number
    #    TT_frac  = TT_event_num/total_event_number
    #    DY_frac  = DY_event_num/total_event_number
    #    W_frac  = W_event_num/total_event_number
    if is_data.data == "data":
        s = hist.Stack.from_dict({f"QCD " + "%.2f"%(QCD_frac): background_histograms["QCD"][var],
                                f"TT " + "%.2f"%(TT_frac) : background_histograms["TT"][var],
                                "DY " + "%.2f"%(DY_frac): background_histograms["DY"][var],       
                                "W" + "%.2f"%(W_frac): background_histograms["W"][var],
                                  })
        s.plot(stack = True, histtype= "fill", color = [colors[0], colors[1], colors[3], colors[2]])
        data_histograms["JetMET"][var].plot(color = 'black', label = 'data')
        box = plt.subplot().get_position()
        plt.subplot().set_position([box.x0, box.y0, box.width * 0.8, box.height])   

        plt.xlabel(var + ' ' + variables_with_bins[var][1])
        plt.ylabel("A.U.")
        plt.yscale('log')
        plt.title(r"$\mathcal{L}_{int}$ = 26.7 fb$^{-1}$")
        plt.ylim(top=data_histograms["JetMET"][var].view().max()*10)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = {"size": 8})
        if is_data.folder not in os.listdir("../www/"):
            os.mkdir(f"../www/{is_data.folder}")
        plt.savefig(f"../www/{is_data.folder}/data_histogram_{var}.png")

        plt.cla()
        plt.clf()

    if is_data.data == "MC":
        for lifetime in stau_dict.keys():
            s = hist.Stack.from_dict({f"QCD " + "%.2f"%(QCD_frac): background_histograms["QCD"][var],
                                    f"TT " + "%.2f"%(TT_frac) : background_histograms["TT"][var],
                                    "DY " + "%.2f"%(DY_frac): background_histograms["DY"][var],       
                                    "W " + "%.2f"%(W_frac): background_histograms["W"][var],
                                      })
            s.plot(stack = True, histtype= "fill", color = [colors[0], colors[1], colors[3]])
            stau_counter = 0
            for mass in stau_dict[lifetime].keys():
                stau_histograms[lifetime][mass][var].plot(color = Stau_colors[stau_counter], label = r"$\tilde{\tau}$ " + mass + "_" + lifetime)
                stau_counter += 1

            box = plt.subplot().get_position()
            plt.subplot().set_position([box.x0, box.y0, box.width * 0.8, box.height])   

            plt.xlabel(var + ' ' + variables_with_bins[var][1])
            plt.ylabel("A.U.")
            plt.yscale('log')
            plt.ylim(top=get_stack_maximum(s)*10)
            plt.title(r"L$_{int}$ = 26.7 fb$^{-1}$")
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = {"size": 8})
            if is_data.folder not in os.listdir("../www/"):
                os.mkdir(f"../www/{is_data.folder}")
            plt.savefig(f"../www/{is_data.folder}/muon_histogram_{var}_{lifetime}.png")

            plt.cla()
            plt.clf()
    if is_data.data == "trig":
        background_histograms["TT"][var].plot(histtype= "fill", color = colors[0], label = "MET triggers")
        TT_histograms[var].plot(histtype= "step", color = colors[1], label = "all triggers")            
        box = plt.subplot().get_position()
        plt.subplot().set_position([box.x0, box.y0, box.width * 0.8, box.height])   

        plt.xlabel(var + ' ' + variables_with_bins[var][1])
        plt.ylabel("A.U.")
        plt.yscale('log')
        plt.ylim(top=get_stack_maximum(s)*10)
        plt.title(r"L$_{int}$ = 26.7 fb$^{-1}$")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = {"size": 8})
        if is_data.folder not in os.listdir("../www/"):
            os.mkdir(f"../www/{is_data.folder}")
        plt.savefig(f"../www/{is_data.folder}/muon_histogram_{var}_{lifetime}.png")

        plt.cla()
        plt.clf()
        




end_time = time.time()

elapsed_time = end_time - start_time

print(f"Code started at: {time.ctime(start_time)}")
print(f"Code ended at: {time.ctime(end_time)}")
print(f"Total time taken: {elapsed_time:.2f} seconds")
