import os
import json
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
import argparse
from coffea.lumi_tools import LumiMask
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, PFNanoAODSchema
import pdb, glob

PFNanoAODSchema.warn_missing_crossrefs = False
PFNanoAODSchema.mixins["DisMuon"] = "Muon"

hep.style.use("CMS")

sample_folder = "/eos/cms/store/user/fiorendi/displacedTaus/skim/prompt_mutau/selected/merged/"


## if a sample is not ready yet, comment it out
all_samples_dict = {
    "DY" : [
#         "DY",
      ],
    "QCD" : [
        "QCD_PT-80to120",
        "QCD_PT-120to170",
        "QCD_PT-170to300",
        "QCD_PT-300to470",
        "QCD_PT-470to600",
        "QCD_PT-600to800",
        "QCD_PT-800to1000",
        "QCD_PT-1000to1400",
        "QCD_PT-1400to1800",
        "QCD_PT-1800to2400",
      ],
    "Wto2Q" : [
        "Wto2Q-2Jets_PTQQ-100to200_1J",
        "Wto2Q-2Jets_PTQQ-100to200_2J",
        "Wto2Q-2Jets_PTQQ-200to400_1J",
        "Wto2Q-2Jets_PTQQ-200to400_2J",
#         "Wto2Q-2Jets_PTQQ-400to600_1J",
        "Wto2Q-2Jets_PTQQ-400to600_2J",
#         "Wto2Q-2Jets_PTQQ-600_1J",
#         "Wto2Q-2Jets_PTQQ-600_2J",
     ],
    "WtoLNu" : [
#         "WtoLNu-2Jets_0J",
        "WtoLNu-2Jets_1J",
#         "WtoLNu-2Jets_2J",
        "WtoLNu-4Jets_3J",
        "WtoLNu-4Jets_4J",
      ],
    "TT" : [
      "TTto2L2Nu", 
#       "TTtoLNu2Q", 
      "TTto4Q"
      ],
    "Stau" : []
}

## build reverse lookup dict to be used when retrieving sum gen events etc
reverse_samples_lookup = {
    subsample: key
    for key, subs in all_samples_dict.items()
    for subsample in subs
}

available_processes = [sub for subs in all_samples_dict.values() for sub in subs]

## load xsec and br settings from JSON file
with open("./plots_config/xsec_and_br.json", "r") as file:
    xsec_and_br = json.load(file)
xsec = xsec_and_br['xsec']
br = xsec_and_br['br']

## load sum_gen_w from another JSON file
with open("./plots_config/all_total_sumw.json", "r") as file:
    sum_gen_w = json.load(file)
    
## target lumi, for now fixed value
target_lumi = 50

# int_lumi = ak.from_parquet("../sample_processing/my_skimData2018C").intLumi[0]
#print(int_lumi)
#print(ak.from_parquet("../sample_processing/my_skimData2018C").intLumi)


## Parser setup
# parser = argparse.ArgumentParser(description="Make control plots for Z to mu mu studies.")
# parser.add_argument("--dataset", choices=["all"] + available_processes, default = "all", type=str, help="specifying which plot you want") # changeable in future... 
# parser.add_argument("--groupProcesses", action="store_true", default = "false", help="saying which processes do you want to group")
# args = parser.parse_args()

dataset_name = 'all'

# Load plot settings from JSON file
with open("./plots_config/plot_settings.json", "r") as file:
    plot_settings = json.load(file)

# Directory creation for plots (directory_path = "plots/" + args.process)
directory_path = "plots/"
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
    print(f"Directory '{directory_path}' created successfully.")

# Dictionary for histograms and binnings
histogram_dict = {}
binning_dict = {}

for process in available_processes:
    # print(process)
    tmp_string = f"merged_{process}/merged_{process}.root"
    tmp_file = sample_folder +  tmp_string
    events = NanoEventsFactory.from_root({tmp_file:"Events"}, schemaclass= PFNanoAODSchema).events()

    ## disable for now
#     print ('need to put back weights')
    weights = events.run / events.run
    if "data" in process.lower():
        weights = events.weight
    else:
        lumi_weight = target_lumi * xsec[process] * br[process] * 1000 / sum_gen_w[reverse_samples_lookup[process]][process]
        weights = events.weight * lumi_weight ## am i missing the sumGenW here?

    for plot_name, settings in plot_settings.items():
        ## test, not clear why
        var_values = getattr(getattr(events, settings["field"]), settings["variable"])
#         print("vals type:", type(var_values), "vals layout:", var_values.layout.form)
        if not settings["per_event"]:
            vals_flat = ak.flatten(var_values)
            weights_broadcast = ak.broadcast_arrays(var_values, weights)[1]
            weights_flat = ak.flatten(weights_broadcast)
            hist, _ = np.histogram(vals_flat, weights=weights_flat, bins=np.linspace(*settings["binning_linspace"]))

        else:
            hist, _ = np.histogram(getattr(getattr(events, settings["field"]), settings["variable"]), weights=weights, bins=np.linspace(*settings["binning_linspace"]))

        if plot_name not in histogram_dict:
            histogram_dict[plot_name] = {}

        if process not in histogram_dict[plot_name]:
            histogram_dict[plot_name][process] = []

        histogram_dict[plot_name][process].append(hist)

        if plot_name not in binning_dict:
            binning_dict[plot_name] = {}

        if process not in binning_dict[plot_name]:
            binning_dict[plot_name][process] = np.linspace(*settings["binning_linspace"])



# Plotting part
groupProcesses = True

for plot_name, histograms in histogram_dict.items():
    print('INFO: Now making plot for', plot_name, '...')
    binning = np.linspace(*plot_settings[plot_name].get("binning_linspace"))
    do_stack = not plot_settings[plot_name].get("density")
    # if args.groupProcesses:
    if groupProcesses:
        hist_Sig = np.zeros(len(binning)-1)
        hist_W   = np.zeros(len(binning)-1)
        hist_EWK = np.zeros(len(binning)-1)
        hist_Top = np.zeros(len(binning)-1)
        hist_QCD = np.zeros(len(binning)-1)
    
    hists_to_plot = []
    labels = []

    for process, histogram in histograms.items():
#         print('INFO: Now looking at process', process, '...')
        histogram = np.asarray(histogram)
        histogram = histogram.flatten()
        # Fix later, hist should not be a list in the first place and should be 60 1d and not (60,1)
        # if args.groupProcesses:
        if groupProcesses:
            if process == "Data2018C":
                data_hist = histogram
            elif process in all_samples_dict['TT']:
                hist_Top += histogram
            elif process in all_samples_dict['DY']:
                hist_EWK += histogram
            elif process in all_samples_dict['WtoLNu']:
                hist_W += histogram
            elif process in all_samples_dict['Wto2Q']:
                hist_W += histogram
            elif process in all_samples_dict['QCD']:
                hist_QCD += histogram
            elif process in all_samples_dict['Stau']:
                hist_Sig += histogram
        else:
            if process == "Data2018C":
                data_hist = histogram
            elif process == "DYJetsToLL":
                hists_to_plot.append(histogram)
                labels.append(process)
            else:
                hists_to_plot.append(histogram)
                labels.append(process)

    # print (hist_EWK)
    if groupProcesses:
    # if args.groupProcesses:
        hists_to_plot.append(hist_EWK)
        labels.append('DY')
        hists_to_plot.append(hist_Top)
        labels.append('Top')
        hists_to_plot.append(hist_W)
        labels.append('WJets')
        hists_to_plot.append(hist_QCD)
        labels.append('QCD')
        hists_to_plot.append(hist_Sig)
        labels.append('signal')

    colours = hep.style.cms.cmap_petroff
    
    fig, (ax_main, ax_ratio) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    fig.subplots_adjust(hspace=0.0)
    hep.histplot(hists_to_plot, bins=binning, stack=do_stack, histtype='fill', 
                 label=labels, sort='label_r', #color=colours,
                 density=plot_settings[plot_name].get("density"), ax=ax_main)
    # hep.histplot(data_hist, xerr=True, bins=binning, stack=False, histtype='errorbar', 
                 # color='black', label='Data', density=plot_settings[plot_name].get("density"), ax=ax_main)
    ax_main.set_ylabel(plot_settings[plot_name].get("ylabel"))
    ax_main.legend()
    
    ## this part is still to be done 
    if groupProcesses:
    # # # if args.groupProcesses:
        sum_histogram = np.sum(np.asarray(hists_to_plot), axis=0)
        # ratio_hist = np.asarray(histogram_dict[plot_name]["Data2018C"]).flatten() / (sum_histogram + np.finfo(float).eps)
    # #     # Adding relative sqrtN Poisson uncertainty for now, should be improved when using the hist package
    # #     rel_unc = np.sqrt(np.asarray(histogram_dict[plot_name]["Data2018C"]).flatten()) / np.asarray(histogram_dict[plot_name]["Data2018C"]).flatten()
    # #     rel_unc *= ratio_hist
    # #     rel_unc[rel_unc < 0] = 0 # Not exactly sure why we have negative values, but this solves it for the moment
    # #     hep.histplot(ratio_hist, bins=binning, histtype='errorbar', yerr=rel_unc, color='black', label='Ratio', ax=ax_ratio)
    # #     ax_ratio.axhline(1, color='gray', linestyle='--')
    ax_ratio.set_xlabel(plot_settings[plot_name].get("xlabel"), usetex=True)
    # #     ax_ratio.set_ylabel('Data / MC')
    # #     ax_ratio.set_xlim(binning[0], binning[-1])
    # #     ax_ratio.set_ylim(0.6, 1.4)
    
    # Decorating with CMS label
    hep.cms.label(data=True, loc=0, label="Private Work", com=13.6, lumi=round(target_lumi, 1), ax=ax_main)
    
    # Saving with special name
    filename = f"./plots/{dataset_name}_{plot_name}"
    # #if args.groupProcesses:
    if plot_settings[plot_name].get("density"):
        filename += "_normalized"
    else: 
        filename += "_stacked"
    filename += ".pdf"
    plt.savefig(filename)
    ax_main.set_yscale('log')
    plt.savefig(filename.replace('.pdf', '_log.pdf'))
    plt.clf()