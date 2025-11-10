import os
import json
import numpy as np
import awkward as ak
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import mplhep as hep
import argparse
from coffea.lumi_tools import LumiMask
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, PFNanoAODSchema
import pdb, glob

PFNanoAODSchema.warn_missing_crossrefs = False
PFNanoAODSchema.mixins["DisMuon"] = "Muon"

hep.style.use("CMS")

nanov = 'Summer22_CHS_v10/'
# nanov = ''
sample_folder = f"/eos/uscms/store/user/dally/skim/{nanov}prompt_mutau/v8/selected/faster_trial/"

## if a sample is not ready yet, comment it out
all_samples_dict = {
    "DY" : [
        "DYJetsToLL_M-50",
      ],
    "QCD" : [
##       "QCD_PT-50to80",
        "QCD_PT-80to120",
##        "QCD_PT-120to170",
#        "QCD_PT-170to300",
        "QCD_PT-300to470",
        "QCD_PT-470to600",
        "QCD_PT-600to800",
#        "QCD_PT-800to1000",
##        "QCD_PT-1000to1400",
##        "QCD_PT-1400to1800",
##        "QCD_PT-1800to2400",
##        "QCD_PT-3200",
     ],
    "Wto2Q" : [
        "Wto2Q-2Jets_PTQQ-100to200_1J",
##        "Wto2Q-2Jets_PTQQ-100to200_2J",
        "Wto2Q-2Jets_PTQQ-200to400_1J",
        "Wto2Q-2Jets_PTQQ-200to400_2J",
##        "Wto2Q-2Jets_PTQQ-400to600_1J",
##        "Wto2Q-2Jets_PTQQ-400to600_2J",
##        "Wto2Q-2Jets_PTQQ-600_1J",
##        "Wto2Q-2Jets_PTQQ-600_2J",
     ],
    "WtoLNu" : [
        "WtoLNu-4Jets",
##         "WtoLNu-2Jets_0J",
##        "WtoLNu-2Jets_1J",
##         "WtoLNu-2Jets_2J",
##        "WtoLNu-4Jets_3J",
      ],
    "TT" : [
      "TTto2L2Nu", 
#      "TTtoLNu2Q", 
#      "TTto4Q"
      ],
    "singleT": [
#      "TbarWplustoLNu2Q",
      "TbarWplusto2L2Nu",
      "TWminusto2L2Nu",
#      "TBbarQ_t-channel_4FS",
#      "TWminustoLNu2Q",
#      "TbarBQ_t-channel_4FS",
      ],  
    #"JetMET": [
    #  "JetMET_Run2022E",
    #  "JetMET_Run2022F",
    #  "JetMET_Run2022G",
    #  ], 
    #"JetMET_Muon": [
    #  "JetMET_Muon_Run2022E",
    #  "JetMET_Muon_Run2022F",
    #  "JetMET_Muon_Run2022G",
    #  ], 
    #"JetMET_None": [
    #  "JetMET_Run2022E_NewTriggers",
    #  "JetMET_Muon_Run2022F",
    #  "JetMET_Muon_Run2022G",
    #  ], 
    "Muon": [
        "Muon_Run2022E",
        "Muon_Run2022F",
        "Muon_Run2022G",
    #    "Muon_Run2022_VVLooseDeepTauVsE",
    ],
    #"JetMET_Run2022E": [
    #  "JetMET_Run2022E",
    #],
    #"JetMET_Run2022F": [
    #  "JetMET_Run2022F",
    #],
    #"JetMET_Run2022G": [
    #  "JetMET_Run2022G",
    #],
    "Stau" : [],
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
with open(f"./plots_config/{nanov}/all_total_sumw.json", "r") as file:
    sum_gen_w = json.load(file)
    
## target lumi, for now fixed value
target_lumi = 26.7

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
    tmp_string = f"faster_trial_{process}/faster_trial_{process}.root"
    tmp_file = sample_folder +  tmp_string
    events = NanoEventsFactory.from_root({tmp_file:"Events"}, schemaclass= PFNanoAODSchema).events()
    events = events[ak.ravel(abs(events.Tau.eta) > 1.2)]
    
    mutau_dr = events.Muon.metric_table(events.Tau)
    mutau_pt = events.Muon.pt + events.Tau.pt

    met = events.PuppiMET.pt
    met_phi = events.PuppiMET.phi
    if "Muon" not in process:
        met = events.CorrectedPuppiMET.pt
        met_phi = events.CorrectedPuppiMET.phi
    dphi = abs(events.Tau.phi - met_phi)
    dphi = np.where(dphi > np.pi, 2*np.pi - dphi, dphi)
    mT = np.sqrt(2 * events.Muon.pt * met * (1 - np.cos(dphi)))
    events = ak.with_field(events, mT, "Puppi_mT")

    events["mutau"] = ak.with_field(events.mutau, mutau_dr, where = 'dR')
    events["mutau"] = ak.with_field(events.mutau, mutau_pt, where = 'pt') 

    ## disable for now
#     print ('need to put back weights')
    weights = events.run / events.run
#    if "jetmet" in process.lower():
#        weights = weights
    if "muon" in process.lower():
        weights = weights
    if "muon" not in process.lower():
        events["PuppiMET"] = events.CorrectedPuppiMET
    else:
        lumi_weight = target_lumi * xsec[process] * br[process] * 1000 / sum_gen_w[process]
#         lumi_weight = target_lumi * xsec[process] * br[process] * 1000 / sum_gen_w[reverse_samples_lookup[process]][process]
        weights = events.weight * lumi_weight ## am i missing the sumGenW here?

    for plot_name, settings in plot_settings.items():
        ## test, not clear why
#         print("vals type:", type(var_values), "vals layout:", var_values.layout.form)
        if not settings["per_event"]:
            var_values = getattr(getattr(events, settings["field"]), settings["variable"])
            vals_flat = ak.flatten(var_values)
            weights_broadcast = ak.broadcast_arrays(var_values, weights)[1]
            weights_flat = ak.flatten(weights_broadcast)
            hist, _ = np.histogram(vals_flat, weights=weights_flat, bins=np.linspace(*settings["binning_linspace"]))

        else:
            if settings["variable"] == "":
                hist, _ = np.histogram(getattr(events, settings["field"]), weights=ak.broadcast_arrays(getattr(events, settings["field"]), weights)[1], bins=np.linspace(*settings["binning_linspace"]))
            else:
                #if settings["variable"] == "mass" and settings["field"] == "mutau":
                    #mutau = events.Muon + events.Tau
                    #events = events[ak.ravel(mutau.charge == 0)]
                    #weights = weights[ak.ravel(mutau.charge == 0)]
                hist, _ = np.histogram(getattr(getattr(events, settings["field"]), settings["variable"]), weights=ak.broadcast_arrays(getattr(getattr(events, settings["field"]), settings["variable"]), weights)[1], bins=np.linspace(*settings["binning_linspace"]))

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
        hist_Sig       = np.zeros(len(binning)-1)
        hist_Wto2Q     = np.zeros(len(binning)-1)
        hist_WtoLNu    = np.zeros(len(binning)-1)
        hist_EWK       = np.zeros(len(binning)-1)
        hist_TT        = np.zeros(len(binning)-1)
        hist_singleT   = np.zeros(len(binning)-1)
        hist_Top   = np.zeros(len(binning)-1)
        hist_WJets   = np.zeros(len(binning)-1)
        hist_QCD       = np.zeros(len(binning)-1)
        hist_Data      = np.zeros(len(binning)-1)
        #hist_Data_Muon      = np.zeros(len(binning)-1)
        #hist_Data_None      = np.zeros(len(binning)-1)
        #hist_Data_E      = np.zeros(len(binning)-1)
        #hist_Data_F      = np.zeros(len(binning)-1)
        #hist_Data_G      = np.zeros(len(binning)-1)
    
    hists_to_plot = []
    data_hists = []
    data_muon_hists = []
    data_none_hists = []
    labels = []
    data_labels = []

    for process, histogram in histograms.items():
#         print('INFO: Now looking at process', process, '...')
        histogram = np.asarray(histogram)
        histogram = histogram.flatten()
        # Fix later, hist should not be a list in the first place and should be 60 1d and not (60,1)
        # if args.groupProcesses:
        if groupProcesses:
            #if process in all_samples_dict["JetMET_Run2022E"]:
            #    hist_Data_E += histogram
            #elif process in all_samples_dict["JetMET_Run2022F"]:
            #    hist_Data_F += histogram
            #eliif process in all_samples_dict["JetMET_Run2022G"]:
            #    hist_Data_G += histogram
            if process in all_samples_dict['Muon']:
                hist_Data += histogram
            #if process in all_samples_dict['JetMET_Muon']:
            #    hist_Data_Muon += histogram
            elif process in all_samples_dict['TT']:
                hist_TT += histogram
                hist_Top += histogram
            elif process in all_samples_dict['singleT']:
                hist_singleT += histogram
                hist_Top += histogram
            elif process in all_samples_dict['DY']:
                hist_EWK += histogram
            elif process in all_samples_dict['WtoLNu']:
                hist_WtoLNu += histogram
                hist_WJets += histogram
            elif process in all_samples_dict['Wto2Q']:
                hist_Wto2Q += histogram
                hist_WJets += histogram
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
     #if args.groupProcesses:
        hists_to_plot.append(hist_QCD)
        labels.append('QCD')
        hists_to_plot.append(hist_Wto2Q)
        labels.append('Wto2Q')
        hists_to_plot.append(hist_WtoLNu)
        labels.append('WtoLNu')
        hists_to_plot.append(hist_singleT)
        labels.append('singleT')
        hists_to_plot.append(hist_TT)
        labels.append('TT')
        hists_to_plot.append(hist_EWK)
        labels.append('DY')
        #hists_to_plot.append(hist_Top)
        #labels.append('tt + singlet')
        #hists_to_plot.append(hist_WJets)
        #labels.append('W to jets')
        #hists_to_plot.append(hist_Sig)
        #labels.append('signal')
        data_hists.append(hist_Data)
        #data_muon_hists.append(hist_Data_Muon)
        #data_hists.append(hist_Data_E)
        #data_labels.append('E')
        #data_hists.append(hist_Data_F)
        #data_labels.append('F')
        #data_hists.append(hist_Data_G)
        #data_labels.append('G')

    colours = hep.style.cms.cmap_petroff
    
    fig, (ax_main, ax_ratio) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    fig.subplots_adjust(hspace=0.0)
    hep.histplot(hists_to_plot, bins=binning, stack=do_stack, histtype='fill', 
                 label=labels, #sort='label_r', color=colours,
                 density=plot_settings[plot_name].get("density"), ax=ax_main)
    hep.histplot(data_hists, xerr=True, bins=binning, stack=False, histtype='errorbar', 
                  color='black', label='data', density=plot_settings[plot_name].get("density"), ax=ax_main)
    #hep.histplot(data_muon_hists, xerr=True, bins=binning, stack=False, histtype='errorbar', 
    #              color='red', label='JetMET Muon Triggers Only', density=plot_settings[plot_name].get("density"), ax=ax_main)
    #hep.histplot(data_hists, bins=binning, stack=do_stack, histtype='fill', 
    #              label=data_labels, density=plot_settings[plot_name].get("density"), ax=ax_main)
    ax_main.set_ylabel(plot_settings[plot_name].get("ylabel"))
    ax_main.legend()
    
    ## this part is still to be done 
    if groupProcesses:
    # # # if args.groupProcesses:
        sum_histogram = np.sum(np.asarray(hists_to_plot), axis=0)
        sum_data_histogram = np.sum(np.asarray(data_hists), axis=0)
        ratio_hist = sum_data_histogram / (sum_histogram + np.finfo(float).eps)
    # #     # Adding relative sqrtN Poisson uncertainty for now, should be improved when using the hist package
        rel_unc = np.sqrt(sum_data_histogram) / sum_data_histogram
        rel_unc *= ratio_hist
        rel_unc[rel_unc < 0] = 0 # Not exactly sure why we have negative values, but this solves it for the moment
        hep.histplot(ratio_hist, bins=binning, histtype='errorbar', yerr=rel_unc, color='black', label='Ratio', ax=ax_ratio)
        ax_ratio.axhline(1, color='gray', linestyle='--')
    ax_ratio.set_xlabel(plot_settings[plot_name].get("xlabel"), usetex=False)
    ax_main.xaxis.set_major_locator(MultipleLocator(plot_settings[plot_name].get("x_major_ticks")))
    ax_main.xaxis.set_minor_locator(MultipleLocator(plot_settings[plot_name].get("x_minor_ticks")))
    ax_ratio.xaxis.set_major_locator(MultipleLocator(plot_settings[plot_name].get("x_major_ticks")))
    ax_ratio.xaxis.set_minor_locator(MultipleLocator(plot_settings[plot_name].get("x_minor_ticks")))
    ax_ratio.set_ylabel('Data / MC')
    ax_ratio.set_xlim(binning[0], binning[-1])
    ax_ratio.set_ylim(0.6, 1.4)
    
    # Decorating with CMS label
    hep.cms.label(data=True, loc=0, label="Private Work", com=13.6, lumi=round(target_lumi, 1), ax=ax_main)
    #hep.cms.add_text(r'$|\eta|$ > 1.2', loc = 'upper left')
    
    # Saving with special name
    #filename = f"/eos/uscms/store/user/dally/DisplacedTauAnalysis/plots/{dataset_name}_{plot_name}"
    filedir = "Zpeak_Run2022EE_v4"
    if filedir not in os.listdir('plots/'):
        os.mkdir(f'plots/{filedir}')
    filename = f"./plots/{filedir}/{dataset_name}_{plot_name}"
    # #if args.groupProcesses:
    if plot_settings[plot_name].get("density"):
        filename += "_normalized"
    else: 
        filename += "_stacked"
    #filename += "_etaleq1p2.pdf"
    filename += "_etag1p2.pdf"
    #filename += ".pdf"
    print(filename)
    plt.savefig(filename)
    plt.savefig(filename.replace('.pdf', '.png'))
    plt.savefig(filename.replace('.pdf', '.eps'))
    ax_main.set_yscale('log')
    plt.savefig(filename.replace('.pdf', '_log.pdf'))
    plt.savefig(filename.replace('.pdf', '_log.png'))
    plt.savefig(filename.replace('.pdf', '_log.eps'))
    plt.clf()
