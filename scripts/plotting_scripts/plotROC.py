import coffea
import sys
import os
import pickle
import uproot
import scipy
import dask
import hist 
import hist.dask as hda
import time
import json, gzip, correctionlib
import dask_awkward as dak
import warnings
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from coffea import processor
from collections import defaultdict
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.dataset_tools import (
    apply_to_fileset,
    max_chunks,
    preprocess,
)
from dask import config as cfg
cfg.set({'distributed.scheduler.worker-ttl': None}) # Check if this solves some dask issues
import dask_awkward as dak
from dask_jobqueue import HTCondorCluster
from dask.distributed import Client, wait, progress, LocalCluster
from xsec import *

import time
from distributed import Client
from lpcjobqueue import LPCCondorCluster

import warnings
warnings.filterwarnings("ignore", module="coffea") # Suppress annoying deprecation warnings for coffea vector, c.f. https://github.com/CoffeaTeam/coffea/blob/master/src/coffea/nanoevents/methods/candidate.py
import logging

# Prevent branch definition problems
NanoAODSchema.mixins["DisMuon"] = "Muon"
NanoAODSchema.mixins["StauTau"] = "Tau"

# Silence obnoxious warnings
NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore", module="coffea.nanoevents.methods")

# --- DEFINITIONS --- #
lifetimes = ['1mm', '10mm', '100mm', '1000mm']
masses    = ['100', '200', '300', '500']
score_granularity = 100

stau_colors = {'100': '#EA7AF4', '200': '#B43E8F', '300': '#6200B3', '500': '#218380'}
stau_lifetime_colors = {'1mm': '#EA7AF4', '10mm': '#B43E8F', '100mm': '#6200B3', '1000mm': '#218380'}
jetId_colors = {'tight': '#EA7AF4', 'tightLV': '#B43E8F', 'tight_chHEF': '#6200B3', 'tightLV_chHEF': '#218380'}
bkgds = ["QCD", "TT", "W"]
cuts = ["tight", "tightLV", "tight_chHEF", "tightLV_chHEF"]

out = {}
files = os.listdir("/eos/uscms/store/user/dally/DisplacedTauAnalysis/ROC_hists/")
for file in files:
    if "300_100mm" not in file: continue
    samp = file.split('.')[0]
    print(samp)
    with open("/eos/uscms/store/user/dally/DisplacedTauAnalysis/ROC_hists/" + file, "rb") as f:
        out[samp] = pickle.load(f)[samp]

stau_dict = {}

fig, ax = plt.subplots(2, 1, figsize=(8,10))
out[samp]["pt_num"].plot_ratio(
                                out[samp]["pt_den"],
#                                rp_uncertainty_type = "efficiency",
                                ax_dict = {"main_ax": ax[1], "ratio_ax": ax[0]},
)
#ax[1].remove()
ax[0].set_ylim((0, 1.1))
plt.savefig(f"gvt_pt_eff.png")
plt.savefig(f"gvt_pt_eff.pdf")


'''
unmatched_histo =  {}
all_histo = {}
for bkd in bkgds:
    unmatched_histo[bkd] = {}
    all_histo[bkd] = hda.hist.Hist(hist.axis.Regular(score_granularity, 0, 1, name = "all_jet_score", label = 'score', overflow = True)).compute()
    unmatched_histo[bkd]['tight'] = hda.hist.Hist(hist.axis.Regular(score_granularity, 0, 1, name = 'unmatched_jet_score', label = 'score', overflow = True)).compute()
    unmatched_histo[bkd]['tightLV'] = hda.hist.Hist(hist.axis.Regular(score_granularity, 0, 1, name = 'unmatched_jet_score', label = 'score', overflow = True)).compute()
    unmatched_histo[bkd]['tight_chHEF'] = hda.hist.Hist(hist.axis.Regular(score_granularity, 0, 1, name = 'unmatched_jet_score', label = 'score', overflow = True)).compute()
    unmatched_histo[bkd]['tightLV_chHEF'] = hda.hist.Hist(hist.axis.Regular(score_granularity, 0, 1, name = 'unmatched_jet_score', label = 'score', overflow = True)).compute()

for samp in out.keys():
    if "Stau" in samp:
        lifetime = samp.split('_')[-1]
        mass     = samp.split('_')[-2]
        if lifetime in stau_dict.keys():
            stau_dict[lifetime][mass] = {}
            for cut in cuts:
                stau_dict[lifetime][mass][cut] = {}
        else:
            stau_dict[lifetime] = {}
            stau_dict[lifetime][mass] = {}
            for cut in cuts:
                stau_dict[lifetime][mass][cut] = {}
        for cut in cuts:
            out[samp][cut]['matched_jet_histo'].values(flow = True)[-2] = out[samp][cut]['matched_jet_histo'].values(flow = True)[-2] + out[samp][cut]['matched_jet_histo'].values(flow = True)[-1]

            matched_binned_scores = out[samp][cut]['matched_jet_histo'].values()

            matched_score_series = np.cumsum(matched_binned_scores[::-1])[::-1]

            #for i in range(score_granularity):
            #    matched_score_series.append(np.sum(matched_binned_scores[i:]))

            matched_score_series = ak.Array(matched_score_series)
            passed_matched_jets =  {}
            print(f"samp: {samp}")
            print(f"matched_binned_scores: {matched_binned_scores}")
            print(f"matched_score_series: {matched_score_series}")
            stau_dict[lifetime][mass][cut]['matched'] = matched_score_series/out[samp]['nGenVisStauTau']
        #passed_matched_jets =  {}
        #for score_list in out[samp]['set_s_pmj_pfj']:
        #    passed_matched_jets[score_list[1]] = score_list[2]
        #    stau_dict[lifetime][mass][score_list[1]] = passed_matched_jets[score_list[1]]/out[samp]["total_matched_jets"]  
    if "Stau" not in samp:
        if "QCD" in samp:
            all_histo["QCD"] += out[samp]['all_jet_histo']
            for cut in cuts: 
                unmatched_histo["QCD"][cut] += out[samp][cut]['unmatched_jet_histo']
        if "TT" in samp:
            all_histo["TT"] += out[samp]['all_jet_histo']
            for cut in cuts: 
                unmatched_histo["TT"][cut] += out[samp][cut]['unmatched_jet_histo']
        if "W" in samp:
            all_histo["W"] += out[samp]['all_jet_histo']
            for cut in cuts: 
                unmatched_histo["W"][cut] += out[samp][cut]['unmatched_jet_histo']
#
#    #if 'set_s_pmj_pfj' not in out[samp]: continue
#    #for score_list in out[samp]['set_s_pmj_pfj']:
#    #    s_sums[score_list[1]][0] += score_list[3]
#    
#  
  
tprocessor = time.time() - tstart
print(f"{tprocessor/60} minutes for processor to finish")
    #
    ## --- ROC Calculations --- #
    ## Totals
    #all_matched = sum(
    #    val["total_matched_jets"]
    #    for val in out.values()
    #    if "total_matched_jets" in val
    #)
    #print(f"all_matched is {all_matched}")
    #all_jets = sum(
    #    val["total_number_jets"]
    #    for val in out.values()
    #    if "total_number_jets" in val
    #)
    #print(f"all_jets is {all_jets}")
    #print(f"{all_jets} total jets, with {all_matched} matched")
    #
    
thresholds   = []
fake_rate = {}
for bkd in bkgds:
    fake_rate[bkd] = {}

    all_histo[bkd].values(flow = True)[-2] = all_histo[bkd].values(flow = True)[-2] + all_histo[bkd].values(flow = True)[-1]        
    all_binned_scores = all_histo[bkd].values()
    all_score_series = np.cumsum(all_binned_scores[::-1])[::-1]

    for cut in unmatched_histo[bkd].keys():
        unmatched_histo[bkd][cut].values(flow = True)[-2] = unmatched_histo[bkd][cut].values(flow = True)[-2] + unmatched_histo[bkd][cut].values(flow = True)[-1]
        
        unmatched_binned_scores = unmatched_histo[bkd][cut].values()
        unmatched_score_series = np.cumsum(unmatched_binned_scores[::-1])[::-1]
        
        #for i in range(score_granularity): 
        #    unmatched_score_series.append(np.sum(unmatched_binned_scores[i:]))
        unmatched_score_series = ak.Array(unmatched_score_series)
        fake_rate[bkd][cut] = unmatched_score_series / all_score_series[0]
        score = np.arange(0, 1, 1/score_granularity)
        fake_rate_standard = (fake_rate[bkd][cut] >= 5E-3)
        if ak.any(fake_rate_standard):
            print(f"For {cut} in {bkd}, the score that corresponds to 5E-3 fake rate is {score[fake_rate_standard][-1]}")
##for lifetime in lifetimes:
##    for mass in masses:
##        stau_dict[lifetime][mass][cut]['eff'] = []
#
##for s, vals in s_sums.items():
##    thresholds.append(s)
##    fake_rate = vals[0] / all_jets
##    fake_rates.append(fake_rate)
##    for lifetime in lifetimes:
##        for mass in masses:
##            stau_dict[lifetime][mass]['eff'].append(stau_dict[lifetime][mass][s])
#
#
##print("sets")
##print(sets)
##print("Score thresholds:")
##print(thresholds)
##print("Fake rates:")
##print(fake_rates)
##print("Efficiencies:")
##print(efficiencies)
##print(f"{tcalc} seconds for calculations to finish")
#
# Plot stuff
#for lifetime in lifetimes:
#    fig, ax = plt.subplots()
#    roc = {}
#    for mass in masses:
#        roc[mass] = ax.plot(fake_rate, stau_dict[lifetime][mass]['matched'], color = stau_colors[mass], label = mass + ' GeV')
#
#    plt.xlabel(r"Fake rate $\left(\frac{fake\_passing\_jets}{total\_jets}\right)$")
#    plt.ylabel(r"Tau tagger efficiency $\left(\frac{matched\_passing\_jets}{total\_matched\_jets}\right)$")
#    plt.title(f"{lifetime}")
#    plt.xscale('log')
#    plt.legend(loc='lower right')
#    
#    #plt.grid()
#    plt.savefig(f'limited-log-skimmed-bg-tau-tagger-roc-scatter-{lifetime}_test.pdf')
    
#cbar = fig.colorbar(roc, ax=ax, label='Score threshold')

#ax.set_xscale("log")
#ax.set_xlim(-1e-1, 6e-3)
#ax.set_ylim(0.85, 1.05)

for bkd in bkgds:
    roc = {}
    for mass in masses:
        roc[mass] = {}
        for lifetime in lifetimes:
            print(lifetime)
            roc[mass][lifetime] = {}
            fig, ax = plt.subplots()
            for cut in unmatched_histo[bkd].keys():
                roc[mass][lifetime][cut]= ax.plot(fake_rate[bkd][cut], stau_dict[lifetime][mass][cut]['matched'], color = jetId_colors[cut], label = cut)
            plt.xlabel(r"Fake rate $\left(\frac{fake\_passing\_jets}{total\_jets}\right)$")
            plt.ylabel(r"Tau tagger efficiency $\left(\frac{matched\_passing\_jets}{all\_GenVisTaus}\right)$")
            plt.title(f"Stau {mass}GeV {lifetime} + {bkd}")
            plt.xscale('log')
            plt.legend(loc='lower right')
            
            #plt.grid()
            plt.savefig(f'roc-gvt-scatter-{mass}-{lifetime}-jetId_log_{bkd}.pdf')
            plt.savefig(f'roc-gvt-scatter-{mass}-{lifetime}-jetId_log_{bkd}.png')
    
    roc = {}
    for mass in masses:
        roc[mass] = {}
        for lifetime in lifetimes:
            print(lifetime)
            roc[mass][lifetime] = {}
            fig, ax = plt.subplots()
            for cut in unmatched_histo[bkd].keys():
                roc[mass][lifetime][cut]= ax.plot(fake_rate[bkd][cut], stau_dict[lifetime][mass][cut]['matched'], color = jetId_colors[cut], label = cut)
            plt.xlabel(r"Fake rate $\left(\frac{fake\_passing\_jets}{total\_jets}\right)$")
            plt.ylabel(r"Tau tagger efficiency $\left(\frac{matched\_passing\_jets}{total\_matched\_jets}\right)$")
            plt.title(f"Stau {mass}GeV {lifetime} + {bkd}")
            plt.xscale('log')
            plt.ylim([0, 1])
            plt.legend(loc='lower right')
            
            #plt.grid()
            plt.savefig(f'roc-gvt-scatter-{mass}-{lifetime}-jetId_log_samescale_{bkd}.pdf')
            plt.savefig(f'roc-gvt-scatter-{mass}-{lifetime}-jetId_log_samescale_{bkd}.png')
'''



