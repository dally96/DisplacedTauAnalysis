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
bkgd_colors = {"QCD": '#56CBF9', "TT":  '#FDCA40', "W": '#D3C0CD'}
bkgds = ["QCD", "TT", "W", "JetMET"]
cuts = ["tight", "tightLV", "tight_chHEF", "tightLV_chHEF"]
colors = ['#56CBF9', '#FDCA40', '#5DFDCB', '#D3C0CD', '#3A5683', '#FF773D']

def get_stack_maximum(stack):
    max_value = 0 

    for hist in stack:
        max_value = max(max_value, hist.view().max())

    return max_value 

out = {}
files = os.listdir("/eos/uscms/store/user/dally/DisplacedTauAnalysis/prelimSelections/")
for file in files:
    if "Good" not in file: continue
    samp = '_'.join(file.split('_')[:-1])
    with open("/eos/uscms/store/user/dally/DisplacedTauAnalysis/prelimSelections/" + file, "rb") as f:
        out[samp] = pickle.load(f)[samp]

hist_vars = {'npvs': [80, 0, 80, ' '],
             'npvsGood': [80, 0, 80, ' '],
            }

hist_dict = {}
for samp  in bkgds: 
    hist_dict[samp] = {}
    for var in hist_vars.keys():
        hist_dict[samp][var] = hda.hist.Hist(hist.axis.Regular(hist_vars[var][0], hist_vars[var][1], hist_vars[var][2], name = var, label =var + ' ' + hist_vars[var][3])).compute()

for samp in out.keys():
    for var in hist_vars.keys():
        print(f"{samp} hist has bins {out[samp][var].axes[var].edges}")
        if "QCD" in samp:
            hist_dict["QCD"][var] += out[samp][var]
        if "TT" in samp:
            hist_dict["TT"][var] += out[samp][var]
        if "W" in samp:
            hist_dict["W"][var] += out[samp][var]
        if "JetMET" in samp:
            hist_dict["JetMET"][var] += out[samp][var]
'''
for var in hist_vars.keys():    
    s = hist.Stack.from_dict( {"QCD": hist_dict["QCD"][var],
                               r"TT":  hist_dict["TT"][var],
                               r"W":   hist_dict["W"][var],
                               } )
    s.plot( stack = True, histtype = "fill", color = [colors[0], colors[1], colors[2]] )
    hist_dict["JetMET"][var].plot( color = 'black', label = 'data' )
    box = plt.subplot().get_position()
    plt.subplot().set_position([box.x0, box.y0, box.width * 0.8, box.height])   

    plt.xlabel(var + ' ' + hist_vars[var][3])
    plt.ylabel("A.U.")
    #plt.yscale('log')
    plt.title(r"$\mathcal{L}_{int}$ = 26.3 fb$^{-1}$")
    #plt.ylim( (100, max(get_stack_maximum(s), hist_dict["JetMET"][var].view().max() ) * 2) )
    plt.ylim( (1E2, 5E9) ) 
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = {"size": 8})
    if "nPV" not in os.listdir("/eos/uscms/store/user/dally/www/"):
        os.mkdir(f"/eos/uscms/store/user/dally/www/nPV")
    plt.savefig(f"/eos/uscms/store/user/dally/www/nPV/data_histogram_{var}.png")
    plt.savefig(f"/eos/uscms/store/user/dally/www/nPV/data_histogram_{var}.pdf")

    plt.cla()
    plt.clf()
'''

for var in hist_vars.keys():    
    if "Good" in var: continue
    print(f"data bins are {hist_dict['JetMET'][var].axes[var].edges}")
    data_integral = np.sum(hist_dict["JetMET"][var].values())
    data_hist = hist_dict["JetMET"][var]/data_integral
    for  bkgd in bkgds:
        if "JetMET" in bkgd: continue
        integral = np.sum(hist_dict[bkgd][var].values())
        #print(integral)
        mc_hist = hist_dict[bkgd][var]/integral
        #mc_hist.plot(histtype = "fill", color = bkgd_colors[bkgd], label = bkgd)
        #data_hist.plot( color = 'black', label = 'data' )
        #box = plt.subplot().get_position()
        #plt.subplot().set_position([box.x0, box.y0, box.width * 0.8, box.height])   

        #plt.xlabel(var + ' ' + hist_vars[var][3])
        #plt.ylabel("A.U.")
        ##plt.yscale('log')
        #plt.title(f"{bkgd}" + r" $\mathcal{L}_{int}$ = 26.3 fb$^{-1}$")
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = {"size": 8})
        #if "nPV" not in os.listdir("/eos/uscms/store/user/dally/www/"):
        #    os.mkdir(f"/eos/uscms/store/user/dally/www/nPV")
        #plt.savefig(f"/eos/uscms/store/user/dally/www/nPV/data_histogram_{bkgd}_{var}.png")
        #plt.savefig(f"/eos/uscms/store/user/dally/www/nPV/data_histogram_{bkgd}_{var}.pdf")

        #plt.cla()
        #plt.clf()

        print(mc_hist.values())
        print(data_hist.values())
        fig, ax = plt.subplots(2,1)
        mc_hist.plot_ratio(
                            data_hist,
                            rp_num_label = bkgd,
                            rp_denom_label = 'data',
                            #rp_uncert_draw_type="none",
                            #rp_uncertainty_type="poisson",
                            ax_dict = {'main_ax': ax[0], f"ratio_ax": ax[1]},
        )
        ax[1].set_ylim((0,2))
        plt.savefig(f"/eos/uscms/store/user/dally/www/nPV/data_histogram_{bkgd}_{var}_ratio.png")
        plt.savefig(f"/eos/uscms/store/user/dally/www/nPV/data_histogram_{bkgd}_{var}_ratio.pdf")
        
        plt.cla()
        plt.clf()
