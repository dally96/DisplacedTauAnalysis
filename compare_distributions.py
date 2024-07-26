#!/usr/bin/env python
# coding: utf-8

# In[312]:


from pdb import set_trace

import matplotlib
from matplotlib import pyplot as plt

import ROOT
import sys
from ROOT import TH2F, TH1F, TChain, TCanvas, gROOT
ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch(True)
import numpy as np
import pandas#, root_numpy
pandas.options.mode.chained_assignment = None  # default='warn'
import uproot, math
import pdb
import awkward as ak
plt.ion()
from hist import Hist

import vector
vector.register_awkward()


# In[313]:


events = uproot.open("Stau_M_100_100mm_Summer22EE_NanoAOD.root:Events")


# In[339]:


type(events.arrays())


# In[314]:


## code from
## https://github.com/jpivarski-talks/2021-07-06-pyhep-uproot-awkward-tutorial/blob/main/uproot-awkward-tutorial.ipynb
jets_ar = events.arrays(filter_name="Jet_*")
jets_ar = ak.zip({
    "pt":  jets_ar.Jet_pt,
    "eta": jets_ar.Jet_eta,
    "phi": jets_ar.Jet_phi,
    "mass": 0.140,
})

cut = ((jets_ar.pt >= 20) & (abs(jets_ar.eta) < 2.4))
skim_jets = jets_ar[cut]


# In[255]:


pfcands_ar = events.arrays(filter_name="PFCandidate_*")
pfcands_ar = ak.zip({
    "pt":  pfcands_ar.PFCandidate_pt,
    "eta": pfcands_ar.PFCandidate_eta,
    "phi": pfcands_ar.PFCandidate_phi,
    "mass": 0.140,
    "dxy": pfcands_ar.PFCandidate_dxy,
    "dz": pfcands_ar.PFCandidate_dz,
    "lostInnerHits": pfcands_ar.PFCandidate_lostInnerHits,
})


# In[303]:


pfcands_cut = ((pfcands_ar.pt >= 5) & (abs(pfcands_ar.eta) < 2.4))
skim_pfcands = pfcands_ar[pfcands_cut]


# In[304]:


pfcands = ak.with_name(skim_pfcands, "Momentum4D")
jets = ak.with_name(skim_jets, "Momentum4D")


# In[316]:

jet_pf = ak.cartesian({"jets": jets, "pfcands": pfcands}, nested=True)
jet, pf = ak.unzip(jet_pf)
dR = jet.deltaR(pf)

## add min dR wrt any jet
pfcands["min_dR_jet"] = ak.min(dR, axis = -1)
## get index of best matched jet
best_dR = ak.argmin(dR, axis=-1, keepdims=True)
### this can be further improved in the future, to only allow one combination 
### see also code at 
### https://gitlab.cern.ch/cms-analysis/general/PocketCoffea/-/blob/main/pocket_coffea/lib/deltaR_matching.py


# In[315]:


## select only matched pf cands
matched_pfs = pfcands[pfcands.min_dR_jet < 0.4]
#matched_pfs.show(type=True)
print (len(matched_pfs)) ## they are the same length
print (len(pfcands))


# In[307]:


ax = hist.axis.Regular(100, 0, 1, flow=False, name="dR")
cax = hist.axis.StrCategory(["all_pfcands", "matched"], name="c")
full_hist = Hist(ax, cax)

full_hist.fill(ak.flatten(matched_pfs.min_dR_jet, axis=None), c="matched")
full_hist.fill(ak.flatten(pfcands.min_dR_jet, axis=None), c="all_pfcands")

the_s = full_hist.stack("c")
the_s.plot()
plt.legend()
plt.yscale('log')
plt.show()


# In[317]:


## define variables to be plotted
to_plot = [
    ['pt', 90, [10, 100]],
    ]


# In[322]:


'''
for var in to_plot:
    nbins = var[1]  
    xmin = var[2][0]  
    xmax = var[2][1]  
    ax = hist.axis.Regular(nbins, xmin, xmax, flow=False, name="pt")
    cax = hist.axis.StrCategory(["all_pfcands", "matched"], name="c")
    full_hist = Hist(ax, cax)

    full_hist.fill(ak.flatten(getattr(matched_pfs,var[0]), axis=None), c="matched")
    full_hist.fill(ak.flatten(getattr(pfcands,var[0]), axis=None), c="all_pfcands")

    the_s = full_hist.stack("c")
    the_s.plot()
    plt.legend()
    plt.show()
'''

nbins = 90  
xmin = 10
xmax = 100  
var = ['pt']
ax = hist.axis.Regular(nbins, xmin, xmax, flow=False, name="pt")
cax = hist.axis.StrCategory(["all_pfcands", "matched"], name="c")
full_hist = Hist(ax, cax)

full_hist.fill(ak.flatten(getattr(matched_pfs,var[0]), axis=None), c="matched")
full_hist.fill(ak.flatten(getattr(pfcands,var[0]), axis=None), c="all_pfcands")

the_s = full_hist.stack("c")
the_s.plot()
plt.legend()
plt.show()


# In[334]:


def get_normalised_info(full_hist, category):
    ## get normalised bin content
    h = full_hist[:, category].density()  # equivalent of h1_vals = h[:, "one"].values()/h[:, "one"].sum()
    ## compute uncertainties on normalised values
    err = np.sqrt(full_hist[:, "all_pfcands"].values())
    err = err/full_hist[:, "all_pfcands"].sum()

    hist = ax.errorbar(full_hist.axes[0].centers, 
                       h, 
                       xerr=full_hist.axes[0].widths/2, 
                       yerr=err, 
                       linestyle='none', 
                       label=category)
    return hist


# In[335]:


#h1, err_1 = get_normalised_info(full_hist, "all_pfcands")
#h2, err_2 = get_normalised_info(full_hist, "matched")


# In[336]:


fig, ax = plt.subplots()
hist_all = get_normalised_info(full_hist, "all_pfcands")
hist_match = get_normalised_info(full_hist, "matched")

ax.set_xlabel('pt')

plt.legend()

