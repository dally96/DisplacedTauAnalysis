import uproot
import awkward as ak
import numpy as np 

from xsec import * 



rel_lumi = {}

for samp in num_events:
    rel_lumi[samp] = num_events[samp] / (1000 * xsecs[samp])
    print ("The relative luminosity for " + samp + " is", rel_lumi[samp])
