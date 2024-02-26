import uproot
import scipy
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import awkward as ak
import math
import ROOT
import array
import pandas as pd
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from leptonPlot import *

NanoAODSchema.warn_missing_crossrefs = False

file = "Staus_M_100_100mm_13p6TeV_Run3Summer22_lpcdisptau_NanoAOD_ExtraDisMuonBranches.root"

#events = NanoEventsFactory.from_root({file:"Events"}, schemaclass=NanoAODSchema).events()[0:21]
events = NanoEventsFactory.from_root({file:"Events"}).events()

gpart = events.GenPart
electrons = events.Electron
muons = events.Muon
staus = gpart[(abs(gpart.pdgId) == 1000015) & (gpart.hasFlags("isLastCopy"))]

#taus = gpart[(abs(gpart.pdgId) == 15) & (abs(gpart.parent.pdgId) == 1000015) & (gpart.hasFlags("isLastCopy"))]
staus_taus = staus.distinctChildren[(abs(staus.distinctChildren.pdgId) == 15) & (staus.distinctChildren.hasFlags("isLastCopy"))]
staus_taus = ak.flatten(staus_taus, axis=2)

gen_mus = staus_taus.distinctChildren[(abs(staus_taus.distinctChildren.pdgId) == 13)]
gen_electrons = staus_taus.distinctChildren[(abs(staus_taus.distinctChildren.pdgId) == 11)]
RecoElectronsFromGen = electrons[(abs(electrons.matched_gen.distinctParent.distinctParent.pdgId) == 1000015)]

gen_electrons = gen_electrons[(gen_electrons.pt > 20) & (abs(gen_electrons.eta) < 2.4)]
RecoElectronsFromGen = RecoElectronsFromGen.matched_gen[(RecoElectronsFromGen.matched_gen.pt > 20) & (abs(RecoElectronsFromGen.matched_gen.eta) < 2.4)]


makeEffPlot("e", "coffea_100GeV_100mm", [""], "pt", 16, 20, 100, 5, "[GeV]", [gen_electrons.pt.compute()], [RecoElectronsFromGen.pt.compute()], 0, file) 

#for ev in range(len(events)):
  #print("The pdgId of the mus in event", ev, "is", gen_mus.pt.compute()[ev])
#  print("The pt of the electrons in event", ev, "is", gen_electrons.pt.compute()[ev])
  #print("The gen idx of the matched muon for event", ev, "is", electrons.matched_gen.compute()[ev]) 
  #print("The dr between reco ele and gen ele for event", ev, "is", dr.compute()[ev])
  #print("The reco electrons that trace back to gen electrons whcih decay from staus for event", ev, "are", RecoElectronsFromGen.genPartIdx.compute()[ev]) 
