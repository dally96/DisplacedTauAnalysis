import uproot
import scipy
import numpy as np
import awkward as ak
import math
import ROOT
import array
import pandas as pd
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

NanoAODSchema.warn_missing_crossrefs = False

file = "Staus_M_100_100mm_13p6TeV_Run3Summer22_lpcdisptau_NanoAOD_ExtraDisMuonBranches.root"

#events = NanoEventsFactory.from_root({file:"Events"}, schemaclass=NanoAODSchema).events()[0:21]
events = NanoEventsFactory.from_root({file:"Events"}).events()[0:11]

gpart = events.GenPart
electrons = events.Electron
muons = events.Muon
staus = gpart[(abs(gpart.pdgId) == 1000015) & (gpart.hasFlags("isLastCopy"))]

#taus = gpart[(abs(gpart.pdgId) == 15) & (abs(gpart.parent.pdgId) == 1000015) & (gpart.hasFlags("isLastCopy"))]
staus_taus = staus.distinctChildren[(abs(staus.distinctChildren.pdgId) == 15) & (staus.distinctChildren.hasFlags("isLastCopy"))]
staus_taus = ak.flatten(staus_taus, axis=2)

gen_mus = staus_taus.distinctChildren[(abs(staus_taus.distinctChildren.pdgId) == 13)]
gen_electrons = staus_taus.distinctChildren[(abs(staus_taus.distinctChildren.pdgId) == 11)]
print(gen_electrons.fields)
#print("Gen electrons before flattening", gen_electrons.pt.compute())
#print("Awkward gen electrons pt is", gen_electrons_pt)
  #print("for event", ev, "gen electron pt is", ak.flatten(gen_electrons.pt[ev]))
  #gen_electrons = ak.flatten(gen_electrons, axis = 2)
#print("Gen electrons after flattening", gen_electrons_pt)

electrons_togen = events.Electron.gen_match.parent.pdgId



for ev in range(len(events)):
  #print("The pdgId of the mus in event", ev, "is", gen_mus.pt.compute()[ev])
#  print("The pt of the electrons in event", ev, "is", gen_electrons.pt.compute()[ev])
  #print("The gen idx of the matched muon for event", ev, "is", electrons.matched_gen.compute()[ev]) 
  #print("The dr between reco ele and gen ele for event", ev, "is", dr.compute()[ev])
  print("The reco electron gen part idx for event", ev, "is", events.Electron.gen_match.parent.pdgId
