import numpy as  np
import matplotlib as mpl
from matplotlib import pyplot as plt
import awkward as ak
import math
import array
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, PFNanoAODSchema
from leptonPlot import *

PFNanoAODSchema.warn_missing_crossrefs = False
PFNanoAODSchema.mixins["DisMuon"] = "Muon"

file = "/eos/uscms/store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-100_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_0_0.root"

events = NanoEventsFactory.from_root({file:"Events"}, schemaclass=PFNanoAODSchema).events()
#events = NanoEventsFactory.from_root({"skimmed_muon_Stau_100_100mm_root/*":"Events"}, schemaclass= PFNanoAODSchema).events()


events = events[ak.num(events.DisMuon.pt) > 0]


events["Jet"] = events.Jet[ak.argsort(events["Jet"]["pt"], ascending=False, axis = 1)]
events["Stau"] = events.GenPart[(abs(events.GenPart.pdgId) == 1000015) & (events.GenPart.hasFlags("isLastCopy"))]

events["StauTau"] = events.Stau.distinctChildren[(abs(events.Stau.distinctChildren.pdgId) == 15) &\
                                                 (events.Stau.distinctChildren.hasFlags("isLastCopy"))]
print(events.StauTau.pt.compute())


events = events[ak.num(events.StauTau.pt) > 0]
events["StauTau"] = ak.flatten(events.StauTau, axis = 1)

print(events.Jet.pt.compute())

print(events.StauTau.pt.compute())
#print(events.StauTau.metric_table(events.Jet).compute())



#gpart = events.GenPart
#electrons = events.Electron
#muons = events.Muon
#staus = gpart[(abs(gpart.pdgId) == 1000015) & (gpart.hasFlags("isLastCopy"))]
#
##taus = gpart[(abs(gpart.pdgId) == 15) & (abs(gpart.parent.pdgId) == 1000015) & (gpart.hasFlags("isLastCopy"))]
#staus_taus = staus.distinctChildren[(abs(staus.distinctChildren.pdgId) == 15) & (staus.distinctChildren.hasFlags("isLastCopy"))]
#staus_taus = ak.flatten(staus_taus, axis=2)
#
#gen_mus = staus_taus.distinctChildren[(abs(staus_taus.distinctChildren.pdgId) == 13)]
#gen_electrons = staus_taus.distinctChildren[(abs(staus_taus.distinctChildren.pdgId) == 11)]
#RecoElectronsFromGen = electrons[(abs(electrons.matched_gen.distinctParent.distinctParent.pdgId) == 1000015)]
#
#gen_electrons = gen_electrons[(gen_electrons.pt > 20) & (abs(gen_electrons.eta) < 2.4)]
#RecoElectronsFromGen = RecoElectronsFromGen.matched_gen[(RecoElectronsFromGen.matched_gen.pt > 20) & (abs(RecoElectronsFromGen.matched_gen.eta) < 2.4)]
#
#
#makeEffPlot("e", "coffea_100GeV_100mm", [""], "pt", 16, 20, 100, 5, "[GeV]", [gen_electrons.pt.compute()], [RecoElectronsFromGen.pt.compute()], 0, file) 

#for ev in range(len(events)):
  #print("The pdgId of the mus in event", ev, "is", gen_mus.pt.compute()[ev])
#  print("The pt of the electrons in event", ev, "is", gen_electrons.pt.compute()[ev])
  #print("The gen idx of the matched muon for event", ev, "is", electrons.matched_gen.compute()[ev]) 
  #print("The dr between reco ele and gen ele for event", ev, "is", dr.compute()[ev])
  #print("The reco electrons that trace back to gen electrons whcih decay from staus for event", ev, "are", RecoElectronsFromGen.genPartIdx.compute()[ev]) 
