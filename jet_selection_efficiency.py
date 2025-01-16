import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema
from hist import Hist, axis, intervals

# Load the file
filename = "/eos/user/d/dally/DisplacedTauAnalysis/SMS-TStauStau_MStau-100_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_NanoAOD.root"
events = NanoEventsFactory.from_root({filename: "Events"}, 
                                         schemaclass=PFNanoAODSchema,
                                         metadata={"dataset": "MC"},
                                         ).events()

## find staus and their tau children  
gpart = events.GenPart
events['staus'] = gpart[(abs(gpart.pdgId) == 1000015) & (gpart.hasFlags("isLastCopy"))] # most likely last copy of stay in the chain

events['staus_taus'] = events.staus.distinctChildren[ (abs(events.staus.distinctChildren.pdgId) == 15) & \
                                                  (events.staus.distinctChildren.hasFlags("isLastCopy")) & \
                                                  (events.staus.distinctChildren.pt > 5) \
                                                 ]
staus_taus = events['staus_taus']

mask_taul = ak.any((abs(staus_taus.distinctChildren.pdgId) == 11) | (abs(staus_taus.distinctChildren.pdgId) == 13), axis=-1)
mask_tauh = ~mask_taul

one_tauh_evt = (ak.sum(mask_tauh, axis=1) > 0) & (ak.sum(mask_tauh, axis=1) < 3)
one_taul_evt = (ak.sum(mask_taul, axis=1) > 0) & (ak.sum(mask_taul, axis=1) < 3)

flat_one_tauh_evt = ak.flatten(one_tauh_evt, axis=None)
flat_one_taul_evt = ak.flatten(one_taul_evt, axis=None)

filtered_events = events[flat_one_tauh_evt & flat_one_taul_evt]  # Filtered events are events with one hadronic tau and one leptonic tau

tau_selections = ak.any((filtered_events.staus_taus.pt > 20) & (abs(filtered_events.staus.eta < 2.4)), axis=-1)
num_taus = ak.num(filtered_events.staus_taus[tau_selections]).compute()
num_tau_mask = num_taus > 1
cut_filerted_events = filtered_events[num_tau_mask]

jets = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.4) & (cut_filtered_events.Jet.pt > 20)]
sorted_jets = jets[ak.argsort(jets.pt, ascending=False)]
leading_jets = ak.singletons(ak.firsts(sorted_jets))

gen_taus = ak.flatten(cut_filtered_events.staus_taus, axis = 2)
gen_taus_jet_matched = leading_jets.nearest(gen_taus, threshold = 0.4)

new_mask_taul = ak.any((abs(gen_taus_jet_matched.distinctChildren.pdgId) == 11) | (abs(gen_taus_jet_matched.distinctChildren.pdgId) == 13), axis=-1)
new_mask_tauh = ~new_mask_taul

gen_taus_jet_matched = gen_taus_jet_matched[new_mask_tauh]
gen_taus_jet_matched = ak.drop_none(gen_taus_jet_matched)
ak.flatten(gen_taus_jet_matched, axis = 1).compute()