import awkward as ak
import numpy as np
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema
import os

def process_events(events):
    """Process events to select staus and their tau children and filter events."""
    ## find staus and their tau children
    gpart = events.GenPart
    events['staus'] = gpart[(abs(gpart.pdgId) == 1000015) & (gpart.hasFlags("isLastCopy"))] # most likely last copy of stay in the chain

    events['staus_taus'] = events.staus.distinctChildren[ (abs(events.staus.distinctChildren.pdgId) == 15) & \
                                                      (events.staus.distinctChildren.hasFlags("isLastCopy")) \
                                                     ]
    events['staus_taus'] = ak.firsts(events.staus_taus[ak.argsort(events.staus_taus.pt, ascending=False)], axis = 2)
    staus_taus = events['staus_taus']

    mask_taul = ak.any((abs(staus_taus.distinctChildren.pdgId) == 11) | (abs(staus_taus.distinctChildren.pdgId) == 13), axis=-1)
    mask_tauh = ~mask_taul

    one_tauh_evt = (ak.sum(mask_tauh, axis=-1) > 0) & (ak.sum(mask_tauh, axis=-1) < 3)
    one_taul_evt = (ak.sum(mask_taul, axis=-1) > 0) & (ak.sum(mask_taul, axis=-1) < 3)

    filtered_events = events[one_tauh_evt & one_taul_evt]  # Filtered events are events with one hadronic tau and one leptonic tau

    tau_selections = ak.any((filtered_events.staus_taus.pt > 20) & (abs(filtered_events.staus.eta) < 2.4), axis=-1)
    num_taus = ak.num(filtered_events.staus_taus[tau_selections])
    num_tau_mask = num_taus > 1
    cut_filtered_events = filtered_events[num_tau_mask]
    return cut_filtered_events

def select_and_define_leading_jets(cut_filtered_events):
    """
    Select jets from events with |eta| < 2.4 and pt > 20, and then
    define two sets of leading jets:
      - leading_pt_jets: the leading jet in each event based on pt.
      - leading_score_jets: the leading jet in each event based on disTauTag_score1.
    """
    # Select jets with |eta| < 2.4 and pt > 20
    jets = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.4) & (cut_filtered_events.Jet.pt > 20)]
    
    # Sort the selected jets by pt (descending) and take the first jet per event
    sorted_by_pt = jets[ak.argsort(jets.pt, ascending=False)]
    leading_pt_jets = ak.singletons(ak.firsts(sorted_by_pt))
    
    # Sort the selected jets by disTauTag_score1 (descending) and take the first jet per event
    sorted_by_score = jets[ak.argsort(jets.disTauTag_score1, ascending=False)]
    leading_score_jets = ak.singletons(ak.firsts(sorted_by_score))
    
    return (jets, leading_pt_jets, leading_score_jets)


def match_gen_taus(cut_filtered_events, leading_pt_jets, leading_score_jets):
    """
    Match gen taus to jets using two different selections.
    """
    # Create a mask to select the gen taus of interest.
    mask_taul_filtered = ak.any(
        (cut_filtered_events.staus_taus.distinctChildren.pdgId == 11) |
        (cut_filtered_events.staus_taus.distinctChildren.pdgId == 13),
        axis=-1
    )
    mask_tauh_filtered = ~mask_taul_filtered

    # Select gen taus.
    gen_taus = cut_filtered_events.staus_taus[mask_tauh_filtered]

    # Calculate dxy for gen taus.
    gen_taus["dxy"] = (gen_taus.vy - cut_filtered_events.GenVtx.y) * np.cos(gen_taus.phi) - \
                      (gen_taus.vx - cut_filtered_events.GenVtx.x) * np.sin(gen_taus.phi)

    # Matching using the pt-based leading jets.
    gen_taus_matched_by_pt = leading_pt_jets.nearest(gen_taus, threshold=0.4)
    gen_taus_matched_by_pt = ak.drop_none(gen_taus_matched_by_pt)
    jet_matched_gen_taus_pt = gen_taus.nearest(leading_pt_jets, threshold=0.4)
    jet_matched_gen_taus_pt = ak.drop_none(jet_matched_gen_taus_pt)

    # Matching using the leading-score jets.
    gen_taus_matched_by_score = leading_score_jets.nearest(gen_taus, threshold=0.4)
    gen_taus_matched_by_score = ak.drop_none(gen_taus_matched_by_score)
    jet_matched_gen_taus_score = gen_taus.nearest(leading_score_jets, threshold=0.4)
    jet_matched_gen_taus_score = ak.drop_none(jet_matched_gen_taus_score)

    return (gen_taus,
            gen_taus_matched_by_pt, jet_matched_gen_taus_pt,
            gen_taus_matched_by_score, jet_matched_gen_taus_score)


def flatten_gen_tau_vars(gen_taus, gen_taus_matched_by_pt):

    # Flatten the dxy fields
    gen_taus_flat_dxy = ak.flatten(gen_taus.dxy, axis=1)
    gen_taus_flat_pt = ak.flatten(gen_taus.pt, axis=1)

    gen_taus_matched_by_pt_flat_dxy = ak.flatten(gen_taus_matched_by_pt.dxy, axis=1)
    gen_taus_matched_by_pt_flat_pt = ak.flatten(gen_taus_matched_by_pt.pt, axis=1)

    return (gen_taus_flat_dxy, gen_taus_flat_pt,
            gen_taus_matched_by_pt_flat_dxy, gen_taus_matched_by_pt_flat_pt)

