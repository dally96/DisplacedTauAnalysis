import awkward as ak
import numpy as np
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema
import os
import json
'''
def delta_r_mask(first: ak.highlevel.Array, second: ak.highlevel.Array, threshold: float) -> ak.highlevel.Array:
            mval = first.metric_table(second)
            return ak.all(mval > threshold, axis=-1)
'''
def process_events(events):
    """Process events to select staus and their tau children and filter events."""
    
    # add dxy to jet fields
    charged_sel = events.Jet.constituents.pf.charge != 0
    dxy = ak.flatten(events.Jet.constituents.pf[ak.argmax(events.Jet.constituents.pf[charged_sel].pt, axis=2, keepdims=True)].d0, axis = 2)
    events['Jet'] = ak.with_field(events.Jet, dxy, where="dxy")
    '''
    # Perform the overlap removal with respect to muons, electrons and photons, dR=0.4
    events['Jet'] = events.Jet[delta_r_mask(events.Jet, events.Photon, 0.4)]
    events['Jet'] = events.Jet[delta_r_mask(events.Jet, events.Electron, 0.4)]
    events['Jet'] = events.Jet[delta_r_mask(events.Jet, events.Muon, 0.4)]
    events['Jet'] = events.Jet[delta_r_mask(events.Jet, events.DisMuon, 0.4)]
    '''
    ## find staus and their tau children
    gpart = events.GenPart
    events['staus'] = gpart[(abs(gpart.pdgId) == 1000015) & (gpart.hasFlags("isLastCopy"))] # most likely last copy of stay in the chain

    events['staus_taus'] = events.staus.distinctChildren[ (abs(events.staus.distinctChildren.pdgId) == 15) & \
                                                      (events.staus.distinctChildren.hasFlags("isLastCopy")) \
                                                     ]

    events['GenVisStauTaus'] = events.GenVisTau[(abs(events.GenVisTau.parent.pdgId) == 15) & (abs(events.GenVisTau.parent.distinctParent.pdgId) == 1000015)]

    # for events where the same tau is listed multiple times, the following takes the first iteration
    events['staus_taus'] = ak.firsts(events.staus_taus[ak.argsort(events.staus_taus.pt, ascending=False)], axis = 2)
    staus_taus = events['staus_taus']

    mask_taul = ak.any((abs(staus_taus.distinctChildren.pdgId) == 11) | (abs(staus_taus.distinctChildren.pdgId) == 13), axis=-1)
    mask_tauh = ~mask_taul
    
    debugging_had_gen_taus = events.staus_taus[mask_tauh]
    debugging_num_had_gen_taus = ak.sum(ak.num(debugging_had_gen_taus))
    debugging_num_vis_gen_taus = ak.sum(ak.num(events.GenVisStauTaus))
    print("Number of had gen taus:", debugging_num_had_gen_taus.compute())
    print("Number of gen vis taus:", debugging_num_vis_gen_taus.compute())


    one_tauh_evt = (ak.sum(mask_tauh, axis=-1) > 0) & (ak.sum(mask_tauh, axis=-1) < 3)
    one_taul_evt = (ak.sum(mask_taul, axis=-1) > 0) & (ak.sum(mask_taul, axis=-1) < 3)

    filtered_events = events[one_tauh_evt & one_taul_evt]  # Filtered events are events with one hadronic tau and one leptonic tau
    
    #tau_selections = ak.any((filtered_events.staus_taus.pt > 20) & (abs(filtered_events.staus.eta) < 2.4), axis=-1)
    #num_taus = ak.num(filtered_events.staus_taus[tau_selections])
    #num_tau_mask = num_taus > 1
    #cut_filtered_events = filtered_events[num_tau_mask]
    num_taus = ak.num(filtered_events.staus_taus)
    num_tau_mask = num_taus > 1
    cut_filtered_events = filtered_events[num_tau_mask]

    #cut_filtered_events = filtered_events
    
    return cut_filtered_events

def select_and_define_leading_jets(cut_filtered_events):
    """
    Select jets from events with |eta| < 2.4 and pt > 20, and then
    define two sets of leading jets:
      - leading_pt_jets: the leading jet in each event based on pt
      - highest_score_jets: the leading jet in each event based on disTauTag_score1
    """
    # Select jets with |eta| < 2.4 and pt > 20
    jets = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.4) & (cut_filtered_events.Jet.pt > 20)]
    total_nJets = ak.num(jets)
    
    # Sort the selected jets by pt (descending) and take the first jet per event
    sorted_by_pt = jets[ak.argsort(jets.pt, ascending=False)]
    leading_pt_jets = ak.singletons(ak.firsts(sorted_by_pt))

    # Sort the selected jets by dxy (descending) and take the first jet per event
    sorted_by_dxy = jets[ak.argsort(abs(jets.dxy), ascending=False)]
    highest_dxy_jets = ak.singletons(ak.firsts(sorted_by_dxy))

    # Make cut on highest score jets
    jets = jets[jets.disTauTag_score1 > 0.90]

    # Sort the selected jets by disTauTag_score1 (descending) and take the first jet per event
    sorted_by_score = jets[ak.argsort(jets.disTauTag_score1, ascending=False)]
    highest_score_jets = ak.singletons(ak.firsts(sorted_by_score))
    
    return (jets, leading_pt_jets, highest_score_jets, highest_dxy_jets, total_nJets)


def match_gen_taus(cut_filtered_events, leading_pt_jets, highest_dxy_jets, highest_score_jets, jets, total_nJets, sample_name):
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

    # Get sum of all had gen taus used as denominator for grid plots
    num_had_gen_taus = ak.sum(ak.num(gen_taus))

    #cut_filtered_events.GenVisStauTaus = cut_filtered_events.GenVisStauTaus[(cut_filtered_events.GenVisStauTaus.pt > 20) & (abs(cut_filtered_events.GenVisStauTaus.eta) < 2.4)]

    # Matching using the pt leading jets
    gen_taus_matched_by_pt = leading_pt_jets.nearest(gen_taus, threshold=0.4)
    gen_taus_matched_by_pt = ak.drop_none(gen_taus_matched_by_pt)

    # Get sum of gen_taus_matched_by_pt
    nMatched_gen_taus_by_pt = ak.sum(ak.num(gen_taus_matched_by_pt))

    # Matching using the dxy leading jets
    gen_taus_matched_by_dxy = highest_dxy_jets.nearest(gen_taus, threshold=0.4)
    gen_taus_matched_by_dxy = ak.drop_none(gen_taus_matched_by_dxy)

    # Get sum of gen_taus_matched_by_dxy
    nMatched_gen_taus_by_dxy = ak.sum(ak.num(gen_taus_matched_by_dxy))

    jet_matched_gen_taus_pt = gen_taus.nearest(leading_pt_jets, threshold=0.4)
    jet_matched_gen_taus_pt = ak.drop_none(jet_matched_gen_taus_pt)

    # Matching using the leading-score jets.
    gen_taus_matched_by_score = highest_score_jets.nearest(gen_taus, threshold=0.4)
    gen_taus_matched_by_score = ak.drop_none(gen_taus_matched_by_score)
    jet_matched_gen_taus_score = gen_taus.nearest(highest_score_jets, threshold=0.4)
    jet_matched_gen_taus_score = ak.drop_none(jet_matched_gen_taus_score)

    # Get sum of leading score jets
    num_highest_score_jets = ak.sum(ak.num(highest_score_jets))

    # Get sum of jets that are matched to gen taus based on highest score jet
    nMatched_jets_matched_to_gen_tau_highest_score_jet = ak.sum(ak.num(jet_matched_gen_taus_score))

    # Get all jets matched to a gen_tau
    all_jets_matched_to_gen_tau = gen_taus.nearest(jets, threshold=0.4)
    
    # Compute dxy efficiency
    #efficiency = float(nMatched_gen_taus_by_dxy / num_had_gen_taus).compute() if num_had_gen_taus.compute() > 0 else 0.0

    gen_vis_taus_matched_by_pt = leading_pt_jets.nearest(cut_filtered_events.GenVisStauTaus, threshold=0.4)
    gen_vis_taus_matched_by_pt = ak.drop_none(gen_vis_taus_matched_by_pt)
    nMatched_gen_vis_taus_by_pt = ak.sum(ak.num(gen_vis_taus_matched_by_pt))
    num_vis_gen_taus = ak.sum(ak.num(cut_filtered_events.GenVisStauTaus))

    # Compute pT efficiency
    #efficiency = (nMatched_gen_vis_taus_by_pt / num_vis_gen_taus).compute() if num_vis_gen_taus.compute() > 0 else 0.0

    # Matching using the leading-score jets.
    gen_vis_taus_matched_by_score = highest_score_jets.nearest(cut_filtered_events.GenVisStauTaus, threshold=0.4)
    gen_vis_taus_matched_by_score = ak.drop_none(gen_vis_taus_matched_by_score)
    jet_matched_gen_vis_taus_score = cut_filtered_events.GenVisStauTaus.nearest(highest_score_jets, threshold=0.4)
    jet_matched_gen_vis_taus_score = ak.drop_none(jet_matched_gen_vis_taus_score)
    nMatched_jets_matched_to_gen_vis_tau_highest_score_jet = ak.sum(ak.num(jet_matched_gen_vis_taus_score))

    #print("Number of had gen taus:", num_had_gen_taus.compute())
    #print("Number of gen vis taus:", num_vis_gen_taus.compute())
    #print("Number of highest score jets:", num_highest_score_jets.compute())
    #print("Number of jets matched to gen taus (highest score):", nMatched_jets_matched_to_gen_tau_highest_score_jet.compute())
    #print("Number of jets matched to gen vis taus (highest score):", nMatched_jets_matched_to_gen_vis_tau_highest_score_jet.compute())

    '''
    # Compute score efficiency
    efficiency = (nMatched_jets_matched_to_gen_vis_tau_highest_score_jet / num_vis_gen_taus).compute() if num_vis_gen_taus.compute() > 0 else 0.0

    # Get mass and lifetime from sample_name (e.g., "Stau_100_1mm")
    parts = sample_name.split('_')
    mass = int(parts[1]) 
    lifetime = int(parts[2].replace('mm', '')) # remove mm from 1mm 

    # Create a dictionary with keys: mass, lifetime, efficiency
    efficiency_data = {
        "mass": mass,
        "lifetime": lifetime,
        "efficiency": efficiency
    }

    # Save to JSON file
    json_filename = "score_efficiency_results.json"

    # Ensure JSON file exists and is not empty before loading
    if os.path.exists(json_filename) and os.path.getsize(json_filename) > 0:
        try:
            with open(json_filename, "r") as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: {json_filename} is corrupted. Overwriting with new data.")
            existing_data = []  # Reset the JSON file if it's corrupted
    else:
        print(f"Creating new JSON file: {json_filename}")
        existing_data = []  # If the file doesn't exist or is empty, initialize as an empty list

    existing_data.append(efficiency_data)

    with open(json_filename, "w") as f:
        json.dump(existing_data, f, indent=4)
    '''
       
    # From all jets matched to gen_tau, select highest pT jets
    sort_by_pt = all_jets_matched_to_gen_tau[ak.argsort(all_jets_matched_to_gen_tau.pt, ascending=False)]
    leading_jets_matched_to_gen_tau = ak.singletons(ak.firsts(sort_by_pt))

    ###########################################################################################################
    # The following block will be used for hist to get the pt of all MATCHED jets
    ###########################################################################################################
    # Create mask for when there is always at least one jet matched to gen_tau
    mask_has_match = ak.num(leading_jets_matched_to_gen_tau) > 0 

    # Create mask for when there is no jet matched to gen_tau
    mask_has_no_match = ak.num(leading_jets_matched_to_gen_tau) == 0

    # For events where there is no jet or no jet matched, fill with -999 so flattening works properly
    filled_matched_pts = ak.fill_none(ak.pad_none(leading_jets_matched_to_gen_tau.pt, 1), -999)
    filled_all_pts = ak.fill_none(ak.pad_none(leading_pt_jets.pt, 1), -999)

    # Create mask where the matched jet IS NOT the overall leading pt jet and flatten
    mask_mismatch = (ak.flatten(filled_all_pts) != ak.flatten(filled_matched_pts))

    # Create mask where the matched jet IS the overall leading pt jet and flatten
    mask_matched_leading = (ak.flatten(filled_all_pts) == ak.flatten(filled_matched_pts))

    # Apply mask where there is always at least one jet matched and flatten
    # This will be used for hist to get pt of ALL matched jets
    matched_leading_jets = leading_jets_matched_to_gen_tau[mask_has_match]
    matched_leading_jets_flat = ak.flatten(matched_leading_jets.pt, axis=1)

    ###########################################################################################################
    # The following block will be used for hist to get the pt of all NOT matched jets
    ###########################################################################################################
    # Create mask where there is more than 1 jet matched in order to get next leading pt jet
    mask_has_more_than_one_jets = ak.num(jets) > 1  

    # Apply mask to get events where there is more than one jet matched and it IS the leading pt jet in order to get next leading pt jet
    matched_jet_to_leading_pt = jets[mask_has_more_than_one_jets & mask_has_match & mask_matched_leading]

    # Sort jets where there is more than one jet matched and it IS the leading pt jet and get the next leading pt jet
    matched_jet_to_leading_pt_sort_by_pt = matched_jet_to_leading_pt[ak.argsort(matched_jet_to_leading_pt.pt, ascending=False)]
    leading_pt_matched_jet_to_leading_pt = ak.singletons(ak.firsts(matched_jet_to_leading_pt_sort_by_pt[:, 1:]))

    # Apply mask to get events where there is a matched jet but it is not the overall leading pt jet
    matched_jet_not_leading_pt = jets[mask_has_match & mask_mismatch]

    # Sort jets where there is a matched jet but it is NOT the overall leading jet then get the overall leading jet pt
    matched_jet_not_leading_pt_sort_by_pt = matched_jet_not_leading_pt[ak.argsort(matched_jet_not_leading_pt.pt, ascending=False)]
    leading_pt_of_matched_jet_not_leading_pt = ak.singletons(ak.firsts(matched_jet_not_leading_pt_sort_by_pt))

    # Apply mask where there are no jets matched to gen_tau
    no_matched_jet = jets[mask_has_no_match]

    # Sort jets where there are no jets matched to gen_tau and get highest pt jet
    no_match_sort_by_pt = no_matched_jet[ak.argsort(no_matched_jet.pt, ascending=False)]
    no_match_jet_leading_pt = ak.singletons(ak.firsts(no_match_sort_by_pt))

    # Combine the pt of all below to fill one hist with pt of unmatched jets
    # leading_pt_matched_jet_to_leading_pt: unmatched next leading pt jet
    # leading_pt_of_matched_jet_not_leading_pt: unmatched overall leading pt jet
    # no_match_jet_leading_pt: unmatched jet
    # Flatten each array to ensure they are 1D
    leading_pt_matched_jet_to_leading_pt_flat = ak.flatten(leading_pt_matched_jet_to_leading_pt)
    leading_pt_of_matched_jet_not_leading_pt_flat = ak.flatten(leading_pt_of_matched_jet_not_leading_pt)
    no_match_jet_leading_pt_flat = ak.flatten(no_match_jet_leading_pt)

    # Concatenate all flattened arrays
    all_unmatched_jets_pt = ak.concatenate(
        [leading_pt_matched_jet_to_leading_pt_flat.pt, 
         leading_pt_of_matched_jet_not_leading_pt_flat.pt, 
         no_match_jet_leading_pt_flat.pt])

    return (gen_taus, gen_taus_matched_by_dxy,
            gen_taus_matched_by_pt, jet_matched_gen_taus_pt,
            gen_taus_matched_by_score, jet_matched_gen_taus_score,
            matched_leading_jets_flat, all_unmatched_jets_pt)

def flatten_gen_tau_vars(gen_taus, gen_taus_matched_by_pt, highest_dxy_jets, gen_taus_matched_by_dxy):

    # Flatten the dxy fields
    gen_taus_flat_dxy = ak.flatten(gen_taus.dxy, axis=1)
    gen_taus_matched_by_flat_dxy = ak.flatten(gen_taus_matched_by_dxy.dxy, axis=1)
    highest_dxy_jets_flat_dxy = ak.flatten(ak.drop_none(highest_dxy_jets.dxy, axis=1))

    # Flatten the pt fields
    gen_taus_flat_pt = ak.flatten(gen_taus.pt, axis=1)

    gen_taus_matched_by_pt_flat_dxy = ak.flatten(gen_taus_matched_by_pt.dxy, axis=1)
    gen_taus_matched_by_pt_flat_pt = ak.flatten(gen_taus_matched_by_pt.pt, axis=1)

    return (gen_taus_flat_dxy, gen_taus_matched_by_flat_dxy, highest_dxy_jets_flat_dxy, gen_taus_flat_pt,
            gen_taus_matched_by_pt_flat_dxy, gen_taus_matched_by_pt_flat_pt)

