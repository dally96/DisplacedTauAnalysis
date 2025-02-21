# jet_selection_efficiency.py
import os
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema

# Import functions from jet_selection.py
from jet_selection import (
    process_events,
    select_and_define_leading_jets,
    match_gen_taus,
    flatten_gen_tau_vars,
)

# Import functions from jet_plotting.py
from jet_plotting import (
    plot_dxy_efficiency,
    plot_pt_efficiency,
    plot_overlay_histograms,
    plot_2d_histogram,
    plot_jet_score_efficiency,
    overlay_efficiency,
    compute_overall_efficiency,
    plot_sample_grid
)

# Load the file
filenames = {
    'Stau_100_100' : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-100_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_*.root',
    'Stau_200_100' : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-200_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_*.root',
    'Stau_300_100' : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_*.root',
    'Stau_500_100' : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_*.root',
}

samples = {}
for sample_name, files in filenames.items():
    samples[sample_name] = NanoEventsFactory.from_root(
        {files: "Events"},
        schemaclass=PFNanoAODSchema,
        metadata={"dataset": "MC"}
    ).events()

# Create output folder for histograms
output_dir = "histograms"
os.makedirs(output_dir, exist_ok=True)

# Lists to store efficiency histograms for overlay plots
pt_eff_data = []
pt_zoom_eff_data = []
dxy_eff_data = []

# ----------------------------------------------------------------------
# Main loop: Process each sample and produce histograms.
# ----------------------------------------------------------------------
if __name__ == '__main__':
    for sample_name, events in samples.items():
        print(f"Processing sample: {sample_name}")
        
        # Process events: select staus, taus, and apply event filters
        cut_filtered_events = process_events(events)
        
        # Select jets and define leading jets (using the filtered events)
        jets, leading_pt_jets, leading_score_jets = select_and_define_leading_jets(cut_filtered_events)
        
        # Match gen taus to jets using both pt-based and leading-score methods
        (gen_taus,
         gen_taus_matched_by_pt, jet_matched_gen_taus_pt,
         gen_taus_matched_by_score, jet_matched_gen_taus_score) = match_gen_taus(cut_filtered_events, leading_pt_jets, leading_score_jets)
        
        # Flatten variables for histogram filling 
        (gen_taus_flat_dxy, gen_taus_flat_pt,
         gen_taus_matched_by_pt_flat_dxy, gen_taus_matched_by_pt_flat_pt) = flatten_gen_tau_vars(gen_taus, gen_taus_matched_by_pt)

        # Generate efficiency histograms
        hist_dxy_num, hist_dxy_den = plot_dxy_efficiency(gen_taus_flat_dxy, gen_taus_matched_by_pt_flat_dxy, output_dir, sample_name)
        hist_pt_num, hist_pt_den, hist_pt_num_zoom, hist_pt_den_zoom = plot_pt_efficiency(gen_taus_flat_pt, gen_taus_matched_by_pt_flat_pt, output_dir, sample_name)
        
        # Overlay histograms (pt and dxy)
        plot_overlay_histograms(gen_taus_flat_pt, gen_taus_flat_dxy, gen_taus_matched_by_pt_flat_pt, gen_taus_matched_by_pt_flat_dxy, output_dir, sample_name)
        
        # 2D histogram of gen_tau_dxy vs. gen_tau_pT
        plot_2d_histogram(gen_taus_flat_pt, gen_taus_flat_dxy, output_dir, sample_name)
        
        # Plot jet score efficiency using leading-score jets
        plot_jet_score_efficiency(leading_score_jets, jet_matched_gen_taus_score, output_dir, sample_name)

        # Generate a random color for the sample
        color = np.random.rand(3,)

        # Store efficiency histograms for overlay
        dxy_eff_data.append((hist_dxy_num, hist_dxy_den, sample_name, color))
        pt_eff_data.append((hist_pt_num, hist_pt_den, sample_name, color))
        pt_zoom_eff_data.append((hist_pt_num_zoom, hist_pt_den_zoom, sample_name, color))
    print(f"All single histograms made")
    # ------------------------------------------------------------------
    # Compute efficiency values for the sample grid **AFTER** all samples are processed
    # ------------------------------------------------------------------
    efficiency_grid_dxy = {}
    efficiency_grid_pt = {}

    for sample_name, hist_data in zip(filenames.keys(), dxy_eff_data):
        hist_dxy_num, hist_dxy_den, _, _ = hist_data
        efficiency_grid_dxy[sample_name] = compute_overall_efficiency(hist_dxy_num, hist_dxy_den)

    for sample_name, hist_data in zip(filenames.keys(), pt_eff_data):
        hist_pt_num, hist_pt_den, _, _ = hist_data
        efficiency_grid_pt[sample_name] = compute_overall_efficiency(hist_pt_num, hist_pt_den)

    # ------------------------------------------------------------------
    # Generate and save sample grid plots
    # ------------------------------------------------------------------
    plot_sample_grid(efficiency_grid_dxy, output_dir, "dxy")
    plot_sample_grid(efficiency_grid_pt, output_dir, "pt")

    # ------------------------------------------------------------------
    # Generate overlay efficiency plots for all samples
    # ------------------------------------------------------------------
    overlay_efficiency(pt_eff_data, pt_zoom_eff_data, dxy_eff_data, output_dir)
