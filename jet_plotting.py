# jet_plotting.py
import os
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import hist
from hist import Hist, axis, intervals
import json

def get_ratio_histogram(passing_probes, denominator):
    """Get the ratio (efficiency) of the passing over passing + failing probes.
    NaN values are replaced with 0.

    Parameters
    ----------
        passing_probes : hist.Hist
            The histogram of the passing probes.
         : hist.Hist
            The histogram of the denominator.

    Returns
    -------
        ratio : hist.Hist
            The ratio histogram.
        yerr : numpy.ndarray
            The y error of the ratio histogram.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_values = passing_probes.values(flow=True) / denominator.values(flow=True)
                                 
    ratio = hist.Hist(hist.Hist(*passing_probes.axes))
    ratio[:] = np.nan_to_num(ratio_values)
    yerr = intervals.ratio_uncertainty(passing_probes.values(), denominator.values(), uncertainty_type="efficiency")

    return ratio, yerr


def plot_efficiency(passing_probes, denominator, log=False, **kwargs):
    """Plot the efficiency using the ratio of passing to passing + failing probes.

    Parameters
    ----------
        passing_probes : hist.Hist
            The histogram of the passing probes.
        denominator : hist.Hist
            The histogram of the denominator
        **kwargs
            Keyword arguments to pass to hist.Hist.plot1d.

    Returns
    -------
        List[Hist1DArtists]

    """
    ratio_hist, yerr = get_ratio_histogram(passing_probes, denominator)
    plt.ylabel('efficiency')
    if log:  plt.xscale('log')
    return ratio_hist.plot1d(histtype="errorbar", yerr=yerr, xerr=True, flow="none", **kwargs)


# --------------------------------------------------------------------
# Plotting Functions
# --------------------------------------------------------------------
def plot_dxy_efficiency(gen_taus_flat_dxy, gen_taus_matched_by_pt_flat_dxy, output_dir, sample_name):
    """
    Create and save dxy efficiency plots:
      - One for prompt dxy (|dxy| <= 0.15).
      - One for overall dxy (all values).
    """
    dxy_bins_low = np.linspace(-5, 5, 11)  
    dxy_bins_med = np.linspace(5, 15, 5)  
    dxy_bins_med_neg = np.linspace(-15, -5, 5)
    dxy_bins_high = np.linspace(15, 30, 3)  
    dxy_bins_high_neg = np.linspace(-30, -15, 3)

    # Combine all bins while ensuring no duplicate edges
    dxy_bins = np.concatenate([dxy_bins_high_neg[:-1],  
                                      dxy_bins_med_neg[:-1],   
                                      dxy_bins_low,            
                                      dxy_bins_med[1:],        
                                      dxy_bins_high[1:]        
                                     ])
    dxy_bins_prompt = np.linspace(-20, 20, 41)
    # Create histograms for prompt dxy efficiency
    hist_dxy_prompt_den = Hist(axis.Variable(dxy_bins_prompt, flow=True, name="dxy", label="dxy [cm]"))
    hist_dxy_prompt_num = Hist(axis.Variable(dxy_bins_prompt, flow=True, name="dxy", label="dxy [cm]"))
    # Create histograms for overall dxy efficiency
    hist_dxy_den = Hist(axis.Variable(dxy_bins, flow=True, name="dxy", label="dxy [cm]"))
    hist_dxy_num = Hist(axis.Variable(dxy_bins, flow=True, name="dxy", label="dxy [cm]"))
    
    # Fill prompt histograms
    hist_dxy_prompt_den.fill(gen_taus_flat_dxy.compute()) 
    hist_dxy_prompt_num.fill(gen_taus_matched_by_pt_flat_dxy.compute())
    
    # Fill overall dxy histograms
    hist_dxy_den.fill(gen_taus_flat_dxy.compute())
    hist_dxy_num.fill(gen_taus_matched_by_pt_flat_dxy.compute())
    
    # Plot efficiency for prompt dxy
    plt.clf()
    plot_efficiency(hist_dxy_prompt_num, hist_dxy_prompt_den, label='Leading pT Jet Matching Efficiency dxy (prompt)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"eff_vs_dxy_prompt_leading_pT_{sample_name}.pdf"))
    
    # Plot efficiency for overall dxy
    plt.clf()
    plot_efficiency(hist_dxy_num, hist_dxy_den, label='Leading pT Jet Matching Efficiency dxy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"eff_vs_dxy_leading_pT_{sample_name}.pdf"))

    return (hist_dxy_num, hist_dxy_den)

def plot_pt_efficiency(gen_taus_flat_pt, gen_taus_matched_by_pt_flat_pt, output_dir, sample_name):
    """
    Create and save pt efficiency plots for:
      - Overall pt efficiency.
      - Zoomed pt efficiency (pt between 100 and 250 GeV).
    """
    pt_bins_low = np.arange(20, 101, 20)
    pt_bins_med = np.arange(100, 400, 30)
    pt_bins_high = np.arange(400, 600, 40)
    pt_bins_higher = np.arange(600, 1000, 50)
    pt_bins_eff = np.unique(np.concatenate([pt_bins_low, pt_bins_med, pt_bins_high, pt_bins_higher]))
    pt_bins_zoom = np.arange(100, 600, 20)
    
    var_axes = {'pt': axis.Variable(pt_bins_eff, flow=False, name="tauh_pt")}
    var_axes_zoom = {'pt': axis.Variable(pt_bins_zoom, flow=False, name="tauh_pt_zoom")}
    
    hist_pt_den = Hist(var_axes['pt'])
    hist_pt_num = Hist(var_axes['pt'])
    hist_pt_den_zoom = Hist(var_axes_zoom['pt'])
    hist_pt_num_zoom = Hist(var_axes_zoom['pt'])
    
    hist_pt_den.fill(gen_taus_flat_pt.compute())
    hist_pt_num.fill(gen_taus_matched_by_pt_flat_pt.compute())
    hist_pt_den_zoom.fill(gen_taus_flat_pt.compute())
    hist_pt_num_zoom.fill(gen_taus_matched_by_pt_flat_pt.compute())
    
    # Plot overall pt efficiency
    plt.clf()
    plot_efficiency(hist_pt_num, hist_pt_den)
    plt.title(f"Leading pT Jet Matching Efficiency (pt)")
    plt.savefig(os.path.join(output_dir, f"eff_vs_pt_leading_pT_{sample_name}.pdf"))

    # Plot zoom pt efficiency
    plt.clf()
    plot_efficiency(hist_pt_num_zoom, hist_pt_den_zoom)
    plt.title(f"Leading pT Jet Matching Efficiency (pt Zoom)")
    plt.savefig(os.path.join(output_dir, f"eff_vs_pt_zoom_leading_pT_{sample_name}.pdf"))

    return (hist_pt_num, hist_pt_den, hist_pt_num_zoom, hist_pt_den_zoom)

def plot_overlay_histograms(gen_taus_flat_pt, gen_taus_flat_dxy, gen_taus_matched_by_pt_flat_pt, gen_taus_matched_by_pt_flat_dxy, output_dir, sample_name):
    """
    Create and save overlay plots comparing gen tau and matched gen tau distributions
    for pt and dxy.
    """
    # Overlay for pt
    pt_bins_indiv = list(range(20, 750, 10))
    hist_gen = Hist(axis.Variable(pt_bins_indiv, flow=True, name="pt", label="pt [GeV]"))
    hist_gen_matched = Hist(axis.Variable(pt_bins_indiv, flow=True, name="pt", label="pt [GeV]"))
    
    hist_gen.fill(gen_taus_flat_pt.compute())
    hist_gen_matched.fill(gen_taus_matched_by_pt_flat_pt.compute())
    
    fig_overlay_pt, ax_overlay_pt = plt.subplots(figsize=(8, 6))
    hist_gen.plot1d(ax=ax_overlay_pt, histtype="step", label="gen_taus.pt", color="blue")
    hist_gen_matched.plot1d(ax=ax_overlay_pt, histtype="step", label="jet-matched gen_taus.pt", color="red")
    ax_overlay_pt.set_xlabel("pt [GeV]")
    ax_overlay_pt.set_ylabel("Counts")
    ax_overlay_pt.legend()
    ax_overlay_pt.set_title(f"Overlay of Gen Taus and Matched Gen Taus pt ({sample_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"overlay_pt_{sample_name}.pdf"))
    plt.close(fig_overlay_pt)
    
    # Overlay for dxy
    dxy_bins_indiv = list(np.arange(-30, 30, 0.5))
    hist_dxy = Hist(axis.Variable(dxy_bins_indiv, flow=True, name="dxy", label="dxy [cm]"))
    hist_dxy_matched = Hist(axis.Variable(dxy_bins_indiv, flow=True, name="dxy", label="dxy [cm]"))
    
    hist_dxy.fill(gen_taus_flat_dxy.compute())
    hist_dxy_matched.fill(gen_taus_matched_by_pt_flat_dxy.compute())
    
    fig_overlay_dxy, ax_overlay_dxy = plt.subplots(figsize=(8, 6))
    hist_dxy.plot1d(ax=ax_overlay_dxy, histtype="step", label="gen_taus.dxy", color="blue")
    hist_dxy_matched.plot1d(ax=ax_overlay_dxy, histtype="step", label="jet-matched gen_taus.dxy", color="red")
    ax_overlay_dxy.set_xlabel("dxy [cm]")
    ax_overlay_dxy.set_ylabel("Counts")
    ax_overlay_dxy.legend()
    ax_overlay_dxy.set_title(f"Overlay of Gen Taus and Matched Gen Taus dxy ({sample_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"overlay_dxy_{sample_name}.pdf"))
    plt.close(fig_overlay_dxy)

def overlay_dxy_match_hists(gen_taus_matched_by_flat_dxy, leading_dxy_jets_flat_dxy, output_dir, sample_name):
    dxy_bins = list(np.arange(-5, 5, 0.1))
    
    hist_gen_matched = Hist(hist.axis.Variable(dxy_bins, flow=True, name="dxy", label="dxy [cm]"))
    hist_jet_dxy = Hist(hist.axis.Variable(dxy_bins, flow=True, name="dxy", label="dxy [cm]"))

    # Fill histograms
    hist_gen_matched.fill(gen_taus_matched_by_flat_dxy.compute())
    hist_jet_dxy.fill(leading_dxy_jets_flat_dxy.compute())

    # Plot overlay
    fig, ax = plt.subplots(figsize=(8, 6))
    hist_gen_matched.plot1d(ax=ax, histtype="step", label="Matched Gen Taus dxy", color="blue")
    hist_jet_dxy.plot1d(ax=ax, histtype="step", label="Leading Jets dxy", color="green")

    ax.set_xlabel("dxy [cm]")
    ax.set_ylabel("Counts")
    ax.set_title(f"Overlay of dxy: Matched Gen Taus vs Jets ({sample_name})")
    ax.legend()
    ax.grid(True)

    # Save and close
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"overlay_dxy_genTau_vs_jet_{sample_name}.pdf"))
    plt.close(fig)

def plot_2d_histogram(gen_taus_flat_pt, gen_taus_flat_dxy, output_dir, sample_name):
    """
    Create and save a 2D histogram of gen_tau_dxy vs. gen_tau_pT.
    The x-axis is gen_tau_dxy (range -20 to 20 cm) and the y-axis is gen_tau_pT (range 0 to 500 GeV).
    """
    dxy_bins = np.linspace(-15, 15, 61)  
    pt_bins = np.linspace(0, 900, 91)  
    h2d = hist.Hist(
        axis.Variable(dxy_bins, name="dxy", label="gen_tau_dxy [cm]"),
        axis.Variable(pt_bins, name="pT", label="gen_tau_pT [GeV]")
    )

    h2d.fill(dxy=gen_taus_flat_dxy.compute() , pT=gen_taus_flat_pt.compute())
    plt.clf()
    fig, ax = plt.subplots(figsize=(8, 6))
    h2d.plot2d(ax=ax)
    ax.set_title(f"2D dxy vs. pT — {sample_name}", fontsize=14)
    plt.savefig(os.path.join(output_dir, f"2d_gen_tau_dxy_vs_pt_{sample_name}.pdf"))
    plt.close()

def plot_jet_score_efficiency(leading_score_jets, jet_matched_gen_taus_score, output_dir, sample_name):
    """
    Creates efficiency plots (using plot_efficiency) and separate normalized plots
    for the jet score histograms.
    
    Normalization is done by dividing each bin by the total number of jets.
    """
    # Define jet score binning and create the axis.
    jet_score_bins = np.linspace(0, 1.0, 61)  # 61 edges for 60 bins
    jet_score_axis = axis.Variable(jet_score_bins, flow=True, name="jet_score", label="disTauTag score")
    
    h_den = Hist(jet_score_axis)
    h_num = Hist(jet_score_axis)
    
    jet_scores = ak.flatten(leading_score_jets.disTauTag_score1, axis=1).compute()
    matched_jet_scores = ak.flatten(jet_matched_gen_taus_score.disTauTag_score1, axis=1).compute()
    
    h_den.fill(jet_scores.compute())
    h_num.fill(matched_jet_scores.compute())
    
    plt.clf()
    plot_efficiency(h_num, h_den, label=f"Jet Score Efficiency")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"eff_vs_jet_score_{sample_name}.pdf"))
    plt.close()
    
    # For normalized standalone plots, extract bin contents and edges.
    den_vals = h_den.values(flow=True)
    num_vals = h_num.values(flow=True)
    total_den = den_vals.sum()
    total_num = num_vals.sum()
    norm_den_vals = den_vals / total_den if total_den > 0 else den_vals
    norm_num_vals = num_vals / total_num if total_num > 0 else num_vals
    edges = np.array(jet_score_axis.edges)

def plot_numJets_histogram(total_nJets, output_dir, sample_name):
    """
    Create and save a histogram of the number of jets per event

    Parameters:
    - numJets: Array of number of jets per event
    - output_filename: Name of the file to save the histogram
    """
    # Define binning for number of jets
    numJets_axis = hist.axis.Regular(11, -0.5, 10.5, name="Number of Jets per Event")
    
    # Create histogram
    numJets_hist = hist.Hist(numJets_axis)
    
    # Fill histogram with the number of jets per event
    numJets_hist.fill(total_nJets.compute())

    # Create output file path
    output_file = os.path.join(output_dir, f"numJets_distribution_{sample_name}.pdf")

    # Plot the histogram
    plt.clf()
    plt.figure(figsize=(8,6))
    numJets_hist.plot1d()
    plt.xlabel("Number of Jets per Event")
    plt.ylabel("Number of Events")
    plt.title(f"Number of Jet After Selection — {sample_name}", fontsize=14)
    plt.grid(True)
    
    # Save plot
    plt.savefig(output_file)
    plt.close()

def plot_matched_vs_unmatched_jets(matched_jets, unmatched_jets, output_dir, sample_name):
    """
    Create and save a histogram comparing matched and unmatched jet pT distributions.

    Parameters:
    - matched_jets: Array of pT values for matched jets.
    - unmatched_jets: Array of pT values for unmatched jets.
    - output_dir: Directory where the plot will be saved.
    - sample_name: Sample name used for naming the output file.
    """
    # Define variable binning
    bins_0_200 = np.linspace(0, 200, 6)  # Few bins for high stats
    bins_200_400 = np.linspace(200, 400, 6)  # Medium bins for moderate stats
    bins_400_up = np.linspace(400, 900, 11)  # More bins for low stats
    bins = np.concatenate([bins_0_200[:-1], bins_200_400[:-1], bins_400_up])  # Combine bins

    # Create histograms
    matched_hist = hist.Hist(hist.axis.Variable(bins, name="pT", label="Jet pT [GeV]"))
    unmatched_hist = hist.Hist(hist.axis.Variable(bins, name="pT", label="Jet pT [GeV]"))

    # Fill histograms
    matched_hist.fill(matched_jets.compute())
    unmatched_hist.fill(unmatched_jets.compute())

    # Create output file path
    output_file = os.path.join(output_dir, f"matched_vs_unmatched_jets_{sample_name}.pdf")

    # Plot histograms
    plt.clf()
    plt.figure(figsize=(8,6))
    
    matched_hist.plot1d(histtype="step", color="blue", label="Matched Jets")
    unmatched_hist.plot1d(histtype="step", color="red", linestyle="dashed", label="Unmatched Jets")

    plt.xlabel("Jet pT [GeV]")
    plt.ylabel("Number of Leading pT Jets")
    plt.title(f"Matched vs Unmatched Jets — {sample_name}", fontsize=14)
    plt.grid(True)
    plt.legend()

    # Save plot
    plt.savefig(output_file)
    plt.close()


def plot_efficiency_from_json(json_filename, output_file):
    """
    Reads efficiency data from a JSON file and plots a heatmap with efficiency-based coloring.
    """

    # Read the JSON file
    with open(json_filename, "r") as f:
        efficiency_data = json.load(f)

    # Extract unique masses and lifetimes
    masses = sorted(set(entry["mass"] for entry in efficiency_data))
    lifetimes = sorted(set(entry["lifetime"] for entry in efficiency_data))  # Still in mm

    # Create a mapping from mass/lifetime to an array index
    mass_idx = {m: i for i, m in enumerate(masses)}
    lifetime_idx = {lt: i for i, lt in enumerate(lifetimes)}

    # Build efficiency grid (rows=lifetimes, columns=masses)
    Z = np.zeros((len(lifetimes), len(masses)))

    for entry in efficiency_data:
        m = entry["mass"]
        lt = entry["lifetime"]
        Z[lifetime_idx[lt], mass_idx[m]] = entry["efficiency"]

    # Reverse the y-axis order so lifetimes go from smallest to largest
    Z = Z[::-1]
    lifetimes = lifetimes[::-1]

    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Display the grid as an image
    cmap = cm.get_cmap("plasma")  # Use a color map where efficiency determines shade
    norm = mcolors.Normalize(vmin=0, vmax=1)  # Efficiency is in range [0,1]
    im = ax.imshow(Z, cmap=cmap, norm=norm)

    # Set x-axis (Mass) and y-axis (Lifetime)
    ax.set_xticks(range(len(masses)))
    ax.set_xticklabels([str(m) for m in masses])  # Mass values as labels

    ax.set_yticks(range(len(lifetimes)))
    ax.set_yticklabels([f"{lt} mm" for lt in lifetimes])  # Lifetimes in mm
    
    ax.set_xlabel("Mass [GeV]")
    ax.set_ylabel("Lifetime [mm]")
    ax.set_title("(had_gen_taus matched to highest dxy jet)/(total had_gen_taus)", fontsize=14, pad=15)

    # Loop over data dimensions and create text annotations.
    for i in range(len(lifetimes)):
        for j in range(len(masses)):
            efficiency_value = Z[i, j]
            if efficiency_value > 0:  # Only display values where efficiency is nonzero
                text_color = "white" if efficiency_value < 0.5 else "black"
                ax.text(j, i, f"{efficiency_value:.3f}", ha="center", va="center", 
                        color=text_color, fontsize=9)

    # Add colorbar to indicate efficiency scale
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Efficiency")

    plt.savefig(output_file)
    plt.close()
    print(f"Saved efficiency plot to {output_file}")
