# jet_plotting.py
import os
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import hist
from hist import Hist, axis, intervals

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
    dxy_bins_prompt = np.linspace(-0.12, 0.12, 30)
    dxy_bins = np.linspace(-20, 20, 41)
    # Create histograms for prompt dxy efficiency
    hist_dxy_prompt_den = Hist(axis.Variable(dxy_bins_prompt, flow=True, name="dxy", label="dxy [cm]"))
    hist_dxy_prompt_num = Hist(axis.Variable(dxy_bins_prompt, flow=True, name="dxy", label="dxy [cm]"))
    # Create histograms for overall dxy efficiency
    hist_dxy_den = Hist(axis.Variable(dxy_bins, flow=True, name="dxy", label="dxy [cm]"))
    hist_dxy_num = Hist(axis.Variable(dxy_bins, flow=True, name="dxy", label="dxy [cm]"))
    
    # Fill prompt histograms
    hist_dxy_prompt_den.fill(gen_taus_flat_dxy) 
    hist_dxy_prompt_num.fill(gen_taus_matched_by_pt_flat_dxy)
    
    # Fill overall dxy histograms
    hist_dxy_den.fill(gen_taus_flat_dxy)
    hist_dxy_num.fill(gen_taus_matched_by_pt_flat_dxy)
    
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
    pt_bins_med = np.arange(100, 600, 10)
    pt_bins_high = np.arange(600, 1000, 20)
    pt_bins_eff = np.unique(np.concatenate([pt_bins_low, pt_bins_med, pt_bins_high]))
    pt_bins_zoom = np.arange(100, 600, 5)
    
    var_axes = {'pt': axis.Variable(pt_bins_eff, flow=False, name="tauh_pt")}
    var_axes_zoom = {'pt': axis.Variable(pt_bins_zoom, flow=False, name="tauh_pt_zoom")}
    
    hist_pt_den = Hist(var_axes['pt'])
    hist_pt_num = Hist(var_axes['pt'])
    hist_pt_den_zoom = Hist(var_axes_zoom['pt'])
    hist_pt_num_zoom = Hist(var_axes_zoom['pt'])
    
    hist_pt_den.fill(gen_taus_flat_pt)
    hist_pt_num.fill(gen_taus_matched_by_pt_flat_pt)
    hist_pt_den_zoom.fill(gen_taus_flat_pt[(gen_taus_flat_pt >= 100) & (gen_taus_flat_pt <= 250)])
    hist_pt_num_zoom.fill(gen_taus_matched_by_pt_flat_pt[(gen_taus_matched_by_pt_flat_pt >= 100) & (gen_taus_matched_by_pt_flat_pt <= 250)])
    
    # Plot overall pt efficiency
    plt.clf()
    plot_efficiency(hist_pt_num, hist_pt_den, label='Leading pT Jet Matching Efficiency (pt)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"eff_vs_pt_leading_pT_{sample_name}.pdf"))
    
    # Plot zoom pt efficiency
    plt.clf()
    plot_efficiency(hist_pt_num_zoom, hist_pt_den_zoom, label='Leading pT Jet Matching Efficiency (pt Zoom)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"eff_vs_pt_zoom_leading_pT_{sample_name}.pdf"))

    return (hist_pt_num, hist_pt_den, hist_pt_num_zoom, hist_pt_den_zoom)

def overlay_efficiency(pt_eff_data, pt_zoom_eff_data, dxy_eff_data, output_dir):
    """
    Overlay efficiency histograms for multiple samples on a single plot.

    Parameters:
      pt_eff_data: List of tuples (hist_pt_num, hist_pt_den, sample_label, color).
      pt_zoom_eff_data: List of tuples (hist_pt_num_zoom, hist_pt_den_zoom, sample_label, color).
      dxy_eff_data: List of tuples (hist_dxy_num, hist_dxy_den, sample_label, color).
      output_dir: Directory where the output plot will be saved.
    """

    # ----------- Overlay pt Efficiency -----------
    fig, ax = plt.subplots(figsize=(8, 6))
    for num_hist, den_hist, sample_label, color in pt_eff_data:
        ratio_hist, yerr = get_ratio_histogram(num_hist, den_hist)
        ratio_hist.plot1d(ax=ax, histtype="errorbar", yerr=yerr, xerr=True, flow="none",
                          label=sample_label, color=color)

    ax.set_xlabel("pt [GeV]")
    ax.set_ylabel("Efficiency")
    ax.set_title("Overlay of pt Efficiencies")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overlay_eff_pt.pdf"))
    plt.close(fig)

    # ----------- Overlay pt Efficiency (Zoom) -----------
    fig, ax = plt.subplots(figsize=(8, 6))
    for num_hist, den_hist, sample_label, color in pt_zoom_eff_data:
        ratio_hist, yerr = get_ratio_histogram(num_hist, den_hist)
        ratio_hist.plot1d(ax=ax, histtype="errorbar", yerr=yerr, xerr=True, flow="none",
                          label=sample_label, color=color)

    ax.set_xlabel("pt [GeV]")
    ax.set_ylabel("Efficiency")
    ax.set_title("Overlay of pt Efficiencies (Zoomed)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overlay_eff_pt_zoom.pdf"))
    plt.close(fig)

    # ----------- Overlay dxy Efficiency -----------
    fig, ax = plt.subplots(figsize=(8, 6))
    for num_hist, den_hist, sample_label, color in dxy_eff_data:
        ratio_hist, yerr = get_ratio_histogram(num_hist, den_hist)
        ratio_hist.plot1d(ax=ax, histtype="errorbar", yerr=yerr, xerr=True, flow="none",
                          label=sample_label, color=color)

    ax.set_xlabel("dxy [cm]")
    ax.set_ylabel("Efficiency")
    ax.set_title("Overlay of dxy Efficiencies")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overlay_eff_dxy.pdf"))
    plt.close(fig)

def plot_overlay_histograms(gen_taus_flat_pt, gen_taus_flat_dxy, gen_taus_matched_by_pt_flat_pt, gen_taus_matched_by_pt_flat_dxy, output_dir, sample_name):
    """
    Create and save overlay plots comparing gen tau and matched gen tau distributions
    for pt and dxy.
    """
    # Overlay for pt
    pt_bins_indiv = list(range(20, 750, 10))
    hist_gen = Hist(axis.Variable(pt_bins_indiv, flow=True, name="pt", label="pt [GeV]"))
    hist_gen_matched = Hist(axis.Variable(pt_bins_indiv, flow=True, name="pt", label="pt [GeV]"))
    
    hist_gen.fill(gen_taus_flat_pt)
    hist_gen_matched.fill(gen_taus_matched_by_pt_flat_pt)
    
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
    
    hist_dxy.fill(gen_taus_flat_dxy)
    hist_dxy_matched.fill(gen_taus_matched_by_pt_flat_dxy)
    
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

def plot_2d_histogram(gen_taus_flat_pt, gen_taus_flat_dxy, output_dir, sample_name):
    """
    Create and save a 2D histogram of gen_tau_dxy vs. gen_tau_pT.
    The x-axis is gen_tau_dxy (range -20 to 20 cm) and the y-axis is gen_tau_pT (range 0 to 500 GeV).
    """
    dxy_bins = np.linspace(-10, 10, 41)  
    pt_bins = np.linspace(0, 500, 51)  
    h2d = hist.Hist(
        axis.Variable(dxy_bins, name="dxy", label="gen_tau_dxy [cm]"),
        axis.Variable(pt_bins, name="pT", label="gen_tau_pT [GeV]")
    )

    h2d.fill(dxy=gen_taus_flat_dxy , pT=gen_taus_flat_pt)
    h2d.plot2d()
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
    
    h_den.fill(jet_scores)
    h_num.fill(matched_jet_scores)
    
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

def compute_overall_efficiency(num_hist, den_hist):
    """
    Compute the overall efficiency from numerator and denominator histograms.

    Parameters:
        num_hist (hist.Hist): Histogram of passing events.
        den_hist (hist.Hist): Histogram of total events.

    Returns:
        float: The overall efficiency (sum of num bins / sum of den bins).
    """
    num_total = num_hist.values(flow=True).sum()
    den_total = den_hist.values(flow=True).sum()

    if den_total > 0:
        return num_total / den_total
    else:
        return 0.0  # Avoid division by zero

def plot_sample_grid(efficiency_data, title, output_file):
    """
    Generate a grid plot of efficiency vs stau mass.

    Parameters:
        efficiency_data (dict): Dictionary mapping sample names to efficiency values.
        title (str): Plot title.
        output_file (str): Filename for saving the plot.
    """
    # Extract stau masses from sample names
    stau_masses = [int(name.split('_')[1]) for name in efficiency_data.keys()]
    efficiencies = list(efficiency_data.values())

    plt.figure(figsize=(6, 5))
    plt.scatter(stau_masses, efficiencies, marker='o', color='blue', label="Efficiency")

    plt.xlabel("Stau Mass [GeV]")
    plt.ylabel("Efficiency")
    plt.title(title)
    plt.xticks(stau_masses)  # Ensure correct x-axis labels
    plt.ylim(0, 1.2)  # Efficiency should be between 0 and 1
    plt.legend()
    plt.grid(True)

    plt.savefig(output_file)
    plt.close()