import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema
from hist import Hist, axis, intervals
import hist
import os

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
    gen_taus_flat_dxy = ak.flatten(gen_taus.dxy, axis=1).compute()
    gen_taus_flat_pt = ak.flatten(gen_taus.pt, axis=1).compute()

    gen_taus_matched_by_pt_flat_dxy = ak.flatten(gen_taus_matched_by_pt.dxy, axis=1).compute()
    gen_taus_matched_by_pt_flat_pt = ak.flatten(gen_taus_matched_by_pt.pt, axis=1).compute()

    return (gen_taus_flat_dxy, gen_taus_flat_pt,
            gen_taus_matched_by_pt_flat_dxy, gen_taus_matched_by_pt_flat_pt)

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

def plot_pt_efficiency(gen_taus_flat_pt, gen_taus_matched_by_pt_flat_pt, output_dir, sample_name):
    """
    Create and save pt efficiency plots for:
      - Overall pt efficiency.
      - Zoomed pt efficiency (pt between 100 and 250 GeV).
    """
    pt_bins_low = np.arange(20, 101, 20)
    pt_bins_med = np.arange(100, 251, 10)
    pt_bins_high = np.arange(250, 401, 20)
    pt_bins_vhigh = np.arange(400, 701, 50)
    pt_bins_eff = np.unique(np.concatenate([pt_bins_low, pt_bins_med, pt_bins_high, pt_bins_vhigh]))
    pt_bins_zoom = np.arange(100, 251, 5)
    
    var_axes = {'pt': axis.Variable(pt_bins_eff, flow=False, name="tauh_pt")}
    var_axes_zoom = {'pt': axis.Variable(pt_bins_zoom, flow=False, name="tauh_pt_zoom")}
    
    den_hist = Hist(var_axes['pt'])
    num_hist = Hist(var_axes['pt'])
    den_hist_zoom = Hist(var_axes_zoom['pt'])
    num_hist_zoom = Hist(var_axes_zoom['pt'])
    
    den_hist.fill(gen_taus_flat_pt)
    num_hist.fill(gen_taus_matched_by_pt_flat_pt)
    den_hist_zoom.fill(gen_taus_flat_pt[(gen_taus_flat_pt >= 100) & (gen_taus_flat_pt <= 250)])
    num_hist_zoom.fill(gen_taus_matched_by_pt_flat_pt[(gen_taus_matched_by_pt_flat_pt >= 100) & (gen_taus_matched_by_pt_flat_pt <= 250)])
    
    # Plot overall pt efficiency
    plt.clf()
    plot_efficiency(num_hist, den_hist, label='Leading pT Jet Matching Efficiency (pt)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"eff_vs_pt_leading_pT_{sample_name}.pdf"))
    
    # Plot zoom pt efficiency
    plt.clf()
    plot_efficiency(num_hist_zoom, den_hist_zoom, label='Leading pT Jet Matching Efficiency (pt Zoom)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"eff_vs_pt_zoom_leading_pT_{sample_name}.pdf"))

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
    dxy_bins_2d = np.linspace(-20, 20, 41)
    pt_bins_2d = np.linspace(0, 500, 51)
    # Create a 2D histogram with dxy on x-axis and pt on y-axis.
    h2d = hist.Hist.new(
        "gen_tau_dxy_vs_pt",
        hist.Bin("dxy", "gen_tau_dxy [cm]", dxy_bins_2d),
        hist.Bin("pT", "gen_tau_pT [GeV]", pt_bins_2d)
    )
    
    h2d.fill(dxy=gen_taus_flat_dxy, pT=gen_taus_flat_pt)
    # Use built-in plot2d (or convert to numpy arrays for a custom plot)
    h2d.plot2d()
    plt.savefig(os.path.join(output_dir, f"2d_gen_tau_dxy_vs_pt_{sample_name}.pdf"))
    plt.close()

def plot_jet_score_efficiency(leading_score_jets, jet_matched_gen_taus_score, label_prefix, file_suffix):
    """
    Creates efficiency plots (using plot_efficiency) and separate normalized plots
    for the jet score histograms.
    
    Normalization is done by dividing each bin by the total number of jets.
    """
    # Define jet score axis (assumed to be defined globally)
    global jet_score_axis  # jet_score_axis should be defined before calling this function
    h_den = Hist(jet_score_axis)
    h_num = Hist(jet_score_axis)
    
    jet_scores = ak.flatten(leading_score_jets.disTauTag_score1, axis=1).compute()
    matched_jet_scores = ak.flatten(jet_matched_gen_taus_score.disTauTag_score1, axis=1).compute()
    
    h_den.fill(jet_scores)
    h_num.fill(matched_jet_scores)
    
    plt.clf()
    plot_efficiency(h_num, h_den, label=f"{label_prefix} Jet Score Efficiency")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"eff_vs_jet_score_{file_suffix}.pdf"))
    plt.close()
    
    # For normalized standalone plots, extract bin contents and edges.
    den_vals = h_den.values(flow=True)
    num_vals = h_num.values(flow=True)
    total_den = den_vals.sum()
    total_num = num_vals.sum()
    norm_den_vals = den_vals / total_den if total_den > 0 else den_vals
    norm_num_vals = num_vals / total_num if total_num > 0 else num_vals
    edges = np.array(jet_score_axis.edges)
    
    fig_den, ax_den = plt.subplots(figsize=(8, 6))
    ax_den.step(edges[:-1], norm_den_vals, where='post', label=f"All {label_prefix} Jets (Normalized)", color="blue")
    ax_den.set_xlabel("disTauTag score")
    ax_den.set_ylabel("Normalized Counts")
    ax_den.legend()
    ax_den.set_title(f"Jet Score - Denom ({label_prefix} Jets)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"jet_score_denom_{file_suffix}.pdf"))
    plt.close(fig_den)
    
    fig_num, ax_num = plt.subplots(figsize=(8, 6))
    ax_num.step(edges[:-1], norm_num_vals, where='post', label=f"{label_prefix} Matched Jets (Normalized)", color="red")
    ax_num.set_xlabel("disTauTag score")
    ax_num.set_ylabel("Normalized Counts")
    ax_num.legend()
    ax_num.set_title(f"Jet Score - Numerator ({label_prefix} Matched Jets)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"jet_score_num_{file_suffix}.pdf"))
    plt.close(fig_num)
