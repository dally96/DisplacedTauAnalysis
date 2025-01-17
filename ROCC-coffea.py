import coffea
import uproot
import scipy
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

# Silence obnoxious warning
NanoAODSchema.warn_missing_crossrefs = False

# Import dataset
signal_fname = "/eos/user/d/dally/DisplacedTauAnalysis/SMS-TStauStau_MStau-100_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_NanoAOD.root" # signal
bg_fname = "/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_1-1.root" # background

# Pass dataset info to coffea objects
signal_events = NanoEventsFactory.from_root(
    {signal_fname: "Events"},
    schemaclass=NanoAODSchema,
    metadata={"dataset": "signal"},
    delayed = False).events()
bg_events = NanoEventsFactory.from_root(
    {bg_fname: "Events"},
    schemaclass=NanoAODSchema,
    metadata={"dataset": "background"},
    delayed = False).events()

# Selection cut parameters
min_pT = 20
max_eta = 2.4
max_dr = 0.3
max_lep_dr = 0.4
score_increment_scale_factor = 4

def delta_r_mask(first: ak.highlevel.Array, second: ak.highlevel.Array, threshold: float) -> ak.highlevel.Array:
            mval = first.metric_table(second)
            return ak.all(mval > threshold, axis=-1)

def apply_cuts(collection):
    pt_mask = collection.pt >= min_pT
    eta_mask = abs(collection.eta) < max_eta
    valid_mask = collection.genJetIdx > 0
    
    collection_length = ak.sum(ak.num(collection.partonFlavour))
    inclusion_mask = collection.genJetIdx < collection_length
    
    cut_collection = collection[
        pt_mask & eta_mask & valid_mask & inclusion_mask ]
    
    return cut_collection

def apply_lepton_veto(evt_collection: ak.highlevel.Array):
    evt_collection['Jet'] = evt_collection.Jet[
        delta_r_mask(evt_collection.Jet, evt_collection.Photon, max_lep_dr) ]
    evt_collection['Jet'] = evt_collection.Jet[
        delta_r_mask(evt_collection.Jet, evt_collection.Electron, max_lep_dr) ]
    evt_collection['Jet'] = evt_collection.Jet[
        delta_r_mask(evt_collection.Jet, evt_collection.DisMuon, max_lep_dr) ]
    return evt_collection

# Signal processing
taus = signal_events.GenPart[signal_events.GenVisTau.genPartIdxMother] # hadronically-decaying taus
stau_taus = taus[abs(taus.distinctParent.pdgId) == 1000015] # h-decay taus with stau parents
#cut_signal_jets = apply_cuts(apply_lepton_veto(signal_events).Jet)
cut_signal = apply_lepton_veto(signal_events)
cut_signal_jets = apply_cuts(cut_signal.Jet)
true_tau_jets = stau_taus.nearest(cut_signal_jets, threshold = max_dr) # jets dr-matched to stau_taus
matched_signal_scores = true_tau_jets.disTauTag_score1

# Background processing
bg_jets = bg_events.Jet
cut_bg_jets = apply_cuts(bg_jets)
bg_scores = cut_bg_jets.disTauTag_score1
false_tau_jets = cut_bg_jets # No staus present in bg
matched_bg_scores = bg_scores

# Jet totals
total_true_tau_jets = ak.sum(ak.num(true_tau_jets))
total_false_tau_jets = ak.sum(ak.num(false_tau_jets))

total_jets = (
    ak.sum(ak.num(cut_signal_jets)) +
    ak.sum(ak.num(cut_bg_jets)) )

# ROC calculations
thresholds = []
fake_rates = []
efficiencies = []

for increment in range(0, score_increment_scale_factor+1):
    threshold = increment / score_increment_scale_factor

    passing_signal_mask = matched_signal_scores >= threshold
    true_passing_jets = true_tau_jets[passing_signal_mask]

    passing_bg_mask = matched_bg_scores >= threshold
    false_passing_jets = false_tau_jets[passing_bg_mask]

    # --- Totals --- #
    total_true_passing_jets = ak.sum(ak.num(true_passing_jets))
    total_false_passing_jets = ak.sum(ak.num(false_passing_jets))

    # --- Results --- #
    efficiency = total_true_passing_jets / total_true_tau_jets
    fake_rate = total_false_passing_jets / total_jets

    thresholds.append(threshold)
    fake_rates.append(fake_rate)
    efficiencies.append(efficiency)

# Helper function for plot colormap
def colored_line(x, y, c, ax, **lc_kwargs):
    """ 
    From https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html#sphx-glr-gallery-lines-bars-and-markers-multicolored-line-py
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)
    
    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)

# Plot stuff
fig, ax = plt.subplots()
color = np.linspace(0, 1, len(thresholds))
roc = colored_line(fake_rates, efficiencies, color, ax, linewidth=2, cmap='plasma')
cbar = fig.colorbar(roc)
cbar.set_label('Score threshold')

ax.set_xscale("log")
#ax.set_ylim(0.7, 1.0)

plt.xlabel(r"Fake rate $\left(\frac{false\_passing\_jets}{total\_jets}\right)$")
plt.ylabel(r"Tau tagger efficiency $\left(\frac{true\_passing\_jets}{total\_true\_jets}\right)$")

plt.grid()
plt.savefig('pt-eta-cut-scaled-tau-tagger-rocc.pdf')
plt.savefig('pt-eta-cut-scaled-tau-tagger-rocc.png')
