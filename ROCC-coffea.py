import coffea
import uproot
import scipy
import dask
import warnings
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from coffea import processor
from coffea.nanoevents.methods import vector
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.dataset_tools import (
    apply_to_fileset,
    max_chunks,
    preprocess,
)

# Prevent lepton veto problems
NanoAODSchema.mixins["DisMuon"] = "Muon"

# Silence obnoxious warning
NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore", module="coffea.nanoevents.methods")

# --- FUNCTIONS & DEFINITIONS --- #
signal_file_name = 'Stau_100_100mm'
score_increment_scale_factor = 500 

def get_bg(collection):
    bg = {}
    for dataset in collection:
        if dataset = signal_file_name:
            continue
        bg.update(collection[dataset][dataset])
    return bg

class BGProcessor(processor.ProcessorABC):
    def __init__(self):
        pass

    def process(self,events):
        dataset = events.metadata['dataset']
        jets = ak.zip(
            {
                "pt": events.Jet.pt,
                "eta": events.Jet.eta,
                "phi": events.Jet.phi,
                "mass": events.Jet.mass,
                "genJetIdx": events.Jet.genJetIdx,
                "partonFlavour": events.Jet.partonFlavour,
                "disTauTag_score1": events.Jet.disTauTag_score1,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )
        return {
            dataset: {
                "jets": cut,
            }
        }
    def postprocess(self,accumulator):
        pass

# Import dataset
fileset = {
    'QCD300_470' : {
        "files" : {
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/skimmed_muon_QCD300_470_root/part0.root': "Events",
            }
    }
    'QCD470_600' : {
        "files" : {
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/skimmed_muon_QCD470_600_root/part0.root': "Events",
            }
    }
    'QCD50_80' : {
        "files" : {
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/skimmed_muon_QCD50_80_root/part05.root': "Events",
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/skimmed_muon_QCD50_80_root/part06.root': "Events",
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/skimmed_muon_QCD50_80_root/part07.root': "Events",
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/skimmed_muon_QCD50_80_root/part08.root': "Events",
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/skimmed_muon_QCD50_80_root/part09.root': "Events",
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/skimmed_muon_QCD50_80_root/part27.root': "Events",
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/skimmed_muon_QCD50_80_root/part29.root': "Events",
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/skimmed_muon_QCD50_80_root/part31.root': "Events",
            }
    }
    'QCD80_120' : {
        "files" : {
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/part096.root': "Events",
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/part097.root': "Events",
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/part099.root': "Events",
        }
    }
    'TTto2L2Nu' : {
        "files" : {
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/skimmed_muon_TTto2L2Nu_root/part0.root': "Events",
        }
    }
    'TTtoLNu2Q' : {
        "files" : {
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/skimmed_muon_TTto2L2Nu_root/part0.root': "Events",
        }
    }
    'Stau_100_100mm' : {
        "files" : {
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/skimmed_muon_Stau_100_100mm_root/part00.root': "Events",
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/skimmed_muon_Stau_100_100mm_root/part01.root': "Events",
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/skimmed_muon_Stau_100_100mm_root/part02.root': "Events",
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/skimmed_muon_Stau_100_100mm_root/part03.root': "Events",
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/skimmed_muon_Stau_100_100mm_root/part04.root': "Events",
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/skimmed_muon_Stau_100_100mm_root/part05.root': "Events",
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/skimmed_muon_Stau_100_100mm_root/part06.root': "Events",
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/skimmed_muon_Stau_100_100mm_root/part07.root': "Events",
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/skimmed_muon_Stau_100_100mm_root/part08.root': "Events",
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/skimmed_muon_Stau_100_100mm_root/part09.root': "Events",
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/skimmed_muon_Stau_100_100mm_root/part10.root': "Events",
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/skimmed_muon_Stau_100_100mm_root/part11.root': "Events",
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/skimmed_muon_Stau_100_100mm_root/part12.root': "Events",
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/skimmed_muon_Stau_100_100mm_root/part13.root': "Events",
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/skimmed_muon_Stau_100_100mm_root/part14.root': "Events",
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/skimmed_muon_Stau_100_100mm_root/part15.root': "Events",
        }
    }
}

dataset_runnable, dataset_updated = preprocess(
    fileset,
    align_clusters=False,
    step_size=100_000,
    files_per_batch=1,
    skip_bad_files=False,
    save_form=False,
)

to_compute = apply_to_fileset(
    BGProcessor(),
    max_chunks(dataset_runnable, 10),
    schemaclass=NanoAODSchema,
)

(cut_jets,) = dask.compute(to_compute)

# --- SIGNAL PROCESSING --- #
signal_events = cut_jets['Stau_100_100mm']['Stau_100_100mm']
stau_taus = signal_events['StauTau']  # h-decay taus with stau parents
cut_signal_jets = signal_events['Jet'] # imported files are precut
matched_tau_jets = stau_taus.nearest(cut_signal_jets, threshold = max_dr) # jets dr-matched to stau_taus
matched_signal_scores = matched_tau_jets.disTauTag_score1

# --- BG PROCESSING --- #
all_bg = get_bg(cut_jets)
bg_scores = cut_bg_jets['TT to 4Q']['TT to 4Q']['jets']['disTauTag_score1']
fake_tau_jets = cut_bg_jets['TT to 4Q']['TT to 4Q']['jets'] # No staus present in bg
matched_bg_scores = bg_scores

# Jet totals
total_matched_tau_jets = ak.sum(ak.num(matched_tau_jets))
total_fake_tau_jets = ak.sum(ak.num(bg_scores))

total_jets = (
    ak.sum(ak.num(cut_signal_jets)) +
    ak.sum(ak.num(bg_scores)) )

# ROC calculations
thresholds = []
fake_rates = []
efficiencies = []

for increment in range(0, score_increment_scale_factor+1):
    threshold = increment / score_increment_scale_factor

    passing_signal_mask = matched_signal_scores >= threshold
    matched_passing_jets = matched_tau_jets[passing_signal_mask]

    passing_bg_mask = matched_bg_scores >= threshold
    fake_passing_jets = fake_tau_jets[passing_bg_mask]

    # --- Totals --- #
    total_matched_passing_jets = ak.sum(ak.num(matched_passing_jets))
    total_fake_passing_jets = ak.sum(ak.num(fake_passing_jets))

    # --- Results --- #
    efficiency = total_matched_passing_jets / total_matched_tau_jets
    fake_rate = total_fake_passing_jets / total_jets

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
ax.set_ylim(0.85, 1.05)

plt.xlabel(r"Fake rate $\left(\frac{fake\_passing\_jets}{total\_jets}\right)$")
plt.ylabel(r"Tau tagger efficiency $\left(\frac{matched\_passing\_jets}{total\_matched\_jets}\right)$")

plt.grid()
plt.savefig('small-RC-TT-4Q-bg-tau-tagger-rocc.pdf')
plt.savefig('small-RC-TT-4Q-bg-tau-tagger-rocc.png')
