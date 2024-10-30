import coffea
import uproot
import scipy
import matplotlib as mpl
import awkward as ak
import numpy as np
import math
# import ROOT
import array
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema, PFNanoAODSchema

coffea.__version__

from hist import Hist
import hist
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 

from hist import intervals
import hist
## adapted from 
## https://github.com/ikrommyd/egamma-tnp/blob/babb73aa7e3654e1cfca5fe8172031e7df7f5071/src/egamma_tnp/utils/histogramming.py#L26
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


# Dictionary for histograms and binnings
num_hist_dict = {}
den_hist_dict = {}

## variables as a function of which I'll plot the efficiency, and binning definition
var_list = ['pt', 'lxy', 'eta', 'dz']#, 'dm']
bins = [0, 0.001, 0.01, 1, 5, 10, 15, 20, 25, 30, 40, 50, 100]
pt_bins = [0, 20, 22, 24, 27, 30, 35, 40, 45, 50, 60, 80, 100, 120]
bins_dz = [0, 0.01, 0.3, 1, 5, 10, 15, 20, 25, 30, 40, 50]
bins_dz_zoom = [0, 0.01, 0.1, 0.3, 0.5, 1]

var_axes = {
    'pt': hist.axis.Variable(pt_bins, flow=False, name="Denominator, pt"),
    'lxy': hist.axis.Variable(bins, flow=False, name="lxy (cm)"),
    'eta': hist.axis.Regular(25, -2.5, 2.5, flow=False, name="eta"),
    'dz': hist.axis.Variable(bins_dz, flow=False, name="dz (cm)"),
}


def fill_efficiency_ingredients(filename):
    jet_type = filename.split('/')[0]
    all_events = NanoEventsFactory.from_root({filename: "Events"}, 
                                         schemaclass=PFNanoAODSchema,
                                         metadata={"dataset": "MC"},
                                         delayed = False).events()

    events = all_events
    print ('sample from ', filename, 'has', len(events), ' events')

    ## find staus and their tau children  
    gpart = events.GenPart
    events['staus'] = gpart[ (abs(gpart.pdgId) == 1000015) & (gpart.hasFlags("isLastCopy"))]

    events['staus_taus'] = events.staus.distinctChildren[ (abs(events.staus.distinctChildren.pdgId) == 15) & \
                                                          (events.staus.distinctChildren.hasFlags("isLastCopy")) & \
                                                          (events.staus.distinctChildren.pt > 5) \
                                                         ]
    staus_taus = events['staus_taus']

    ## first remove events with for example 3 taus in the final state (all from staus, not expected)
    mask_tauh = ak.any(abs(staus_taus.distinctChildren.pdgId)==211, axis=-1) 
    one_tauh_evt = (ak.sum(mask_tauh, axis = 1) > 0) & (ak.sum(mask_tauh, axis = 1) < 3)
    flat_one_tauh_evt = ak.flatten(one_tauh_evt, axis=0)
    one_tauh_mask = ak.sum(flat_one_tauh_evt, axis = 1) < 2  
    events = events[one_tauh_mask]

    ## now filter on events with one or two tau-h
    mask_tauh = ak.any(abs(events.staus_taus.distinctChildren.pdgId)==211, axis=-1) 
    one_tauh_evt = (ak.sum(mask_tauh, axis = 1) > 0) & (ak.sum(mask_tauh, axis = 1) < 3)
    one_tauh_evt = ak.any(one_tauh_evt, axis=1)
    t_events = events[one_tauh_evt]
    print ('n one-tau_h events:', len(t_events),  '(%.3f)' %(len(t_events)/len(events)))

    ## Acceptance cuts
    staus_taus = t_events.staus_taus
    mask_acceptance = ak.any( (abs(staus_taus.eta) < 2.4) & (staus_taus.pt > 20) , axis=-1) 
    acc_tauh_evt = ak.sum(mask_acceptance, axis = 1) >= 1
    ## acc_tau_evt is already flat
    a_events = t_events[acc_tauh_evt]
    print ('n tau accepted events:', len(a_events),  '(%.3f)' %(len(a_events)/len(t_events)))


    ##now find jets matched to gen tau_h
    jets = a_events.Jet[(abs(a_events.Jet.eta) < 2.4) & (a_events.Jet.pt > 20) ]
    staus_taus = a_events.staus_taus
    gen_taus = staus_taus[(abs(staus_taus.eta) < 2.4) & (staus_taus.pt > 20) ]
    gen_taus['lxy'] = np.sqrt((gen_taus.distinctParent.vx - gen_taus.vx) ** 2 + \
                              (gen_taus.distinctParent.vy - gen_taus.vy) ** 2 )

    gen_taus['dz'] = np.abs((gen_taus.distinctParent.vz - gen_taus.vz) )
    
    ## reduce one dimension 
    gen_taus = ak.flatten(gen_taus, axis=2)
    
    gen_taus_jet_matched = jets.nearest(gen_taus) ## this will have the lenght of the jets
    ##compute the delta_R
    jets.delta_r(gen_taus_jet_matched)
    ##select jets with delta_R to gen taus < 0.4 
    jets = jets[jets.delta_r(gen_taus_jet_matched) < 0.4]

    ## now re-do the association with GEN taus
    ## to only keep the GEN taus matched to reco jets
    gen_taus_jet_matched_again = jets.nearest(gen_taus)
    pt_gen_taus_jet_matched_again = gen_taus_jet_matched_again.pt
    ak.drop_none(pt_gen_taus_jet_matched_again)

    for var in var_list:
        ax = var_axes[var]
        # ax = hist.axis.Regular(100, 0, 100, flow=False, name=var)
        full_hist = Hist(ax)
        full_hist.fill(ak.flatten(getattr(gen_taus,var)))
    
        reco_hist = Hist(ax)
        reco_hist.fill(ak.flatten(getattr(gen_taus_jet_matched_again,var)))

        if jet_type not in num_hist_dict.keys():
            num_hist_dict[jet_type] = {}
            num_hist_dict[jet_type][var] = []
        
            den_hist_dict[jet_type] = {}
            den_hist_dict[jet_type][var] = []

        if var not in num_hist_dict[jet_type].keys():
            num_hist_dict[jet_type][var] = []
            den_hist_dict[jet_type][var] = []

        num_hist_dict[jet_type][var].append(reco_hist)
        den_hist_dict[jet_type][var].append(full_hist)


fill_efficiency_ingredients("chs/stau_m300_ctau100_nano_39_0.root")
fill_efficiency_ingredients("puppi/stau_m300_ctau100_nano_39_0.root")
fill_efficiency_ingredients("puppiV18/stau_m300_ctau100_nano_39_0.root")

### now start plotting
plt.xscale('log')
plot_efficiency(num_hist_dict['puppiV18']['lxy'][0], den_hist_dict['puppi']['lxy'][0], label='puppi v18')
plot_efficiency(num_hist_dict['puppi']['lxy'][0], den_hist_dict['puppi']['lxy'][0], label='puppi v17')
plot_efficiency(num_hist_dict['chs']['lxy'][0], den_hist_dict['chs']['lxy'][0], label='chs')
plt.legend()
plt.savefig('eff_vs_lxy.pdf')


plt.clf()
plot_efficiency(num_hist_dict['puppiV18']['eta'][0], den_hist_dict['puppi']['eta'][0], label='puppi v18')
plot_efficiency(num_hist_dict['puppi']['eta'][0], den_hist_dict['puppi']['eta'][0], label='puppi v17')
plot_efficiency(num_hist_dict['chs']['eta'][0], den_hist_dict['chs']['eta'][0], label='chs')
plt.legend()
plt.savefig('eff_vs_eta.pdf')

plot_efficiency(num_hist_dict['puppiV18']['pt'][0], den_hist_dict['puppi']['pt'][0], label='puppi v18')
plot_efficiency(num_hist_dict['puppi']['pt'][0], den_hist_dict['puppi']['pt'][0], label='puppi v17')
plot_efficiency(num_hist_dict['chs']['pt'][0], den_hist_dict['chs']['pt'][0], label='chs')
plt.legend()
plt.savefig('eff_vs_pt.pdf')

plt.clf()
plot_efficiency(num_hist_dict['puppiV18']['dz'][0], den_hist_dict['puppi']['dz'][0], label='puppi v18')
plot_efficiency(num_hist_dict['puppi']['dz'][0], den_hist_dict['puppi']['dz'][0], label='puppi v17')
plot_efficiency(num_hist_dict['chs']['dz'][0], den_hist_dict['chs']['dz'][0], label='chs')
plt.legend()
plt.savefig('eff_vs_dz.pdf')
