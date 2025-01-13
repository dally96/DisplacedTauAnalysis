import coffea
import uproot
import scipy
import numpy
import awkward as ak
import matplotlib.pyplot as plt
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, PFNanoAODSchema
from coffea.analysis_tools import PackedSelection

# Silence obnoxious warning
NanoAODSchema.warn_missing_crossrefs = False

# Import dataset

fname = "/eos/user/d/dally/DisplacedTauAnalysis/SMS-TStauStau_MStau-100_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_NanoAOD.root" # signal
#fname = "/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_1-1.root" # background


# Pass dataset info to coffea object
events = NanoEventsFactory.from_root(
    {fname: "Events"},
    schemaclass=NanoAODSchema,
    metadata={"dataset": "signal"},
    delayed = False).events()

dr_max = 0.3
threshold = 0.5

taus = events.GenPart[events.GenVisTau.genPartIdxMother]
stau_taus = taus[abs(taus.distinctParent.pdgId) == 1000015]

tau_jets = stau_taus.nearest(events.Jet, threshold = dr_max)
matched_scores = tau_jets.disTauTag_score1
matched_scores_mask = tau_jets.disTauTag_score1 > threshold
passing_tau_jets = tau_jets[matched_scores_mask]

passing_jet_mask = events.Jet.disTauTag_score1 > threshold
passing_jets = events.Jet[passing_jet_mask]
false_jet_mask = 

fake_rate = total_false_passing_jets / total_scores

# Convert to a flat array for histogram
flattened_scores = ak.flatten(matched_scores, axis=None)

# Define histogram bins
bins = numpy.linspace(0, 1, 50)

# Plot the histogram
plt.hist(flattened_scores, bins=bins, histtype='step', label='Truth Distribution')
plt.xlabel('disTauTag_score1')
plt.ylabel('Frequency')
plt.title('Fake Rate Histogram of disTauTag_score1')
plt.legend()
plt.savefig("tau-truth-hist.png")
