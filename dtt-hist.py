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
fname = "/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215454/0000/nanoaod_output_1.root"

# Pass dataset info to coffea object
events = NanoEventsFactory.from_root(
    {fname: "Events"},
    schemaclass=PFNanoAODSchema,
    metadata={"dataset": "signal"},
    delayed = True).events()

# Selection cut parameters
min_pT = 20
max_eta = 2.4
dr_max = 0.3

# Prompt and signal selection
gpart = events.GenPart
prompt_sig_mask = gpart.hasFlags(['isPrompt', 'isLastCopy'])
dr_mask = events.Jet.nearest(gpart, threshold = dr_max) 

# Tau stuff
taus = events.GenPart[events.GenVisTau.genPartIdxMother]
stau_taus = taus[abs(taus.distinctParent.pdgId) == 1000015]
tau_selection = taus[
        (numpy.abs(taus.eta) < max_eta) &
        (taus.pt > min_pT)]

# Jet stuff
tagger_scores = events.Jet.disTauTag_score1
tau_jets = stau_taus.nearest(events.Jet, threshold = dr_max)
matched_scores = tau_jets.disTauTag_score1

# Convert to a flat array for histogram
flattened_scores = ak.flatten(matched_scores, axis=None)

# Define histogram bins
bins = numpy.linspace(0, 1, 50)

# Plot the histogram
plt.hist(flattened_scores, bins=bins, histtype='step', label='Truth Distribution')
plt.xlabel('disTauTag_score1')
plt.ylabel('Frequency')
plt.title('Truth Histogram of disTauTag_score1')
plt.legend()
plt.savefig("tau-truth-hist.png")

# debug
#print("Tau selection \n ", tau_selection)
#print("Tau score \n ", tau_score)
#print("events.Jet.disTauTag_score1 fields \n ", tagger_scores)
#print("GenPart \n", gpart.compute())
#print("Status flags \n", prompt_sig_mask.compute())
