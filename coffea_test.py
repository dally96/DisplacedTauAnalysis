import uproot
import scipy
import awkward as ak
import numpy as np
from coffea import processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, PFNanoAODSchema
from coffea.nanoevents.methods.vector import LorentzVector
from hist import Hist
import matplotlib.pyplot as plt
import vector

NanoAODSchema.warn_missing_crossrefs = False

class LeptonProcessor:
    def __init__(self, eta_cut, pt_cut):
        self.eta_cut = eta_cut
        self.pt_cut = pt_cut

    def process(self, events):
        # Select reconstructed leptons (Muons)
        leptons = events.Muon  # or events.Electron
        
        # Access GenPart for matching
        gen_parts = events.GenPart
        
        # Match leptons to GenPart
        gen_match_mask = (gen_parts.statusFlags & (1 << 13) != 0) & (gen_parts.statusFlags & (1 << 0) != 0)
        matched_gen_parts = gen_parts[gen_match_mask]
        
        # Match reconstructed leptons with generator particles
        gen_dr = leptons.nearest(matched_gen_parts, axis=1, return_metric=True)
        matched_leptons = gen_dr[1] < 0.1  # deltaR threshold

        leptons = leptons[matched_leptons]

        # Apply eta and pT cuts
        leptons = leptons[(abs(leptons.eta) < self.eta_cut) & (leptons.pt > self.pt_cut)]

        ### --- debugging --- ###
        # Count leptons per event
        n_leptons = np.array([len(leptons[event]) for event in range(len(leptons))])

        return leptons

processor_instance = LeptonProcessor(eta_cut=2.4, pt_cut=20)

fname = "/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215454/0000/nanoaod_output_1.root"

events = NanoEventsFactory.from_root(
    {fname: "Events"},
    schemaclass=PFNanoAODSchema,
    metadata={"dataset": "signal"},
    delayed = False).events()

output = processor_instance.process(events)

### --- debugging --- ###
# print(output["n_leptons"])
print(output)

# Scoring
gen_parts = events.GenPart
gen_jets = events.GenJet

# True labels using GenPart (prompt leptons)
is_prompt = (gen_parts.statusFlags & (1 << 13) != 0) & (gen_parts.statusFlags & (1 << 0) != 0)
true_prompt = gen_parts[is_prompt]

# Match leptons to true prompts
dr2_prompt = output.metric_table(true_prompt, return_metric=True)[1] ** 2
dr2_prompt_matched = np.min(dr2_prompt, axis=1) < 0.01  # Adjust threshold as needed

# Match leptons to background jets
dr2_jet = output.metric_table(gen_jets, return_metric=True)[1] ** 2
dr2_jet_matched = np.min(dr2_jet, axis=1) < 0.01  # Adjust threshold as needed

# True labels
labels = dr2_prompt_matched  # 1 for prompt leptons, 0 for background

# Delta R^2 scores for ROC
scores = np.min(dr2_prompt, axis=1)  # Use delta R^2 as the score
# --- end scoring

# Histograms for ROC computation
bins = np.linspace(0, 0.5, 50)  
hist_prompt = Hist.new.Reg(bins.size - 1, bins.min(), bins.max(), name="score").Double()
hist_background = Hist.new.Reg(bins.size - 1, bins.min(), bins.max(), name="score").Double()

hist_prompt.fill(score=scores[labels == 1])
hist_background.fill(score=scores[labels == 0])

# Calculate TPR and FPR
cumulative_prompt = np.cumsum(hist_prompt.view()[::-1])[::-1]
cumulative_background = np.cumsum(hist_background.view()[::-1])[::-1]
tpr = cumulative_prompt / cumulative_prompt[0]  # True positive rate
fpr = cumulative_background / cumulative_background[0]  # False positive rate

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label="ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.savefig("ROC-curve.pdf")
