import uproot
import scipy
import awkward as ak
import numpy as np
from coffea import processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, PFNanoAODSchema

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

        # Count leptons per event
        n_leptons = np.array([len(leptons[event]) for event in range(len(leptons))])

        return {
            "n_leptons": n_leptons,
        }

processor_instance = LeptonProcessor(eta_cut=2.4, pt_cut=20)

fname = "/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215454/0000/nanoaod_output_1.root"

events = NanoEventsFactory.from_root(
    {fname: "Events"},
    schemaclass=PFNanoAODSchema,
    metadata={"dataset": "signal"},
    delayed = False).events()

output = processor_instance.process(events)

print(output["n_leptons"])
