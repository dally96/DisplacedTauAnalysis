import numpy as np
import awkward as ak
from coffea import processor
from coffea.processor import Runner
from coffea.nanoevents import NanoAODSchema
import matplotlib.pyplot as plt

deltaR = 0.4
score_resolution = 500
score_thresholds = np.linspace(0, 1, score_resolution)

class TauTaggerProcessor(processor.ProcessorABC):
    def __init__(self, score_thresholds, dr_match=deltaR):
        self.score_thresholds = score_thresholds
        self.dr_match = dr_match

    def process(self, events):
        jets = events.Jet
        gen = events.GenPart

        # Select taus from staus
        stau_taus = gen[
            (abs(gen.pdgId) == 15)
            & gen.hasFlags(["isLastCopy"])
            & (abs(gen.mother(0).pdgId) == 1000015)
        ]

        if len(jets) == 0 or len(stau_taus) == 0:
            return {
                "eff": np.zeros(len(self.score_thresholds)),
                "fake": np.zeros(len(self.score_thresholds)),
                "counts": 0,
            }

        # Jet-tau matching
        tau_eta = ak.broadcast_arrays(jets.eta, stau_taus.eta)[1]
        tau_phi = ak.broadcast_arrays(jets.phi, stau_taus.phi)[1]
        dphi = (jets.phi - tau_phi + np.pi) % (2 * np.pi) - np.pi
        deta = jets.eta - tau_eta
        dR = np.sqrt(deta**2 + dphi**2)

        matched = ak.any(dR < self.dr_match, axis=-1)

        score = jets.disTauTag_score1
        total_matched = ak.sum(matched)
        total_jets = len(jets)

        eff = []
        fake = []

        for t in self.score_thresholds:
            passing = score > t
            matched_passing = passing & matched
            fake_passing = passing & ~matched

            eff.append(
                ak.sum(matched_passing) / total_matched if total_matched > 0 else 0
            )
            fake.append(ak.sum(fake_passing) / total_jets if total_jets > 0 else 0)

        return {
            "eff": np.array(eff),
            "fake": np.array(fake),
            "counts": 1,
        }

    def postprocess(self, accumulator):
        return accumulator

# Load files
with open("rocc-filelist.txt") as f:
    files = [line.strip() for line in f if line.strip()]

# Basic runner
runner = Runner(
    executor=processor.futures_executor,
    executor_args={"workers": 4, "schema": NanoAODSchema},
    chunksize=100_000,  # tune as needed
)

# Run processor
output = runner(
    files,
    treename="Events",
    processor_instance=TauTaggerProcessor(score_thresholds=thresholds),
    executor_args={
        "schema": NanoAODSchema,
        "client": client,
    },
)

# ROC calculations
thresholds = score_thresholds
fake_rates = output["fake"]
efficiencies = output["eff"]

# Plot stuff
fig, ax = plt.subplots()
roc = ax.scatter(fake_rates, efficiencies, c=thresholdes, cmap='plasma')
cbar = fig.colorbar(roc, ax=ax, label='Score threshold')

ax.set_xscale("log")
ax.set_ylim(0.85, 1.05)

plt.xlabel(r"Fake rate $\left(\frac{fake\_passing\_jets}{total\_jets}\right)$")
plt.ylabel(r"Tau tagger efficiency $\left(\frac{matched\_passing\_jets}{total\_matched\_jets}\right)$")

plt.grid()
plt.savefig('RC-skimmed-bg-tau-tagger-roc-scatter.pdf')
