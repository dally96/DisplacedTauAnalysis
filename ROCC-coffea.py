import coffea
import uproot
import scipy
import dask
import time
import warnings
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from coffea import processor
from collections import defaultdict
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.dataset_tools import (
    apply_to_fileset,
    max_chunks,
    preprocess,
)

# Prevent branch definition problems
NanoAODSchema.mixins["DisMuon"] = "Muon"
NanoAODSchema.mixins["StauTau"] = "Tau"

# Silence obnoxious warnings
NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore", module="coffea.nanoevents.methods")

# --- DEFINITIONS --- #
max_dr = 0.3
score_granularity = 500

def passing_mask(jets, score):
    return jets['score'] >= score

def get_passing_jets(jets, score):
    return jets[passing_mask(jets, score)]

class BGProcessor(processor.ProcessorABC):
    def __init__(self):
        pass
    
    def process(self,events):
        if len(events) == 0:
            return {}
        dataset        = events.metadata['dataset']
        total_jets     = ak.sum( ak.num(events.Jet) )
        tau_jets       = events.StauTau.nearest(events.Jet, threshold = max_dr)
        match_mask     = ak.num(tau_jets) > 0 
        matched_jets   = ak.zip({
                        "jets": events.Jet[match_mask],
                        "score": events.Jet[match_mask].disTauTag_score1
                        })
        unmatched_jets = ak.zip({
                        "jets": events.Jet[~match_mask],
                        "score": events.Jet[~match_mask].disTauTag_score1
                        })
    

        results = []
        scores = np.linspace(0, 1, score_granularity)
        for s in scores:
            pmj = get_passing_jets(matched_jets, s) # passing matched jets
            pfj = get_passing_jets(unmatched_jets, s) # passing fake jets
            num_pmj = ak.sum( ak.num(pmj) )
            num_pfj = ak.sum( ak.num(pfj) )
            results.append( (dataset, s, num_pmj, num_pfj) )

        return {
            'total_number_jets': total_jets,
            'total_matched_jets': ak.sum( ak.num(matched_jets) ),
            'total_unmatched_jets': ak.sum( ak.num(unmatched_jets) ),
            'set_s_pmj_pfj': results,
            }

    def postprocess(self,accumulator):
        pass

# --- IMPORT DATASETS --- #
with open("root_files.txt") as f:
    lines = [line.strip() for line in f if line.strip()]

xrootd_prefix    = 'root://cmseos.fnal.gov/'
base_prefix_full = '/eos/uscms/store/user/dally/DisplacedTauAnalysis/skimmed_muon_'
base_prefix      = '/eos/uscms'
base_suffix      = '_root:'

paths = []
sets  = []

i = 0 
while i < len(lines):
    if '.root' in lines[i]:
        i += 1
        continue

    base_path = lines[i]
    dataset_name = base_path.removeprefix(base_prefix_full).removesuffix(base_suffix)
    sets.append(dataset_name)
    xrootd_path = base_path.removeprefix(base_prefix).removesuffix(':')

    # Look ahead for .root file
    if i + 1 < len(lines) and '.root' in lines[i + 1]: 
        root_file = lines[i + 1]
        paths.append(xrootd_prefix + xrootd_path + '/' + root_file)
        i += 2  # Move past both lines
    else:
        i += 1  # Only move past base path

fileset = {}

for data in sets:
    matched_paths = [p for p in paths if data in p]
    fileset[data] = {
        "files": {p: "Events" for p in matched_paths}
    }

tstart = time.time()

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
    max_chunks(dataset_runnable, ), # add 10 to run over 10
    schemaclass=NanoAODSchema,
)

(out,) = dask.compute(to_compute)

tprocessor = time.time() - tstart
print(f"{tprocessor} seconds for processor to finish")

# --- ROC Calculations --- #
# Totals
all_matched = sum(
    val["total_matched_jets"]
    for val in out.values()
    if "total_matched_jets" in val
)
all_jets = sum(
    val["total_number_jets"]
    for val in out.values()
    if "total_number_jets" in val
)

print(f"{all_jets} total jets, with {all_matched} matched")

# Aggregation dict: s → [sum of 2nd elements, sum of 3rd elements]

s_sums = defaultdict(lambda: [0, 0])

for entry in out.values():
    if 'set_s_pmj_pfj' not in entry:
        continue
    for s, val2, val3 in entry['set_s_pmj_pfj']:
        s_sums[s][0] += val2
        s_sums[s][1] += val3

thresholds   = []
fake_rates   = []
efficiencies = []

for s, vals in s_sums.items():
    thresholds.append(s)
    efficiency = vals[0] / all_matched
    efficiencies.append(efficiency)
    fake_rate = vals[1] / all_jets
    fake_rates.append(fake_rate)

tcalc = time.time() - tstart - tprocessor

print("sets")
print(sets)
print("Score thresholds:")
print(thresholds)
print("Fake rates:")
print(fake_rates)
print("Efficiencies:")
print(efficiencies)
print(f"{tcalc} seconds for calculations to finish")

# Plot stuff
fig, ax = plt.subplots()
roc = ax.scatter(fake_rates, efficiencies, c=thresholds, cmap='plasma')
cbar = fig.colorbar(roc, ax=ax, label='Score threshold')

#ax.set_xscale("log")
ax.set_xlim(-1e-1, 6e-3)
#ax.set_ylim(0.85, 1.05)

plt.xlabel(r"Fake rate $\left(\frac{fake\_passing\_jets}{total\_jets}\right)$")
plt.ylabel(r"Tau tagger efficiency $\left(\frac{matched\_passing\_jets}{total\_matched\_jets}\right)$")

plt.grid()
plt.savefig('limited-log-skimmed-bg-tau-tagger-roc-scatter.pdf')

tplotting = time.time() - tstart - tprocessor - tcalc
tfinish   = time.time() - tstart
print(f"{tplotting} seconds for plotting to finish")
print(f"{tfinish} seconds total")

