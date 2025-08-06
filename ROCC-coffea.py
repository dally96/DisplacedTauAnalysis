import coffea
import pickle
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

lifetimes = ['1mm', '10mm', '100mm', '1000mm']
masses    = ['100', '200', '300', '500']

stau_colors = {'100': '#EA7AF4', '200': '#B43E8F', '300': '#6200B3', '500': '#218380'}

def passing_mask(jets, score):
    return jets['score'] >= score

def get_passing_jets(jets, score):
    return jets[passing_mask(jets, score)]

def delta_r_mask(first: ak.highlevel.Array, second: ak.highlevel.Array, threshold: float) -> ak.highlevel.Array:
    mval = first.metric_table(second)
    return ak.all(mval > threshold, axis=-1)

class BGProcessor(processor.ProcessorABC):
    def __init__(self):
        pass
    
    def process(self,events):
        if len(events) == 0:
            return {}

        dataset        = events.metadata['dataset']
        print(f"Starting {dataset}")
        # Determine if dataset is MC or Data
        is_MC = True if "Stau" in dataset else False

        if is_MC:
            vx = events.GenVisTau.parent.vx - events.GenVisTau.parent.parent.vx
            vy = events.GenVisTau.parent.vy - events.GenVisTau.parent.parent.vy
            Lxy = np.sqrt(vx**2 + vy**2)
            events["GenVisTau"] = ak.with_field(events.GenVisTau, Lxy, where="lxy")            

        events["GenVisStauTau"] = events.GenVisTau[(abs(events.GenVisTau.parent.distinctParent.pdgId) == 1000015)   &\
                                               (events.GenVisTau.pt > 20)                                       &\
                                               (abs(events.GenVisTau.eta) < 2.4)                                &\
                                               (events.GenVisTau.parent.distinctParent.hasFlags("isLastCopy"))  &\
                                               (events.GenVisTau.parent.hasFlags("fromHardProcess"))
                                              ]

        if is_MC: 
             events["GenVisTau"] = events.GenVisTau[(abs(events.GenVisTau.lxy) < 100)]

        total_jets     = ak.sum( ak.num(events.Jet) )
        tau_jets       = events.GenVisTau.nearest(events.Jet, threshold = max_dr)
        match_mask     = ak.num(tau_jets) > 0 
        matched_jets   = ak.zip({
                        "jets": tau_jets,
                        "score": tau_jets.disTauTag_score1
                        })
        unmatched_jets = ak.zip({
                        "jets": events.Jet[delta_r_mask(events.Jet, events.GenVisTau, 0.3)],
                        "score": events.Jet[delta_r_mask(events.Jet, events.GenVisTau, 0.3)].disTauTag_score1
                        })
    

        results = []
        scores = np.linspace(0, 1, score_granularity)
        #print(f"scores is {scores}")
        for s in scores:
            pmj = get_passing_jets(matched_jets, s) # passing matched jets
            pfj = get_passing_jets(unmatched_jets, s) # passing fake jets
            num_pmj = ak.sum( ak.num(pmj) )
            #print(f"num_pmj is {num_pmj.compute()}")
            num_pfj = ak.sum( ak.num(pfj) )
            #print(f"num_pfj is {num_pfj.compute()}")
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
#with open("root_files.txt") as f:
#    lines = [line.strip() for line in f if line.strip()]
#
#xrootd_prefix    = 'root://cmseos.fnal.gov/'
#base_prefix_full = '/eos/uscms/store/user/dally/DisplacedTauAnalysis/skimmed_muon_'
#base_prefix      = '/eos/uscms'
#base_suffix      = '_root:'
#
#paths = []
#sets  = []
#
#i = 0 
#while i < len(lines):
#    if '.root' in lines[i]:
#        i += 1
#        continue
#
#    base_path = lines[i]
#    dataset_name = base_path.removeprefix(base_prefix_full).removesuffix(base_suffix)
#    sets.append(dataset_name)
#    xrootd_path = base_path.removeprefix(base_prefix).removesuffix(':')
#
#    # Look ahead for .root file
#    if i + 1 < len(lines) and '.root' in lines[i + 1]: 
#        root_file = lines[i + 1]
#        paths.append(xrootd_prefix + xrootd_path + '/' + root_file)
#        i += 2  # Move past both lines
#    else:
#        i += 1  # Only move past base path
#
#fileset = {}
#
#for data in sets:
#    matched_paths = [p for p in paths if data in p]
#    fileset[data] = {
#        "files": {p: "Events" for p in matched_paths}
#    }
#
tstart = time.time()
#
#dataset_runnable, dataset_updated = preprocess(
#    fileset,
#    align_clusters=False,
#    step_size=100_000,
#    files_per_batch=1,
#    skip_bad_files=False,
#    save_form=False,
#)

all_matched = 0

all_jets = 0

# Aggregation dict: s → [sum of 2nd elements, sum of 3rd elements]

s_sums = defaultdict(lambda: [0])

dataset_runnable = {}
stau_dict = {}

with open("second_skim_preprocessed_fileset.pkl", "rb") as f:
    sub_dataset_runnable = pickle.load(f)
    for dataset in sub_dataset_runnable.keys():
        if "JetMET" in dataset: continue
        dataset_runnable[dataset] = sub_dataset_runnable[dataset] 
for samp in dataset_runnable.keys():
    samp_runnable = {}
    samp_runnable[samp] = dataset_runnable[samp]
    
    to_compute = apply_to_fileset(
        BGProcessor(),
        max_chunks(samp_runnable, ), # add 10 to run over 10
        schemaclass=NanoAODSchema,
    )

    (out, ) = dask.compute(to_compute)

    if "Stau" in samp:
        lifetime = samp.split('_')[-1]
        mass     = samp.split('_')[-2]
        if lifetime in stau_dict.keys():
            stau_dict[lifetime][mass] = {}
        else:
            stau_dict[lifetime] = {}
            stau_dict[lifetime][mass] = {}

        passed_matched_jets =  {}
        for score_list in out[samp]['set_s_pmj_pfj']:
            passed_matched_jets[score_list[1]] = score_list[2]
            stau_dict[lifetime][mass][score_list[1]] = passed_matched_jets[score_list[1]]/out[samp]["total_matched_jets"]  
         
    all_jets    += out[samp]["total_number_jets"]

    if 'set_s_pmj_pfj' not in out[samp]: continue
    for score_list in out[samp]['set_s_pmj_pfj']:
        s_sums[score_list[1]][0] += score_list[3]
    
    
tprocessor = time.time() - tstart
print(f"{tprocessor} seconds for processor to finish")

'''
# --- ROC Calculations --- #
# Totals
all_matched = sum(
    val["total_matched_jets"]
    for val in out.values()
    if "total_matched_jets" in val
)
print(f"all_matched is {all_matched}")
all_jets = sum(
    val["total_number_jets"]
    for val in out.values()
    if "total_number_jets" in val
)
print(f"all_jets is {all_jets}")
print(f"{all_jets} total jets, with {all_matched} matched")
'''

thresholds   = []
fake_rates   = []

for lifetime in lifetimes:
    for mass in masses:
        stau_dict[lifetime][mass]['eff'] = []

for s, vals in s_sums.items():
    thresholds.append(s)
    fake_rate = vals[0] / all_jets
    fake_rates.append(fake_rate)
    for lifetime in lifetimes:
        for mass in masses:
            stau_dict[lifetime][mass]['eff'].append(stau_dict[lifetime][mass][s])

tcalc = time.time() - tstart - tprocessor

#print("sets")
#print(sets)
#print("Score thresholds:")
#print(thresholds)
#print("Fake rates:")
#print(fake_rates)
#print("Efficiencies:")
#print(efficiencies)
print(f"{tcalc} seconds for calculations to finish")

# Plot stuff
for lifetime in lifetimes:
    fig, ax = plt.subplots()
    roc = {}
    for mass in masses:
        roc[mass] = ax.plot(fake_rates, stau_dict[lifetime][mass]['eff'], color = stau_colors[mass], label = mass + ' GeV')

    plt.xlabel(r"Fake rate $\left(\frac{fake\_passing\_jets}{total\_jets}\right)$")
    plt.ylabel(r"Tau tagger efficiency $\left(\frac{matched\_passing\_jets}{total\_matched\_jets}\right)$")
    plt.title(f"{lifetime}")
    plt.legend(loc='bottom right')
    
    #plt.grid()
    plt.savefig(f'limited-log-skimmed-bg-tau-tagger-roc-scatter-{lifetime}.pdf')
    
#cbar = fig.colorbar(roc, ax=ax, label='Score threshold')

#ax.set_xscale("log")
#ax.set_xlim(-1e-1, 6e-3)
#ax.set_ylim(0.85, 1.05)


tplotting = time.time() - tstart - tprocessor - tcalc
tfinish   = time.time() - tstart
print(f"{tplotting} seconds for plotting to finish")
print(f"{tfinish} seconds total")

