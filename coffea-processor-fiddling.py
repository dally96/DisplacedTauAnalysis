import coffea
import uproot
import scipy
import dask
import time
import warnings
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea import processor
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
score_granularity = 5

def passing_mask(jets, score):
    return jets['score'] >= score

def get_fake_rate(func_unmatched_jets, all_jets, score):
    passing_jets  = func_unmatched_jets[passing_mask(func_unmatched_jets, score)]
    total_passing = ak.sum( ak.num(passing_jets) )
    return total_passing / all_jets

def get_efficiency(func_matched_jets, score): 
    passing_jets   = func_matched_jets[passing_mask(func_matched_jets, score)]
    total_passing  = ak.sum( ak.num(passing_jets) )
    number_matched = ak.sum( ak.num(func_matched_jets) )
    return total_passing / number_matched

class BGProcessor(processor.ProcessorABC):
    def __init__(self):
        pass
    
    def process(self,events):
        if len(events) == 0:
            return {}
        dataset        = events.metadata['dataset']
        total_jets     = ak.sum( ak.num(events.Jet) )
        matched_jets   = ak.zip({"score": events.StauTau.nearest(events.Jet, threshold = max_dr).disTauTag_score1},)
        unmatched_jets = ak.zip({"score": events.Jet[ak.min( events.Jet.metric_table(events.StauTau), axis=-1 ) > (max_dr**2)].disTauTag_score1},)

        results = []
        scores = np.linspace(0, 1, score_granularity)
        for s in scores:
            fr = get_fake_rate(unmatched_jets, total_jets, s)
            eff = get_efficiency(matched_jets, s)
            results.append((s, fr, eff))

        return {
            's-fr-eff': results,
            }

    def postprocess(self,accumulator):
        pass

# --- IMPORT DATASETS --- #
fileset = {
    'QCD300_470' : {
        "files" : {
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/skimmed_muon_QCD300_470_root/part0.root': "Events",
            }
    },
    'QCD470_600' : {
        "files" : {
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/skimmed_muon_QCD470_600_root/part0.root': "Events",
            }
    },
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
    },
    'QCD80_120' : {
        "files" : {
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/skimmed_muon_QCD80_120_root/part096.root': "Events",
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/skimmed_muon_QCD80_120_root/part097.root': "Events",
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/skimmed_muon_QCD80_120_root/part099.root': "Events",
        }
    },
    'TTto2L2Nu' : {
        "files" : {
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/skimmed_muon_TTto2L2Nu_root/part0.root': "Events",
        }
    },
    'TTtoLNu2Q' : {
        "files" : {
            'root://cmseos.fnal.gov//store/user/dally/DisplacedTauAnalysis/skimmed_muon_TTto2L2Nu_root/part0.root': "Events",
        }
    },
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

tstart = time.time()

dataset_runnable, dataset_updated = preprocess(
    fileset,
    align_clusters=False,
    step_size=1_000,
    files_per_batch=1,
    skip_bad_files=True,
    save_form=False,
)

to_compute = apply_to_fileset(
    BGProcessor(),
    max_chunks(dataset_runnable, 10), # remove 10 to run over all
    schemaclass=NanoAODSchema,
)

(out,) = dask.compute(to_compute)

print(out)

elapsed_time = time.time() - tstart
print(elapsed_time)
