import coffea
import uproot
import scipy
import dask
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

# Prevent lepton veto problems
NanoAODSchema.mixins["DisMuon"] = "Muon"

# Silence obnoxious warning
NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore", module="coffea.nanoevents.methods")

# --- FUNCTIONS & DEFINITIONS --- #
# Selection cut parameters
min_pT = 20
max_eta = 2.4 
max_dr = 0.3 
max_lep_dr = 0.4 
score_increment_scale_factor = 500 

def delta_r_mask(first: ak.highlevel.Array, second: ak.highlevel.Array, threshold: float) -> ak.highlevel.Array:
    mval = first.metric_table(second)
    return ak.all(mval > threshold, axis=-1)

def apply_lepton_veto(evt_collection: ak.highlevel.Array):
    evt_collection['Jet'] = evt_collection.Jet[
        delta_r_mask(evt_collection.Jet, evt_collection.Photon, max_lep_dr) ]
    evt_collection['Jet'] = evt_collection.Jet[
        delta_r_mask(evt_collection.Jet, evt_collection.Electron, max_lep_dr) ]
    evt_collection['Jet'] = evt_collection.Jet[
        delta_r_mask(evt_collection.Jet, evt_collection.DisMuon, max_lep_dr) ]
    return evt_collection

def apply_jet_cuts(collection):
    cut_collection = apply_lepton_veto(collection)
    jets = cut_collection.Jet
    
    pt_mask = jets.pt >= min_pT
    eta_mask = abs(jets.eta) < max_eta
    valid_mask = jets.genJetIdx > 0 
    
    collection_length = ak.sum(ak.num(jets.partonFlavour))
    inclusion_mask = jets.genJetIdx < collection_length

    cut_jets = jets[
        pt_mask & eta_mask & valid_mask & inclusion_mask ]

    return cut_jets

def apply_gpart_cuts(collection):
    pt_mask = collection.pt >= min_pT
    eta_mask = abs(collection.eta) < max_eta
    #valid_mask = collection.genPartIdx > 0 
    
    #collection_length = ak.sum(ak.num(collection.partonFlavour))
    #inclusion_mask = collection.genPartIdx < collection_length

    #cut_collection = collection[
    #    pt_mask & eta_mask & valid_mask & inclusion_mask ]

    cut_collection = collection[
        pt_mask & eta_mask]
    
    return cut_collection

class BGProcessor(processor.ProcessorABC):
    def __init__(self):
        pass

    def process(self,events):
        dataset = events.metadata['dataset']
        cut = apply_jet_cuts(events)

        return {
            dataset: {
                "jets": cut,
            }
        }
    def postprocess(self,accumulator):
        pass

# --- SIGNAL PROCESSING --- #
# Import dataset
signal_fname = "/eos/user/d/dally/DisplacedTauAnalysis/SMS-TStauStau_MStau-100_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_NanoAOD.root" # signal

# Pass dataset info to coffea objects
signal_events = NanoEventsFactory.from_root(
    {signal_fname: "Events"},
    schemaclass=NanoAODSchema,
    metadata={"dataset": "signal"},
    delayed = False).events()

# Signal processing
taus = signal_events.GenPart[signal_events.GenVisTau.genPartIdxMother] # hadronically-decaying taus
cut_taus = apply_gpart_cuts(taus) 
stau_taus = cut_taus[abs(taus.distinctParent.pdgId) == 1000015] # h-decay taus with stau parents
cut_signal_jets = apply_jet_cuts(signal_events)
matched_tau_jets = stau_taus.nearest(cut_signal_jets, threshold = max_dr) # jets dr-matched to stau_taus
matched_signal_scores = matched_tau_jets.disTauTag_score1

# --- BG PROCESSING --- #
# Import dataset
fileset = {
    'TT to 4Q' : {
        "files": {
            '/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_1-1.root': "Events",
            '/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_1-2.root': "Events",
            '/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_1-3.root': "Events",
            '/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_1.root': "Events",
            '/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_10.root': "Events",
            '/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_11.root': "Events",
            '/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_12.root': "Events",
            '/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_13.root': "Events",
            '/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_14.root': "Events",
            '/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_15.root': "Events",
            '/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_16.root': "Events",
            '/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_17.root': "Events",
            '/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_18.root': "Events",
            '/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_2.root': "Events",
            '/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_20.root': "Events",
            '/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_21.root': "Events",
            '/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_22.root': "Events",
            '/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_23.root': "Events",
            '/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_24.root': "Events",
            '/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_25.root': "Events",
            '/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_26.root': "Events",
            '/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_27.root': "Events",
            '/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_28.root': "Events",
            '/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_29.root': "Events",
            '/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_30.root': "Events",
            '/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_3.root': "Events",
            '/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_4.root': "Events",
            '/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_5.root': "Events",
            '/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_6.root': "Events",
            '/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_7.root': "Events",
            '/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_8.root': "Events",
            '/eos/user/d/dally/DisplacedTauAnalysis/080924_BG_Out/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/crab_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/240809_215611/0000/nanoaod_output_9.root': "Events",
            }
    }
}

dataset_runnable, dataset_updated = preprocess(
    fileset,
    align_clusters=False,
    step_size=1_000,
    files_per_batch=1,
    skip_bad_files=False,
    save_form=False,
)

to_compute = apply_to_fileset(
    BGProcessor(),
    max_chunks(dataset_runnable, ),
    schemaclass=NanoAODSchema,
)

(cut_bg_jets,) = dask.compute(to_compute)

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

# Plot stuff
fig, ax = plt.subplots()
rocs = ax.scatter(fake_rates, efficiencies, c=thresholds, cmap='plasma')

cbar = fig.colorbar(rocs, ax=ax, label='Score threshold')

ax.set_xscale("log")
ax.set_ylim(0.8, 1.025)

plt.xlabel(r"Fake rate $\left(\frac{fake\_passing\_jets}{total\_jets}\right)$")
plt.ylabel(r"Tau tagger efficiency $\left(\frac{matched\_passing\_jets}{total\_matched\_jets}\right)$")

plt.grid()
plt.savefig('inef-full-TT-4Q-bg-tau-tagger-scatter.pdf')
