import hist
import hist.dask as hda
import matplotlib.pyplot as plt
import numpy as np
import awkward as ak
from scipy.optimize import curve_fit

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, PFNanoAODSchema
from coffea.analysis_tools import PackedSelection
import dask_awkward
import dask
import coffea.nanoevents.methods
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema, NanoAODSchema                                                            
from coffea.dataset_tools import (                                                                                                         
    apply_to_fileset,                       
    max_chunks,                                                                                                                            
    preprocess,                                                                                                                            
)        
from coffea import processor

import vector
import XRootD

NanoAODSchema.warn_missing_crossrefs = False

def exponential_func(x, a, b):
    """
    Defines a general exponential function of the form a * exp(-b * x) + c.
    """
    return a * np.exp(-x * b)

fileset = {"Stau_300_100mm": {"files": {
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_0_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_10_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_11_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_12_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_13_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_14_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_15_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_16_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_17_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_18_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_19_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_1_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_20_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_21_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_22_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_23_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_24_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_25_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_26_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_27_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_28_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_2_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_30_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_31_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_32_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_33_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_34_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_35_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_36_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_37_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_38_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_39_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_3_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_40_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_4_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_5_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_6_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_7_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_8_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_9_0.root": "Events",
        }}}



class MyProcessor(processor.ProcessorABC):
    
    def __init__(self):
        self._accumulator = {}
    
    def process(self, events):
        
        hist_dict = {}        

        events["Stau"] = events.GenPart[(abs(events.GenPart.pdgId) == 1000015) & (events.GenPart.hasFlags("isLastCopy"))]

        events["StauTau"] = events.Stau.distinctChildren[(abs(events.Stau.distinctChildren.pdgId) == 15) & (events.Stau.distinctChildren.hasFlags("isLastCopy")) & (events.Stau.distinctChildren.hasFlags("fromHardProcess"))]
        events["StauTau"] = ak.flatten(events["StauTau"], axis = 2)

        lxy = np.sqrt((events["StauTau"].vx - events["Stau"].vx)**2 + (events["StauTau"].vy - events["Stau"].vy)**2)
        events["Stau"] = ak.with_field(events.Stau, lxy, where='lxy')
        time = events.Stau.lxy*events.Stau.mass/(events.Stau.pt * 3 * 10**10)*10**9
        events["Stau"] = ak.with_field(events.Stau, time, where = 'time')

        Stau_time = ak.ravel(events.Stau.time)
        time_bins = np.linspace(0, 2, 101)

        time_hist = hda.hist.Hist(hist.axis.Regular(100, 0, 2, name = 'time', label = 'time'))
        time_hist.fill(Stau_time)

        hist_dict["time"] = time_hist

        return hist_dict
    
    def postprocess(self, accumulator):
        pass

skimmed_dataset_runnable, skimmed_dataset_updated = preprocess(                                                                        
    fileset,
    align_clusters=False,                                                                                                              
    step_size=10_000,
    files_per_batch=100,                                                                                                               
    skip_bad_files=True,                                                                                                               
    save_form=False,
#     file_exceptions=(OSError, KeyInFileError),                                                                                         
    allow_empty_datasets=False,
)

to_compute = apply_to_fileset(
                 MyProcessor(),
                 max_chunks(skimmed_dataset_runnable, 10000),
                 schemaclass=PFNanoAODSchema
                )   
out = dask.compute(to_compute)

time_bin_entries = out[0]["Stau_300_100mm"]['time'].values()
time_bin_entries[-1] += out[0]["Stau_300_100mm"]['time'].values(flow=True)[-1]

time_bins = np.linspace(0, 2, 100)

N = np.cumsum(time_bin_entries[::-1])[::-1]
popt, pcov = curve_fit(exponential_func, time_bins, N)

a_fit, b_fit = popt

plt.plot(time_bins, N, 'ko', label='Remaining staus (proper time)')
plt.plot(time_bins, exponential_func(time_bins, a_fit, b_fit), 'r-', label=r'Exponential Fit, $c\tau$ =' + '{:.2f} mm'.format(1/b_fit * 300))
plt.plot(time_bins, exponential_func(time_bins, N[0], 1/(100/300)), 'g-', label = r"Expected $N_{0} e^{-t/\tau}$")
plt.xlabel("Time [ns]")
plt.ylabel("Counts")
plt.title(Stau_300_100mm)
plt.legend()

plt.savefig("exp_decay_stau.pdf")
plt.savefig("exp_decay_stau.png")
