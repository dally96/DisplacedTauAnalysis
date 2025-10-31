import hist
import hist.dask as hda
import matplotlib.pyplot as plt
import numpy as np
import awkward as ak

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


fileset = {"Stau_300_100mm": {"files": {
          "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_0_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_10_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_11_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_13_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_15_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_16_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_17_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_18_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_19_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_1_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_20_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_21_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_22_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_23_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_24_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_25_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_26_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_27_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_28_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_2_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_30_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_31_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_33_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_34_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_35_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_36_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_37_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_38_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_39_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_3_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_40_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_4_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_5_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_7_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_8_0.root": "Events",
                "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_9_0.root": "Events",
        }}}

class MyProcessor(processor.ProcessorABC):
    
    def __init__(self):
        self._accumulator = {}
    
    def process(self, events):
        events["Jet"] = events.Jet[((events.Jet.pt > 30) & (abs(events.Jet.eta) < 2.4))]
        vx = events.GenVisTau.parent.vx - events.GenVisTau.parent.parent.vx
        vy = events.GenVisTau.parent.vy - events.GenVisTau.parent.parent.vy
        Lxy = np.sqrt(vx**2 + vy**2)
        events["GenVisTau"] = ak.with_field(events.GenVisTau, Lxy, where="lxy")    

        events["GenVisStauTau"] = events.GenVisTau[(abs(events.GenVisTau.parent.distinctParent.pdgId) == 1000015) &
                                               (events.GenVisTau.pt > 50)                                       &
                                               (abs(events.GenVisTau.eta) < 2.4)                                &
                                               (events.GenVisTau.parent.distinctParent.hasFlags("isLastCopy"))  & 
                                               (events.GenVisTau.parent.hasFlags("fromHardProcess"))            &
                                               (abs(events.GenVisTau.lxy) < 100)
                                              ]   
        events = events[(ak.num(events.Jet) == 1) & (ak.num(events.GenVisStauTau) == 1)]
        
#         events["hadTau"] = events.GenVisStauTau.parent.distinctChildrenDeep
        Jet_vec = ak.zip(
                                {'pt': events.Jet.pt,
                                 'phi': events.Jet.phi,
                                 'eta': events.Jet.eta,
                                 'mass': events.Jet.mass,
                                },
                                with_name = "PtEtaPhiMLorentzVector",
                                behavior = coffea.nanoevents.methods.vector.behavior,
        
        )
        GVT_vec = ak.zip(
                                {'pt': events.GenVisStauTau.pt,
                                 'phi': events.GenVisStauTau.phi,
                                 'eta': events.GenVisStauTau.eta,
                                 'mass': events.GenVisStauTau.mass,
                                },
                                with_name = "PtEtaPhiMLorentzVector",
                                behavior = coffea.nanoevents.methods.vector.behavior,
        
        )
    
#         gpart = events.GenPart[1].compute()
        
#         gvst_pt = events.GenVisStauTau.pt.compute()
#         jet_pt = events.Jet.pt.compute()
        
#         gvst_eta = events.GenVisStauTau.eta.compute()
#         jet_eta = events.Jet.eta.compute()
#         print((abs(events.Jet.pt - events.GenVisStauTau.pt)/events.Jet.pt < 0.3).compute())
        
        events = events[ak.flatten((abs(Jet_vec.energy - GVT_vec.energy)/Jet_vec.energy < 0.3)) 
#                         ak.flatten((abs(events.Jet.eta - events.GenVisStauTau.eta) < 0.1))
                       ]
        dxy = abs((events.GenVisStauTau.parent.vy - events.GenVtx.y) * np.cos(events.GenVisStauTau.parent.phi) - \
              (events.GenVisStauTau.parent.vx - events.GenVtx.x) * np.sin(events.GenVisStauTau.parent.phi))
        events["Stau"] = events.GenVisStauTau.parent.distinctParent
        events["GenVisStauTau"] = ak.with_field(events.GenVisStauTau, dxy, where = "dxy")
        
        pt_bins_low    = np.arange(40, 100, 20)
        pt_bins_med    = np.arange(100, 400, 30)
        pt_bins_high   = np.arange(400, 600, 40)
        pt_bins_higher = np.arange(600, 1000, 50)
        pt_bins_low_zoom = np.arange(0, 210, 10)
        pt_bins_eff = np.array(np.concatenate([pt_bins_low, pt_bins_med, pt_bins_high, pt_bins_higher]))
        matched_taus_to_jets = events.Jet.nearest(events.GenVisStauTau, threshold = 0.4)
        dxy_bins_low_zoom = np.arange(0, 5.1, 0.1)
        dxy_bins_med = np.arange(0, 20, 1)
        dxy_bins_high = np.arange(20, 110, 10)
        dxy_bins_eff = np.concatenate([dxy_bins_med, dxy_bins_high])
        
        
        
        gvt_pt = hda.hist.Hist(hist.axis.Variable(pt_bins_eff, name = "gvt_pt", label = "gvt_pt"))
        
        matched_pt = hda.hist.Hist(hist.axis.Variable(pt_bins_eff, name = "pt", label = "matched_pt"))
        regular_pt = hda.hist.Hist(hist.axis.Variable(pt_bins_eff, name = "pt", label = "regular_pt"))
        stau_pt = hda.hist.Hist(hist.axis.Variable(pt_bins_eff, name = "stau_pt", label = "stau_pt"))
        matched_pt_zoom = hda.hist.Hist(hist.axis.Variable(pt_bins_low_zoom, name = "pt_zoom", label = "matched_pt"))
        regular_pt_zoom = hda.hist.Hist(hist.axis.Variable(pt_bins_low_zoom, name = "pt_zoom", label = "regular_pt"))
        
        matched_dxy = hda.hist.Hist(hist.axis.Variable(dxy_bins_eff, name = "dxy", label = "matched_dxy"))
        regular_dxy = hda.hist.Hist(hist.axis.Variable(dxy_bins_eff, name = "dxy", label = "regular_dxy"))
        matched_dxy_zoom = hda.hist.Hist(hist.axis.Variable(dxy_bins_low_zoom, name = "dxy_zoom", label = "matched_dxy"))
        regular_dxy_zoom = hda.hist.Hist(hist.axis.Variable(dxy_bins_low_zoom, name = "dxy_zoom", label = "regular_dxy"))
    
        pt_dxy_hist2 = hda.hist.Hist(hist.new.Var(dxy_bins_eff, name = "dxy", label = "dxy").Var(pt_bins_eff, name = 'pt', label = 'pt').Double())
    
        gvt_pt.fill(ak.flatten(events.GenVisTau.pt, axis = None))
    
        matched_pt.fill(ak.flatten(matched_taus_to_jets.pt, axis = None))
        regular_pt.fill(ak.flatten(events.GenVisStauTau.pt, axis = None))
        stau_pt.fill(ak.flatten(events.Stau.pt, axis = None))
        matched_pt_zoom.fill(ak.flatten(matched_taus_to_jets.pt, axis = None))
        regular_pt_zoom.fill(ak.flatten(events.GenVisStauTau.pt, axis = None))
        matched_dxy.fill(ak.flatten(matched_taus_to_jets.dxy, axis = None))
        regular_dxy.fill(ak.flatten(events.GenVisStauTau.dxy, axis = None))
        matched_dxy_zoom.fill(ak.flatten(matched_taus_to_jets.dxy, axis = None))
        regular_dxy_zoom.fill(ak.flatten(events.GenVisStauTau.dxy, axis = None))
        
        
        pt_dxy_hist2.fill(dxy = ak.flatten(matched_taus_to_jets.dxy, axis = None), pt = ak.flatten(matched_taus_to_jets.pt, axis = None))
        
        events["GenVisStauTau"] = events.GenVisStauTau[events.GenVisStauTau.dxy < 15]
        matched_taus_to_jets_prompt = events.Jet.nearest(events.GenVisStauTau, threshold = 0.4)
        
        var_pt = hist.axis.Variable(pt_bins_eff, name = "pt_prompt", label = "pt_prompt")
        
        matched_pt_prompt = hda.hist.Hist(var_pt)
        regular_pt_prompt = hda.hist.Hist(var_pt)

        matched_pt_prompt.fill(ak.flatten(matched_taus_to_jets_prompt.pt, axis = None))
        regular_pt_prompt.fill(ak.flatten(events.GenVisStauTau.pt, axis = None))
        hist_dict = {"matched": matched_pt, "regular": regular_pt,
                     "matched_prompt": matched_pt_prompt, "regular_prompt": regular_pt_prompt,
                     "matched_zoom": matched_pt_zoom, "regular_zoom": regular_pt_zoom,
                     "matched_dxy": matched_dxy, "regular_dxy": regular_dxy,
                     "matched_dxy_zoom": matched_dxy_zoom, "regular_dxy_zoom": regular_dxy_zoom, "pt_dxy_hist2": pt_dxy_hist2,
                     "gvt_pt": gvt_pt, "stau_pt": stau_pt
                    }
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


fig, ax = plt.subplots(2, 1, figsize=(8,10))


out[0]['Stau_300_100mm']['matched'].plot_ratio(out[0]['Stau_300_100mm']['regular'], 
                                               rp_num_label = "Matched GVT",
                                               rp_denom_label = "All GVT",
                                               rp_uncertainty = 'efficiency',
                                               ax_dict = {"main_ax": ax[0], "ratio_ax": ax[1]}
                                              )

plt.savefig(f"gvt_pt_eff.png")
plt.savefig(f"gvt_pt_eff.pdf")


fig, ax = plt.subplots(2, 1, figsize=(8,10))
out[0]['Stau_300_100mm']['matched_dxy'].plot_ratio(out[0]['Stau_300_100mm']['regular_dxy'], 
                                               rp_num_label = "Matched GVT",
                                               rp_denom_label = "All GVT",
                                               rp_uncertainty = 'efficiency',
                                                ax_dict = {"main_ax": ax[1], "ratio_ax": ax[0]}
                                              )

# ax[1].remove()
# ax[0].set_ylim((0.5, 1.2))
ax[0].set_ylabel("Matched GVT/Total GVT")
ax[1].set_xlabel("GVT_dxy [cm]")
ax[1].set_ylabel("Number of GVT")
plt.savefig(f"gvt_dxy_eff.png")
plt.savefig(f"gvt_dxy_eff.pdf")


fig, ax = plt.subplots(2, 1, figsize=(8,10))
out[0]['Stau_300_100mm']['matched_prompt'].plot_ratio(out[0]['Stau_300_100mm']['regular_prompt'], 
                                               rp_num_label = "Matched GVT",
                                               rp_denom_label = "All GVT",
                                               rp_uncertainty = 'efficiency',
                                               ax_dict = {"main_ax": ax[0], "ratio_ax": ax[1]}
                                              )

plt.savefig(f"gvt_pt_prompt_15_eff.png")
plt.savefig(f"gvt_pt_prompt_15_eff.pdf")





