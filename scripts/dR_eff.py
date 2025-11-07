import hist, os
import hist.dask as hda
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
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
import pickle

import vector
import XRootD

NanoAODSchema.warn_missing_crossrefs = False

plot_dir = "nominal"


class MyProcessor(processor.ProcessorABC):
    
    def __init__(self):
        self._accumulator = {}
    
    def process(self, events):
        events["Jet"] = events.Jet[((events.Jet.pt > 30) & (abs(events.Jet.eta) < 2.4))]

        global_vx = events.GenVtx.x - ak.firsts(events.GenVisTau.parent.distinctChildren.vx, axis =2)
        global_vy = events.GenVtx.y - ak.firsts(events.GenVisTau.parent.distinctChildren.vy, axis =2)
        Dxy = np.sqrt(global_vx**2 + global_vy**2)
        events = ak.with_field(events, Dxy, where="Dxy") 
        tau_vx = events.GenVisTau.parent.vx - ak.firsts(events.GenVisTau.parent.distinctChildren.vx, axis =2)
        tau_vy = events.GenVisTau.parent.vy - ak.firsts(events.GenVisTau.parent.distinctChildren.vy, axis =2)
        Lxy = np.sqrt(tau_vx**2 + tau_vy**2)
        events["Staus"] = events.GenPart[(abs(events.GenPart.pdgId) == 1000015) & (events.GenPart.hasFlags("isLastCopy"))]
        events["GenVisTau"] = ak.with_field(events.GenVisTau, Lxy, where="lxy")    

        events["GenVisStauTau"] = events.GenVisTau[(abs(events.GenVisTau.parent.distinctParent.pdgId) == 1000015) &
                                               (events.GenVisTau.pt > 50)                                       &
                                               (abs(events.GenVisTau.eta) < 2.4)                                &
                                               (events.GenVisTau.parent.distinctParent.hasFlags("isLastCopy"))  & 
                                               (events.GenVisTau.parent.hasFlags("fromHardProcess"))            &
                                               (abs(events.GenVisTau.lxy) < 100)
                                              ]   
        events = events[(ak.num(events.Jet) == 1) & (ak.num(events.GenVisStauTau) == 1)]

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
    
        
        events = events[ak.flatten((abs(Jet_vec.energy - GVT_vec.energy)/Jet_vec.energy < 0.3)) 
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
        stau_pt = hda.hist.Hist(hist.axis.Regular(96, 40, 1000, name = "stau_pt", label = "stau_pt"))
        all_stau_pt = hda.hist.Hist(hist.axis.Regular(96, 40, 1000, name = "all_stau_pt", label = "all_stau_pt"))
        matched_pt_zoom = hda.hist.Hist(hist.axis.Variable(pt_bins_low_zoom, name = "pt_zoom", label = "matched_pt"))
        regular_pt_zoom = hda.hist.Hist(hist.axis.Variable(pt_bins_low_zoom, name = "pt_zoom", label = "regular_pt"))
       
        tau_lxy = hda.hist.Hist(hist.axis.Regular(20, 0, 100, name = "lxy", label = "lxy")) 
        matched_dxy = hda.hist.Hist(hist.axis.Variable(dxy_bins_eff, name = "dxy", label = "matched_dxy"))
        regular_dxy = hda.hist.Hist(hist.axis.Variable(dxy_bins_eff, name = "dxy", label = "regular_dxy"))
        matched_dxy_zoom = hda.hist.Hist(hist.axis.Variable(dxy_bins_low_zoom, name = "dxy_zoom", label = "matched_dxy"))
        regular_dxy_zoom = hda.hist.Hist(hist.axis.Variable(dxy_bins_low_zoom, name = "dxy_zoom", label = "regular_dxy"))
    
        pt_dxy_hist2 = hda.hist.Hist(hist.new.Var(dxy_bins_eff, name = "dxy", label = "dxy").Var(pt_bins_eff, name = 'pt', label = 'pt').Double())
    
#        gvt_pt.fill(ak.flatten(events.GenVisTau.pt, axis = None))
   
#        matched_pt.fill(ak.flatten(matched_taus_to_jets.pt, axis = None))
#        regular_pt.fill(ak.flatten(events.GenVisStauTau.pt, axis = None))
#        stau_pt.fill(ak.flatten(events.Stau.pt, axis = None))
#        all_stau_pt.fill(ak.flatten(events.Staus.pt, axis = None))
#        matched_pt_zoom.fill(ak.flatten(matched_taus_to_jets.pt, axis = None))
#        regular_pt_zoom.fill(ak.flatten(events.GenVisStauTau.pt, axis = None))
#        tau_lxy.fill(ak.flatten(events.Dxy, axis = None))
#        matched_dxy.fill(ak.flatten(matched_taus_to_jets.dxy, axis = None))
#        regular_dxy.fill(ak.flatten(events.GenVisStauTau.dxy, axis = None))
#        matched_dxy_zoom.fill(ak.flatten(matched_taus_to_jets.dxy, axis = None))
#        regular_dxy_zoom.fill(ak.flatten(events.GenVisStauTau.dxy, axis = None))
       
       
#        pt_dxy_hist2.fill(dxy = ak.flatten(matched_taus_to_jets.dxy, axis = None), pt = ak.flatten(matched_taus_to_jets.pt, axis = None))
        
        prompt_GVT = events.GenVisStauTau[events.GenVisStauTau.dxy < 5]
        matched_taus_to_jets_prompt = events.Jet.nearest(prompt_GVT, threshold = 0.4)
        
        var_pt = hist.axis.Variable(pt_bins_eff, name = "pt_prompt", label = "pt_prompt")
        
        matched_pt_prompt = hda.hist.Hist(var_pt)
        regular_pt_prompt = hda.hist.Hist(var_pt)

#        matched_pt_prompt.fill(ak.flatten(matched_taus_to_jets_prompt.pt, axis = None))
#        regular_pt_prompt.fill(ak.flatten(prompt_GVT, axis = None))

        displaced_GVT = events.GenVisStauTau[events.GenVisStauTau.dxy > 25]
        num_displaced_GVT = ak.num(displaced_GVT) > 0
        displaced_jet = events.Jet[num_displaced_GVT]

        jet_GVT_pt_hist2 =  hda.hist.Hist(hist.new.Reg(40, 0, 800, name = 'Jet_pt', label = 'Jet_pt').Reg(40, 0, 800, name = 'GVT_pt', label = 'GVT_pt'))
#        jet_GVT_pt_hist2.fill(Jet_pt = ak.flatten(displaced_jet.pt, axis = None), GVT_pt = ak.flatten(displaced_GVT.pt, axis = None))
        displaced_jet_pfc_pdgid = hda.hist.Hist(hist.axis.Regular(300, 0, 300, name="pdgIds", label = "pdgIds"))
#        displaced_jet_pfc_pdgid.fill(ak.ravel(displaced_jet.constituents.pf.pdgId))

        DM0_GVT = events.GenVisStauTau[events.GenVisStauTau.status == 0]
        matched_taus_to_jets_DM0 = events.Jet.nearest(DM0_GVT, threshold = 0.4)

        var_dxy = hist.axis.Variable(dxy_bins_eff, name = "DM0_dxy", label = "DM0_dxy")
        
        matched_dxy_DM0 = hda.hist.Hist(var_dxy)
        regular_dxy_DM0 = hda.hist.Hist(var_dxy)

        matched_dxy_DM0.fill(ak.flatten(matched_taus_to_jets_DM0.dxy, axis = None))
        regular_dxy_DM0.fill(ak.flatten(DM0_GVT.dxy, axis = None))


        hist_dict = {"matched": matched_pt, "regular": regular_pt,
                     "matched_prompt": matched_pt_prompt, "regular_prompt": regular_pt_prompt,
                     "matched_zoom": matched_pt_zoom, "regular_zoom": regular_pt_zoom,
                     "matched_dxy": matched_dxy, "regular_dxy": regular_dxy, "matched_dxy_DM0":  matched_dxy_DM0, "regular_dxy_DM0": regular_dxy_DM0,
                     "matched_dxy_zoom": matched_dxy_zoom, "regular_dxy_zoom": regular_dxy_zoom, "pt_dxy_hist2": pt_dxy_hist2,
                     "gvt_pt": gvt_pt, "stau_pt": stau_pt, 'all_stau_pt': all_stau_pt, 'tau_lxy': tau_lxy, "jet_GVT_pt": jet_GVT_pt_hist2, "pdgId": ak.ravel(displaced_jet.constituents.pf.pdgId),
                    }
        return hist_dict
    
    def postprocess(self, accumulator):
        pass

with open("processed_Stau_300_100mm.pkl", "rb") as f:
    dataset_runnable = pickle.load(f)

to_compute = apply_to_fileset(
                 MyProcessor(),
                 max_chunks(dataset_runnable, 10000),
                 schemaclass=PFNanoAODSchema
                )   
out = dask.compute(to_compute)

if not plot_dir in os.listdir('.'):
    os.mkdir(plot_dir)

plot_dir = plot_dir + '/'

#fig, ax = plt.subplots(2, 1, figsize=(8,10))
#
#
#out[0]['Stau_300_100mm']['matched'].plot_ratio(out[0]['Stau_300_100mm']['regular'], 
#                                               rp_num_label = "Matched GVT",
#                                               rp_denom_label = "All GVT",
#                                               rp_uncertainty = 'efficiency',
#                                               ax_dict = {"main_ax": ax[0], "ratio_ax": ax[1]}
#                                              )
#ax[1].set_ylim((0.5, 1.2))
#ax[1].xaxis.set_major_locator(MultipleLocator(50))
#ax[1].xaxis.set_minor_locator(MultipleLocator(10))
#ax[0].xaxis.set_major_locator(MultipleLocator(50))
#ax[0].xaxis.set_minor_locator(MultipleLocator(10))
#ax[1].set_ylabel("Matched GVT/Total GVT")
#ax[0].set_ylabel("Number of GVT")
#plt.savefig(f"{plot_dir}gvt_pt_eff.png")
#plt.savefig(f"{plot_dir}gvt_pt_eff.pdf")
#
#
#fig, ax = plt.subplots(2, 1, figsize=(8,10))
#out[0]['Stau_300_100mm']['matched_dxy'].plot_ratio(out[0]['Stau_300_100mm']['regular_dxy'], 
#                                               rp_num_label = "Matched GVT",
#                                               rp_denom_label = "All GVT",
#                                               rp_uncertainty = 'efficiency',
#                                                ax_dict = {"main_ax": ax[0], "ratio_ax": ax[1]}
#                                              )
#
## ax[1].remove()
#ax[1].set_ylim((0.5, 1.2))
#ax[1].set_ylabel("Matched GVT/Total GVT")
#ax[1].set_xlabel("GVT_dxy [cm]")
#ax[0].set_ylabel("Number of GVT")
#ax[1].xaxis.set_major_locator(MultipleLocator(10))
#ax[1].xaxis.set_minor_locator(MultipleLocator(5))
#ax[0].xaxis.set_major_locator(MultipleLocator(10))
#ax[0].xaxis.set_minor_locator(MultipleLocator(5))
#plt.savefig(f"{plot_dir}gvt_dxy_eff.png")
#plt.savefig(f"{plot_dir}gvt_dxy_eff.pdf")
#
#
#fig, ax = plt.subplots(2, 1, figsize=(8,10))
#out[0]['Stau_300_100mm']['matched_prompt'].plot_ratio(out[0]['Stau_300_100mm']['regular_prompt'], 
#                                               rp_num_label = "Matched GVT",
#                                               rp_denom_label = "All GVT",
#                                               rp_uncertainty = 'efficiency',
#                                               ax_dict = {"main_ax": ax[0], "ratio_ax": ax[1]}
#                                              )
#
#ax[1].set_ylim((0.5, 1.2))
#ax[1].set_ylabel("Matched GVT/Total GVT")
#ax[1].set_xlabel("GVT pt [GeV]")
#ax[0].set_ylabel("Number of GVT")
#plt.savefig(f"{plot_dir}gvt_pt_prompt_5_eff.png")
#plt.savefig(f"{plot_dir}gvt_pt_prompt_5_eff.pdf")

fig, ax = plt.subplots(2, 1, figsize=(8,10))
out[0]['Stau_300_100mm']['matched_dxy_DM0'].plot_ratio(out[0]['Stau_300_100mm']['regular_dxy_DM0'],
                                                        rp_num_label = "Matched GVT with decay mode 0",
                                                        rp_denom_label = "All GVT with decay mode 0",
                                                        rp_uncertainty = 'efficiency',
                                                        ax_dict = {"main_ax": ax[0], "ratio_ax": ax[1]}
                                              )
ax[1].set_ylim((0.5, 1.2))
ax[1].set_ylabel("Matched GVT/Total GVT")
ax[1].set_xlabel("GVT_dxy [cm]")
ax[0].set_ylabel("Number of GVT")
ax[1].xaxis.set_major_locator(MultipleLocator(10))
ax[1].xaxis.set_minor_locator(MultipleLocator(5))
ax[0].xaxis.set_major_locator(MultipleLocator(10))
ax[0].xaxis.set_minor_locator(MultipleLocator(5))
plt.title("Stau_300_100mm")
plt.savefig(f"{plot_dir}gvt_dxy_DM0_eff.png")
plt.savefig(f"{plot_dir}gvt_dxy_DM0_eff.pdf")


#fig, ax = plt.subplots()
#out[0]['Stau_300_100mm']['stau_pt'].plot()
#ax.set_xlabel(r'$p_{T}$ [GeV]')
#ax.set_ylabel('Staus')
#ax.xaxis.set_major_locator(MultipleLocator(100))
#ax.xaxis.set_minor_locator(MultipleLocator(20))
#fig.savefig(f"{plot_dir}stau_pt.png")
#fig.savefig(f"{plot_dir}stau_pt.pdf")
#
#
#fig, ax = plt.subplots()
#out[0]['Stau_300_100mm']['all_stau_pt'].plot()
#ax.set_xlabel(r'$p_{T}$ [GeV]')
#ax.set_ylabel('Staus')
#ax.xaxis.set_major_locator(MultipleLocator(100))
#ax.xaxis.set_minor_locator(MultipleLocator(20))
#fig.savefig(f"{plot_dir}all_stau_pt.png")
#fig.savefig(f"{plot_dir}all_stau_pt.pdf")
#
#fig, ax = plt.subplots()
#out[0]['Stau_300_100mm']['tau_lxy'].plot()
#ax.set_xlabel(r'$L_{xy}$ [cm]')
#ax.set_ylabel('taus')
#fig.savefig(f"{plot_dir}stau_lxy.png")
#fig.savefig(f"{plot_dir}stau_lxy.pdf")
#
#
#fig, ax = plt.subplots()
#out[0]['Stau_300_100mm']['jet_GVT_pt'].plot()
#ax.set_xlabel(r'Jet $p_{T}$ [GeV]')
#ax.set_ylabel(r'$\tau_{h}$ $p_{T}$ [GeV]')
#fig.savefig(f"{plot_dir}jet_GVT_pt.png")
#fig.savefig(f"{plot_dir}jet_GVT_pt.pdf")
#
#fig, ax = plt.subplots()
#counts, bins, patches = ax.hist(abs(out[0]['Stau_300_100mm']['pdgId']), bins = np.arange(0, 51, 1))
#ax.set_xlabel(r'pdgId [GeV]')
#ax.set_ylabel(r'Counts')
#ax.xaxis.set_major_locator(MultipleLocator(5))
#ax.xaxis.set_minor_locator(MultipleLocator(1))
#
#fig.savefig(f"{plot_dir}pdgId_0_50.png")
#fig.savefig(f"{plot_dir}pdgId_0_50.pdf")
#
#fig, ax = plt.subplots()
#counts, bins, patches = ax.hist(abs(out[0]['Stau_300_100mm']['pdgId']), bins = np.arange(50, 101, 1))
#ax.set_xlabel(r'pdgId [GeV]')
#ax.set_ylabel(r'Counts')
#ax.xaxis.set_major_locator(MultipleLocator(5))
#ax.xaxis.set_minor_locator(MultipleLocator(1))
#
#fig.savefig(f"{plot_dir}pdgId_50_100.png")
#fig.savefig(f"{plot_dir}pdgId_50_100.pdf")
#
#fig, ax = plt.subplots()
#counts, bins, patches = ax.hist(abs(out[0]['Stau_300_100mm']['pdgId']), bins = np.arange(100, 151, 1))
#ax.set_xlabel(r'pdgId [GeV]')
#ax.set_ylabel(r'Counts')
#ax.xaxis.set_major_locator(MultipleLocator(5))
#ax.xaxis.set_minor_locator(MultipleLocator(1))
#
#fig.savefig(f"{plot_dir}pdgId_100_150.png")
#fig.savefig(f"{plot_dir}pdgId_100_150.pdf")
#
#fig, ax = plt.subplots()
#counts, bins, patches = ax.hist(abs(out[0]['Stau_300_100mm']['pdgId']), bins = np.arange(150, 201, 1))
#ax.set_xlabel(r'pdgId [GeV]')
#ax.set_ylabel(r'Counts')
#ax.xaxis.set_major_locator(MultipleLocator(5))
#ax.xaxis.set_minor_locator(MultipleLocator(1))
#
#fig.savefig(f"{plot_dir}pdgId_150_200.png")
#fig.savefig(f"{plot_dir}pdgId_150_200.pdf")
#
#fig, ax = plt.subplots()
#counts, bins, patches = ax.hist(abs(out[0]['Stau_300_100mm']['pdgId']), bins = np.arange(200, 251, 1))
#ax.set_xlabel(r'pdgId [GeV]')
#ax.set_ylabel(r'Counts')
#ax.xaxis.set_major_locator(MultipleLocator(5))
#ax.xaxis.set_minor_locator(MultipleLocator(1))
#
#fig.savefig(f"{plot_dir}pdgId_200_250.png")
#fig.savefig(f"{plot_dir}pdgId_200_250.pdf")
#
#fig, ax = plt.subplots()
#counts, bins, patches = ax.hist(abs(out[0]['Stau_300_100mm']['pdgId']), bins = np.arange(250, 301, 1))
#ax.set_xlabel(r'pdgId [GeV]')
#ax.set_ylabel(r'Counts')
#ax.xaxis.set_major_locator(MultipleLocator(5))
#ax.xaxis.set_minor_locator(MultipleLocator(1))
#
#fig.savefig(f"{plot_dir}pdgId_250_300.png")
#fig.savefig(f"{plot_dir}pdgId_250_300.pdf")
#
#fig, ax = plt.subplots()
#counts, bins, patches = ax.hist(abs(out[0]['Stau_300_100mm']['pdgId']), bins = np.arange(300, 351, 1))
#ax.set_xlabel(r'pdgId [GeV]')
#ax.set_ylabel(r'Counts')
#ax.xaxis.set_major_locator(MultipleLocator(5))
#ax.xaxis.set_minor_locator(MultipleLocator(1))
#
#fig.savefig(f"{plot_dir}pdgId_300_350.png")
#fig.savefig(f"{plot_dir}pdgId_300_350.pdf")
#
#fig, ax = plt.subplots()
#counts, bins, patches = ax.hist(abs(out[0]['Stau_300_100mm']['pdgId']), bins = np.arange(350, 401, 1))
#ax.set_xlabel(r'pdgId [GeV]')
#ax.set_ylabel(r'Counts')
#ax.xaxis.set_major_locator(MultipleLocator(5))
#ax.xaxis.set_minor_locator(MultipleLocator(1))
#
#fig.savefig(f"{plot_dir}pdgId_350_400.png")
#fig.savefig(f"{plot_dir}pdgId_350_400.pdf")
#
#fig, ax = plt.subplots()
#counts, bins, patches = ax.hist(abs(out[0]['Stau_300_100mm']['pdgId']), bins = np.arange(400, 1000, 1))
#ax.set_xlabel(r'pdgId [GeV]')
#ax.set_ylabel(r'Counts')
#ax.xaxis.set_major_locator(MultipleLocator(100))
#ax.xaxis.set_minor_locator(MultipleLocator(20))
#
#fig.savefig(f"{plot_dir}pdgId_400_1000.png")
#fig.savefig(f"{plot_dir}pdgId_400_1000.pdf")


