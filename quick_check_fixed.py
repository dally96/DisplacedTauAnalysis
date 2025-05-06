import os
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import hist
import vector
from hist import Hist, axis, intervals
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema
import coffea.nanoevents.methods
np.set_printoptions(precision=6, suppress=False, threshold=np.inf)

# Load the file
filenames = {
    'Stau_100_1mm'    : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-100_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_100_10mm'   : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-100_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_100_100mm'  : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-100_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_100_1000mm' : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-100_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_200_1mm'    : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-200_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_200_10mm'   : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-200_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_200_100mm'  : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-200_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_200_1000mm' : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-200_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_300_1mm'    : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_300_10mm'   : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_300_100mm'  : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_300_1000mm' : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_500_1mm'    : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_500_10mm'   : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_500_100mm'  : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_500_1000mm' : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
}

PFNanoAODSchema.mixins["DisMuon"] = "Muon"
samples = {}
for sample_name, files in filenames.items():
    samples[sample_name] = NanoEventsFactory.from_root(
        {files: "Events"},
        schemaclass=PFNanoAODSchema,
        metadata={"dataset": "MC"}
    ).events()

def delta_r_mask(first: ak.highlevel.Array, second: ak.highlevel.Array, threshold: float) -> ak.highlevel.Array: 
    mval = first.metric_table(second) 
    return ak.all(mval < threshold, axis=-1)

# ----------------------------------------------------------------------
# Main loop: Process each sample and produce histograms.
# ----------------------------------------------------------------------
if __name__ == '__main__':
    for sample_name, events in samples.items():
        print(f"Processing sample: {sample_name}")
        # add dxy to jet fields
        charged_sel = events.Jet.constituents.pf.charge != 0
        dxy = ak.flatten(events.Jet.constituents.pf[ak.argmax(events.Jet.constituents.pf[charged_sel].pt, axis=2, keepdims=True)].d0, axis = 2)
        events['Jet'] = ak.with_field(events.Jet, dxy, where="dxy")

        ## find staus and their tau children
        gpart = events.GenPart
        events['staus'] = gpart[(abs(gpart.pdgId) == 1000015) & (gpart.hasFlags("isLastCopy"))] # most likely last copy of stay in the chain

        events['staus_taus'] = events.staus.distinctChildren[ (abs(events.staus.distinctChildren.pdgId) == 15) & \
                                                          (events.staus.distinctChildren.hasFlags("isLastCopy")) \
                                                         ]
        #events['GenVisStauTaus'] = events.GenVisTau
        events['GenVisStauTaus'] = events.GenVisTau[(abs(events.GenVisTau.parent.pdgId) == 15) & (abs(events.GenVisTau.parent.distinctParent.pdgId) == 1000015) & (events.GenVisTau.parent.distinctParent.hasFlags("isLastCopy"))]

        events['staus_taus'] = ak.firsts(events.staus_taus[ak.argsort(events.staus_taus.pt, ascending=False)], axis = 2)
        staus_taus = events['staus_taus']

        mask_taul = ak.any((abs(staus_taus.distinctChildren.pdgId) == 11) | (abs(staus_taus.distinctChildren.pdgId) == 13), axis=-1)
        mask_tauh = ~mask_taul

        # Note: does the delta_r_mask work as expected
        debugging_had_gen_taus = events.staus_taus[mask_tauh]

        # Apply the mask to select distinctChildren (excluding neutrinos) these are the daughters
        gen_had_distinctChildren = debugging_had_gen_taus.distinctChildren[(abs(debugging_had_gen_taus.distinctChildren.pdgId) != 16)]

        hadTauDaughterVec = vector.zip(
            {
                "pt": gen_had_distinctChildren.pt,
                "eta": gen_had_distinctChildren.eta,
                "phi": gen_had_distinctChildren.phi,
                "mass": gen_had_distinctChildren.mass,
            },
        )
        hadTauDaughterVec = ak.sum(hadTauDaughterVec, axis = -1)

        hadTauDaughterAkVec = ak.zip(
                                {
                                    "pt":   hadTauDaughterVec.pt,
                                    "phi":  hadTauDaughterVec.phi,
                                    "eta":  hadTauDaughterVec.eta,
                                    "mass": hadTauDaughterVec.mass,
                                },
                                with_name = "PtEtaPhiMLorentzVector",
                                behavior = coffea.nanoevents.methods.vector.behavior,
        
        )

        hadTauDaughterAkVec = hadTauDaughterAkVec[delta_r_mask(hadTauDaughterAkVec, events.GenVisStauTaus, 0.4)]

        pt_diff = hadTauDaughterAkVec.pt.compute() - events.GenVisStauTaus.pt.compute()
        flat_pt_diff = ak.to_numpy(ak.flatten(pt_diff, axis=None))

        # Plot
        plt.hist(flat_pt_diff, bins=30, range=(-10, 10), histtype='step', linewidth=1.5)
        plt.xlabel("pT (daughter) - pT (GenVisTau) [GeV]")
        plt.ylabel("Counts")
        plt.title(f"{sample_name}: pT Difference per Tau Daughter")
        plt.grid(True)
        plt.savefig(f"{sample_name}_HadTauDaughterVsGenVisTau_pTDiff.pdf")
        plt.close()

        '''
        debugging_num_had_gen_taus = ak.num(hadTauDaughterAkVec)
        debugging_num_vis_gen_taus = ak.num(events.GenVisStauTaus)
        
        had_gen_tau_etas = ak.flatten(hadTauDaughterAkVec.eta, axis=None).compute()

        num_had = ak.num(hadTauDaughterAkVec.pt).compute() 
        num_vis = ak.num(events.GenVisStauTaus).compute()

        num_had = np.asarray(num_had)
        num_vis = np.asarray(num_vis)

        unique_had_gen_counts = np.unique(num_had)
        unique_vis_counts = np.unique(num_vis)
        print("Unique counts of had gen taus per event:", unique_had_gen_counts)
        print("Unique counts of gen vis taus per event:", unique_vis_counts)      

        # Sum the pt of the distinct children (visible tau decay products) per event
        sum_per_event_pt = ak.sum(hadTauDaughterAkVec.pt, axis=-1).compute()
        sum_per_event_eta = ak.sum(hadTauDaughterAkVec.eta, axis=-1).compute()

        # Flatten to get a 1D array for histogramming
        flat_pts  = ak.to_numpy(ak.flatten(sum_per_event_pt, axis=None))
        flat_etas = ak.to_numpy(ak.flatten(sum_per_event_eta, axis=None))

        # Flatten eta sums and convert to NumPy array
        flat_etas_cut = flat_etas[flat_pts > 10.0]

        # Define eta bin edges: -5 to 5 in steps of 0.5
        eta_bins = np.arange(-5, 5.5, 0.5)

        # Plot eta histogram
        plt.hist(flat_etas_cut, bins=eta_bins, histtype='step', linewidth=1.5)
        plt.xlabel("Hadronic Gen Tau Decay Product Minus Neutrino $\\eta$")
        plt.ylabel("Counts")
        plt.title(f"{sample_name}: Sum of Hadronic Gen Tau Daughters $\\eta$")
        plt.grid(True)
        plt.savefig(f"{sample_name}_HadGenTauDecayProductsETA.pdf")
        plt.close()
        
        # Apply cut for pT > 10 GeV
        flat_pts_cut = flat_pts_np[flat_pts_np > 10.0]

        # Histogram 1: 10–20 GeV (bin width = 1)
        mask_mid = (flat_pts_cut > 10) & (flat_pts_cut <= 20)
        mid_pt = flat_pts_cut[mask_mid]
        bins_mid = np.arange(10, 21, 1)

        plt.hist(mid_pt, bins=bins_mid, histtype='step', linewidth=1.5)
        plt.xlabel("Hadronic Gen Tau Decay Product Minus Neutrino $p_T$ [GeV]")
        plt.ylabel("Counts")
        plt.title(f"{sample_name}: $p_T$ [10–20 GeV]")
        plt.grid(True)
        plt.savefig(f"{sample_name}_HadGenTauDecayProductsPT_10to20.pdf")
        plt.close()

        # Histogram 2: >20–500 GeV (bin width = 20)
        mask_high = flat_pts_cut > 20
        high_pt = flat_pts_cut[mask_high]
        bins_high = np.arange(20, 520, 20)

        plt.hist(high_pt, bins=bins_high, histtype='step', linewidth=1.5)
        plt.xlabel("Hadronic Gen Tau Decay Product Minus Neutrino $p_T$ [GeV]")
        plt.ylabel("Counts")
        plt.title(f"{sample_name}: $p_T$ [>20 GeV]")
        plt.grid(True)
        plt.savefig(f"{sample_name}_HadGenTauDecayProductsPT_gt20.pdf")
        plt.close()
        '''

        '''
        num_vis_per_event = ak.num(events.GenVisStauTaus)
        mask_vis_eq = (num_vis_per_event > 2)

        selected_vis_taus = events.GenVisStauTaus[mask_vis_eq]
        selected_had_gen_taus = debugging_had_gen_taus[mask_vis_eq]

        evt_numbers = ak.local_index(events.GenVisStauTaus, axis=0) 
        evt_numbers = evt_numbers[mask_vis_eq]                    
        evt_numbers = ak.flatten(evt_numbers, axis=None) 

        vis_tau_pts         = ak.flatten(selected_vis_taus.pt,              axis=None)
        gen_part_idx_mother = ak.flatten(selected_vis_taus.genPartIdxMother, axis=None)        

        evt_numbers         = evt_numbers.compute()
        vis_tau_pts         = vis_tau_pts.compute()
        gen_part_idx_mother = gen_part_idx_mother.compute()

        evt_numbers         = np.asarray(evt_numbers)
        vis_tau_pts         = np.asarray(vis_tau_pts)
        gen_part_idx_mother = np.asarray(gen_part_idx_mother)

        for evt, pt, mother in zip(evt_numbers, vis_tau_pts, gen_part_idx_mother):
            print(f"{sample_name}\t event {int(evt):6d}\t pT = {pt:7.2f} GeV\t gen_part_idx_mother = {mother}")

        '''
