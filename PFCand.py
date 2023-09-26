import uproot 
import statistics
import math
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import awkward as ak
import tau_selections as ts 
import pandas as pd
from pdb import set_trace
import ROOT
import sys
from ROOT import TH2F, TH1F, TChain, TCanvas, gROOT
ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch(True)
pd.options.mode.chained_assignment = None  # default='warn'
import pdb
plt.ion()
from hist import Hist

import vector 
vector.register_awkward()

Run2 = uproot.open("SUS-RunIISummer20UL18GEN-stau100_lsp1_ctau100mm_v6_cms-xrd-global_ntuple.root")
Run2_tree = Run2["tree"]
Run2_nevt = Run2_tree.num_entries

Run3 = uproot.open("Staus_M_100_100mm_13p6TeV_Run3Summer22EE_cms-xrd-global_ntuple.root")
Run3_tree = Run3["tree"]
Run3_nevt = Run3_tree.num_entries

Run2_branchDict = {}
Run3_branchDict = {}

for branch in Run2_tree.keys():
  Run2_branchDict[branch] = Run2_tree[branch].array()
for branch in Run3_tree.keys():
  Run3_branchDict[branch] = Run3_tree[branch].array()

Run2_PF_df = ak.to_dataframe(Run2_tree.arrays(filter_name = "PF_*", library='ak'))
Run2_PF_df_charged = Run2_PF_df.query("abs(PF_charge) > 0")
Run2_Jet_df = ak.to_dataframe(Run2_tree.arrays(filter_name = "jet_*", library='ak'))
Run2_GenPart_df = ak.to_dataframe(Run2_tree.arrays(filter_name = "genPart_*", library='ak'))
Run2_GenVis_df = ak.to_dataframe(Run2_tree.arrays(filter_name = "genVisTau_*", library='ak'))
Run2_GenVisTau_idx = list(zip(list(zip(*Run2_GenVis_df["genVisTau_genPartIdx"].index))[0], Run2_GenVis_df["genVisTau_genPartIdx"].to_numpy()))
Run2_GenVisTau_df = (Run2_GenPart_df.loc[Run2_GenVisTau_idx])

Run3_PF_df = ak.to_dataframe(Run3_tree.arrays(filter_name = "PF_*", library='ak'))
Run3_PF_df_charged = Run3_PF_df.query("abs(PF_charge) > 0")
Run3_Jet_df = ak.to_dataframe(Run3_tree.arrays(filter_name = "jet_*", library='ak'))
Run3_GenPart_df = ak.to_dataframe(Run3_tree.arrays(filter_name = "genPart_*", library='ak'))
Run3_GenVis_df = ak.to_dataframe(Run3_tree.arrays(filter_name = "genVisTau_*", library='ak'))
Run3_GenVisTau_idx = list(zip(list(zip(*Run3_GenVis_df["genVisTau_genPartIdx"].index))[0], Run3_GenVis_df["genVisTau_genPartIdx"].to_numpy()))
Run3_GenVisTau_df = (Run3_GenPart_df.loc[Run3_GenVisTau_idx])

Run3_Jet_df_selections = Run3_Jet_df.query("jet_pt > @ts.jetPtMin & jet_eta < @ts.jetEtaMax") 
Run3_GenVisTau_df_selections = Run3_GenVisTau_df.query("genPart_pt > @ts.genTauPtMin & genPart_eta < @ts.genTauEtaMax & genPart_vertexZ < @ts.genTauVtxZMax & genPart_vertexRho < @ts.genTauVtxRhoMax")

Run2_Jet_df_selections = Run2_Jet_df.query("jet_pt > @ts.jetPtMin & jet_eta < @ts.jetEtaMax") 
Run2_GenVisTau_df_selections = Run2_GenVisTau_df.query("genPart_pt > @ts.genTauPtMin & genPart_eta < @ts.genTauEtaMax & genPart_vertexZ < @ts.genTauVtxZMax & genPart_vertexRho < @ts.genTauVtxRhoMax")
Run3_jets = ak.zip({
            "pt":  Run3_tree["jet_pt"].array(),
            "eta": Run3_tree["jet_eta"].array(),
            "phi": Run3_tree["jet_phi"].array(),
            "mass": 0.140,
})

Run3_taus = ak.zip({
            "pt":  Run3_tree["genPart_pt"].array(),
            "eta": Run3_tree["genPart_eta"].array(),
            "phi": Run3_tree["genPart_phi"].array(),
            "mass": 0.140,
            "vertexZ": Run3_tree["genPart_vertexZ"].array(),
            "vertexRho": Run3_tree["genPart_vertexRho"].array(),
            "index": ak.local_index(Run3_tree["genPart_pt"].array()),
})

Run3_PF_dict = {}
for branch in Run3_tree.keys():
  if "PF" in branch:
    Run3_PF_dict[branch] = Run3_tree[branch].array()
Run3_PFs = ak.zip(Run3_PF_dict)

Run3_taus = Run3_taus[Run3_tree["genVisTau_genPartIdx"].array()]

Run3_Jet_selections = ((Run3_jets.pt > ts.genTauPtMin) & (abs(Run3_jets.eta) < ts.jetEtaMax))
Run3_Tau_selections = ((Run3_taus.pt > ts.genTauPtMin) & (abs(Run3_taus.eta) < ts.genTauEtaMax) & (Run3_taus.vertexZ < ts.genTauVtxZMax) & (Run3_taus.vertexRho < ts.genTauVtxRhoMax))

Run3_selected_jets = Run3_jets[Run3_Jet_selections]
Run3_selected_taus = Run3_taus[Run3_Tau_selections] 

Run3_awk_taus = ak.with_name(Run3_selected_taus, "Momentum4D")
Run3_awk_jets = ak.with_name(Run3_selected_jets, "Momentum4D")

Run3_jet_tau = ak.cartesian({"taus": Run3_awk_taus, "jets": Run3_awk_jets}, axis =-1, nested=True) 
Run3_tau, Run3_jet = ak.unzip(Run3_jet_tau)
Run3_dR = Run3_tau.deltaR(Run3_jet)

## add min dR wrt any jet
#taus["min_dR_jet"] = ak.min(dR, axis = -1,keepdims=True)
## get index of best matched jet
Run3_best_dR = ak.argmin(Run3_dR, axis=-1, keepdims=False)
### this can be further improved in the future, to only allow one combination 
Run3_matched_PFs = Run3_PFs[Run3_best_dR]
Run3_charged_PF_selections = ((abs(Run3_matched_PFs.PF_charge) > 0))
Run3_greaterpt_selections = (Run3_matched_PFs.PF_pt >= 0.5)
Run3_lowerpt_selections = (Run3_matched_PFs.PF_pt < 0.5)
Run3_matched_charged_PFs = Run3_matched_PFs[Run3_charged_PF_selections]
Run3_matched_charged_greaterPt_PFs = Run3_matched_PFs[Run3_charged_PF_selections & Run3_greaterpt_selections]
Run3_matched_charged_lowerPt_PFs = Run3_matched_PFs[Run3_charged_PF_selections & Run3_lowerpt_selections]

Run3_matched_charged_greaterPt_PF = ak.unzip(Run3_matched_charged_greaterPt_PFs)
Run3_matched_charged_lowerPt_PF = ak.unzip(Run3_matched_charged_lowerPt_PFs)

Run3_matched_charged_greaterPt_PF_dict = {}
Run3_matched_charged_lowerPt_PF_dict = {}

for i in range(len(ak.fields(Run3_matched_charged_greaterPt_PFs))):
  Run3_matched_charged_greaterPt_PF_dict[ak.fields(Run3_matched_charged_greaterPt_PFs)[i]] = Run3_matched_charged_greaterPt_PF[i]
for i in range(len(ak.fields(Run3_matched_charged_lowerPt_PFs))):
  Run3_matched_charged_lowerPt_PF_dict[ak.fields(Run3_matched_charged_lowerPt_PFs)[i]] = Run3_matched_charged_lowerPt_PF[i]

Run2_jets = ak.zip({
            "pt":  Run2_tree["jet_pt"].array(),
            "eta": Run2_tree["jet_eta"].array(),
            "phi": Run2_tree["jet_phi"].array(),
            "mass": 0.140,
})

Run2_taus = ak.zip({
            "pt":  Run2_tree["genPart_pt"].array(),
            "eta": Run2_tree["genPart_eta"].array(),
            "phi": Run2_tree["genPart_phi"].array(),
            "mass": 0.140,
            "vertexZ": Run2_tree["genPart_vertexZ"].array(),
            "vertexRho": Run2_tree["genPart_vertexRho"].array(),
            "index": ak.local_index(Run2_tree["genPart_pt"].array()),
})

Run2_PF_dict = {}
for branch in Run2_tree.keys():
  if "PF" in branch:
    Run2_PF_dict[branch] = Run2_tree[branch].array()
Run2_PFs = ak.zip(Run2_PF_dict)

Run2_taus = Run2_taus[Run2_tree["genVisTau_genPartIdx"].array()]

Run2_Jet_selections = ((Run2_jets.pt > ts.genTauPtMin) & (abs(Run2_jets.eta) < ts.jetEtaMax))
Run2_Tau_selections = ((Run2_taus.pt > ts.genTauPtMin) & (abs(Run2_taus.eta) < ts.genTauEtaMax) & (Run2_taus.vertexZ < ts.genTauVtxZMax) & (Run2_taus.vertexRho < ts.genTauVtxRhoMax))

Run2_selected_jets = Run2_jets[Run2_Jet_selections]
Run2_selected_taus = Run2_taus[Run2_Tau_selections] 

Run2_awk_taus = ak.with_name(Run2_selected_taus, "Momentum4D")
Run2_awk_jets = ak.with_name(Run2_selected_jets, "Momentum4D")

Run2_jet_tau = ak.cartesian({"taus": Run2_awk_taus, "jets": Run2_awk_jets}, axis =-1, nested=True) 
Run2_tau, Run2_jet = ak.unzip(Run2_jet_tau)
Run2_dR = Run2_tau.deltaR(Run2_jet)

## add min dR wrt any jet
#taus["min_dR_jet"] = ak.min(dR, axis = -1,keepdims=True)
## get index of best matched jet
Run2_best_dR = ak.argmin(Run2_dR, axis=-1, keepdims=False)
### this can be further improved in the future, to only allow one combination 
Run2_matched_PFs = Run2_PFs[Run2_best_dR]
Run2_charged_PF_selections = ((abs(Run2_matched_PFs.PF_charge) > 0))
Run2_greaterpt_selections = (Run2_matched_PFs.PF_pt >= 0.5)
Run2_lowerpt_selections = (Run2_matched_PFs.PF_pt < 0.5)
Run2_matched_charged_PFs = Run2_matched_PFs[Run2_charged_PF_selections]
Run2_matched_charged_greaterPt_PFs = Run2_matched_PFs[Run2_charged_PF_selections & Run2_greaterpt_selections]
Run2_matched_charged_lowerPt_PFs = Run2_matched_PFs[Run2_charged_PF_selections & Run2_lowerpt_selections]

Run2_matched_charged_greaterPt_PF = ak.unzip(Run2_matched_charged_greaterPt_PFs)
Run2_matched_charged_lowerPt_PF = ak.unzip(Run2_matched_charged_lowerPt_PFs)

Run2_matched_charged_greaterPt_PF_dict = {}
Run2_matched_charged_lowerPt_PF_dict = {}

for i in range(len(ak.fields(Run2_matched_charged_greaterPt_PFs))):
  Run2_matched_charged_greaterPt_PF_dict[ak.fields(Run2_matched_charged_greaterPt_PFs)[i]] = Run2_matched_charged_greaterPt_PF[i]
for i in range(len(ak.fields(Run2_matched_charged_lowerPt_PFs))):
  Run2_matched_charged_lowerPt_PF_dict[ak.fields(Run2_matched_charged_lowerPt_PFs)[i]] = Run2_matched_charged_lowerPt_PF[i]


for branch in Run2_matched_charged_greaterPt_PF_dict:
  Run2_matched_charged_greaterPt_PF_dict[branch] = ak.flatten(Run2_matched_charged_greaterPt_PF_dict[branch], axis = None)
  Run3_matched_charged_greaterPt_PF_dict[branch] = ak.flatten(Run3_matched_charged_greaterPt_PF_dict[branch], axis = None)
  Run2_matched_charged_greaterPt_PF_dict[branch] = [x for x in Run2_matched_charged_greaterPt_PF_dict[branch] if abs(x) < 9999]
  Run3_matched_charged_greaterPt_PF_dict[branch] = [x for x in Run3_matched_charged_greaterPt_PF_dict[branch] if abs(x) < 9999]
  
for branch in Run2_matched_charged_lowerPt_PF_dict:
  Run2_matched_charged_lowerPt_PF_dict[branch] =  ak.flatten(Run2_matched_charged_lowerPt_PF_dict[branch], axis = None)
  Run3_matched_charged_lowerPt_PF_dict[branch] =  ak.flatten(Run3_matched_charged_lowerPt_PF_dict[branch], axis = None)
  Run2_matched_charged_lowerPt_PF_dict[branch] = [x for x in Run2_matched_charged_lowerPt_PF_dict[branch] if abs(x) < 9999]
  Run3_matched_charged_lowerPt_PF_dict[branch] = [x for x in Run3_matched_charged_lowerPt_PF_dict[branch] if abs(x) < 9999]


##          Input variable            bins                        xlabel            ylabel
histDict = {
           "pt"               : [np.linspace(0, 100, 51),       "pt [GeV]",         "A.U."        ], 
           "eta"              : [np.linspace(-2.4, 2.4, 49),    "eta",              "A.U."        ],
           "phi"              : [np.linspace(-3.2, 3.2, 65),    "phi",              "A.U."        ],
           "mass"             : [np.linspace(0, 0.2, 41),       "mass [GeV]",       "A.U."        ],
           "charge"           : [np.linspace(-1, 2, 4),         "charge",           "A.U."        ],
           "puppiWeight"      : [np.linspace(0, 1, 51),         "puppiWeight",      "A.U."        ],
           "lostInnerHits"    : [np.linspace(-1, 3, 5),         "lostInnerHits",    "A.U."        ],
           "nPixelHits"       : [np.linspace(0, 12, 13),        "nPixelHits",       "A.U."        ],
           "nHits"            : [np.linspace(0, 50, 51),        "nHits",            "A.U."        ],
           "caloFraction"     : [np.linspace(0, 2.6, 53),       "caloFraction",     "A.U."        ],
           "hcalFraction"     : [np.linspace(0, 1.2, 25),       "hcalFraction",     "A.U."        ],
           "hasTrackDetails"  : [np.linspace(0, 2, 3),          "hasTrackDetails",  "A.U."        ],
           "dz"               : [np.linspace(-100, 100, 201),   "dz [cm]",          "A.U."        ],
           "dz_mod"           : [np.linspace(-25, 25, 101),      "dz [cm]",          "A.U."        ],
           "dzError"          : [np.linspace(0, 20, 41),        "dzError [cm]",     "A.U."        ],
           "dxy"              : [np.linspace(-50, 50, 101),     "dxy [cm]",         "A.U."        ],
           "dxyError"         : [np.linspace(0, 6, 61),         "dxyError [cm]",    "A.U."        ],
           "deta"             : [np.linspace(-7, 7, 201),        "deta",             "A.U."        ],
           "dphi"             : [np.linspace(-1, 1, 101),       "dphi",             "A.U."        ],
           "puppiWeightNoLep" : [np.linspace(0, 1, 51),         "puppiWeightNoLep", "A.U."        ],
           "rawCaloFraction"  : [np.linspace(0, 2.6, 53),       "rawCaloFraction",  "A.U."        ],
           "rawHcalFraction"  : [np.linspace(0, 1.2, 25),       "rawHcalFraction",  "A.U."        ],
           "chi2"             : [np.linspace(0, 20, 41),     "Chi^2",            "A.U."        ],
           "ndof"             : [np.linspace(0, 60, 31),        "ndof",             "A.U"         ],
           "nchi2"             : [np.linspace(0, 10, 21),     "Normalized Chi^2",            "A.U."        ],
           }



for branch in Run2_matched_charged_greaterPt_PF_dict:
  plt.cla()
  plt.clf()
  y2, bin2, other2 = plt.hist(Run2_matched_charged_greaterPt_PF_dict[branch], bins=histDict[branch.split("_")[-1]][0], weights=[1/len(Run2_matched_charged_greaterPt_PF_dict[branch]),]*len(Run2_matched_charged_greaterPt_PF_dict[branch]), histtype = 'step', label="Run2 M = 100 GeV, ctau = 100mm, RMS = " + "{:.1f}".format(np.std(Run2_matched_charged_greaterPt_PF_dict[branch])))
  y3, bin3, other3 = plt.hist(Run3_matched_charged_greaterPt_PF_dict[branch], bins=histDict[branch.split("_")[-1]][0], weights=[1/len(Run3_matched_charged_greaterPt_PF_dict[branch]),]*len(Run3_matched_charged_greaterPt_PF_dict[branch]), histtype = 'step', label="Run3 M = 100 GeV, ctau = 100mm, RMS = " + "{:.1f}".format(np.std(Run3_matched_charged_greaterPt_PF_dict[branch])))
  bincenters2 = 0.5*(bin2[1:]+bin2[:-1])
  bincenters3 = 0.5*(bin3[1:]+bin3[:-1])
  y2err = np.sqrt(y2 * len(Run2_matched_charged_greaterPt_PF_dict[branch]))/len(Run2_matched_charged_greaterPt_PF_dict[branch])
  y3err = np.sqrt(y3 * len(Run3_matched_charged_greaterPt_PF_dict[branch]))/len(Run3_matched_charged_greaterPt_PF_dict[branch])
  
  plt.errorbar(bincenters2, y2, yerr = y2err, ls = 'none')
  plt.errorbar(bincenters3, y3, yerr = y3err, ls = 'none')
  plt.title("Charged PF Cand w/ pt > 0.5 GeV " + branch.split("_")[-1] + " for matched jets"); plt.xlabel(histDict[branch.split("_")[-1]][1]); plt.ylabel(histDict[branch.split("_")[-1]][2])
  #if "pt" in branch or "Frac" in branch or "Weight" in branch or "Error" in branch or "mass" in branch or "dxy" in branch or "dz" in branch:
  plt.yscale("log")
  legend = plt.legend(fontsize="5.5")
  legend.get_frame().set_alpha(0)
  plt.savefig("PFCandPlotImages/matchPF_" + branch.split("_")[-1] + "_chargedPF_greater0p5_log.png")

for branch in histDict:
  if "mod" in branch:
    plt.cla()
    plt.clf()
    y2, bin2, other2 = plt.hist(Run2_matched_charged_lowerPt_PF_dict["PF_" + branch.split("_")[0]], bins=histDict[branch][0], weights=[1/len(Run2_matched_charged_lowerPt_PF_dict["PF_" + branch.split("_")[0]]),]*len(Run2_matched_charged_lowerPt_PF_dict["PF_" + branch.split("_")[0]]), histtype = 'step', label="Run2 M = 100 GeV, ctau = 100mm, RMS = " + "{:.1f}".format(np.std(Run2_matched_charged_lowerPt_PF_dict["PF_" + branch.split("_")[0]])))
    y3, bin3, other3 = plt.hist(Run3_matched_charged_lowerPt_PF_dict["PF_" + branch.split("_")[0]], bins=histDict[branch][0], weights=[1/len(Run3_matched_charged_lowerPt_PF_dict["PF_" + branch.split("_")[0]]),]*len(Run3_matched_charged_lowerPt_PF_dict["PF_" + branch.split("_")[0]]), histtype = 'step', label="Run3 M = 100 GeV, ctau = 100mm, RMS = " + "{:.1f}".format(np.std(Run3_matched_charged_lowerPt_PF_dict["PF_" + branch.split("_")[0]])))
    bincenters2 = 0.5*(bin2[1:]+bin2[:-1])
    bincenters3 = 0.5*(bin3[1:]+bin3[:-1])
    y2err = np.sqrt(y2 * len(Run2_matched_charged_lowerPt_PF_dict["PF_" + branch.split("_")[0]]))/len(Run2_matched_charged_lowerPt_PF_dict["PF_" + branch.split("_")[0]])
    y3err = np.sqrt(y3 * len(Run3_matched_charged_lowerPt_PF_dict["PF_" + branch.split("_")[0]]))/len(Run3_matched_charged_lowerPt_PF_dict["PF_" + branch.split("_")[0]])
    
    plt.errorbar(bincenters2, y2, yerr = y2err, ls = 'none')
    plt.errorbar(bincenters3, y3, yerr = y3err, ls = 'none')
    plt.title("Charged PF Cand w/ pt < 0.5 GeV" + branch.split("_")[0] + " for matched jets"); plt.xlabel(histDict[branch][1]); plt.ylabel(histDict[branch][2])
    #if "pt" in branch or "Frac" in branch or "Weight" in branch or "Error" in branch or "mass" in branch or "dxy" in branch:
    #  plt.yscale("log")
    legend = plt.legend(fontsize="5.5")
    legend.get_frame().set_alpha(0)
    plt.savefig("PFCandPlotImages/matchPF_" + branch + "_chargedPF_lower0p5.png")
