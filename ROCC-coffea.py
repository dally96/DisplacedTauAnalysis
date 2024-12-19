# Standard python packages
import ROOT
import uproot
import scipy
import numpy
import awkward as ak
import dill as pickle
from array import array

# Coffea modules
import coffea
from coffea import processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, PFNanoAODSchema
from coffea.analysis_tools import PackedSelection
from coffea.dataset_tools import (
    apply_to_fileset,
    max_chunks,
    preprocess,
)
# Silence obnoxious warning
NanoAODSchema.warn_missing_crossrefs = False

# Dask related packages and modules
import hist.dask as hda
import hist
import dask 
import dask_awkward as dak

# Python files
from fileset import *
from xsec import * 

# Lets us use functions specific for Muon branches for DisMuon
PFNanoAODSchema.mixins["DisMuon"] = "Muon"

lumi = 38.01 ##fb-1

for samp in fileset: 
    fileset[samp]["dataset"] = samp

SAMP = [ 
        #'Stau_100_100mm',
        #'QCD50_80',
        #'QCD80_120',
        #'QCD120_170',
        #'QCD170_300',
        #'QCD300_470',
        #'QCD470_600',
        #'QCD600_800',
        #'QCD800_1000',
        #'QCD1000_1400',
        #'QCD1400_1800',
        #'QCD1800_2400',
        #'QCD2400_3200',
        #'QCD3200',
      ]

color_dict = {'QCD':'#56CBF9', 
              #'TT': '#FDCA40', 
              #'W': '#5DFDCB', 
              #'DY': '#D3C0CD', 
              'Stau_100_100mm': '#B80C09',
             }

variables = ["pfRelIso03_all",
             "pfRelIso03_chg",
             "pfRelIso04_all",
            ]

selections = { 
              "muon_pt":                    30, ##GeV
              "muon_eta":                   1.5,
              "muon_ID":                    "muon_tightId",
              "muon_dxy_prompt_max":        50E-4, ##cm
              "muon_dxy_prompt_min":        0E-4, ##cm
              "muon_dxy_displaced_min":     100E-4, ##cm
              "muon_dxy_displaced_max":     10, ##cm
 
              "jet_score":                  0.9, 
              "jet_pt":                     32, ##GeV
 
              "MET_pT":                     105, ##GeV
             }

num_bins = 25
range_min = 0
range_max = 1

with open("muon_QCD_hists_firsts.pkl", "rb") as file:
    QCD_hists = pickle.load(file)

# Can also be put in a utils file later
def delta_r_mask(first: ak.highlevel.Array, second: ak.highlevel.Array, threshold: float) -> ak.highlevel.Array:
            mval = first.metric_table(second)
            return ak.all(mval > threshold, axis=-1)

class IsoROCC(processor.ProcessorABC): 
    def __init__(self):
        pass 

    def initialize_hist(self):
        isoHist = {}
        for var in variables: 
            isoHist[var] = hda.hist.Hist(hist.axis.Regular(num_bins, range_min, range_max, name = var, label = var, underflow = True, overflow = True))
        return isoHist

    def process(self, events):
        
        # Define the "good muon" condition for each muon per event
        good_muon_mask = (
             (events.DisMuon.pt > 20)
            & (abs(events.DisMuon.eta) < 2.4) # Acceptance of the CMS muon system
        )      

        events['DisMuon'] = ak.drop_none(events.DisMuon[good_muon_mask])
        num_evts = ak.num(events, axis=0)
        num_good_muons = ak.count_nonzero(good_muon_mask, axis=1)
        events = events[num_good_muons >= 1]
        print("Processed muon preselection")
              
        # Perform the overlap removal with respect to muons, electrons and photons, dR=0.4
        events['Jet'] = events.Jet[delta_r_mask(events.Jet, events.Photon, 0.4)]
        events['Jet'] = events.Jet[delta_r_mask(events.Jet, events.Electron, 0.4)]
        events['Jet'] = events.Jet[delta_r_mask(events.Jet, events.DisMuon, 0.4)]

        # Define the "good jet" condition for each jet per event              
        good_jet_mask = (
            (events.Jet.pt > 20)
            & (abs(events.Jet.eta) < 2.4) 
        )     
              
        events['Jet'] = events.Jet[good_jet_mask]
        num_good_jets = ak.count_nonzero(good_jet_mask, axis=1)
        events = events[num_good_jets >= 1]
        print("Processed jet preselection")

        #Noise filter
        noise_mask = (
                     (events.Flag.goodVertices == 1) 
                     & (events.Flag.globalSuperTightHalo2016Filter == 1)
                     & (events.Flag.EcalDeadCellTriggerPrimitiveFilter == 1)
                     & (events.Flag.BadPFMuonFilter == 1)
                     & (events.Flag.BadPFMuonDzFilter == 1)
                     & (events.Flag.hfNoisyHitsFilter == 1)
                     & (events.Flag.eeBadScFilter == 1)
                     & (events.Flag.ecalBadCalibFilter == 1)
                     )
              
        events = events[noise_mask] 
        print("Processed noise mask")

        # These are more stringent selections on muons for signal vs background discrimination
        better_muons = (
                         (events.DisMuon.pt > selections["muon_pt"])
                       & (events.DisMuon.tightId == 1)
                       & (abs(events.DisMuon.dxy) > selections["muon_dxy_displaced_min"])
                       & (abs(events.DisMuon.dxy) < selections["muon_dxy_displaced_max"])
                       )

        # These are more stringent selections on jets for signal vs background discrimination
        better_jets =  ( 
                         (events.Jet.disTauTag_score1 > selections["jet_score"])
                       & (events.Jet.pt > selections["jet_pt"])
                       )

        # These are event level selections also for signal vs background discrimination
        better_events = (events.MET.pt > selections["MET_pT"])

        num_better_muons = ak.num(events.DisMuon[better_muons])
        num_better_jets  = ak.num(events.Jet[better_jets])
        
        better_muon_event_mask = num_better_muons > 0 
        better_jet_event_mask  = num_better_jets  > 0

        events = events[better_muon_event_mask & better_jet_event_mask & better_events]
        print("Applied current event selections")

        # Now that our events have been selected, for our signal, choose the reco muons that match to a gen muon which  comes from the stau decay chain
        RecoMuons = events.DisMuon[(abs(events.GenPart[events.DisMuon.genPartIdx].pdgId) == 13)
                  & (abs(events.GenPart[events.DisMuon.genPartIdx].distinctParent.distinctParent.pdgId) == 1000015)
                  ]
        RecoMuons = ak.firsts(RecoMuons)
         
        isoHist = self.initialize_hist()

        for var in variables:
            isoHist[var].fill(**{var: dak.flatten(RecoMuons[var], axis = None)}, weight = xsecs[events.metadata['dataset']] * lumi * 1000 * 1/num_events[events.metadata['dataset']])

        return isoHist

    def postprocess(self):
        pass
    
dataset_runnable, dataset_updated = preprocess(
    fileset,
    align_clusters=False,
    step_size=100_000_000,
    files_per_batch=1,
    skip_bad_files=True,
    save_form=False,
)


to_compute = apply_to_fileset(
             IsoROCC(),
             max_chunks(dataset_runnable, 10000000),
             schemaclass=PFNanoAODSchema,
)
(out,) = dask.compute(to_compute)
print(out)

eff = {}
num = {}
den = {}

for proc in color_dict:
    eff[proc] = {}
    num[proc] = {}
    den[proc] = {}

    for var in variables:

        num[proc][var] = ROOT.TH1F(f"h_{proc}_{var}_num", f";{var};Numerator", num_bins,  range_min, range_max)
        den[proc][var] = ROOT.TH1F(f"h_{proc}_{var}_den", f";{var};Denominator", num_bins, range_min, range_max)

        if "QCD" in proc:        
            for i in range(num_bins): 
                num[proc][var].SetBinContent(i + 1, ak.sum(QCD_hists[f"muon_{var}"].view(flow=True)[1:num_bins + 1 - i]))
                den[proc][var].SetBinContent(i + 1, ak.sum(QCD_hists[f"muon_{var}"].view(flow=True)))                    
        else: 
            for i in range(num_bins):
                num[proc][var].SetBinContent(i + 1, ak.sum(out[proc][var].view(flow=True)[1:num_bins + 1 - i]))
                den[proc][var].SetBinContent(i + 1, ak.sum(out[proc][var].view(flow=True))) 

        if (ROOT.TEfficiency.CheckConsistency(num[proc][var], den[proc][var])):
            eff[proc][var] =  ROOT.TEfficiency(num[proc][var], den[proc][var])
        else:
            print("Numerator and denominator histograms are not the same!!")
        


canvas = ROOT.TCanvas("canvas", "Efficiency Plot", 800, 600)
            
for var in variables:
    print(f"Starting efficiency plot for {var}")
    legend = ROOT.TLegend(0.75, 0.3, 0.9, 0.5)
    legend.SetBorderSize(0)  # No border
    legend.SetFillStyle(0)   # Transparent background
    legend.SetTextSize(0.03)
    for i, (proc, proc_eff) in enumerate(eff.items()): 
        eff[proc][var].SetMarkerColor(ROOT.TColor.GetColor(color_dict[proc]))
        eff[proc][var].SetMarkerStyle(20)
        legend.AddEntry(eff[proc][var], proc)
        draw_option = "AP" if i == 0 else "P SAME"
        eff[proc][var].Draw(draw_option)
        ROOT.gPad.Update()
        eff[proc][var].GetPaintedGraph().GetYaxis().SetRangeUser(0, 1.1)
        eff[proc][var].GetPaintedGraph().GetYaxis().SetTitle("Efficiency")

    canvas.Update()
    legend.Draw()
    canvas.SaveAs(f"{var}_eff.pdf")
    canvas.SaveAs(f"../www/roc_curves/{var}_eff.png")

eff_list = {}

for proc in eff:
    eff_list[proc] = {}
    for var in variables:
        print(f"For sample {proc} and variable {var}, get efficiencies")
        eff_list[proc][var] = array('d') 
    
        for i in range(num_bins): 
            eff_list[proc][var].append(eff[proc][var].GetEfficiency(i + 1))

ROCC = {}
rocc_colors = ["#c0a9b0", "#7880b5", "#6bbaec"]
#rocc_legend = ROOT.TLegend()
#rocc_legend.SetBorderSize(0)  # No border
#rocc_legend.SetFillStyle(0)   # Transparent background
#for i, var in enumerate(variables):
#    print(f"Making ROC curve for {var}")
#    ROCC[var] = ROOT.TGraph(num_bins, eff_list["QCD"][var], eff_list["Stau_100_100mm"][var])
#    ROCC[var].SetLineColor(ROOT.TColor.GetColor(rocc_colors[i]))
#    ROCC[var].SetMarkerColor(ROOT.TColor.GetColor(rocc_colors[i]))
#    ROCC[var].SetMarkerStyle(20 + i)
#    ROCC[var].SetLineWidth(2)
#
#    if i == 0:
#        ROCC[var].Draw("APL")
#        ROCC[var].GetXaxis().SetTitle("QCD")
#        ROCC[var].GetYaxis().SetTitle("Stau")
#        ROCC[var].SetTitle("Isolation ROC Curves")
#    else:
#        ROCC[var].Draw("PL SAME")
#
#    rocc_legend.AddEntry(ROCC[var], var, "LP")
#
#rocc_legend.Draw()
#canvas.SaveAs("IsoROCC.pdf")
#canvas.SaveAs("../www/roc_curves/isolation_roc_curves.png")

#for i, var in enumerate(variables):
#    print(f"Making separate ROC curve for {var}")
#    ROCC[var] = ROOT.TGraph(num_bins, eff_list["QCD"][var], eff_list["Stau_100_100mm"][var])
#    ROCC[var].SetLineColor(ROOT.TColor.GetColor(rocc_colors[i]))
#    ROCC[var].SetMarkerColor(ROOT.TColor.GetColor(rocc_colors[i]))
#    ROCC[var].SetMarkerStyle(20 + i)
#    ROCC[var].SetLineWidth(2)
#
#    ROCC[var].Draw("APL")
#    ROCC[var].GetXaxis().SetTitle("QCD")
#    ROCC[var].GetYaxis().SetTitle("Stau")
#    ROCC[var].SetTitle(f"{var} ROC Curve")
#
#    latex = ROOT.TLatex()
#    for j in range(num_bins): 
#        latex.DrawLatex(eff_list["QCD"][var][j] + 0.01, eff_list["Stau_100_100mm"][var][j], f"{1 - 1/num_bins * j:.2f}")
#        latex.SetTextColor(ROOT.TColor.GetColor(rocc_colors[i]))
#        latex.SetTextSize(0.01)
#
#
#    canvas.SaveAs(f"{var}_ROCC.pdf")
#    canvas.SaveAs(f"../www/roc_curves/{var}_ROCC.png")
    
 
