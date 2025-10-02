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
from final_fileset import *
from xsec import * 

# Lets us use functions specific for Muon branches for DisMuon
PFNanoAODSchema.mixins["DisMuon"] = "Muon"

lumi = 38.01 ##fb-1

for samp in final_fileset: 
    final_fileset[samp]["dataset"] = samp

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

color_dict = {'QCD_disp':'#56CBF9', 
              'QCD_notdisp': '#FDCA40', 
              'Stau_100_100mm_NotDisplaced': '#5DFDCB', 
              #'DY': '#D3C0CD', 
              'Stau_100_100mm_Displaced': '#B80C09',
             }

variables = ["pfRelIso03_all",
             "pfRelIso03_chg",
             "pfRelIso04_all",
             "tkRelIso",
            ]


num_bins = 50
range_min = 0
range_max = 1

with open("muon_QCD_hists_Iso_Displaced.pkl", "rb") as file:
    QCD_disp_hists = pickle.load(file)
with open("muon_QCD_hists_Iso_NotDisplaced.pkl", "rb") as file:
    QCD_notdisp_hists = pickle.load(file)

signal = [
          "Stau_100_100mm_Displaced",
          "Stau_100_100mm_NotDisplaced",
         ]


# Can also be put in a utils file later
def delta_r_mask(first: ak.highlevel.Array, second: ak.highlevel.Array, threshold: float) -> ak.highlevel.Array:
            mval = first.metric_table(second)
            return ak.all(mval > threshold, axis=-1)

class IsoROCC(processor.ProcessorABC): 
    def __init__(self):
        pass 

    def initialize_hist(self):
        isoHist = {}
        for sig in signal:
            isoHist[sig] = {}
            for var in variables: 
                isoHist[sig][var] = hda.hist.Hist(hist.axis.Regular(num_bins, range_min, range_max, name = var, label = var, underflow = True, overflow = True))
        return isoHist

    def process(self, events):
        
        # Now that our events have been selected, for our signal, choose the reco muons that match to a gen muon which  comes from the stau decay chain
        RecoMuons = events.DisMuon[(abs(events.GenPart[events.DisMuon.genPartIdx].pdgId) == 13)
                  & (abs(events.GenPart[events.DisMuon.genPartIdx].distinctParent.distinctParent.pdgId) == 1000015)
                  ]
        RecoMuons = ak.firsts(RecoMuons)
         
        isoHist = self.initialize_hist()

        for sig in signal:
            if sig == events.metadata['dataset']
            for var in variables:
                isoHist[sig][var].fill(**{var: dak.flatten(RecoMuons[var], axis = None)}, weight = xsecs[events.metadata['dataset']] * lumi * 1000 * 1/num_events[events.metadata['dataset']])

        return isoHist

    def postprocess(self):
        pass
    
dataset_runnable, dataset_updated = preprocess(
    final_fileset,
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

        num[proc][var].Sumw2()
        den[proc][var].Sumw2()

        if "QCD_disp" in proc:        
            for i in range(num_bins): 
                num[proc][var].SetBinContent(i + 1, ak.sum(QCD_disp_hists[f"DisMuon_{var}"].view(flow=True)[1:num_bins + 1 - i]))
                den[proc][var].SetBinContent(i + 1, ak.sum(QCD_disp_hists[f"DisMuon_{var}"].view(flow=True)))                    
        else if "QCD_notdisp" in proc:        
            for i in range(num_bins): 
                num[proc][var].SetBinContent(i + 1, ak.sum(QCD_notdisp_hists[f"DisMuon_{var}"].view(flow=True)[1:num_bins + 1 - i]))
                den[proc][var].SetBinContent(i + 1, ak.sum(QCD_notdisp_hists[f"DisMuon_{var}"].view(flow=True)))                    
        else if "Stau_100_100mm_NotDisplaced" in proc: 
            for i in range(num_bins):
                num[proc][var].SetBinContent(i + 1, ak.sum(out[proc][var].view(flow=True)[1:num_bins + 1 - i]))
                den[proc][var].SetBinContent(i + 1, ak.sum(out[proc][var].view(flow=True))) 
        else if "Stau_100_100mm_Displaced" in proc: 
            for i in range(num_bins):
                num[proc][var].SetBinContent(i + 1, ak.sum(out[proc][var].view(flow=True)[1:num_bins + 1 - i]))
                den[proc][var].SetBinContent(i + 1, ak.sum(out[proc][var].view(flow=True))) 

        if (ROOT.TEfficiency.CheckConsistency(num[proc][var], den[proc][var])):
            eff[proc][var] =  ROOT.TEfficiency(num[proc][var], den[proc][var])
        else:
            print("Numerator and denominator histograms are not the same!!")
        


canvas = ROOT.TCanvas("canvas", "Efficiency Plot", 800, 600)
            
#for var in variables:
#    print(f"Starting efficiency plot for {var}")
#    legend = ROOT.TLegend(0.75, 0.3, 0.9, 0.5)
#    legend.SetBorderSize(0)  # No border
#    legend.SetFillStyle(0)   # Transparent background
#    legend.SetTextSize(0.03)
#    for i, (proc, proc_eff) in enumerate(eff.items()): 
#        eff[proc][var].SetMarkerColor(ROOT.TColor.GetColor(color_dict[proc]))
#        eff[proc][var].SetMarkerStyle(20)
#        legend.AddEntry(eff[proc][var], proc)
#        draw_option = "AP" if i == 0 else "P SAME"
#        eff[proc][var].Draw(draw_option)
#        ROOT.gPad.Update()
#        eff[proc][var].GetPaintedGraph().GetYaxis().SetRangeUser(0, 1.1)
#        eff[proc][var].SetTitle(f";{var};Efficiency")
#
#    canvas.Update()
#    legend.Draw()
#    canvas.SaveAs(f"{var}_eff.pdf")
#    canvas.SaveAs(f"../www/roc_curves/{var}_eff.png")

eff_list = {}

for proc in eff:
    eff_list[proc] = {}
    for var in variables:
        print(f"For sample {proc} and variable {var}, get efficiencies")
        eff_list[proc][var] = array('d') 
    
        for i in range(num_bins): 
            eff_list[proc][var].append(eff[proc][var].GetEfficiency(i + 1))

ROCC = {}
rocc_colors = ["#c0a9b0", "#7880b5", "#6bbaec", "#1f271b"]
rocc_legend = ROOT.TLegend()
rocc_legend.SetBorderSize(0)  # No border
rocc_legend.SetFillStyle(0)   # Transparent background
for sig in signal
    ROCC[sig] = {}:
    for i, var in enumerate(variables):
        print(f"Making ROC curve for {var}")
        ROCC[sig][var] = ROOT.TGraph(num_bins, eff_list["QCD"][var], eff_list["Stau_100_100mm"][var])
        ROCC[sig][var].SetLineColor(ROOT.TColor.GetColor(rocc_colors[i]))
        ROCC[sig][var].SetMarkerColor(ROOT.TColor.GetColor(rocc_colors[i]))
        ROCC[sig][var].SetMarkerStyle(20 + i)
        ROCC[sig][var].SetLineWidth(2)
    
        if i == 0:
            ROCC[var].Draw("APL")
            ROCC[var].GetXaxis().SetTitle("QCD")
            ROCC[var].GetYaxis().SetTitle("Stau")
            ROCC[var].SetTitle("Isolation ROC Curves")
        else:
            ROCC[var].Draw("PL SAME")
    
        rocc_legend.AddEntry(ROCC[var], var, "LP")
    
    rocc_legend.Draw()
    canvas.SaveAs("IsoROCC.pdf")
    canvas.SaveAs("../www/roc_curves/isolation_roc_curves.png")

for i, var in enumerate(variables):
    print(f"Making separate ROC curve for {var}")
    ROCC[var] = ROOT.TGraph(num_bins, eff_list["QCD"][var], eff_list["Stau_100_100mm"][var])
    ROCC[var].SetLineColor(ROOT.TColor.GetColor(rocc_colors[i]))
    ROCC[var].SetMarkerColor(ROOT.TColor.GetColor(rocc_colors[i]))
    ROCC[var].SetMarkerStyle(20 + i)
    ROCC[var].SetLineWidth(2)

    ROCC[var].Draw("APL")
    ROCC[var].GetXaxis().SetTitle("QCD")
    ROCC[var].GetYaxis().SetTitle("Stau")
    ROCC[var].SetTitle(f"{var} ROC Curve")

    latex = ROOT.TLatex()
    for j in range(num_bins): 
        latex.DrawLatex(eff_list["QCD"][var][j] + 0.04, eff_list["Stau_100_100mm"][var][j], f"{1 - 1/num_bins * j:.2f}")
        latex.SetTextColor(ROOT.TColor.GetColor(rocc_colors[i]))
        latex.SetTextSize(0.01)


    canvas.SaveAs(f"{var}_ROCC.pdf")
    canvas.SaveAs(f"../www/roc_curves/{var}_ROCC.png")
    
 
