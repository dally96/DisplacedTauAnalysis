import numpy as np
import awkward as ak
import ROOT 
import math
import scipy
import array
import pandas as pd
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from leptonPlot import *

ptRANGE = ["20-25", "25-30", "30-35", "35-40", "40-45", "45-50", "50-55", "55-60", 
           "60-65", "65-70", "70-75", "75-80", "80-85", "85-90", "90-95", "95-100"]

etaRANGE = ["-2.4--2.2", "-2.2--2.0", "-2.0--1.8", "-1.8--1.6", "-1.6--1.4", "-1.4--1.2", "-1.2--1.0", "-1.0--0.8", "-0.8--0.6", "-0.6--0.4", "-0.4--0.2", "-0.2-0",
            "0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0", "1.0-1.2", "1.2-1.4", "1.4-1.6", "1.6-1.8", "1.8-2.0", "2.0-2.2", "2.2-2.4"]

#dxyRANGE = ["0-0.5", "0.5-1", "1-1.5", "1.5-2", "2-2.5","2.5-3", "3-3.5", "3.5-4", "4-4.5", "4.5-5",
#            "5-5.5", "5.5-6", "6-6.5", "6.5-7", "7-7.5","7.5-8", "8-8.5", "8.5-9", "9-9.5", "9.5-10",
#            "10-10.5", "10.5-11", "11-11.5", "11.5-12", "12-12.5","12.5-13", "13-13.5", "13.5-14", "14-14.5", "14.5-15"]
dxyRANGE = ["0-0.5", "0.5-1", "1-1.5", "1.5-2", "2-2.5", "2.5-3", "3-3.5", "3.5-4", "4-4.5", "4.5-5", "5-7", "7-10", "10-15"]
dxyBins = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 7, 10, 15]
dxyBins = array.array('d', dxyBins)
lxyRANGE = ["0-0.5", "0.5-1", "1-1.5", "1.5-2", "2-2.5","2.5-3", "3-3.5", "3.5-4", "4-4.5", "4.5-5",
            "5-5.5", "5.5-6", "6-6.5", "6.5-7", "7-7.5","7.5-8", "8-8.5", "8.5-9", "9-9.5", "9.5-10",
            "10-10.5", "10.5-11", "11-11.5", "11.5-12", "12-12.5","12.5-13", "13-13.5", "13.5-14", "14-14.5", "14.5-15"]

file = "Staus_M_100_100mm_13p6TeV_Run3Summer22_lpcdisptau_NanoAOD_ExtraDisMuonBranches.root"

events = NanoEventsFactory.from_root({file:"Events"}).events()
ROOT.gStyle.SetOptStat(0)

gpart = events.GenPart
gvtx = events.GenVtx
gvistau = events.GenVisTau
muons = events.DisMuon

staus = gpart[(abs(gpart.pdgId) == 1000015) & (gpart.hasFlags("isLastCopy"))]
staus_taus = staus.distinctChildren[(abs(staus.distinctChildren.pdgId) == 15) & (staus.distinctChildren.hasFlags("isLastCopy"))]
gen_muons = staus_taus.distinctChildren[(abs(staus_taus.distinctChildren.pdgId) == 13)]
gen_muons = gen_muons[(gen_muons.pt > GenPtMin) & (abs(gen_muons.eta) < GenEtaMax)]

RecoMuons = muons[(abs(gpart[muons.genPartIdx.compute()].pdgId) == 13)
                  & (abs(gpart[muons.genPartIdx.compute()].distinctParent.distinctParent.pdgId) == 1000015)
                  & (gpart[muons.genPartIdx.compute()].pt > GenPtMin) & (abs(gpart[muons.genPartIdx.compute()].eta) < GenEtaMax)]

RecoMuonsFromGen = gpart[muons.genPartIdx.compute()]
RecoMuonsFromGen = RecoMuonsFromGen[(abs(RecoMuonsFromGen.pdgId) == 13) & (abs(RecoMuonsFromGen.distinctParent.distinctParent.pdgId) == 1000015)]
RecoMuonsFromGen = RecoMuonsFromGen[(RecoMuonsFromGen.pt > GenPtMin) & (abs(RecoMuonsFromGen.eta) < GenEtaMax)]
RecoMuonsFromGen["dxy"] = (RecoMuonsFromGen.vertexY - gvtx.y) * np.cos(RecoMuonsFromGen.phi) - (RecoMuonsFromGen.vertexX - gvtx.x) * np.sin(RecoMuonsFromGen.phi)
RecoMuonsFromGen["lxy"] = np.sqrt((RecoMuonsFromGen.vertexX - ak.firsts(staus.vertexX)) ** 2 + (RecoMuonsFromGen.vertexY - ak.firsts(staus.vertexY)) ** 2)


makeResPlot("mu", ["DisMuons matched to #tau_{#mu}"], "pt", "dxy", ptRANGE, 20, 100, -0.1, 0.1, 5, [RecoMuonsFromGen.pt.compute()], [(RecoMuons.dxy - RecoMuonsFromGen.dxy).compute()], "[GeV]", "[cm]", file) 
makeResPlot("mu", ["DisMuons matched to #tau_{#mu}"], "eta", "dxy", etaRANGE, -2.4, 2.4, -0.1, 0.1, 0.2, [RecoMuonsFromGen.eta.compute()], [(RecoMuons.dxy - RecoMuonsFromGen.dxy).compute()], "", "[cm]", file) 
makeResPlot_varBin("mu", ["DisMuons matched to #tau_{#mu}"], "dxy", "dxy", dxyRANGE, dxyBins, -0.1, 0.1, [abs(RecoMuonsFromGen.dxy.compute())], [(RecoMuons.dxy - RecoMuonsFromGen.dxy).compute()], "[cm]", "[cm]", file) 
makeResPlot("mu", ["DisMuons matched to #tau_{#mu}"], "lxy", "dxy", lxyRANGE, 0, 15, -0.1, 0.1, 0.5, [RecoMuonsFromGen.lxy.compute()], [(RecoMuons.dxy - RecoMuonsFromGen.dxy).compute()], "[cm]", "[cm]", file) 
#makeResPlot("mu", ["DisMuons matched to #tau_{#mu}"], "dxy", "dxy", lxyRANGE, 0, 15, -0.1, 0.1, 0.5, [RecoMuonsFromGen.dxy.compute()], [(RecoMuons.dxy - RecoMuonsFromGen.dxy).compute()], "[cm]", "[cm]", file) 


