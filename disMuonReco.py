import uproot
import scipy
import matplotlib as mpl
import awkward as ak
import numpy as np
import math
import ROOT
from array import array
import pandas as pd

stauFile = uproot.open("Staus_M_100_100mm_13p6TeV_Run3Summer22_NanoAOD.root")
Events = stauFile["Events"]
muonBranches = {}
genPartBranches = {}

can = ROOT.TCanvas("can", "can")
  


for key in Events.keys():
  if "Muon_" in key:
    muonBranches[key] = Events[key].array()
  if "GenPart_" in key:
    genPartBranches[key] = Events[key].array()

def isMotherStau(evt, part_idx):
  #Find its mother
  mother_idx = genPartBranches["GenPart_genPartIdxMother"][evt][part_idx]
  #If it's a stau, we're done
  if (abs(genPartBranches["GenPart_pdgId"][evt][mother_idx]) == 1000015):
    return 1
  #If it's a tau, go further up the chain
  if (abs(genPartBranches["GenPart_pdgId"][evt][mother_idx]) == 15):
    return isMotherStau(evt, mother_idx)
  #Else, it's  not what we're looking for
  else:
    return 0        

  

# This chooses the gen particles which are muons which are decayed from displaced taus
genDisMuon = []
for evt in range(len(genPartBranches["GenPart_pdgId"])):
  genDisMuon_evt = []
  if (len(genPartBranches["GenPart_pdgId"][evt]) != 0):
    for part_idx in range(len(genPartBranches["GenPart_pdgId"][evt])):
      if (abs(genPartBranches["GenPart_pdgId"][evt][part_idx]) == 13):
        if (isMotherStau(evt, part_idx)):
          genDisMuon_evt.append(part_idx)
  genDisMuon.append(genDisMuon_evt)
print(len(ak.flatten(genDisMuon)))

genDisMuonPt = genPartBranches["GenPart_pt"][genDisMuon]
genDisMuonEta = genPartBranches["GenPart_eta"][genDisMuon]
#genDisMuondz = 

filteredRecoMuon = [np.array([val for val in subarr if val in values]) for subarr, values in zip(muonBranches["Muon_genPartIdx"], genDisMuon)]
filteredRecoMuonIndices = [np.where(np.isin(subarr, values))[0] for subarr, values in zip(muonBranches["Muon_genPartIdx"], genDisMuon)]

filteredRecoMuonPt = muonBranches["Muon_pt"][filteredRecoMuonIndices]
filteredRecoMuonEta = muonBranches["Muon_eta"][filteredRecoMuonIndices]
filteredRecoMuondz = muonBranches["Muon_dz"][filteredRecoMuonIndices]
filteredRecoMuondxy = muonBranches["Muon_dxy"][filteredRecoMuonIndices]

disMuon_pteff = ROOT.TEfficiency("disMuon_pteff", "Staus_M_100_100mm_13p6TeV_Run3Summer22; pt [GeV]; Fraction of gen dis muons which are recon'd", 25, 0, 100)
for i in range(25):
  disMuon_pteff.SetTotalEvents(i + 1, len(ak.flatten(genDisMuonPt[(genDisMuonPt > (i * 4)) & (genDisMuonPt < ((i + 1) * 4))])))
  disMuon_pteff.SetPassedEvents(i + 1, len(ak.flatten(filteredRecoMuonPt[(filteredRecoMuonPt > (i * 4)) & (filteredRecoMuonPt < ((i + 1) * 4))])))

disMuon_etaeff = ROOT.TEfficiency("disMuon_etaeff", "Staus_M_100_100mm_13p6TeV_Run3Summer22; #eta; Fraction of gen dis muons which are recon'd", 16, -2.4, 2.4)
for i in range(16):
  disMuon_etaeff.SetTotalEvents(i + 1, len(ak.flatten(genDisMuonEta[(genDisMuonEta > (-2.4 + 0.3 * i)) & (genDisMuonEta < (-2.4 + (i + 1) * 0.3))])))
  disMuon_etaeff.SetPassedEvents(i + 1, len(ak.flatten(filteredRecoMuonEta[(filteredRecoMuonEta > (i * 0.3 - 2.4)) & (filteredRecoMuonEta < ((i + 1) * 0.3 - 2.4))])))

disMuon_pteff.Draw()
can.SaveAs("disMuon_recoPtEff.pdf")

disMuon_etaeff.Draw()
can.SaveAs("disMuon_recoEtaEff.pdf")

print(len(ak.flatten(filteredRecoMuon)))

