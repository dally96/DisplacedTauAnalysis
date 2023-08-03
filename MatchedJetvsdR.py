import numpy as np
import uproot 
import awkward as ak
import ROOT
import scipy
import math
import os
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
from array import array
exec(open("tau_func.py").read())

colors = [ROOT.kBlue, ROOT.kRed, ROOT.kGreen, ROOT.kViolet, ROOT.kMagenta, ROOT.kBlack]
workingBranches = ["GenVisTau", "Jet", "GenPart"]
can = ROOT.TCanvas("can", "can", 1000, 600)
can.SetLogy(0)
dR_legend = ROOT.TLegend(0.7,0.6,0.8,0.8)
color_index = 1
dRsqu_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
dR_squ = dRsqu_list[-1]


scoreSpace = np.linspace(0, 0.995,200)
file = "Stau_M_100_100mm_Summer18UL_NanoAOD.root"
Sample_file = uproot.open(file)
events = Sample_file["Events"]
eventBranches = events.keys()
nevt = events.num_entries 

histDict = {}
colorDict = {}

Branches = {}
for branch in workingBranches:
  for key in eventBranches:
    if branch in key.split("_")[0]:
      Branches[key] = events[key].array()

print(len(Branches))


matchedJets= []
matcheddR = []
matchedDisTauTagScore = []
    
for evt in range(nevt):
  matchedJets_evt = []
  matcheddR_evt = []
  matchedDisTauTagScore_evt = []
  if (len(Branches["GenVisTau_pt"][evt])== 0): continue
  for tau_idx in range(len(Branches["GenVisTau_pt"][evt])):
    if Branches["GenVisTau_pt"][evt][tau_idx] < 20 or abs(Branches["GenVisTau_eta"][evt][tau_idx]) > 2.4: continue 
    if not stauMother(evt, Branches["GenVisTau_genPartIdxMother"][evt][tau_idx], Branches["GenPart_pdgId"], Branches["GenPart_genPartIdxMother"]): continue 
    mthrTau_idx = Branches["GenVisTau_genPartIdxMother"][evt][tau_idx]
    stau_idx = stauIdx(evt, Branches["GenVisTau_genPartIdxMother"][evt][tau_idx], Branches["GenPart_pdgId"], Branches["GenPart_genPartIdxMother"])
    if Branches["GenPart_vertexZ"][evt][mthrTau_idx] >= 100 or Branches["GenPart_vertexRho"][evt][mthrTau_idx] >= 50: continue
    matchedJets_evt_tau = []
    matcheddR_evt_tau = []
    matchedDisTauTagScore_evt_tau = []
    for jet_idx in range(len(Branches["Jet_pt"][evt])):
      if Branches["Jet_pt"][evt][jet_idx] < 20 or abs(Branches["Jet_eta"][evt][jet_idx]) > 2.4: continue
      if Branches["Jet_genJetIdx"][evt][jet_idx] == -1 or Branches["Jet_genJetIdx"][evt][jet_idx] >= len(Branches["GenJet_partonFlavour"][evt]): continue

      dphi = abs(Branches["GenVisTau_phi"][evt][tau_idx]-Branches["Jet_phi"][evt][jet_idx])
            
      if (dphi > math.pi) :
        dphi -= 2 * math.pi
      
      deta = Branches["GenVisTau_eta"][evt][tau_idx]-Branches["Jet_eta"][evt][jet_idx]

      dR2  = dphi ** 2 + deta ** 2
      if (dR2 <= dR_squ ** 2):
        matchedJets_evt_tau.append(jet_idx)
        matcheddR_evt_tau.append(dR2)
        matchedDisTauTagScore_evt_tau.append(Branches["Jet_disTauTag_score1"][evt][jet_idx])
    matchedJets_evt.append(matchedJets_evt_tau)
    matcheddR_evt.append(matcheddR_evt_tau)
    matchedDisTauTagScore_evt.append(matchedDisTauTagScore_evt_tau)
  matchedJets.append(matchedJets_evt)
  matcheddR.append(matcheddR_evt)
  matchedDisTauTagScore.append(matchedDisTauTagScore_evt)

### If a jet is matched to more than one tau, pick tau with smallest matching ###
for evt in range(len(matchedJets)):
  if len(matchedJets[evt]) > 1:
    for tau in range(len(matchedJets[evt]) - 1):
      for tau2 in range(tau + 1, len(matchedJets[evt])): 
        matchedJets_intersection = set(matchedJets[evt][tau]).intersection(set(matchedJets[evt][tau2]))
        if len(matchedJets_intersection) > 0:
          for intersectJets in matchedJets_intersection:
            if matcheddR[evt][tau][matchedJets[evt][tau].index(intersectJets)] > matcheddR[evt][tau2][matchedJets[evt][tau2].index(intersectJets)]:
              matcheddR[evt][tau].pop(matchedJets[evt][tau].index(intersectJets))
              matchedDisTauTagScore[evt][tau].pop(matchedJets[evt][tau].index(intersectJets))
              matchedJets[evt][tau].pop(matchedJets[evt][tau].index(intersectJets))
            else:
              matcheddR[evt][tau2].pop(matchedJets[evt][tau2].index(intersectJets))
              matchedDisTauTagScore[evt][tau2].pop(matchedJets[evt][tau2].index(intersectJets))
              matchedJets[evt][tau2].pop(matchedJets[evt][tau2].index(intersectJets))
              
### If more than one jet is matched to a tau, pick the jet with the smallest matching ###
for evt in range(len(matchedJets)):
  for tau in range(len(matchedJets[evt])):
    if len(matchedJets[evt][tau]) > 1:
      poppedJetDisTauTagScore = []
      poppedJetdR = []
      poppedJet = []
      print("Event with more than one jet matched to a tau is ", evt)
      print("There are ",  len(matchedJets[evt][tau]), " jets matched to tau ", tau, " in this event")
      for jet in range(len(matchedJets[evt][tau])):
        print("The dR^2 for each jet for this tau is ", matcheddR[evt][tau])
        if matcheddR[evt][tau][jet] > min(matcheddR[evt][tau]):
          poppedJetdR.append(matcheddR[evt][tau][jet])
          poppedJetDisTauTagScore.append(matchedDisTauTagScore[evt][tau][jet])
          poppedJet.append(matchedJets[evt][tau][jet])
      for jet in range(len(poppedJetdR)):
          matcheddR[evt][tau].remove(poppedJetdR[jet])
          matchedDisTauTagScore[evt][tau].remove(poppedJetDisTauTagScore[jet])
          matchedJets[evt][tau].remove(poppedJet[jet])
      print("The dR^2 for each jet for this tau is now ", matcheddR[evt][tau])

matchedJet_dict = {"score" : ak.flatten(matchedDisTauTagScore, axis = None), "dR": ak.flatten(matcheddR, axis = None)}
matchedJet_df = pd.DataFrame(data = matchedJet_dict)

for dRsqu in dRsqu_list:
  histDict[str(dRsqu)] = ROOT.TH1F("h_"+str(dRsqu), "Jet score for different dR^{2} selections for Run 2 #tilde{#tau} M = 100 GeV, c#tau = 100 mm; score 1; Number of Jets", 200, 0, 1)
  colorDict[str(dRsqu)] = color_index
  matchedJet_dR = matchedJet_df.query("dR <= @dRsqu**2")
  print("For dR^2 ", dRsqu, " the number of jets matched to taus is ", len(matchedJet_dR))
  for row in range(len(matchedJet_dR)):
    histDict[str(dRsqu)].Fill(matchedJet_dR.iloc[row]["score"])
  color_index += 1  

for hist in histDict:
  histDict[hist].SetStats(0)
  histDict[hist].SetLineColor(colorDict[hist])
  if colorDict[hist] == 1:
    histDict[hist].Draw("HISTE")
  else:
    histDict[hist].Draw("SAMEHISTE")
  dR_legend.AddEntry(histDict[hist], "dR = "+hist)
dR_legend.Draw()
can.SaveAs(file + ".pdf")
     


