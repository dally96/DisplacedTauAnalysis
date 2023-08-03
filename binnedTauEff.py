import uproot 
import awkward as ak
import ROOT as rt
import scipy
import math
import matplotlib as mpl
import pandas as pd
import numpy as np
from array import array
import tau_selections as ts
import tau_func as tf
import Run3NanoAOD_Dict

Files = Run3NanoAOD_Dict.Run3_Dict
file_keys = Files.keys()

colors = [rt.kBlue, rt.kRed, rt.kGreen, rt.kMagenta]

binningVars = ["pt", "eta", "lxy"]
binningVarsUnits = ["[GeV]", "", "[cm]"]

ptBins = np.linspace(20, 1000, 50)
etaBins = np.linspace(-2.4, 2.4, 17)
lxyBins = np.linspace(0, 20, 11) 

workingPoints = [0.50, 0.65, 0.80, 0.90, 0.98]

histDict = {}
colorDict = {}
branchDict = {}
leafDict = {}
arrayDict = {}
mGraphDict = {}

can = rt.TCanvas("can", "can")



for file in Files:
  if "Stau" not in file: continue
  histDict[file] = {}
  histDict[file]["pt"] = {}
  histDict[file]["eta"] = {}
  histDict[file]["lxy"] = {}
  for score in workingPoints:
    histDict[file]["pt"][str(score)]  = rt.TEfficiency(file+"_eff_pt"+str(score).split(".")[-1], "Tau efficiency binned in p_{T}; p_{T} [GeV]; Tau Efficiency", 49, 20, 1000) 
    histDict[file]["eta"][str(score)] = rt.TEfficiency(file+"_eff_eta"+str(score).split(".")[-1], "Tau efficiency binned in #eta; #eta; Tau Efficiency", 16, -2.4, 2.4) 
    histDict[file]["lxy"][str(score)] =  rt.TEfficiency(file+"_eff_lxy"+str(score).split(".")[-1], "Tau efficiency binned in L_{xy}; L_{xy} [cm]; Tau Efficiency", 20, 0, 40)


for file in Files:
  if "Stau" not in file: continue 

  print("Opening file ", file)
  print("histDict contains ", histDict)

  sampleFile = rt.TFile.Open(Files[file])
  Events = sampleFile.Get("Events")
  nevt = Events.GetEntriesFast()
  
  print("histDict contains ", histDict, " after opening new file")

  print("File ", file, " has ", nevt, " events")

  Branches = Events.GetListOfBranches()
  
  for i in range(Branches.GetEntries()):
    if Branches.At(i).GetName().split("_")[0] == "GenJet" or Branches.At(i).GetName().split("_")[0] == "GenVisTau" or Branches.At(i).GetName().split("_")[0] == "Jet" or Branches.At(i).GetName().split("_")[0] == "GenPart":
      branchDict[Branches.At(i).GetName()] = Branches.At(i).GetName()

  arrayBranchDict = {}
  sampleFileUp = uproot.open(Files[file])
  eventsUp = sampleFileUp["Events"]

  genVisTauDict = {}
  jetDict = {}
  genJetDict = {}

  for branch in branchDict:
    arrayBranchDict[branch] = eventsUp[branch].array()
    if branch.split("_")[0] == "GenVisTau":
      genVisTauDict[branch] = eventsUp[branch].array()
    if branch.split("_")[0] == "Jet":
      jetDict[branch] = eventsUp[branch].array()
    if branch.split("_")[0] == "GenJet":
      genJetDict[branch] = eventsUp[branch].array()

  print("histDict contains ", histDict, " after defining dictionaries for each branch type")

  genVisTauDict["GenVisTau_Lxy"] = [] 
  genVisTauDict["GenVisTau_vertexZ"] = [] 
  genVisTauDict["GenVisTau_vertexRho"] = [] 

  print("histDict contains ", histDict, " after initializing new entries in genVisTauDict")
  
  for evt in range(nevt):
    lxy_evt = []
    vtxz_evt = []
    vtxrho_evt = []
    for tau in arrayBranchDict["GenVisTau_genPartIdxMother"][evt]:
      Particle = 0
      vtxz_evt.append(arrayBranchDict["GenPart_vertexZ"][evt][int(tau)])
      vtxrho_evt.append(arrayBranchDict["GenPart_vertexRho"][evt][int(tau)])
      vertexCreateX = arrayBranchDict["GenPart_vertexX"][evt][int(tau)]
      vertexCreateY = arrayBranchDict["GenPart_vertexY"][evt][int(tau)]
      for part_idx in range(len(arrayBranchDict["GenPart_genPartIdxMother"][evt])):
        if Particle > 0:
          continue
        if arrayBranchDict["GenPart_genPartIdxMother"][evt][part_idx] == tau:
          vertexDecayX = arrayBranchDict["GenPart_vertexX"][evt][part_idx]
          vertexDecayY = arrayBranchDict["GenPart_vertexY"][evt][part_idx]
          lxy_evt.append(math.sqrt((vertexDecayX - vertexCreateX)**2 + (vertexDecayY - vertexCreateY)**2))
          Particle += 1
    genVisTauDict["GenVisTau_Lxy"].append(lxy_evt)
    genVisTauDict["GenVisTau_vertexZ"].append(vtxz_evt)
    genVisTauDict["GenVisTau_vertexRho"].append(vtxrho_evt)

  print("histDict contains ", histDict, " after adding entries to the GenVisTau dict")

  AllHadTauDict = {}
  for branch in genVisTauDict:
    AllHadTauDict[branch] = ak.flatten(genVisTauDict[branch])
  AllHadTaus_df = pd.DataFrame(data = AllHadTauDict)
  SelectedHadTaus_df = AllHadTaus_df.query("GenVisTau_genPartIdxMother >= 0 & GenVisTau_pt > @ts.genTauPtMin & abs(GenVisTau_eta) < @ts.genTauEtaMax & GenVisTau_vertexZ < @ts.genTauVtxZMax & GenVisTau_vertexRho < @ts.genTauVtxRhoMax")

  print("histDict contains ", histDict, " after defining AllHadTauDict and SelectedHadTauDict")
   
  matchedJets = []
  matchedPt = []
  matchedEta = []
  matchedLxy = []
  matcheddRsqu = []
  matchedScore = []
  for evt in range(nevt):
    
    matchedJets_evt = []
    matchedPt_evt = []
    matchedEta_evt = []
    matchedLxy_evt = []
    matcheddRsqu_evt = []
    matchedScore_evt = []

    if (len(genVisTauDict["GenVisTau_pt"][evt]) == 0): continue
    for tau_idx in range(len(genVisTauDict["GenVisTau_pt"][evt])):
      if genVisTauDict["GenVisTau_pt"][evt][tau_idx] <= ts.genTauPtMin or abs(genVisTauDict["GenVisTau_eta"][evt][tau_idx]) >= ts.genTauEtaMax: continue
      if genVisTauDict["GenVisTau_genPartIdxMother"][evt][tau_idx] == -1: continue 
      if genVisTauDict["GenVisTau_vertexZ"][evt][tau_idx] >= ts.genTauVtxZMax or genVisTauDict["GenVisTau_vertexRho"][evt][tau_idx] >= ts.genTauVtxRhoMax: continue 

      matchedJets_evt_tau = []
      matchedPt_evt_tau = []
      matchedEta_evt_tau = []
      matchedLxy_evt_tau = []
      matcheddRsqu_evt_tau = []
      matchedScore_evt_tau = []

      for jet_idx in range(len(jetDict["Jet_pt"][evt])):  
        if jetDict["Jet_pt"][evt][jet_idx] <= ts.jetPtMin or abs(jetDict["Jet_eta"][evt][jet_idx]) >= ts.jetEtaMax: continue
        if jetDict["Jet_genJetIdx"][evt][jet_idx] == -1 or jetDict["Jet_genJetIdx"][evt][jet_idx] >= len(genJetDict["GenJet_pt"][evt]): continue
        if genJetDict["GenJet_pt"][evt][jetDict["Jet_genJetIdx"][evt][jet_idx]] <= ts.genJetPtMin or abs(genJetDict["GenJet_eta"][evt][jetDict["Jet_genJetIdx"][evt][jet_idx]]) >= ts.genJetEtaMax: continue
        dphi = abs(genVisTauDict["GenVisTau_phi"][evt][tau_idx]-jetDict["Jet_phi"][evt][jet_idx])
            
        if (dphi > math.pi) :
          dphi -= 2 * math.pi
      
        deta = genVisTauDict["GenVisTau_eta"][evt][tau_idx]-jetDict["Jet_eta"][evt][jet_idx]
        dR2  = dphi ** 2 + deta ** 2
        if (dR2 <= ts.dR_squ):
          matchedJets_evt_tau.append(jet_idx)
          matcheddRsqu_evt_tau.append(dR2)
          matchedScore_evt_tau.append(jetDict["Jet_disTauTag_score1"][evt][jet_idx])
          matchedPt_evt_tau.append(genVisTauDict["GenVisTau_pt"][evt][tau_idx])
          matchedEta_evt_tau.append(genVisTauDict["GenVisTau_eta"][evt][tau_idx])
          matchedLxy_evt_tau.append(genVisTauDict["GenVisTau_Lxy"][evt][tau_idx])
      matchedJets_evt.append(matchedJets_evt_tau)
      matcheddRsqu_evt.append(matcheddRsqu_evt_tau)
      matchedScore_evt.append(matchedScore_evt_tau)
      matchedPt_evt.append(matchedPt_evt_tau)
      matchedEta_evt.append(matchedEta_evt_tau)
      matchedLxy_evt.append(matchedLxy_evt_tau)
    matchedJets.append(matchedJets_evt)
    matcheddRsqu.append(matcheddRsqu_evt)
    matchedScore.append(matchedScore_evt)
    matchedPt.append(matchedPt_evt)
    matchedEta.append(matchedEta_evt)
    matchedLxy.append(matchedLxy_evt)

  print("histDict contains ", histDict, " after defining tau jets")

### If a jet is matched to more than one tau, pick tau with smallest matching ###
  for evt in range(len(matchedJets)):
    if len(matchedJets[evt]) > 1:
      for tau in range(len(matchedJets[evt]) - 1):
        for tau2 in range(tau + 1, len(matchedJets[evt])): 
          matchedJets_intersection = set(matchedJets[evt][tau]).intersection(set(matchedJets[evt][tau2]))
          if len(matchedJets_intersection) > 0:
            for intersectJets in matchedJets_intersection:
              if matcheddRsqu[evt][tau][matchedJets[evt][tau].index(intersectJets)] > matcheddRsqu[evt][tau2][matchedJets[evt][tau2].index(intersectJets)]:
                matcheddRsqu[evt][tau].pop(matchedJets[evt][tau].index(intersectJets))
                matchedScore[evt][tau].pop(matchedJets[evt][tau].index(intersectJets))
                matchedPt[evt][tau].pop(matchedJets[evt][tau].index(intersectJets))
                matchedEta[evt][tau].pop(matchedJets[evt][tau].index(intersectJets))
                matchedLxy[evt][tau].pop(matchedJets[evt][tau].index(intersectJets))
                matchedJets[evt][tau].pop(matchedJets[evt][tau].index(intersectJets))
              else:
                matcheddRsqu[evt][tau2].pop(matchedJets[evt][tau2].index(intersectJets))
                matchedScore[evt][tau2].pop(matchedJets[evt][tau2].index(intersectJets))
                matchedPt[evt][tau2].pop(matchedJets[evt][tau2].index(intersectJets))
                matchedEta[evt][tau2].pop(matchedJets[evt][tau2].index(intersectJets))
                matchedLxy[evt][tau2].pop(matchedJets[evt][tau2].index(intersectJets))
                matchedJets[evt][tau2].pop(matchedJets[evt][tau2].index(intersectJets))

### If more than one jet is matched to a tau, pick the jet with the smallest matching ###
  for evt in range(len(matchedJets)):
    for tau in range(len(matchedJets[evt])):
      if len(matchedJets[evt][tau]) > 1:
        for jet in range(len(matchedJets[evt][tau]) - 1):
          if matcheddRsqu[evt][tau][jet] > matcheddRsqu[evt][tau][jet + 1]:
            matcheddRsqu[evt][tau].pop(jet)
            matchedScore[evt][tau].pop(jet)
            matchedJets[evt][tau].pop(jet)
            matchedPt[evt][tau].pop(jet)
            matchedEta[evt][tau].pop(jet)
            matchedLxy[evt][tau].pop(jet)
          else:
            matcheddRsqu[evt][tau].pop(jet + 1)
            matchedScore[evt][tau].pop(jet + 1)
            matchedJets[evt][tau].pop(jet + 1)
            matchedPt[evt][tau].pop(jet + 1)
            matchedEta[evt][tau].pop(jet + 1)
            matchedLxy[evt][tau].pop(jet + 1)

  print("histDict contains ", histDict, " after making sure each tau gets matched to 1 jet and vice versa")

  matchedJetDict = {"jet": ak.flatten(matchedJets, axis=None), "dRsqu": ak.flatten(matcheddRsqu, axis=None), "score": ak.flatten(matchedScore, axis=None), "pt": ak.flatten(matchedPt, axis=None), "eta": ak.flatten(matchedEta, axis=None), "lxy": ak.flatten(matchedLxy, axis=None)}

  matchedJet_df = pd.DataFrame(data = matchedJetDict)

  for score in workingPoints:
    matchedTauCount_pt = []
    matchedTauCount_eta = []
    matchedTauCount_lxy = []
  
    totalTauCount_pt = []
    totalTauCount_eta = []
    totalTauCount_lxy = []

    for pt_idx in range(1, len(ptBins)):
      matchedTauCount_pt.append(len(matchedJet_df.query("pt > @ptBins[@pt_idx-1] & pt <= @ptBins[@pt_idx] & score >= @score")))
      totalTauCount_pt.append(len(SelectedHadTaus_df.query("GenVisTau_pt >  @ptBins[@pt_idx-1] & GenVisTau_pt <= @ptBins[@pt_idx]")))
    for eta_idx in range(1, len(etaBins)):
      matchedTauCount_eta.append(len(matchedJet_df.query("eta > @etaBins[@eta_idx-1] & eta <= @etaBins[@eta_idx] & score >= @score")))
      totalTauCount_eta.append(len(SelectedHadTaus_df.query("GenVisTau_eta > @etaBins[@eta_idx-1] & GenVisTau_eta <= @etaBins[@eta_idx]")))
    for lxy_idx in range(1, len(lxyBins)):
      matchedTauCount_lxy.append(len(matchedJet_df.query("lxy > @lxyBins[@lxy_idx-1] & lxy <= @lxyBins[@lxy_idx] & score >= @score")))
      totalTauCount_lxy.append(len(SelectedHadTaus_df.query("GenVisTau_Lxy > @lxyBins[@lxy_idx-1] & GenVisTau_Lxy  <= @lxyBins[@lxy_idx]")))
    print("histDict contains ", histDict, " before the new efficiencies are defined")


    for pt in range(1, len(ptBins)):
      print("On bin ", pt)
      histDict[file]["pt"][str(score)].SetTotalEvents(pt, totalTauCount_pt[pt-1])
      print("The total taus considered are ", totalTauCount_pt[pt-1])
      histDict[file]["pt"][str(score)].SetPassedEvents(pt, matchedTauCount_pt[pt-1])
      print("The passed taus considered are ", matchedTauCount_pt[pt-1])
    for eta in range(1, len(etaBins)):
      histDict[file]["eta"][str(score)].SetTotalEvents(eta, totalTauCount_eta[eta-1])
      histDict[file]["eta"][str(score)].SetPassedEvents(eta, matchedTauCount_eta[eta-1])
    for lxy in range(1, len(lxyBins)):
      histDict[file]["lxy"][str(score)].SetTotalEvents(lxy, totalTauCount_lxy[lxy-1])
      histDict[file]["lxy"][str(score)].SetPassedEvents(lxy, matchedTauCount_lxy[lxy-1])
    print("The pt efficiency for file ", file, " for jets with score greater than ", score, " is ", histDict[file]["pt"][str(score)].GetEfficiency(0))


histDict_keys = list(histDict.keys())

for var in binningVars:
  mGraphDict[var] = {}
  for score in workingPoints:
    mGraphDict[var][str(score)] = rt.TMultiGraph(var+"_"+str(score).split(".")[-1], var+"_"+str(score).split(".")[-1])
    mGraphDict[var][str(score)].SetTitle("Tau efficiency binned in "+var+" for matched jets with score >= "+str(score)+";"+var+binningVarsUnits[binningVars.index(var)]+"; Tau Efficiency")
    if var == "pt" or var == "lxy":
      legend = rt.TLegend(0.12, 0.15, 0.35, 0.37)
    if var == "eta": 
      legend = rt.TLegend(0.35, 0.2, 0.6, 0.35) 
    for file in histDict_keys:
      graph = histDict[file][var][str(score)].CreateGraph()
      graph.SetLineColor(colors[histDict_keys.index(file)])
      mGraphDict[var][str(score)].Add(graph) 
      legend.AddEntry(graph, file)
    mGraphDict[var][str(score)].Draw("A")
    legend.Draw()
    can.SaveAs("taueff_"+var+"_"+str(score).split(".")[-1]+".pdf")

    
