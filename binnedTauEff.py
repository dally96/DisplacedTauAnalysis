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
rt.gStyle.SetPalette(55)
colors = [rt.kBlue, rt.kRed, rt.kGreen, rt.kMagenta]

binningVars = ["pt", "eta", "lxy"]
binningVarsUnits = ["[GeV]", "", "[cm]"]

ptBins_500  = (20, 30, 40, 60, 100, 120, 200, 300, 400, 600)
ptBins_100  = (20, 30, 40, 60, 100, 120, 200, 300, 400)
etaBins     = (-2.4, -2.2, -1.4, -1.2, 1.2, 1.4, 2.2, 2.4)
lxyBins_100 = (0, 1E-3, 1E-2, 5E-2, 1E-1, 1, 5, 10, 20) 
lxyBins_10  = (0, 1E-3, 1E-2, 5E-2, 1E-1, 1, 5, 10)

workingPoints = [0.65, 0.80, 0.90, 0.98]
scoreSpace = np.linspace(0.0,0.995,200)
scoreSpace = np.append(scoreSpace, np.linspace(0.9951, 1.0001, 51))
print(len(scoreSpace))

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
  if "100GeV" in file:
    for pt in range(len(ptBins_100)-1):
      histDict[file]["pt"][str(ptBins_100[pt]) + "-" + str(ptBins_100[pt+1])]  = rt.TEfficiency(file+"_eff_pt"+str(ptBins_100[pt]), "Tau efficiency binned for "+str(ptBins_100[pt])+" <  p_{T} <= "+str(ptBins_100[pt+1]) + "; score; Tau Efficiency", 250, scoreSpace)
  if "500GeV" in file:
    for pt in range(len(ptBins_500)-1):
      histDict[file]["pt"][str(ptBins_500[pt]) + "-" + str(ptBins_500[pt+1])]  = rt.TEfficiency(file+"_eff_pt"+str(ptBins_500[pt]), "Tau efficiency binned for "+str(ptBins_500[pt])+" <  p_{T} <= "+str(ptBins_500[pt+1]) + "; score; Tau Efficiency", 250, scoreSpace)
  for eta in range(len(etaBins)-1): 
    histDict[file]["eta"][str(etaBins[eta]) + "-" + str(etaBins[eta+1])] = rt.TEfficiency(file+"_eff_eta"+str(etaBins[eta]), "Tau efficiency for " + str(etaBins[eta]) + " < #eta <= " + str(etaBins[eta+1]) + " ; score; Tau Efficiency", 250, scoreSpace)
  if "10mm" in file:
    for lxy in range(len(lxyBins_10)-1): 
      histDict[file]["lxy"][str(lxyBins_10[lxy]) + "-" + str(lxyBins_10[lxy+1])] =  rt.TEfficiency(file+"_eff_lxy"+str(lxyBins_10[lxy]), "Tau efficiency binned for " + str(lxyBins_10[lxy]) + " < L_{xy} <= " + str(lxyBins_10[lxy+1]) + "; score; Tau Efficiency", 250, scoreSpace)
  if "100mm" in file:
    for lxy in range(len(lxyBins_100)-1): 
      histDict[file]["lxy"][str(lxyBins_100[lxy]) + "-" + str(lxyBins_100[lxy+1])] =  rt.TEfficiency(file+"_eff_lxy"+str(lxyBins_100[lxy]), "Tau efficiency binned for " + str(lxyBins_100[lxy]) + " < L_{xy} <= " + str(lxyBins_100[lxy+1]) + "; score; Tau Efficiency", 250, scoreSpace)


for file in Files:
  if "Stau" not in file: continue 

  print("Opening file ", file)

  sampleFile = rt.TFile.Open(Files[file])
  Events = sampleFile.Get("Events")
  nevt = Events.GetEntriesFast()
  

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


  genVisTauDict["GenVisTau_Lxy"] = [] 
  genVisTauDict["GenVisTau_vertexZ"] = [] 
  genVisTauDict["GenVisTau_vertexRho"] = [] 

  
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


  AllHadTauDict = {}
  for branch in genVisTauDict:
    AllHadTauDict[branch] = ak.flatten(genVisTauDict[branch])
  AllHadTaus_df = pd.DataFrame(data = AllHadTauDict)
  SelectedHadTaus_df = AllHadTaus_df.query("GenVisTau_genPartIdxMother >= 0 & GenVisTau_pt > @ts.genTauPtMin & abs(GenVisTau_eta) < @ts.genTauEtaMax & GenVisTau_vertexZ < @ts.genTauVtxZMax & GenVisTau_vertexRho < @ts.genTauVtxRhoMax")

   
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


  matchedJetDict = {"jet": ak.flatten(matchedJets, axis=None), "dRsqu": ak.flatten(matcheddRsqu, axis=None), "score": ak.flatten(matchedScore, axis=None), "pt": ak.flatten(matchedPt, axis=None), "eta": ak.flatten(matchedEta, axis=None), "lxy": ak.flatten(matchedLxy, axis=None)}

  matchedJet_df = pd.DataFrame(data = matchedJetDict)

  if "100GeV" in file:
   for pt_idx in range(1, len(ptBins_100)):
      matchedTauCount_pt = []
      totalTauCount_pt = []
      for score in scoreSpace:
        matchedTauCount_pt.append(len(matchedJet_df.query("pt > @ptBins_100[@pt_idx-1] & pt <= @ptBins_100[@pt_idx] & score >= @score")))
        totalTauCount_pt.append(len(SelectedHadTaus_df.query("GenVisTau_pt >  @ptBins_100[@pt_idx-1] & GenVisTau_pt <= @ptBins_100[@pt_idx]")))
      for score_idx in range(1, len(scoreSpace)):
        histDict[file]["pt"][str(ptBins_100[pt_idx-1]) + "-" + str(ptBins_100[pt_idx])].SetTotalEvents(score_idx, totalTauCount_pt[score_idx-1])
        histDict[file]["pt"][str(ptBins_100[pt_idx-1]) + "-" + str(ptBins_100[pt_idx])].SetPassedEvents(score_idx, matchedTauCount_pt[score_idx-1])

  if "500GeV" in file:
   for pt_idx in range(1, len(ptBins_500)):
      matchedTauCount_pt = []
      totalTauCount_pt = []
      for score in scoreSpace:
        matchedTauCount_pt.append(len(matchedJet_df.query("pt > @ptBins_500[@pt_idx-1] & pt <= @ptBins_500[@pt_idx] & score >= @score")))
        totalTauCount_pt.append(len(SelectedHadTaus_df.query("GenVisTau_pt >  @ptBins_500[@pt_idx-1] & GenVisTau_pt <= @ptBins_500[@pt_idx]")))
      for score_idx in range(1, len(scoreSpace)):
        histDict[file]["pt"][str(ptBins_500[pt_idx-1]) + "-" + str(ptBins_500[pt_idx])].SetTotalEvents(score_idx, totalTauCount_pt[score_idx-1])
        histDict[file]["pt"][str(ptBins_500[pt_idx-1]) + "-" + str(ptBins_500[pt_idx])].SetPassedEvents(score_idx, matchedTauCount_pt[score_idx-1])


  for eta_idx in range(1, len(etaBins)):

    matchedTauCount_eta = []
    totalTauCount_eta = []
    for score in scoreSpace:
      matchedTauCount_eta.append(len(matchedJet_df.query("eta > @etaBins[@eta_idx-1] & eta <= @etaBins[@eta_idx] & score >= @score")))
      totalTauCount_eta.append(len(SelectedHadTaus_df.query("GenVisTau_eta > @etaBins[@eta_idx-1] & GenVisTau_eta <= @etaBins[@eta_idx]")))
    for score_idx in range(1, len(scoreSpace)):
      histDict[file]["eta"][str(etaBins[eta_idx-1]) + "-" + str(etaBins[eta_idx])].SetTotalEvents(score_idx, totalTauCount_eta[score_idx-1])
      histDict[file]["eta"][str(etaBins[eta_idx-1]) + "-" + str(etaBins[eta_idx])].SetPassedEvents(score_idx, matchedTauCount_eta[score_idx-1])

  if "10mm" in file:
    for lxy_idx in range(1, len(lxyBins_10)):
      matchedTauCount_lxy = []
      totalTauCount_lxy = []
      for score in scoreSpace:
        matchedTauCount_lxy.append(len(matchedJet_df.query("lxy > @lxyBins_10[@lxy_idx-1] & lxy <= @lxyBins_10[@lxy_idx] & score >= @score")))
        totalTauCount_lxy.append(len(SelectedHadTaus_df.query("GenVisTau_Lxy > @lxyBins_10[@lxy_idx-1] & GenVisTau_Lxy  <= @lxyBins_10[@lxy_idx]")))
      for score_idx in range(1, len(scoreSpace)):
        histDict[file]["lxy"][str(lxyBins_10[lxy_idx-1]) + "-" + str(lxyBins_10[lxy_idx])].SetTotalEvents(score_idx, totalTauCount_lxy[score_idx-1])
        histDict[file]["lxy"][str(lxyBins_10[lxy_idx-1]) + "-" + str(lxyBins_10[lxy_idx])].SetPassedEvents(score_idx, matchedTauCount_lxy[score_idx-1])

  if "100mm" in file:
    for lxy_idx in range(1, len(lxyBins_100)):
      matchedTauCount_lxy = []
      totalTauCount_lxy = []
      for score in scoreSpace:
        matchedTauCount_lxy.append(len(matchedJet_df.query("lxy > @lxyBins_100[@lxy_idx-1] & lxy <= @lxyBins_100[@lxy_idx] & score >= @score")))
        totalTauCount_lxy.append(len(SelectedHadTaus_df.query("GenVisTau_Lxy > @lxyBins_100[@lxy_idx-1] & GenVisTau_Lxy  <= @lxyBins_100[@lxy_idx]")))
      for score_idx in range(1, len(scoreSpace)):
        histDict[file]["lxy"][str(lxyBins_100[lxy_idx-1]) + "-" + str(lxyBins_100[lxy_idx])].SetTotalEvents(score_idx, totalTauCount_lxy[score_idx-1])
        histDict[file]["lxy"][str(lxyBins_100[lxy_idx-1]) + "-" + str(lxyBins_100[lxy_idx])].SetPassedEvents(score_idx, matchedTauCount_lxy[score_idx-1])



#aveDict = {}
#
#for file in Files:
#  aveDict[file] = {}
#  aveDict[file]["pt"] =  histDict[file]["pt"][str(workingPoints[0])]
#  for score_idx in range(1, len(workingPoints)):
#    aveDict[file]["pt"].Add(histDict[file]["pt"][str(workingPoints[score_idx])])
#  
#  aveDict[file]["eta"] = histDict[file]["eta"][str(workingPoints[0])]
#  for score_idx in range(1, len(workingPoints)):
#    aveDict[file]["eta"].Add(histDict[file]["eta"][str(workingPoints[score_idx])])
#
#  aveDict[file]["lxy"] = histDict[file]["lxy"][str(workingPoints[0])]
#  for score_idx in range(1, len(workingPoints)):
#    aveDict[file]["lxy"].Add(histDict[file]["lxy"][str(workingPoints[score_idx])])  

histDict_keys = list(histDict.keys())
  

for file in histDict_keys:
  pt_keys = list(histDict[file]["pt"].keys())
  eta_keys =  list(histDict[file]["eta"].keys())
  lxy_keys = list(histDict[file]["lxy"].keys())
  mGraphDict["pt"] = rt.TMultiGraph("pt_"+file, "pt_"+file)
  mGraphDict["pt"].SetTitle("Tau efficiency for pt for " + file + "; score; Tau Efficiency")
  ptLegend = rt.TLegend()

  for bins in pt_keys:
  
    graph = histDict[file]["pt"][bins].CreateGraph()
    graph.SetLineColor(pt_keys.index(bins)+1)
    mGraphDict["pt"].Add(graph)
    ptLegend.AddEntry(graph, bins)
  mGraphDict["pt"].Draw("A")
  ptLegend.Draw()
  can.SaveAs("taueff_pt_"+file+".pdf")

  mGraphDict["eta"] = rt.TMultiGraph("eta_"+file, "eta_"+file)
  mGraphDict["eta"].SetTitle("Tau efficiency for eta for " + file + "; score; Tau Efficiency")
  etaLegend = rt.TLegend()
  for bins in eta_keys:
  
    graph = histDict[file]["eta"][bins].CreateGraph()
    graph.SetLineColor(eta_keys.index(bins)+1)
    mGraphDict["eta"].Add(graph)
    etaLegend.AddEntry(graph, bins)
  mGraphDict["eta"].Draw("A")
  etaLegend.Draw()
  can.SaveAs("taueff_eta_"+file+".pdf")


  mGraphDict["lxy"] = rt.TMultiGraph("lxy_"+file, "lxy_"+file)
  mGraphDict["lxy"].SetTitle("Tau efficiency for lxy for " + file + "; score; Tau Efficiency")
  lxyLegend = rt.TLegend()
  for bins in lxy_keys:
  
    graph = histDict[file]["lxy"][bins].CreateGraph()
    graph.SetLineColor(lxy_keys.index(bins)+1)
    mGraphDict["lxy"].Add(graph)
    lxyLegend.AddEntry(graph, bins)
  mGraphDict["lxy"].Draw("A")
  lxyLegend.Draw()
  can.SaveAs("taueff_lxy_"+file+".pdf")


#  for score in workingPoints:
#    mGraphDict[var][str(score)] = rt.TMultiGraph(var+"_"+str(score).split(".")[-1], var+"_"+str(score).split(".")[-1])
#    mGraphDict[var][str(score)].SetTitle("Tau efficiency binned in "+var+" for matched jets with score >= "+str(score)+";"+var+binningVarsUnits[binningVars.index(var)]+"; Tau Efficiency")
#    if var == "pt" or var == "lxy":
#      legend = rt.TLegend(0.12, 0.15, 0.35, 0.37)
#    if var == "eta": 
#      legend = rt.TLegend(0.35, 0.2, 0.6, 0.35) 
#    for file in histDict_keys:
#      graph = histDict[file][var][str(score)].CreateGraph()
#      graph.SetLineColor(colors[histDict_keys.index(file)])
#      mGraphDict[var][str(score)].Add(graph) 
#      legend.AddEntry(graph, file)
#    mGraphDict[var][str(score)].Draw("A")
#    legend.Draw()
#    can.SaveAs("taueff_"+var+"_"+str(score).split(".")[-1]+".pdf")
#
#mResDict = {}
#
#
#for file in Files:
#  histDict_fileVar_keys = list(histDict[file]["pt"].keys())
#  mResDict[file] = {}
#  mResDict[file]["pt"] = rt.TMultiGraph("pt", "pt")
#  mResDict[file]["pt"].SetTitle("Tau efficiency binned in pt for different scores for file " + file + ";pt " + binningVarsUnits[binningVars.index("pt")] + "; Difference between average and tau efficiencies at different scores")
#  pt_legend = rt.TLegend()
#  
#  aveGraph = aveDict[file]["pt"].CreateGraph()
#  for score in workingPoints:
#    scoreGraph = histDict[file]["pt"][str(score)].CreateGraph()
#    res = []
#    for i in range(aveGraph.GetN()):
#      res.append(aveGraph.GetY()[i] - scoreGraph.GetY()[i])
#    resGraph = rt.TGraph(49, aveGraph.GetX(), res)
#    resGraph.SetLineColor(colors[histDict_fileVar_keys.index(score)])
#    mResDict[file]["pt"].Add(resGraph)
#    legend.AddEntry(resGraph, str(score))
#  mResDict[file]["pt"].Draw("A")
#  legend.Draw()
#  can.SaveAs("resTauEff_" + file + "_pt.pdf")
#
