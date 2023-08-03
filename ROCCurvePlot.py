
import uproot 
import awkward as ak
import ROOT
import scipy
import math
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
from array import array
import tau_selections
exec(open("NanoAOD_Dict.py").read())
exec(open("tau_func.py").read())

f = open("TTToHadronic_Events.txt", 'r')
lines = f.readlines()
eventDict = {}
keys = lines[0].strip().split("\t")
for key in keys:
  eventDict[key] = []
for line in lines[1:]:
  eventDict["event"].append(int(line.strip().split("\t")[0]))
  eventDict["lumiBlock"].append(int(line.strip().split("\t")[1]))
  eventDict["run"].append(int(line.strip().split("\t")[2]))
eventDict["event"] = np.array(eventDict["event"])
eventDict["lumiBlock"] = np.array(eventDict["lumiBlock"])
eventDict["run"] = np.array(eventDict["run"])
#print(eventDict)


colors = [ROOT.kBlue, ROOT.kRed, ROOT.kGreen, ROOT.kBlack, ROOT.kViolet, ROOT.kMagenta, ROOT.kGray]
can = ROOT.TCanvas("can", "can", 1000, 600)
score_legend = ROOT.TLegend(0.7,0.6,0.8,0.8)
dict_idx = 0
scoreSpace = np.linspace(0.0,0.995,200)
score_dict = {}

Files = Nano_Dict
Sample = {}
BG = {}
Run2_Eff = {}
Run2_Fake = {}
Run3_Eff = {}
Run3_Fake = {}
Run3_ROC = {}
Run2_ROC = {}

for file in Files:
  #print(file, " is running")
  nano_file = uproot.open(Files[file])
  nano_file_evt = nano_file["Events"]

  Branches = {
              "GenVisTau_pt": nano_file_evt["GenVisTau_pt"].array(),
              "GenVisTau_phi":  nano_file_evt["GenVisTau_phi"].array(),
              "GenVisTau_eta":  nano_file_evt["GenVisTau_eta"].array(),
              "GenVisTau_genPartIdxMother":  nano_file_evt["GenVisTau_genPartIdxMother"].array(),

              "Jet_pt": nano_file_evt["Jet_pt"].array(),
              "Jet_phi": nano_file_evt["Jet_phi"].array(),
              "Jet_eta": nano_file_evt["Jet_eta"].array(),
              "Jet_disTauTag_score1": nano_file_evt["Jet_disTauTag_score1"].array(),
              "Jet_genJetIdx": nano_file_evt["Jet_genJetIdx"].array(),
              "Jet_partonFlavour": nano_file_evt["Jet_partonFlavour"].array(),
              "Jet_hadronFlavour": nano_file_evt["Jet_hadronFlavour"].array(),
              "Jet_jetId": nano_file_evt["Jet_jetId"].array(),
              
              "GenPart_genPartIdxMother": nano_file_evt["GenPart_genPartIdxMother"].array(),
              "GenPart_pdgId": nano_file_evt["GenPart_pdgId"].array(),
              "GenPart_pt": nano_file_evt["GenPart_pt"].array(),
              "GenPart_phi": nano_file_evt["GenPart_phi"].array(),
              "GenPart_eta": nano_file_evt["GenPart_eta"].array(),
              "GenPart_vertexX": nano_file_evt["GenPart_vertexX"].array(),
              "GenPart_vertexY": nano_file_evt["GenPart_vertexY"].array(),
              "GenPart_vertexZ": nano_file_evt["GenPart_vertexZ"].array(),
              "GenPart_vertexR": nano_file_evt["GenPart_vertexR"].array(),
              "GenPart_vertexRho": nano_file_evt["GenPart_vertexRho"].array(),

              "GenJet_partonFlavour": nano_file_evt["GenJet_partonFlavour"].array(),
              "GenJet_pt": nano_file_evt["GenJet_pt"].array(),
              "GenJet_phi": nano_file_evt["GenJet_phi"].array(),
              "GenJet_eta": nano_file_evt["GenJet_eta"].array(),

              "Electron_pt": nano_file_evt["Electron_pt"].array(),
              "Electron_phi": nano_file_evt["Electron_phi"].array(),
              "Electron_eta": nano_file_evt["Electron_eta"].array(),

              "Muon_pt": nano_file_evt["Muon_pt"].array(),
              "Muon_phi": nano_file_evt["Muon_phi"].array(),
              "Muon_eta": nano_file_evt["Muon_eta"].array(),
              
              "event": nano_file_evt["event"].array(),
              "lumiBlock": nano_file_evt["luminosityBlock"].array(),
              "run": nano_file_evt["run"].array()
  }

  #print(file, " has ", len(Branches), " brranches")

  if ("TT" in file):
    nevt = nano_file_evt.num_entries
    #nevt = 24000
    #print(file, " has ", nevt, " number of entries")

### Match jets that pass score cut to hadronically decaying taus using dR matching ###
    unmatchedJetsPassScore = []
    unmatchedJets_score = []  

    for evt in range(nevt):
      if isLeptonic(evt, Branches["GenPart_pdgId"], Branches["GenPart_genPartIdxMother"]): continue 
      if Branches["event"][evt] not in eventDict["event"] or  Branches["lumiBlock"][evt] not in eventDict["lumiBlock"] or  Branches["run"][evt] not in eventDict["run"]: continue
      unmatchedJetsPassScore_evt = []
      unmatchedJets_score_evt = []  
      if (len(Branches["GenVisTau_pt"][evt])== 0):
        for jet_idx in range(len(Branches["Jet_pt"][evt])):
          #print(vetoLep(Branches["Jet_phi"][evt][jet_idx], Branches["Jet_eta"][evt][jet_idx], evt, "Electron") , vetoLep(Branches["Jet_phi"][evt][jet_idx], Branches["Jet_eta"][evt][jet_idx], evt, "Muon"), " tells us whether we should veto this jet")
          if Branches["Jet_partonFlavour"][evt][jet_idx] == 0 and Branches["Jet_hadronFlavour"][evt][jet_idx] == 0: continue
          #if Branches["Jet_jetId"][evt][jet_idx] < 6: continue
          if Branches["Jet_pt"][evt][jet_idx] < tau_selections.jetPtMin or abs(Branches["Jet_eta"][evt][jet_idx]) > tau_selections.jetEtaMax: continue
          if Branches["Jet_genJetIdx"][evt][jet_idx] == -1 or Branches["Jet_genJetIdx"][evt][jet_idx] >= len(Branches["GenJet_partonFlavour"][evt]): continue
          if vetoGenLep(Branches["GenJet_phi"][evt][Branches["Jet_genJetIdx"][evt][jet_idx]], Branches["GenJet_eta"][evt][Branches["Jet_genJetIdx"][evt][jet_idx]], evt, "Electron") or vetoGenLep(Branches["GenJet_phi"][evt][Branches["Jet_genJetIdx"][evt][jet_idx]], Branches["GenJet_eta"][evt][Branches["Jet_genJetIdx"][evt][jet_idx]], evt, "Muon"): continue  
          if Branches["GenJet_pt"][evt][Branches["Jet_genJetIdx"][evt][jet_idx]] < tau_selections.genJetPtMin  or abs(Branches["GenJet_eta"][evt][Branches["Jet_genJetIdx"][evt][jet_idx]]) > tau_selections.genJetEtaMax: continue
          unmatchedJetsPassScore_evt.append(jet_idx)   
          unmatchedJets_score_evt.append(Branches["Jet_disTauTag_score1"][evt][jet_idx])
      for tau_idx in range(len(Branches["GenVisTau_pt"][evt])):
        if Branches["GenVisTau_pt"][evt][tau_idx] < tau_selections.genTauPtMin  or abs(Branches["GenVisTau_eta"][evt][tau_idx]) > tau_selections.genTauEtaMax: continue
        mthrTau_idx = Branches["GenVisTau_genPartIdxMother"][evt][tau_idx]
        if Branches["GenPart_vertexZ"][evt][mthrTau_idx] >= tau_selections.genTauVtxZMax or Branches["GenPart_vertexRho"][evt][mthrTau_idx] >= tau_selections.genTauVtxRhoMax: continue
        for jet_idx in range(len(Branches["Jet_pt"][evt])):
          #if vetoLep(Branches["Jet_phi"][evt][jet_idx], Branches["Jet_eta"][evt][jet_idx], evt, "Electron") or vetoLep(Branches["Jet_phi"][evt][jet_idx], Branches["Jet_eta"][evt][jet_idx], evt, "Muon"): continue  
          if Branches["Jet_pt"][evt][jet_idx] < tau_selections.jetPtMin or abs(Branches["Jet_eta"][evt][jet_idx]) > tau_selections.jetEtaMax: continue
          if Branches["Jet_genJetIdx"][evt][jet_idx] == -1 or Branches["Jet_genJetIdx"][evt][jet_idx] >= len(Branches["GenJet_partonFlavour"][evt]): continue
          if Branches["GenJet_pt"][evt][Branches["Jet_genJetIdx"][evt][jet_idx]] < tau_selections.genJetPtMin  or abs(Branches["GenJet_eta"][evt][Branches["Jet_genJetIdx"][evt][jet_idx]]) > tau_selections.genJetEtaMax: continue
          dphi = abs(Branches["GenVisTau_phi"][evt][tau_idx]-Branches["Jet_phi"][evt][jet_idx])
            
          if (dphi > math.pi) :
            dphi -= 2 * math.pi
      
          deta = Branches["GenVisTau_eta"][evt][tau_idx]-Branches["Jet_eta"][evt][jet_idx]
          dR2  = dphi ** 2 + deta ** 2
          if (dR2 > tau_selections.dR_squ):
            if jet_idx in unmatchedJetsPassScore_evt: continue
            unmatchedJetsPassScore_evt.append(jet_idx)
            unmatchedJets_score_evt.append(Branches["Jet_disTauTag_score1"][evt][jet_idx])
      unmatchedJetsPassScore.append(unmatchedJetsPassScore_evt)
      unmatchedJets_score.append(unmatchedJets_score_evt)
    #print(file," has been matched") 

    unmatchedJet_dict =  {"score": ak.flatten(unmatchedJets_score)}

    unmatchedJet_df = pd.DataFrame(data = unmatchedJet_dict)
    #print(unmatchedJet_df) 

### Count the number of jets for each score ###
    fakeJetCount = []
    for score in scoreSpace:
      fakeJetCount.append(len(unmatchedJet_df.query("score >= @score")))
      print("There are ", len(unmatchedJet_df.query("score >= @score")), " jets that aren't matched taus with score greater than or equal to ", score) 
      #print("There are ", len(unmatchedJet_df.query("score >= @score")), " unmatches jets for score ", score)

### Calculate total number of unmatched jets with selection cuts###
    fullJetCount = len(unmatchedJet_df)
    print("Total number of jets that aren't matched to taus within selections is ", fullJetCount)

### Make TEfficiency plot ###
    BG[file] = ROOT.TEfficiency("e_fakerate", "M = 100 GeV, Lxy = 100mm; DisTauTagger_Score1; Fraction of jets that are not a tau", 200, -0.0025, 0.9975) 
    for score_idx in range(len(scoreSpace)):
      BG[file].SetTotalEvents(score_idx+1, fullJetCount)
      BG[file].SetPassedEvents(score_idx+1, fakeJetCount[score_idx])

### Match jets that pass score cut to hadronically decaying taus using dR matching ###
  if "Stau" in file:
    nevt = nano_file_evt.num_entries
    matchedJetsPassScore = []
    matchedScore = []
    matchedDisTauTagScore = []
    for evt in range(nevt):
      
      if Branches["event"][evt] not in eventDict["event"] or  Branches["lumiBlock"][evt] not in eventDict["lumiBlock"] or  Branches["run"][evt] not in eventDict["run"]: continue
      matchedJetsPassScore_evt = []
      matchedScore_evt = []
      matchedDisTauTagScore_evt = []
      if (len(Branches["GenVisTau_pt"][evt])== 0): continue
      for tau_idx in range(len(Branches["GenVisTau_pt"][evt])):
        if Branches["GenVisTau_pt"][evt][tau_idx] <= tau_selections.genTauPtMin or abs(Branches["GenVisTau_eta"][evt][tau_idx]) >= tau_selections.genTauEtaMax: continue 
        if not stauMother(evt, Branches["GenVisTau_genPartIdxMother"][evt][tau_idx], Branches["GenPart_pdgId"], Branches["GenPart_genPartIdxMother"]): continue 
        mthrTau_idx = Branches["GenVisTau_genPartIdxMother"][evt][tau_idx]
        stau_idx = stauIdx(evt, Branches["GenVisTau_genPartIdxMother"][evt][tau_idx], Branches["GenPart_pdgId"], Branches["GenPart_genPartIdxMother"])
        if Branches["GenPart_vertexZ"][evt][mthrTau_idx] >= tau_selections.genTauVtxZMax or Branches["GenPart_vertexRho"][evt][mthrTau_idx] >= tau_selections.genTauVtxRhoMax: continue
        #if L(evt, mthrTau_idx, stau_idx) > tau_selections.stauTauVtxDistMax  or  L(evt, mthrTau_idx, stau_idx) < tau_selections.stauTauVtxDistMin: continue 
        matchedJetsPassScore_evt_tau = []
        matchedScore_evt_tau = []
        matchedDisTauTagScore_evt_tau = []
        for jet_idx in range(len(Branches["Jet_pt"][evt])):
          if Branches["Jet_pt"][evt][jet_idx] < tau_selections.jetPtMin or abs(Branches["Jet_eta"][evt][jet_idx]) > tau_selections.jetEtaMax: continue
          if Branches["Jet_genJetIdx"][evt][jet_idx] == -1 or Branches["Jet_genJetIdx"][evt][jet_idx] >= len(Branches["GenJet_partonFlavour"][evt]): continue
          if Branches["GenJet_pt"][evt][Branches["Jet_genJetIdx"][evt][jet_idx]] < tau_selections.genJetPtMin or abs(Branches["GenJet_eta"][evt][Branches["Jet_genJetIdx"][evt][jet_idx]]) > tau_selections.genJetEtaMax: continue
          dphi = abs(Branches["GenVisTau_phi"][evt][tau_idx]-Branches["Jet_phi"][evt][jet_idx])
            
          if (dphi > math.pi) :
            dphi -= 2 * math.pi
      
          deta = Branches["GenVisTau_eta"][evt][tau_idx]-Branches["Jet_eta"][evt][jet_idx]
          dR2  = dphi ** 2 + deta ** 2
          if (dR2 <= tau_selections.dR_squ):
            matchedJetsPassScore_evt_tau.append(jet_idx)
            matchedScore_evt_tau.append(dR2)
            matchedDisTauTagScore_evt_tau.append(Branches["Jet_disTauTag_score1"][evt][jet_idx])
        matchedJetsPassScore_evt.append(matchedJetsPassScore_evt_tau)
        matchedScore_evt.append(matchedScore_evt_tau)
        matchedDisTauTagScore_evt.append(matchedDisTauTagScore_evt_tau)
      matchedJetsPassScore.append(matchedJetsPassScore_evt)
      matchedScore.append(matchedScore_evt)
      matchedDisTauTagScore.append(matchedDisTauTagScore_evt)

### If a jet is matched to more than one tau, pick tau with smallest matching ###
    for evt in range(len(matchedJetsPassScore)):
      if len(matchedJetsPassScore[evt]) > 1:
        for tau in range(len(matchedJetsPassScore[evt]) - 1):
          for tau2 in range(tau + 1, len(matchedJetsPassScore[evt])): 
            matchedJets_intersection = set(matchedJetsPassScore[evt][tau]).intersection(set(matchedJetsPassScore[evt][tau2]))
            if len(matchedJets_intersection) > 0:
              for intersectJets in matchedJets_intersection:
                if matchedScore[evt][tau][matchedJetsPassScore[evt][tau].index(intersectJets)] > matchedScore[evt][tau2][matchedJetsPassScore[evt][tau2].index(intersectJets)]:
                  matchedScore[evt][tau].pop(matchedJetsPassScore[evt][tau].index(intersectJets))
                  matchedDisTauTagScore[evt][tau].pop(matchedJetsPassScore[evt][tau].index(intersectJets))
                  matchedJetsPassScore[evt][tau].pop(matchedJetsPassScore[evt][tau].index(intersectJets))
                else:
                  matchedScore[evt][tau2].pop(matchedJetsPassScore[evt][tau2].index(intersectJets))
                  matchedDisTauTagScore[evt][tau2].pop(matchedJetsPassScore[evt][tau2].index(intersectJets))
                  matchedJetsPassScore[evt][tau2].pop(matchedJetsPassScore[evt][tau2].index(intersectJets))
              
### If more than one jet is matched to a tau, pick the jet with the smallest matching ###
    for evt in range(len(matchedJetsPassScore)):
      for tau in range(len(matchedJetsPassScore[evt])):
        if len(matchedJetsPassScore[evt][tau]) > 1:
          for jet in range(len(matchedJetsPassScore[evt][tau]) - 1):
            if matchedScore[evt][tau][jet] > matchedScore[evt][tau][jet + 1]:
              matchedScore[evt][tau].pop(jet)
              matchedDisTauTagScore[evt][tau].pop(jet)
              matchedJetsPassScore[evt][tau].pop(jet)
            else:
              matchedScore[evt][tau].pop(jet + 1)
              matchedDisTauTagScore[evt][tau].pop(jet + 1)
              matchedJetsPassScore[evt][tau].pop(jet + 1)
    

### Count the number of taus that were matched ###
    matchedTauCount = []
    matchedJet_dict = {"jet" : ak.flatten(matchedJetsPassScore, axis=None), "matching" : ak.flatten(matchedScore, axis=None), "score" : ak.flatten(matchedDisTauTagScore, axis=None)}
    score_dict[file] = matchedJet_dict["score"]
    
    matchedJet_df = pd.DataFrame(data = matchedJet_dict)
    #print(matchedJet_df) 
  
    for score in scoreSpace:
      matchedTauCount.append(len(matchedJet_df.query("score >= @score")))
      print("There are ", len(matchedJet_df.query("score >= @score")), " jets matched to taus with score greater than or equal to ", score) 
      #print("Number of taus with score greater than or equal to ", score, " is ", len(matchedJet_df.query("score >= @score")))

    #matchedJet_hist = ROOT.TH1F("h_matchjets", "Number of jets matched to #tau_{h} above score threshold; score; Number of jets", 100, 0.5, 1)
    #for indx, nTau in enumerate(matchedTauCount):
    #  matchedJet_hist.SetBinContent(indx+1, nTau)
    #matchedJet_legend = ROOT.TLegend(0.6,0.6,0.85,0.88)
    #matchedJet_legend.AddEntry(matchedJet_hist, file)
    #matchedJet_hist.SetStats(0)
    #matchedJet_hist.Draw("HISTE")
    #matchedJet_legend.Draw()
    #can.SaveAs(file + "TauCount.pdf")  


### Count the total number of hadronically decaying taus ###
    pt = []
    eta = []
    vZ = []
    vRho = []
    isStauMother = []
    #Lxyz = []
    for evt in range(nevt):
      if Branches["event"][evt] not in eventDict["event"] or  Branches["lumiBlock"][evt] not in eventDict["lumiBlock"] or  Branches["run"][evt] not in eventDict["run"]: continue
      vZ_evt = []
      vRho_evt = []
      isStauMother_evt = []
      pt_evt = []
      eta_evt = []
      #Lxyz_evt = []
      for tau_idx in range(len(Branches["GenVisTau_genPartIdxMother"][evt])):
        vZ_evt.append(Branches["GenPart_vertexZ"][evt][Branches["GenVisTau_genPartIdxMother"][evt][tau_idx]])
        vRho_evt.append(Branches["GenPart_vertexRho"][evt][Branches["GenVisTau_genPartIdxMother"][evt][tau_idx]])
        isStauMother_evt.append(stauMother(evt, Branches["GenVisTau_genPartIdxMother"][evt][tau_idx], Branches["GenPart_pdgId"], Branches["GenPart_genPartIdxMother"]))
        pt_evt.append(Branches["GenVisTau_pt"][evt][tau_idx])
        eta_evt.append(Branches["GenVisTau_eta"][evt][tau_idx])
        #if not stauMother(evt, Branches["GenVisTau_genPartIdxMother"][evt][tau_idx], Branches["GenPart_pdgId"], Branches["GenPart_genPartIdxMother"]):
          #Lxyz_evt.append(999)
        #if stauMother(evt, Branches["GenVisTau_genPartIdxMother"][evt][tau_idx], Branches["GenPart_pdgId"], Branches["GenPart_genPartIdxMother"]):
          #mthrTau_idx = Branches["GenVisTau_genPartIdxMother"][evt][tau_idx]
          #stau_idx = stauIdx(evt, Branches["GenVisTau_genPartIdxMother"][evt][tau_idx], Branches["GenPart_pdgId"], Branches["GenPart_genPartIdxMother"])
          #Lxyz_evt.append(L(evt, mthrTau_idx, stau_idx))
      vZ.append(vZ_evt)
      vRho.append(vRho_evt)
      isStauMother.append(isStauMother_evt)
      pt.append(pt_evt)
      eta.append(eta_evt)
      #Lxyz.append(Lxyz_evt)
    #FidTau_Dict = {"pt" : ak.flatten(Branches["GenVisTau_pt"]), "eta": np.abs(ak.flatten(Branches["GenVisTau_eta"])), "vertexZ": ak.flatten(vZ), "vertexRho": ak.flatten(vRho), "stauMother": ak.flatten(isStauMother), "Lxyz": ak.flatten(Lxyz)}
    FidTau_Dict = {"pt" : ak.flatten(pt), "eta": np.abs(ak.flatten(eta)), "vertexZ": ak.flatten(vZ), "vertexRho": ak.flatten(vRho), "stauMother": ak.flatten(isStauMother)}
    FidTau_df = pd.DataFrame(data = FidTau_Dict)
    #fullTauCount = len(FidTau_df.query("pt > @tau_selections.genTauPtMin & eta < @tau_selections.genTauEtaMax & vertexZ < @tau_selections.genTauVtxZMax & vertexRho < @tau_selections.genTauVtxRhoMax & stauMother == 1 & @tau_selections.stauTauVtxDistMin < Lxyz < @tau_selections.stauTauVtxDistMax"))
    fullTauCount = len(FidTau_df.query("pt > @tau_selections.genTauPtMin & eta < @tau_selections.genTauEtaMax & vertexZ < @tau_selections.genTauVtxZMax & vertexRho < @tau_selections.genTauVtxRhoMax & stauMother == 1"))
    print("Total number of hadronic taus is ", fullTauCount) 
      
### Make TEfficiency plot ###
    Sample[file] = ROOT.TEfficiency("e_taueff", "M = 100 GeV, Lxy = 100mm for " + str(tau_selections.stauTauVtxDistMin) + " < L(#tau, s#tau) < " +str(tau_selections.stauTauVtxDistMax) + "; DisTauTagger_Score1; Fraction of hadronically decaying taus matched to a jet", 200, -0.0025, 0.9975) 
    for score_idx in range(len(scoreSpace)):
      Sample[file].SetTotalEvents(score_idx+1, fullTauCount)
      Sample[file].SetPassedEvents(score_idx+1, matchedTauCount[score_idx])
    


#h_score_dict = {}
#h_score_leg = ROOT.TLegend(0.6, 0.6, 0.88, 0.85)
#first_hist = True

#for file in score_dict:
#  h_score_dict[file] = ROOT.TH1F(file+"_score", "Score 1; score 1; Fraction of jets", 100, 0.5, 1)
#  for score in score_dict[file]:
#    h_score_dict[file].Fill(score)
#  h_score_dict[file].Scale(1/len(score_dict[file]))
#  h_score_dict[file].SetStats(0)
#  h_score_leg.AddEntry(h_score_dict[file], file)
#  if first_hist:
#    h_score_dict[file].SetLineColor(ROOT.kBlue)
#    h_score_dict[file].Draw("HISTE")
#    first_hist = False
#  else:
#    h_score_dict[file].SetLineColor(ROOT.kRed)
#    h_score_dict[file].Draw("SAMEHISTE")
#h_score_leg.Draw()
#can.SaveAs("Run3_M100v500_score.pdf")

colorDict = {}
colorIdx3 = 0
colorIdx2 = 0
for file in Files:
  if "Run3" in file:    
    Run3_Eff[file] = array( 'd' ) 
    Run3_Fake[file] = array( 'd' ) 
    if "Stau" in file:
      #print(file)
      #print(colorIdx3)
      colorDict[file] = colors[colorIdx3]
      colorIdx3+=1
      for score_idx in range(len(scoreSpace)):
        Run3_Eff[file].append(Sample[file].GetEfficiency(score_idx+1))
    if "TT" in file:
      for score_idx in range(len(scoreSpace)):
        Run3_Fake[file].append(BG[file].GetEfficiency(score_idx+1))


  if "Run2" in file:
    Run2_Eff[file] = array( 'd' ) 
    Run2_Fake[file] = array( 'd' )  
    if "Stau" in file:
      #print(file)
      #print(colorIdx2)
      colorDict[file] = colors[colorIdx2]
      colorIdx2+=1
      for score_idx in range(len(scoreSpace)):
        Run2_Eff[file].append(Sample[file].GetEfficiency(score_idx+1))
    if "TT" in file:
      for score_idx in range(len(scoreSpace)):
        Run2_Fake[file].append(BG[file].GetEfficiency(score_idx+1))

#Run3_ROC["TauEff"] = array( 'd' ) 
#Run3_ROC["FakeRate"] = array( 'd' ) 
#Run2_ROC["TauEff"] = array( 'd' ) 
#Run2_ROC["FakeRate"] = array( 'd' )  
#for file in Files:
#  if "Run3" in file:    
#    if "Stau" in file:
#      #print(file)
#      #print(colorIdx3)
#      #colorDict[file] = colors[colorIdx3]
#      #colorIdx3+=1
#      for score_idx in range(len(scoreSpace)):
#        Run3_ROC["TauEff"].append(Sample[file].GetEfficiency(score_idx+1))
#    if "TT" in file:
#      for score_idx in range(len(scoreSpace)):
#        Run3_ROC["FakeRate"].append(BG[file].GetEfficiency(score_idx+1))
#
#
#  if "Run2" in file:
#    if "Stau" in file:
#      #print(file)
#      #print(colorIdx2)
#      #colorDict[file] = colors[colorIdx2]
#      #colorIdx2+=1
#      for score_idx in range(len(scoreSpace)):
#        Run2_ROC["TauEff"].append(Sample[file].GetEfficiency(score_idx+1))
#        print(Sample[file].GetEfficiency(score_idx+1))
#    if "TT" in file:
#      for score_idx in range(len(scoreSpace)):
#        Run2_ROC["FakeRate"].append(BG[file].GetEfficiency(score_idx+1))
#
#print(Run2_ROC["TauEff"])
#
#
#TauEff = ROOT.TMultiGraph("taueff", "taueff")
#TauEff_legend = ROOT.TLegend(0.7, 0.5, 0.88, 0.65)
#Run2_TauEff = ROOT.TGraph(100, scoreSpace, Run2_ROC["TauEff"])
#Run3_TauEff = ROOT.TGraph(100, scoreSpace, Run3_ROC["TauEff"])
#Run2_TauEff.SetLineColor(ROOT.kRed)
#Run3_TauEff.SetLineColor(ROOT.kBlue)
#Run2_TauEff.SetMaximum(1)
#TauEff.Add(Run2_TauEff)
#TauEff.Add(Run3_TauEff)
#TauEff_legend.AddEntry(Run2_TauEff, "Stau_M_100GeV_c#tau_100mm_13TeV")
#TauEff_legend.AddEntry(Run3_TauEff, "Stau_M_100GeV_c#tau_100mm_13.6TeV")
#TauEff.SetTitle("Tau efficiency for Run2 and Run3 stau samples for " + str(tau_selections.stauTauVtxDistMin) + "  < L(#tau, s#tau) < " + str(tau_selections.stauTauVtxDistMax) + "; score; Tau Efficiency")
#TauEff.Draw("A")
#TauEff_legend.Draw()
#can.SaveAs("TauEfficiency_L" + str(tau_selections.stauTauVtxDistMin) + "to" + str(tau_selections.stauTauVtxDistMax) + ".pdf")
#

Run2_TauEff = ROOT.TMultiGraph("run2_taueff", "run2_taueff")
Run2_TauEff_legend = ROOT.TLegend(0.7, 0.5, 0.88, 0.65)
Run3_TauEff = ROOT.TMultiGraph("run3_taueff", "run3_taueff")
Run3_TauEff_legend = ROOT.TLegend(0.7, 0.5, 0.88, 0.65)
Run2_TauEff_TGraph = {}
Run3_TauEff_TGraph = {}
for file in Run2_Eff:
    Run2_TauEff_TGraph[file] = ROOT.TGraph(len(scoreSpace), scoreSpace, Run2_Eff[file])
    Run2_TauEff.Add(Run2_TauEff_TGraph[file])
    Run2_TauEff_legend.AddEntry(Run2_TauEff_TGraph[file], file)      
for file in Run3_Eff:
    Run3_TauEff_TGraph[file] = ROOT.TGraph(len(scoreSpace), scoreSpace, Run3_Eff[file])
    Run3_TauEff.Add(Run3_TauEff_TGraph[file])
    Run3_TauEff_legend.AddEntry(Run3_TauEff_TGraph[file], file)      
Run2_TauEff.SetMaximum(1)
Run3_TauEff.SetMaximum(1)
Run2_TauEff.SetTitle("Tau efficiency for Run2 stau samples; score; Tau Efficiency")
Run2_TauEff.Draw("A")
Run2_TauEff_legend.Draw()
can.SaveAs("Run2_TauEfficiency.pdf")
Run3_TauEff.SetTitle("Tau efficiency for Run3 stau samples; score; Tau Efficiency")
Run3_TauEff.Draw("A")
Run3_TauEff_legend.Draw()
can.SaveAs("Run3_TauEfficiency.pdf")


#FakeRate = ROOT.TMultiGraph("fakerate", "fakerate")
#FakeRate_legend = ROOT.TLegend(0.7, 0.5, 0.88, 0.65)
#Run2_FakeRate = ROOT.TGraph(100, scoreSpace, Run2_ROC["FakeRate"])
#Run3_FakeRate = ROOT.TGraph(100, scoreSpace, Run3_ROC["FakeRate"])
#Run2_FakeRate.SetLineColor(ROOT.kRed)
#Run3_FakeRate.SetLineColor(ROOT.kBlue)
##Run2_FakeRate.SetMaximum(1)
#FakeRate.Add(Run2_FakeRate)
#FakeRate.Add(Run3_FakeRate)
#FakeRate_legend.AddEntry(Run2_FakeRate, "Run2_TT_13TeV")
#FakeRate_legend.AddEntry(Run3_FakeRate, "Run3_TT_13.6TeV")
#FakeRate.SetTitle("Fake Rate for Run2 and Run3 TT backgrounds; score; Fake Rate")
#FakeRate.Draw("A")
#FakeRate_legend.Draw()
#can.SaveAs("FakeRate.pdf")
#
#ROC_legend = ROOT.TLegend(0.6, 0.6, 0.88, 0.85)
#ROC_legend.SetBorderSize(0)
#ROC_legend.SetFillStyle(0)

ROC2_legend = ROOT.TLegend(0.6, 0.1, 0.88, 0.3)
ROC2_legend.SetBorderSize(0)
ROC2_legend.SetFillStyle(0)
ROC3_legend = ROOT.TLegend(0.6, 0.6, 0.88, 0.85)
ROC3_legend.SetBorderSize(0)
ROC3_legend.SetFillStyle(0)
ROC2 = ROOT.TMultiGraph("roc2","roc2")
ROC3 = ROOT.TMultiGraph("roc3","roc3")

for file in Files:
  if "Run2" in file:
    if "Stau" in file:
      Run2_ROC[file] = ROOT.TGraph(100, Run2_Fake["Run2_TTToHadronic_13TeV"], Run2_Eff[file])
      Run2_ROC[file].SetMaximum(1)
      Run2_ROC[file].SetLineColor(colorDict[file])
      ROC2.Add(Run2_ROC[file])
      ROC2_legend.AddEntry(Run2_ROC[file], file)
  if "Run3" in file:
    if "Stau" in file:
      Run3_ROC[file] = ROOT.TGraph(100, Run3_Fake["Run3_TT_13p6TeV"], Run3_Eff[file])
      Run3_ROC[file].SetMaximum(1)
      Run3_ROC[file].SetLineColor(colorDict[file])
      ROC3.Add(Run3_ROC[file])
      ROC3_legend.AddEntry(Run3_ROC[file], file)
        
ROC2.SetTitle("ROC Curves for different mass points for Run2; Fake Rate; Tau Efficiency")
ROC3.SetTitle("ROC Curves for different mass points for Run3; Fake Rate; Tau Efficiency")

ROC2.Draw("A")
ROC2_legend.Draw()
can.SetLogx(1)
can.SaveAs("Run2_ROC_DiffMass_Logx.pdf")

ROC3.Draw("A")
ROC3_legend.Draw()
can.SetLogx(1)
can.SaveAs("Run3_ROC_DiffMass_Logx.pdf") 

#ROC = ROOT.TMultiGraph("roc","roc")
#
#Run2_ROCCurve = ROOT.TGraph(100, Run2_ROC["FakeRate"], Run2_ROC["TauEff"])
#Run2_ROCCurve.SetLineColor(ROOT.kRed)
#Run2_ROCCurve.SetMaximum(1)
##Run2_ROCCurve.Draw("ACP")
#ROC_legend.AddEntry(Run2_ROCCurve, "Run2")
##Run2_ROCCurve.SetTitle("ROC Curve based on disTauTag_score1;Fake Rate;Tau Efficiency")
#ROC.Add(Run2_ROCCurve)
#
#Run3_ROCCurve = ROOT.TGraph(100, Run3_ROC["FakeRate"], Run3_ROC["TauEff"])
#Run3_ROCCurve.SetLineColor(ROOT.kBlue)
##Run3_ROCCurve.Draw("CP")
#ROC.Add(Run3_ROCCurve)
#ROC_legend.AddEntry(Run3_ROCCurve, "Run3")
#ROC.SetTitle("ROC Curve based on disTauTag_score1 for " + str(tau_selections.stauTauVtxDistMin) + "  < L(#tau, s#tau) < " + str(tau_selections.stauTauVtxDistMax) + ";Fake Rate;Tau Efficiency")
#
#ROC.Draw("A")
#ROC_legend.Draw()
#can.SaveAs("ROC_TT_Run2_100GeV_100mm_Run3_L" + str(tau_selections.stauTauVtxDistMin) + "to" + str(tau_selections.stauTauVtxDistMax) + "_Lxyz.pdf")


      






