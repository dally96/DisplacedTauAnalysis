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

colors = [ROOT.kBlue, ROOT.kRed, ROOT.kGreen, ROOT.kBlack, ROOT.kViolet, ROOT.kMagenta, ROOT.kGray]
can = ROOT.TCanvas("can", "can")
score_legend = ROOT.TLegend(0.7,0.6,0.8,0.8)
dict_idx = 0
scoreSpace = np.linspace(0.0,0.995,200)
scoreSpace = np.append(scoreSpace, np.linspace(0.9951, 1.0001, 51))
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
  print(file, " is running")
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
              "GenPart_statusFlags": nano_file_evt["GenPart_statusFlags"].array(),

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

  lepton_selection = ((abs(Branches["GenPart_pdgId"]) == 11) | (abs(Branches["GenPart_pdgId"]) == 13) | (abs(Branches["GenPart_pdgId"]) == 15))
  prompt_selection = (Branches["GenPart_statusFlags"] & 1 == 1)
  first_selection  = ((Branches["GenPart_statusFlags"] & 4096) == 4096)

  Branches["GenLep_pt"] = Branches["GenPart_pt"][lepton_selection & (prompt_selection & first_selection)]
  Branches["GenLep_phi"] = Branches["GenPart_phi"][lepton_selection & (prompt_selection & first_selection)]
  Branches["GenLep_eta"] =  Branches["GenPart_eta"][lepton_selection & (prompt_selection & first_selection)]

  if ("TT" in file):
    nevt = nano_file_evt.num_entries
    print(file, " has ", nevt, " number of entries")

### Match jets that pass score cut to hadronically decaying taus using dR matching ###
    unmatchedJetsPassScore = []
    unmatchedJets_score = []  

    for evt in range(nevt):
      if isLeptonic(evt, Branches["GenPart_pdgId"], Branches["GenPart_genPartIdxMother"]): continue 
      unmatchedJetsPassScore_evt = []
      unmatchedJets_score_evt = []  
      if (len(Branches["GenVisTau_pt"][evt])== 0):
        for jet_idx in range(len(Branches["Jet_pt"][evt])):
          if Branches["Jet_pt"][evt][jet_idx] < tau_selections.jetPtMin or abs(Branches["Jet_eta"][evt][jet_idx]) > tau_selections.jetEtaMax: continue
          if Branches["Jet_genJetIdx"][evt][jet_idx] == -1 or Branches["Jet_genJetIdx"][evt][jet_idx] >= len(Branches["GenJet_partonFlavour"][evt]): continue
          if Branches["GenJet_pt"][evt][Branches["Jet_genJetIdx"][evt][jet_idx]] < tau_selections.genJetPtMin  or abs(Branches["GenJet_eta"][evt][Branches["Jet_genJetIdx"][evt][jet_idx]]) > tau_selections.genJetEtaMax: continue
          ###Start of lepton veto
          lepVeto = False
          if len(Branches["GenLep_pt"][evt]) > 0:
            for lep_idx in range(len(Branches["GenLep_pt"][evt])):
              lep_dphi = abs(Branches["Jet_phi"][evt][jet_idx] - Branches["GenLep_phi"][evt][lep_idx])
              if (lep_dphi > math.pi) : dphi -= 2 * math.pi
              lep_deta = Branches["Jet_eta"][evt][jet_idx] - Branches["GenLep_eta"][evt][lep_idx]              
              lep_dR2 = lep_dphi ** 2 + lep_deta ** 2
              if lep_dR2 < 0.4**2:
                lepVeto = True
          if lepVeto:continue        

          unmatchedJetsPassScore_evt.append(jet_idx)   
          unmatchedJets_score_evt.append(Branches["Jet_disTauTag_score1"][evt][jet_idx])
      for tau_idx in range(len(Branches["GenVisTau_pt"][evt])):
        if Branches["GenVisTau_pt"][evt][tau_idx] < tau_selections.genTauPtMin  or abs(Branches["GenVisTau_eta"][evt][tau_idx]) > tau_selections.genTauEtaMax: continue
        mthrTau_idx = Branches["GenVisTau_genPartIdxMother"][evt][tau_idx]
        if Branches["GenPart_vertexZ"][evt][mthrTau_idx] >= tau_selections.genTauVtxZMax or Branches["GenPart_vertexRho"][evt][mthrTau_idx] >= tau_selections.genTauVtxRhoMax: continue
        for jet_idx in range(len(Branches["Jet_pt"][evt])):
          if Branches["Jet_pt"][evt][jet_idx] < tau_selections.jetPtMin or abs(Branches["Jet_eta"][evt][jet_idx]) > tau_selections.jetEtaMax: continue
          if Branches["Jet_genJetIdx"][evt][jet_idx] == -1 or Branches["Jet_genJetIdx"][evt][jet_idx] >= len(Branches["GenJet_partonFlavour"][evt]): continue
          if Branches["GenJet_pt"][evt][Branches["Jet_genJetIdx"][evt][jet_idx]] < tau_selections.genJetPtMin  or abs(Branches["GenJet_eta"][evt][Branches["Jet_genJetIdx"][evt][jet_idx]]) > tau_selections.genJetEtaMax: continue
          lepVeto = False
          if len(Branches["GenLep_pt"][evt]) > 0:
            for lep_idx in range(len(Branches["GenLep_pt"][evt])):
              lep_dphi = abs(Branches["Jet_phi"][evt][jet_idx] - Branches["GenLep_phi"][evt][lep_idx])
              if (lep_dphi > math.pi) : dphi -= 2 * math.pi
              lep_deta = Branches["Jet_eta"][evt][jet_idx] - Branches["GenLep_eta"][evt][lep_idx]              
              lep_dR2 = lep_dphi ** 2 + lep_deta ** 2
              if lep_dR2 < 0.4**2:
                lepVeto = True
          if lepVeto:continue        
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

    unmatchedJet_dict =  {"score": ak.flatten(unmatchedJets_score)}

    unmatchedJet_df = pd.DataFrame(data = unmatchedJet_dict)

### Count the number of jets for each score ###
    fakeJetCount = []
    for score in scoreSpace:
      fakeJetCount.append(len(unmatchedJet_df.query("score >= @score")))
      print("There are ", len(unmatchedJet_df.query("score >= @score")), " jets that aren't matched taus with score greater than or equal to ", score) 

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
  
    for score in scoreSpace:
      matchedTauCount.append(len(matchedJet_df.query("score >= @score")))
      print("There are ", len(matchedJet_df.query("score >= @score")), " jets matched to taus with score greater than or equal to ", score) 

### Count the total number of hadronically decaying taus ###
    pt = []
    eta = []
    vZ = []
    vRho = []
    isStauMother = []
    for evt in range(nevt):
      vZ_evt = []
      vRho_evt = []
      isStauMother_evt = []
      pt_evt = []
      eta_evt = []
      for tau_idx in range(len(Branches["GenVisTau_genPartIdxMother"][evt])):
        vZ_evt.append(Branches["GenPart_vertexZ"][evt][Branches["GenVisTau_genPartIdxMother"][evt][tau_idx]])
        vRho_evt.append(Branches["GenPart_vertexRho"][evt][Branches["GenVisTau_genPartIdxMother"][evt][tau_idx]])
        isStauMother_evt.append(stauMother(evt, Branches["GenVisTau_genPartIdxMother"][evt][tau_idx], Branches["GenPart_pdgId"], Branches["GenPart_genPartIdxMother"]))
        pt_evt.append(Branches["GenVisTau_pt"][evt][tau_idx])
        eta_evt.append(Branches["GenVisTau_eta"][evt][tau_idx])
      vZ.append(vZ_evt)
      vRho.append(vRho_evt)
      isStauMother.append(isStauMother_evt)
      pt.append(pt_evt)
      eta.append(eta_evt)
    FidTau_Dict = {"pt" : ak.flatten(pt), "eta": np.abs(ak.flatten(eta)), "vertexZ": ak.flatten(vZ), "vertexRho": ak.flatten(vRho), "stauMother": ak.flatten(isStauMother)}
    FidTau_df = pd.DataFrame(data = FidTau_Dict)
    fullTauCount = len(FidTau_df.query("pt > @tau_selections.genTauPtMin & eta < @tau_selections.genTauEtaMax & vertexZ < @tau_selections.genTauVtxZMax & vertexRho < @tau_selections.genTauVtxRhoMax & stauMother == 1"))
    print("Total number of hadronic taus is ", fullTauCount) 
      
### Make TEfficiency plot ###
    Sample[file] = ROOT.TEfficiency("e_taueff", "M = 100 GeV, Lxy = 100mm for " + str(tau_selections.stauTauVtxDistMin) + " < L(#tau, s#tau) < " +str(tau_selections.stauTauVtxDistMax) + "; DisTauTagger_Score1; Fraction of hadronically decaying taus matched to a jet", 200, -0.0025, 0.9975) 
    for score_idx in range(len(scoreSpace)):
      Sample[file].SetTotalEvents(score_idx+1, fullTauCount)
      Sample[file].SetPassedEvents(score_idx+1, matchedTauCount[score_idx])
    
colorDict = {}
colorIdx3 = 0
colorIdx2 = 0
for file in Files:
  if "Run3" in file:    
    if "Stau" in file:
      Run3_Eff[file] = array( 'd' ) 
      colorDict[file] = colors[colorIdx3]
      colorIdx3+=1
      for score_idx in range(len(scoreSpace)):
        Run3_Eff[file].append(Sample[file].GetEfficiency(score_idx+1))
    if "TT" in file:
      Run3_Fake[file] = array( 'd' ) 
      for score_idx in range(len(scoreSpace)):
        Run3_Fake[file].append(BG[file].GetEfficiency(score_idx+1))

  if "Run2" in file:
    if "Stau" in file:
      Run2_Eff[file] = array( 'd' ) 
      print(file)
      colorDict[file] = colors[colorIdx2]
      colorIdx2+=1
      for score_idx in range(len(scoreSpace)):
        Run2_Eff[file].append(Sample[file].GetEfficiency(score_idx+1))
      print(Run2_Eff[file][-1])
    if "TT" in file:
      Run2_Fake[file] = array( 'd' )  
      for score_idx in range(len(scoreSpace)):
        Run2_Fake[file].append(BG[file].GetEfficiency(score_idx+1))

Run2_TauEff = ROOT.TMultiGraph("run2_taueff", "run2_taueff")
Run2_TauEff_legend = ROOT.TLegend(0.7, 0.5, 0.88, 0.65)
Run3_TauEff = ROOT.TMultiGraph("run3_taueff", "run3_taueff")
Run3_TauEff_legend = ROOT.TLegend(0.7, 0.5, 0.88, 0.65)
Run2_TauEff_TGraph = {}
Run3_TauEff_TGraph = {}
for file in Run2_Eff:
    print(len(Run2_Eff[file]))
    print(len(scoreSpace))
    Run2_TauEff_TGraph[file] = ROOT.TGraph(len(scoreSpace), scoreSpace, Run2_Eff[file])
    Run2_TauEff_TGraph[file].SetLineColor(colorDict[file])
    Run2_TauEff.Add(Run2_TauEff_TGraph[file])
    Run2_TauEff_legend.AddEntry(Run2_TauEff_TGraph[file], file)      
for file in Run3_Eff:
    Run3_TauEff_TGraph[file] = ROOT.TGraph(len(scoreSpace), scoreSpace, Run3_Eff[file])
    Run3_TauEff_TGraph[file].SetLineColor(colorDict[file])
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

FakeRate = ROOT.TMultiGraph("fakerate", "fakerate")
FakeRate_legend = ROOT.TLegend(0.7, 0.5, 0.88, 0.65)
Run2_FakeRate = ROOT.TGraph(100, scoreSpace, Run2_ROC["FakeRate"])
Run3_FakeRate = ROOT.TGraph(100, scoreSpace, Run3_ROC["FakeRate"])
Run2_FakeRate.SetLineColor(ROOT.kRed)
Run3_FakeRate.SetLineColor(ROOT.kBlue)
# Note: the line below was double-commented out
Run2_FakeRate.SetMaximum(1)
FakeRate.Add(Run2_FakeRate)
FakeRate.Add(Run3_FakeRate)
FakeRate_legend.AddEntry(Run2_FakeRate, "Run2_TT_13TeV")
FakeRate_legend.AddEntry(Run3_FakeRate, "Run3_TT_13.6TeV")
FakeRate.SetTitle("Fake Rate for Run2 and Run3 TT backgrounds; score; Fake Rate")
FakeRate.Draw("A")
FakeRate_legend.Draw()
can.SaveAs("FakeRate.pdf")

ROC_legend = ROOT.TLegend(0.6, 0.6, 0.88, 0.85)
ROC_legend.SetBorderSize(0)
ROC_legend.SetFillStyle(0)

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
      Run2_ROC[file] = ROOT.TGraph(len(scoreSpace), Run2_Eff[file], Run2_Fake["Run2_TTToHadronic_13TeV"])
      Run2_ROC[file].SetMaximum(1)
      Run2_ROC[file].SetLineColor(colorDict[file])
      ROC2.Add(Run2_ROC[file])
      ROC2_legend.AddEntry(Run2_ROC[file], file)
  if "Run3" in file:
    if "Stau" in file:
      Run3_ROC[file] = ROOT.TGraph(len(scoreSpace), Run3_Fake["Run3_TT_13p6TeV"], Run3_Eff[file])
      Run3_ROC[file].SetMaximum(1)
      Run3_ROC[file].SetLineColor(colorDict[file])
      ROC3.Add(Run3_ROC[file])
      ROC3_legend.AddEntry(Run3_ROC[file], file)
        
ROC2.SetTitle("ROC Curves for different mass points for Run2; Tau Efficiency; Fake Rate")
ROC3.SetTitle("ROC Curves for different mass points for Run3; Fake Rate; Tau Efficiency")

ROC2.Draw("A")
ROC2_legend.Draw()
can.SetLogy(1)
can.SaveAs("Run2_ROC_M_400GeV_100mm.pdf")

ROC3.Draw("A")
ROC3_legend.Draw()
can.SetLogx(0)
can.SaveAs("Run3_ROC_DiffMass.pdf") 
