import numpy as np
import uproot 
import awkward as ak
import ROOT
import scipy
import math
from matplotlib import pyplot as plt
import matplotlib as mpl
exec(open("NanoAOD_Dict.py").read()) 

Files = Nano_Dict
Num_jet = {}
Hists = {}
Dict_idx = {}

colors = [ROOT.kBlue, ROOT.kRed, ROOT.kGreen, ROOT.kBlack, ROOT.kViolet, ROOT.kMagenta]
score_legend = ROOT.TLegend(0.7,0.6,0.85,0.8)
dict_idx = 0

for file in Files:
  if ("TT" in file): 
    nano_file = uproot.open(Files[file])
    nano_file_evt = nano_file["Events"]

    nevt = nano_file_evt.num_entries
    score = 0.6
    dR_squ = 0.09

    Branches = {
                "GenVisTau_pt": nano_file_evt["GenVisTau_pt"].array(),
                "GenVisTau_phi":  nano_file_evt["GenVisTau_phi"].array(),
                "GenVisTau_eta":  nano_file_evt["GenVisTau_pt"].array(),
                "GenVisTau_genPartIdxMother":  nano_file_evt["GenVisTau_pt"].array(),

                "Jet_pt": nano_file_evt["Jet_pt"].array(),
                "Jet_phi": nano_file_evt["Jet_phi"].array(),
                "Jet_eta": nano_file_evt["Jet_eta"].array(),
                "Jet_disTauTag_score1": nano_file_evt["Jet_disTauTag_score1"].array(),
                "Jet_genJetIdx": nano_file_evt["Jet_genJetIdx"].array(),
                
                "GenPart_genPartIdxMother": nano_file_evt["GenPart_genPartIdxMother"].array(),
                "GenPart_pdgId": nano_file_evt["GenPart_pdgId"].array(),

                "GenJet_partonFlavour": nano_file_evt["GenJet_partonFlavour"].array()
    }


    jets_pass_score = []
    
    for evt in range(nevt):
      jets_pass_score_evt = []
      for jet_idx in range(len(Branches["Jet_disTauTag_score1"][evt])):
        if (Branches["Jet_disTauTag_score1"][evt][jet_idx] >= score):
          jets_pass_score_evt.append(jet_idx)
      jets_pass_score.append(jets_pass_score_evt)

    unmatch_jets_pass_score = []
    unmatch_jets_pass_score_pt = []

    for evt in range(nevt):
      unmatch_jets_pass_score_evt = []
      unmatch_jets_pass_score_pt_evt = []
      
      if (len(Branches["GenVisTau_pt"][evt])== 0):
        for jet_idx in jets_pass_score[evt]:
          unmatch_jets_pass_score_evt.append(jet_idx)
          unmatch_jets_pass_score_pt_evt.append(Branches["Jet_pt"][evt][jet_idx])
      
      else:
        for tau_idx in range(len(Branches["GenVisTau_pt"][evt])):
          for jet_idx in jets_pass_score[evt]:
            dphi = abs(Branches["GenVisTau_phi"][evt][tau_idx]-Branches["Jet_phi"][evt][jet_idx])
            if (dphi > math.pi) :
              dphi -= 2 * math.pi
            deta = Branches["GenVisTau_eta"][evt][tau_idx]-Branches["Jet_eta"][evt][jet_idx]
            if (dphi * dphi + deta * deta > dR_squ):
              if jet_idx in unmatch_jets_pass_score_evt: continue
              unmatch_jets_pass_score_evt.append(jet_idx)
              unmatch_jets_pass_score_pt_evt.append(Branches["Jet_pt"][evt][jet_idx])
      unmatch_jets_pass_score.append(unmatch_jets_pass_score_evt)
      unmatch_jets_pass_score_pt.append(unmatch_jets_pass_score_pt_evt)

    Num_jet[file] = len(ak.flatten(unmatch_jets_pass_score))
    print(len(ak.flatten(unmatch_jets_pass_score)))
    
    Hists[file] = ROOT.TH1F("h_umut_jetscore_"+file, file+";score 1;Fraction of jets unmatched to #tau_{h} that can't be traced to a genJet", 81, 0.6, 1)
    print(Hists[file].GetName())
    
    Dict_idx[file] = dict_idx
    dict_idx += 1

    for evt in range(nevt):
      for jet_idx in unmatch_jets_pass_score[evt]:
        if (Branches["Jet_genJetIdx"][evt][jet_idx] == -1 and Branches["Jet_pt"][evt][jet_idx] > 20 and abs(Branches["Jet_eta"][evt][jet_idx]) < 2.4):
          Hists[file].Fill(Branches["Jet_disTauTag_score1"][evt][jet_idx])

    Branches.clear()

can = ROOT.TCanvas("can", "can", 1000, 600)
can.Draw()
for hists in Hists:
  Hists[hists].SetLineColor(colors[Dict_idx[hists]])
  Hists[hists].Scale(1/Num_jet[hists])
  Hists[hists].SetStats(0)
  #Hists[hists].Draw("HISTE")
  Hists[hists].Draw("SAMEHISTE")
  score_legend.AddEntry(Hists[hists], hists)
score_legend.Draw()
can.SaveAs("Unmatched_Untraced_Jets.pdf")


    

      
        
