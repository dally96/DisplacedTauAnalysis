import uproot
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import scipy 
import ROOT as rt
import awkward as ak

Run2_Cust = uproot.open("Stau_M_100_100mm_Summer18UL_NanoAOD.root")
Run2_Tree = Run2_Cust["Events"]

Sample_Cust = uproot.open("Stau_M_100_100mm_Summer22EE_NanoAOD.root")
Sample_Tree = Sample_Cust["Events"]

Run2_DisTau_Score0 = Run2_Tree["Jet_disTauTag_score0"].array()
Run2_DisTau_Score1 = Run2_Tree["Jet_disTauTag_score1"].array()
BG_nJets = Run2_Tree["nJet"].array()
BG_genJetIdx = Run2_Tree["Jet_genJetIdx"].array()
BG_nTau = Run2_Tree["nTau"].array()
BG_nGenVisTau = Run2_Tree["nGenVisTau"]
BG_TauJetIdx = Run2_Tree["Tau_jetIdx"].array()
BG_TauFlav = Run2_Tree["Tau_genPartFlav"].array()
#print(BG_TauFlav)
Run2_GenPart_vertexRho = Run2_Tree["GenPart_vertexRho"].array()
Run3_GenPart_vertexRho = Sample_Tree["GenPart_vertexRho"].array()
Run2_GenVisTau_genPartIdxMother = Run2_Tree["GenVisTau_genPartIdxMother"].array()
Run3_GenVisTau_genPartIdxMother = Sample_Tree["GenVisTau_genPartIdxMother"].array()

h_r2_lxy = rt.TH1F("h_r2_lxy", "Lxy of Run 2 Gen Taus; L_{xy} [cm]; Normalized fraction of taus", 50, 0, 100)
h_r3_lxy = rt.TH1F("h_r3_lxy", "Lxy of Run 3 Gen Taus; L_{xy} [cm]; Number of #tau", 50, 0, 100)

for evt in range(len(Run2_GenPart_vertexRho)):
  for indx in Run2_GenVisTau_genPartIdxMother[evt]:
     h_r2_lxy.Fill(Run2_GenPart_vertexRho[evt][indx])

for evt in range(len(Run3_GenPart_vertexRho)):
  for indx in Run3_GenVisTau_genPartIdxMother[evt]:
     h_r3_lxy.Fill(Run3_GenPart_vertexRho[evt][indx])

can = rt.TCanvas("can", "can", 1000, 600)
leg = rt.TLegend(0.6,0.6,0.88,0.85)
h_r2_lxy.Scale(1/h_r2_lxy.Integral(1, 50))
h_r2_lxy.Draw("HISTE")
h_r2_lxy.SetTitle("L_{xy} of Gen Taus")
h_r2_lxy.SetStats(0)
h_r2_lxy.SetLineColor(rt.kBlue)

leg.AddEntry(h_r2_lxy, "Run 2 M = 100 GeV, c#tau = 100 mm")

h_r3_lxy.Scale(1/h_r3_lxy.Integral(1, 50))
h_r3_lxy.Draw("SAMEHISTE")
h_r3_lxy.SetStats(0)
h_r3_lxy.SetLineColor(rt.kRed)

leg.AddEntry(h_r3_lxy, "Run 3 M = 100 GeV, c#tau = 100 mm")
leg.Draw()
can.SaveAs("Lxy.pdf")

#for i in range(len(BG_genJetIdx)):
#  for j in BG_genJetIdx[i]:
#    if BG_genJetIdx[i][j] < 0:
#      print("No index jet")
#      break
#  break

BG_Tau_jets = []

for evt_idx, evt in enumerate(BG_TauJetIdx):
  tau_jets_evt = []
  for tau_idx, tau in enumerate(BG_TauJetIdx[evt_idx]):
    if BG_genJetIdx[evt_idx][tau] != -1:
      tau_jets_evt.append(tau)
  BG_Tau_jets.append(tau_jets_evt)

Sample_DisTau_Score0 = Sample_Tree["Jet_disTauTag_score0"].array()
Sample_DisTau_Score1 = Sample_Tree["Jet_disTauTag_score1"].array()
Sample_genJetIdx = Sample_Tree["Jet_genJetIdx"].array()
Sample_TauJetIdx = Sample_Tree["Tau_jetIdx"].array()
Sample_TauGenPartIdx = Sample_Tree["Tau_genPartIdx"].array()

Sample_TauJetIdx_Tau = []

for i in range(len(Sample_TauJetIdx)):
  for j in range(len(Sample_TauJetIdx[i])):
    Sample_TauJetIdx_Tau.append([x for x in Sample_TauJetIdx[i] if (Sample_TauGenPartIdx[i][j] == 1 or Sample_TauGenPartIdx[i][j] == 0) and x >= 0])

print(Sample_TauJetIdx_Tau)


Sample_Tau_jets = []

for evt_idx, evt in enumerate(Sample_TauJetIdx):
  tau_jets_evt = []
  for tau_idx, tau in enumerate(Sample_TauJetIdx[evt_idx]):
    if Sample_genJetIdx[evt_idx][tau] != -1:
      tau_jets_evt.append(tau)
  Sample_Tau_jets.append(tau_jets_evt)

BG_GenJet_DisTau_Score0 = []
BG_GenJet_DisTau_Score1 = []

for evt_idx, evt in enumerate(BG_genJetIdx):
  Score0_Evt = []
  Score1_Evt = []
  for tau_idx, tau in enumerate(BG_Tau_jets[evt_idx]):
    Score0_Evt.append(Run2_DisTau_Score0[evt_idx][tau])
    Score1_Evt.append(Run2_DisTau_Score1[evt_idx][tau])
  BG_GenJet_DisTau_Score0.append(Score0_Evt)
  BG_GenJet_DisTau_Score1.append(Score1_Evt)

Sample_GenJet_DisTau_Score0 = []
Sample_GenJet_DisTau_Score1 = []

for evt_idx, evt in enumerate(Sample_genJetIdx):
  Score0_Evt = []
  Score1_Evt = []
  for tau_idx, tau in enumerate(Sample_TauJetIdx_Tau[evt_idx]):
    Score0_Evt.append(Sample_DisTau_Score0[evt_idx][tau])
    Score1_Evt.append(Sample_DisTau_Score1[evt_idx][tau])
  Sample_GenJet_DisTau_Score0.append(Score0_Evt)
  Sample_GenJet_DisTau_Score1.append(Score1_Evt)

#BG_DisTau_Score0 = ak.flatten(Run2_DisTau_Score0)
#BG_DisTau_Score1 = ak.flatten(Run2_DisTau_Score1)

#Sample_DisTau_Score0 = ak.flatten(Sample_DisTau_Score0)
#Sample_DisTau_Score1 = ak.flatten(Sample_DisTau_Score1)

Sample_GenJet_DisTau_Score0 = ak.flatten(Sample_GenJet_DisTau_Score0)
Sample_GenJet_DisTau_Score1 = ak.flatten(Sample_GenJet_DisTau_Score1)

print(np.add(Sample_GenJet_DisTau_Score0, Sample_GenJet_DisTau_Score1))

BG_GenJet_DisTau_Score0 = ak.flatten(BG_GenJet_DisTau_Score0)
BG_GenJet_DisTau_Score1 = ak.flatten(BG_GenJet_DisTau_Score1)

print(len(ak.flatten(BG_genJetIdx)))
#BG_DisTau_Total = np.add(BG_DisTau_Score0, BG_DisTau_Score1)
#Sample_DisTau_Total = np.add(Sample_DisTau_Score0, Sample_DisTau_Score1)

weights_samp = [100*1.1678E-1/len(Sample_GenJet_DisTau_Score1),]*len(Sample_GenJet_DisTau_Score1)
weights_BG = [100*7.623E5/len(BG_GenJet_DisTau_Score1),]*len(BG_GenJet_DisTau_Score1)

plt.hist(Sample_GenJet_DisTau_Score1, bins=np.arange(0,1.02,0.02), weights=weights_samp, histtype='step', label = "Sample")
plt.hist(BG_GenJet_DisTau_Score1, bins=np.arange(0,1.02,0.02), weights=weights_BG, histtype='step', label = "BG")
plt.yscale('log')
plt.xlabel("Score 1")
plt.ylabel("Number of jets")
plt.legend()
plt.show()

