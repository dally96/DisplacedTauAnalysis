import uproot
import math
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import scipy 
import awkward as ak

#Using open the root file that contains the sample and call the branches of matter
###>>>Hoping to do this in a separate file later<<<
#Sample_Cust = uproot.open("Staus_M_500_10mm_13p6TeV_Run3Summer22EE_with-disTauTagScore.root")
Sample_Cust = uproot.open("041223_FullSample_with-disTauTagScore.root")
Sample_Tree = Sample_Cust["Events"]

DisTau_Score0 = Sample_Tree["Jet_disTauTag_score0"].array()
DisTau_Score1 = Sample_Tree["Jet_disTauTag_score1"].array()

Jet_genJetIdx = Sample_Tree["Jet_genJetIdx"].array()
Jet_phi = Sample_Tree["Jet_phi"].array()
Jet_eta = Sample_Tree["Jet_eta"].array()
Jet_pt = Sample_Tree["Jet_pt"].array()

Tau_jetIdx = Sample_Tree["Tau_jetIdx"].array()
Tau_genPartIdx = Sample_Tree["Tau_genPartIdx"].array()
Tau_pt = Sample_Tree["Tau_pt"].array()
Tau_eta = Sample_Tree["Tau_eta"].array()
Tau_dxy = Sample_Tree["Tau_dxy"].array()
Tau_genPartFlav = Sample_Tree["Tau_genPartFlav"].array()

GenVisTau_pt = Sample_Tree["GenVisTau_pt"].array()
GenVisTau_eta = Sample_Tree["GenVisTau_eta"].array()
GenVisTau_genPartIdxMother = Sample_Tree["GenVisTau_genPartIdxMother"].array()

GenPart_pdgId = Sample_Tree["GenPart_pdgId"].array()
GenPart_pt = Sample_Tree["GenPart_pt"].array()
GenPart_phi = Sample_Tree["GenPart_phi"].array()
GenPart_eta = Sample_Tree["GenPart_eta"].array()

GenJet_pt = Sample_Tree["GenJet_pt"].array()
GenJet_phi = Sample_Tree["GenJet_phi"].array()
GenJet_eta = Sample_Tree["GenJet_eta"].array()
GenJet_partonFlavour = Sample_Tree["GenJet_partonFlavour"].array()

#Search for which Gen Particles are taus and return the index that corresponds to the taus
GenTauIdx = []
for event_idx, event in enumerate(GenPart_pdgId):
  GenTauIdx_evt = []
  for part_idx,  part in enumerate(GenPart_pdgId[event_idx]):
    if abs(part) == 15:
      GenTauIdx_evt.append(part_idx)
  GenTauIdx.append(GenTauIdx_evt)

#Now compare those taus and see if they match any of the hadronic decaying taus
#Pretty confident that these GenVisTaus actually contain the decay parameters: https://git.covolunablu.org/devoncode/cmssw/-/blob/4c69dceec90cd40989b58bad380425107f226eaa/PhysicsTools/HepMCCandAlgos/plugins/GenVisTauProducer.cc
HadTaus = []

for evt in range(len(GenVisTau_genPartIdxMother)):
  jets = np.intersect1d(GenVisTau_genPartIdxMother[evt], GenTauIdx[evt]) 
  HadTaus.append(np.array(jets))
  if len(jets) == 0:
    if len(GenVisTau_genPartIdxMother[evt]) != 0:
      print("The event that doesn't match a tau is ",evt)
      print(GenVisTau_genPartIdxMother[evt])
      print(GenPart_pdgId[evt])
      print("Had Tau does not match an index for an identified gen tau")
      break

#Filter the gen taus so that we only look at the gen taus which decay hadronically
HadTau_phi = GenPart_phi[HadTaus]
HadTau_eta = GenPart_eta[HadTaus]
HadTau_pt = GenPart_pt[HadTaus]

#We want to compare the GenVisTau decays to a jet by using a dR^2 selection, but we don't know what the dR^2 value should be. Thus, we scan through the values and see where efficiency falls off
dR_squared = np.linspace(0.01,0.25,25)

#Now look at the reco jets and see if they match any of the gen hadronic taus by using a dR^2 comparison. 
tau_cand_per_dR = []
for dRsqu in dR_squared:
  jet_cand = []
  tau_cand = []
  for evt in range(len(HadTau_phi)):
    jet_cand_evt = []
    tau_cand_evt = []
    for tau_idx, tau in enumerate(HadTau_phi[evt]):
      jet_cand_evt_tau = []
      for jet_idx, jet in enumerate(Jet_phi[evt]):
        dphi = tau - jet
        if dphi > math.pi:
          dphi -= 2*math.pi
        deta = HadTau_eta[evt][tau_idx] - Jet_eta[evt][jet_idx]
        if dphi**2 + deta**2 < dRsqu:
          jet_cand_evt_tau.append(jet_idx)
      if len(jet_cand_evt_tau) > 0:
        tau_cand_evt.append(HadTaus[evt][tau_idx])
      jet_cand_evt.append(jet_cand_evt_tau)
    tau_cand.append(tau_cand_evt)
    jet_cand.append(jet_cand_evt)
  tau_cand_per_dR.append(tau_cand)

print(len(tau_cand_per_dR))
print(len(tau_cand_per_dR[0]), len(tau_cand_per_dR[1]))
for i in range(len(tau_cand_per_dR)):
  ak.flatten(tau_cand_per_dR[i])

print(len(tau_cand_per_dR))
print(len(tau_cand_per_dR[0]), len(tau_cand_per_dR[1]))
##Because of the way we compared the taus to jets above, jet_cand has more structure to it than usual. The following line flattens the contents of an event.
#jet_cand = ak.flatten(jet_cand, axis=2)

##Similarly here filter the jets so that we only use use the ones that correspond to gen had tau
#RecoJetTau_pt = Jet_pt[jet_cand]
#RecoJetTau_eta =  Jet_eta[jet_cand]
#RecoJetTau_phi = Jet_phi[jet_cand]

#To get an efficiency plot, we sort both the filtered jets and all of the gen had taus into the same pt bins and then divide the former by the latter. This way we see how many of the hadronic taus are actually being reconstructed into jets 

Jet_Tau_Pt_dR = []

for i in range(len(tau_cand_per_dR)):
  Jet_Tau_Pt_dR.append(GenPart_pt[tau_cand_per_dR[i]])


#JetTau_pt = GenPart_pt[tau_cand]
#JetTauPt = ak.flatten(JetTau_pt)

HadTauPt = ak.flatten(HadTau_pt)

bins = np.linspace(0, 1900, 20)

HadTauPt_Hist, HadTauPt_Bins = np.histogram(HadTauPt, bins = bins)
JetTauPt_HistdR, JetTauPt_Bins = np.histogram(Jet_Tau_Pt_dR[0], bins = bins)
print(JetTauPt_HistdR)

#JetTauPt_HistdR = {}
#for i in range(len(tau_cand_per_dR)):
  #JetTauPt_HistdR[i], JetTauPt_Bins = np.histogram(Jet_Tau_Pt_dR[i], bins = bins)
#print(JetTauPt_HistdR)
#
#Eff = np.divide(JetTauPt_Hist, HadTauPt_Hist)
#
#bincenters = 0.5*(bins[1:]+bins[:-1])
#
#plt.scatter(bincenters, Eff)
#plt.xlabel("Had Tau pt [GeV]")
#plt.ylabel("Efficiency")
#plt.title("Sample m = 500 GeV, lxy = 100 mm")
#yerr = np.divide(np.sqrt(JetTauPt_Hist), HadTauPt_Hist)
#plt.errorbar(bincenters, Eff, yerr = yerr, ls = 'none')
##plt.show()
#
#
#
##For fake rate let's look at the hadronically decaying reco taus. This is where Tau_genPartFlav is equal to 5 
#Reco_HadTaus = []
#for evt in range(len(Tau_genPartFlav)):
#  Reco_HadTaus_evt = []
#  for indx, tau_decay in enumerate(Tau_genPartFlav[evt]):
#    if tau_decay == 5 and Tau_jetIdx[evt][indx] >= 0:
#      Reco_HadTaus_evt.append(Tau_jetIdx[evt][indx])
#  Reco_HadTaus.append(Reco_HadTaus_evt)
#
##We take the jets where the reco tau has partFlav = 5 and the jets that have been associated with gen taus.
##Then, we want the reco tau jets that are not associated with the gen taus (the xor function)
#FakeTaus = []
#TotalTaus = []
#for evt in range(len(jet_cand)):
#  FakeTaus.append([x for x in np.setxor1d(jet_cand[evt], Reco_HadTaus[evt]) if x in Reco_HadTaus[evt]])
#  TotalTaus.append(np.asarray(np.union1d(jet_cand[evt], Reco_HadTaus[evt]), dtype='int'))
#
#FakeTau_pt = Jet_pt[FakeTaus]
#FakeTauPt = ak.flatten(FakeTau_pt)
#
#TotalTau_pt = Jet_pt[TotalTaus]
#TotalTauPt = ak.flatten(TotalTau_pt)
#
#FakeTauPt_Hist, FakeTauPt_Bins = np.histogram(FakeTauPt, bins = bins)
#TotalTauPt_Hist, TotalTauPt_Bins = np.histogram(TotalTauPt, bins = bins)
#
#FakeRate = np.divide(FakeTauPt_Hist, TotalTauPt_Hist)
#
#plt.cla(); plt.clf()
#plt.scatter(bincenters, FakeRate)
#plt.xlabel("Had Tau pt [GeV]")
#plt.ylabel("Fake Rate")
#plt.title("Sample m = 500 GeV, lxy = 100 mm")
#Fakerr = np.divide(np.sqrt(FakeTauPt_Hist), TotalTauPt_Hist)
#plt.errorbar(bincenters, FakeRate, Fakerr, ls = 'none')
#plt.show()
#
#
#
#
