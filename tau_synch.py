import uproot 
import awkward as ak
import scipy
import math
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
from array import array
import tau_selections as ts


file = "SUS-RunIISummer20UL18GEN-stau100_lsp1_ctau100mm_v6_with-disTauTagScore.root"

nano_file = uproot.open(file)
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
            "Jet_nElectrons": nano_file_evt["Jet_nElectrons"].array(),
            "Jet_nMuons": nano_file_evt["Jet_nMuons"].array(),
            
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

}

tau_count = 0
jet_count = 0

nevt = nano_file_evt.num_entries

for evt in range(nevt):

  if len(Branches["Jet_pt"][evt]) > 0 :
    for jet_idx in range(len(Branches["Jet_pt"][evt])):
      has_genTau = False
      if Branches["Jet_jetId"][evt][jet_idx] < 6: continue

      if (len(Branches["GenVisTau_pt"][evt])== 0): continue
      for tau_idx in range(len(Branches["GenVisTau_pt"][evt])):

        if not (Branches["Jet_pt"][evt][jet_idx] > ts.jetPtMin and abs(Branches["Jet_eta"][evt][jet_idx]) < ts.jetEtaMax): continue
        if not (Branches["Jet_genJetIdx"][evt][jet_idx] >= 0 and Branches["Jet_genJetIdx"][evt][jet_idx] < len(Branches["GenJet_partonFlavour"][evt])): continue
        mthrTau_idx = Branches["GenVisTau_genPartIdxMother"][evt][tau_idx]
        assert(abs(Branches["GenPart_pdgId"][evt][mthrTau_idx])==15)
        if not (Branches["GenVisTau_pt"][evt][tau_idx] > ts.genTauPtMin and abs( Branches["GenVisTau_eta"][evt][tau_idx] ) < ts.genTauEtaMax): continue
        if not (abs(Branches["GenPart_vertexZ"][evt][mthrTau_idx]) < ts.genTauVtxZMax and Branches["GenPart_vertexRho"][evt][mthrTau_idx] < ts.genTauVtxRhoMax): continue
        #if Branches["Jet_nElectrons"][evt][jet_idx] > 0 or Branches["Jet_nMuons"][evt][jet_idx] > 0: continue 

        dphi = abs(Branches["GenVisTau_phi"][evt][tau_idx]-Branches["Jet_phi"][evt][jet_idx])

        if (dphi > math.pi) : dphi -= 2 * math.pi
        deta = Branches["GenVisTau_eta"][evt][tau_idx]-Branches["Jet_eta"][evt][jet_idx]
        dR2  = dphi ** 2 + deta ** 2
        if (dR2 <= ts.dR_squ): has_genTau = True

      if has_genTau:
        tau_count+=1 
      else: 
        jet_count+=1


print(jet_count)

#print("The number of total taus is ", len(ak.flatten(Branches["GenVisTau_pt"])))
#print("The number of total jets is ", len(ak.flatten(Branches["Jet_pt"])))




