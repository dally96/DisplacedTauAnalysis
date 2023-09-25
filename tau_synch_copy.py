dR_squ = 0.4**2      #Value of dR that determines whether a jet is matched to a tau 

jetPtMin = 20        #The minimum pt a jet can have for the tagger 
jetEtaMax = 2.4      #The maximum eta a jet can have for tagger 

genJetPtMin = 20     #The minimum pt a gen jet can have 
genJetEtaMax = 2.4

genTauPtMin = 20     #The mimimum pt for a gen tau 
genTauEtaMax = 2.4   #The maximum eta for a gen tau 

genTauVtxZMax = 100  #The maximum distance of the z-coordinate to the tau production vertex
genTauVtxRhoMax = 50 #The maximum distance in the xy-plane of the tau production vertex

import uproot 
import awkward as ak
import scipy
import math
from matplotlib import pyplot as plt 
import matplotlib as mpl 
import pandas as pd
import numpy as np
from array import array

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
            "Jet_genJetIdx": nano_file_evt["Jet_genJetIdx"].array(),
            "Jet_partonFlavour": nano_file_evt["Jet_partonFlavour"].array(),
            "Jet_hadronFlavour": nano_file_evt["Jet_hadronFlavour"].array(),
            "Jet_jetId" : nano_file_evt["Jet_jetId"].array(),
            "Jet_nElectrons": nano_file_evt["Jet_nElectrons"].array(),
            "Jet_nMuons": nano_file_evt["Jet_nMuons"].array(),
            "Jet_matchedGenJetIdx" : [],
         
    
            "GenPart_genPartIdxMother": nano_file_evt["GenPart_genPartIdxMother"].array(),
            "GenPart_pdgId": nano_file_evt["GenPart_pdgId"].array(),
            "GenPart_status": nano_file_evt["GenPart_status"].array(),
            "GenPart_statusFlags": nano_file_evt["GenPart_statusFlags"].array(),
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

lepton_selection = ((abs(Branches["GenPart_pdgId"]) == 11) | (abs(Branches["GenPart_pdgId"]) == 13) | (abs(Branches["GenPart_pdgId"]) == 15))
prompt_selection = (Branches["GenPart_statusFlags"] & 1 == 1)
first_selection  = ((Branches["GenPart_statusFlags"] & 4096) == 4096)

GenPart_genLeptons_eta = Branches["GenPart_eta"][lepton_selection & (prompt_selection & first_selection)]
GenPart_genLeptons_phi = Branches["GenPart_phi"][lepton_selection & (prompt_selection & first_selection)]
GenPart_genLeptons_pt = Branches["GenPart_pt"][lepton_selection & (prompt_selection & first_selection)]


print(len(ak.flatten(GenPart_genLeptons_eta)))

nevt = nano_file_evt.num_entries
print(nevt)

for evt in range(nevt):
  if len(GenPart_genLeptons_phi[evt]) > 2:
    print(evt)
for evt in range(nevt):
#  if len(Branches["Jet_pt"][evt]) > 0 :
#    for jet_idx in range(len(Branches["Jet_pt"][evt])):
#        has_genTau = False
#        if Branches["Jet_jetId"][evt][jet_idx] < 6: continue
#
#        if (len(Branches["GenVisTau_pt"][evt])== 0): continue
#        for tau_idx in range(len(Branches["GenVisTau_pt"][evt])):
#
#            if not (Branches["Jet_pt"][evt][jet_idx] > jetPtMin and abs(Branches["Jet_eta"][evt][jet_idx]) < jetEtaMax): continue
#            if not (Branches["Jet_genJetIdx"][evt][jet_idx] >= 0 and Branches["Jet_genJetIdx"][evt][jet_idx] < len(Branches["GenJet_partonFlavour"][evt])): continue
#            mthrTau_idx = Branches["GenVisTau_genPartIdxMother"][evt][tau_idx]
#            assert(abs(Branches["GenPart_pdgId"][evt][mthrTau_idx])==15)
#            if not (Branches["GenVisTau_pt"][evt][tau_idx] > genTauPtMin and abs( Branches["GenVisTau_eta"][evt][tau_idx] ) < genTauEtaMax): continue
#            #if not (abs(Branches["GenPart_vertexZ"][evt][mthrTau_idx]) < genTauVtxZMax and Branches["GenPart_vertexRho"][evt][mthrTau_idx] < genTauVtxRhoMax): continue
#
#            dphi = abs(Branches["GenVisTau_phi"][evt][tau_idx]-Branches["Jet_phi"][evt][jet_idx])
#
#            if (dphi > math.pi) : dphi -= 2 * math.pi
#            deta = Branches["GenVisTau_eta"][evt][tau_idx]-Branches["Jet_eta"][evt][jet_idx]
#            dR2  = dphi ** 2 + deta ** 2
#            if (dR2 <= dR_squ): has_genTau = True
#
#        if has_genTau:
#            tau_count+=1 
  Jet_matchedGenJetIdx = []
  if len(Branches["Jet_pt"][evt]) > 0 :
    for jet_idx in range(len(Branches["Jet_pt"][evt])):
      for genJet_idx in range(len(Branches["GenJet_pt"][evt])):
        if (len(Jet_matchedGenJetIdx) - 1) == jet_idx: continue
        if genJet_idx in Jet_matchedGenJetIdx : continue
        dphi = abs(Branches["GenJet_phi"][evt][genJet_idx] - Branches["Jet_phi"][evt][jet_idx])
        if (dphi > math.pi) : dphi -= 2 * math.pi
        deta = Branches["GenJet_eta"][evt][genJet_idx] - Branches["Jet_eta"][evt][jet_idx]
        dR2 = dphi ** 2 + deta ** 2
        if (dR2 <= 0.4 ** 2):
          Jet_matchedGenJetIdx.append(genJet_idx)
      if (len(Jet_matchedGenJetIdx) - 1 < jet_idx): Jet_matchedGenJetIdx.append(-1)
  Branches["Jet_matchedGenJetIdx"].append(Jet_matchedGenJetIdx)      
  if len(Branches["Jet_pt"][evt]) > 0 :
    for jet_idx in range(len(Branches["Jet_pt"][evt])):
        lepVeto = False
        
        #if Branches["Jet_nElectrons"][evt][jet_idx] > 0 or Branches["Jet_nMuons"][evt][jet_idx] > 0: continue 
        if not (Branches["Jet_pt"][evt][jet_idx] > jetPtMin and abs(Branches["Jet_eta"][evt][jet_idx]) < jetEtaMax): continue
        #if not (Branches["Jet_genJetIdx"][evt][jet_idx] >= 0 and Branches["Jet_genJetIdx"][evt][jet_idx] < len(Branches["GenJet_partonFlavour"][evt])): continue

        if (Branches["Jet_matchedGenJetIdx"][evt][jet_idx] == -1): continue
        if Branches["GenJet_pt"][evt][Branches["Jet_matchedGenJetIdx"][evt][jet_idx]] < genJetPtMin  or abs(Branches["GenJet_eta"][evt][Branches["Jet_matchedGenJetIdx"][evt][jet_idx]]) > genJetEtaMax: continue
        if len(GenPart_genLeptons_eta[evt]) > 0:
          for tau_idx in range(len(GenPart_genLeptons_eta[evt])):
            lep_dphi = abs(Branches["Jet_phi"][evt][jet_idx] - GenPart_genLeptons_phi[evt][tau_idx])
            if (lep_dphi > math.pi) : dphi -= 2 * math.pi
            lep_deta = Branches["Jet_eta"][evt][jet_idx] - GenPart_genLeptons_eta[evt][tau_idx]
            lep_dR2 = lep_dphi ** 2 + lep_deta ** 2
            if lep_dR2 < 0.4**2:
              lepVeto = True
        if lepVeto: continue
        jet_count += 1 
print(jet_count)
#for i in range(nevt):
#  if not (np.array_equal(Branches["Jet_matchedGenJetIdx"][i], Branches["Jet_genJetIdx"][i])):
#    print("Event ", i, " does not match")
#    print("With manual matching, ", Branches["Jet_matchedGenJetIdx"][i])
#    print("With auto matching, ", Branches["Jet_genJetIdx"][i])
#    print("The length of the genJet array for this event is ", len(Branches["GenJet_pt"][i]))
