dR_squ = 0.4**2      #Value of dR that determines whether a jet is matched to a tau 

jetPtMin = 20        #The minimum pt a jet can have for the tagger 
jetEtaMax = 2.4      #The maximum eta a jet can have for tagger 

genJetPtMin = 20     #The minimum pt a gen jet can have 
genJetEtaMax = 2.4

genTauPtMin = 20     #The mimimum pt for a gen tau 
genTauEtaMax = 2.4   #The maximum eta for a gen tau 

genTauVtxZMax = 100  #The maximum distance of the z-coordinate to the tau production vertex
genTauVtxRhoMax = 50 #The maximum distance in the xy-plane of the tau production vertex

genLepton_iso_dR = 0.4

import uproot 
import ROOT
import awkward as ak
import scipy
import math
from matplotlib import pyplot as plt 
import matplotlib as mpl 
import pandas as pd
import numpy as np
from array import array

#file = "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_106X_upgrade2018_realistic_v16_L1v1-v2_with-disTauTagScore.root"
file = "SUS-RunIISummer20UL18GEN-stau100_lsp1_ctau100mm_v6_with-disTauTagScore.root"
nano_file = uproot.open(file)
nano_file_evt = nano_file["Events"]


Branches = { 
            "GenVisTau_pt": nano_file_evt["GenVisTau_pt"].array(),
            "GenVisTau_phi":  nano_file_evt["GenVisTau_phi"].array(),
            "GenVisTau_eta":  nano_file_evt["GenVisTau_eta"].array(),
            "GenVisTau_mass":  nano_file_evt["GenVisTau_mass"].array(),
            "GenVisTau_genPartIdxMother":  nano_file_evt["GenVisTau_genPartIdxMother"].array(),

            "Jet_pt": nano_file_evt["Jet_pt"].array(),
            "Jet_phi": nano_file_evt["Jet_phi"].array(),
            "Jet_eta": nano_file_evt["Jet_eta"].array(),
            "Jet_mass": nano_file_evt["Jet_mass"].array(),
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
            "GenPart_mass": nano_file_evt["GenPart_mass"].array(),
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
last_selection =   ((Branches["GenPart_statusFlags"] &  2 ** 13) == (2 ** 13))

#def isFirstCopy(pdgId):
  

#GenPart_genLeptons_eta = Branches["GenPart_eta"][lepton_selection & (prompt_selection & first_selection)]
#GenPart_genLeptons_phi = Branches["GenPart_phi"][lepton_selection & (prompt_selection & first_selection)]
#GenPart_genLeptons_pt = Branches["GenPart_pt"][lepton_selection & (prompt_selection & first_selection)]
#GenPart_genLeptons_mass = Branches["GenPart_mass"][lepton_selection & (prompt_selection & first_selection)]
#GenPart_genLeptons_pdgId = Branches["GenPart_pdgId"][lepton_selection & (prompt_selection & first_selection)]
#GenPart_genLeptons_genPartIdxMother = Branches["GenPart_genPartIdxMother"][lepton_selection & (prompt_selection & first_selection)]
#GenPart_genLeptons_statusFlags = Branches["GenPart_statusFlags"][lepton_selection & (prompt_selection & first_selection)]

def isLastfromFirst(evt, part, motherIdx):
  if (((Branches["GenPart_statusFlags"][evt][part] & (2 ** 12)) == (2 ** 12)) & ((Branches["GenPart_statusFlags"][evt][part] & (2 ** 13)) == (2 ** 13)) & ((Branches["GenPart_statusFlags"][evt][part] & (2 ** 0)) == (2 ** 0))): 
    return 1
  if ((Branches["GenPart_statusFlags"][evt][part] & (2 ** 13)) != (2 ** 13)):
    return 0
  else:
    if (Branches["GenPart_pdgId"][evt][motherIdx] == Branches["GenPart_pdgId"][evt][part]): 

      if ((Branches["GenPart_statusFlags"][evt][motherIdx] & (2 ** 12)) == (2 ** 12)):

        if ((Branches["GenPart_statusFlags"][evt][motherIdx] & (2 ** 0)) == (2 ** 0)):
          return 1
        else: 
          return 0  

      else:
        return isLastfromFirst(evt, part, Branches["GenPart_genPartIdxMother"][evt][motherIdx])  

    else:
      return 0

  
LastGenLeptons = []
for evt in range(len(Branches["GenPart_pt"])):
  LastGenLeptons_evt = []
  for part in range(len(Branches["GenPart_pt"][evt])):
    #if ((abs(Branches["GenPart_pdgId"][evt][part]) == 11)): continue
    if ((abs(Branches["GenPart_pdgId"][evt][part]) != 11) & (abs(Branches["GenPart_pdgId"][evt][part]) != 13) & (abs(Branches["GenPart_pdgId"][evt][part]) != 15)): continue
    if isLastfromFirst(evt, part, Branches["GenPart_genPartIdxMother"][evt][part]):
      LastGenLeptons_evt.append(part)
  LastGenLeptons.append(LastGenLeptons_evt)
print(len(ak.flatten(LastGenLeptons)))
#Check how many daughters a particle has
def numberOfDaughters(part, evt):
  daughters = []
  for nextPart in range(part + 1, len(Branches["GenPart_pdgId"][evt])):
    if (abs(Branches["GenPart_pdgId"][evt][nextPart]) in [12, 14, 16]): continue
    if (Branches["GenPart_genPartIdxMother"][evt][nextPart] == part):
      daughters.append(nextPart)
  return len(daughters)    

#Checks if particle has daughters
def finalDaughters(lep, evt, daughters):
  for nextPart in range(lep + 1, len(Branches["GenPart_pdgId"][evt])):
    if (abs(Branches["GenPart_pdgId"][evt][nextPart]) in [12, 14, 16]): continue
    if (Branches["GenPart_genPartIdxMother"][evt][nextPart] == lep):
      daughter = nextPart
      if (numberOfDaughters(daughter, evt) == 0):
        daughters.append(daughter)
      else: 
        finalDaughters(daughter, evt, daughters)

GenLep_vis_p4 = []
for evt in range(len(Branches["GenPart_pt"])):
  GenLep_vis_p4_evt = []  
  if len(LastGenLeptons[evt]) == 0: continue
  for lep in LastGenLeptons[evt]:
    daughters = []
    finalDaughter_p4 = []
    finalDaughters(lep, evt, daughters)
    for part in daughters:
      part_p4 = ROOT.Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<double>")(Branches["GenPart_pt"][evt][part], Branches["GenPart_eta"][evt][part], Branches["GenPart_phi"][evt][part], Branches["GenPart_mass"][evt][part])
      finalDaughter_p4.append(part_p4)
    if len(daughters) == 0:
      GenLep_vis_p4_evt.append(ROOT.Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<double>")(Branches["GenPart_pt"][evt][lep], Branches["GenPart_eta"][evt][lep], Branches["GenPart_phi"][evt][lep], Branches["GenPart_mass"][evt][lep]))
    else:
      for daughter in range(1, len(finalDaughter_p4)):
        finalDaughter_p4[0] += finalDaughter_p4[daughter]
      GenLep_vis_p4_evt.append(finalDaughter_p4[0])
  GenLep_vis_p4.append(GenLep_vis_p4_evt)

#for i in range(len(GenLep_vis_p4)):
#  for lep in GenLep_vis_p4[i]:
#    print("The status flag of this lepton is :", GenPart_genLeptons_statusFlags[i][lep])
#    print("The pdgId of the particle in the genLep collection is :", GenPart_genLeptons_pdgId[i][lep])
#    print("Event:", i)
#    print("The pt of the particle in the genLep collection is :", lep.pt())
#    print("The phi of the particle in the genLep collection is : %.5f"%(lep.phi()))
#    print("The eta of the particle in the genLep collection is : %.5f"%(lep.eta()))
    #if (abs(GenPart_genLeptons_pdgId[i][lep]) == 11):
      #print("The event of this lepton is", i, "and index", LastGenLeptons[i])

GenPart_genLeptons_eta = Branches["GenPart_eta"][LastGenLeptons]
GenPart_genLeptons_phi = Branches["GenPart_phi"][LastGenLeptons]
GenPart_genLeptons_pt = Branches["GenPart_pt"][LastGenLeptons]
GenPart_genLeptons_mass = Branches["GenPart_mass"][LastGenLeptons]
GenPart_genLeptons_pdgId = Branches["GenPart_pdgId"][LastGenLeptons]
GenPart_genLeptons_genPartIdxMother = Branches["GenPart_genPartIdxMother"][LastGenLeptons]
GenPart_genLeptons_statusFlags = Branches["GenPart_statusFlags"][LastGenLeptons]

GenPart_firstlastLep = Branches["GenPart_eta"][lepton_selection & (last_selection & first_selection) & prompt_selection]


#for i in range(len(GenPart_genLeptons_eta)):
#  print("Event is:", i)
#  for lep in range(len(GenPart_genLeptons_eta[i])):
#    print("The status flag of this lepton is :", GenPart_genLeptons_statusFlags[i][lep])
#    print("The pdgId of the particle in the genLep collection is :", GenPart_genLeptons_pdgId[i][lep])
#    print("The pt of the particle in the genLep collection is :", GenPart_genLeptons_pt[i][lep])
#    print("The phi of the particle in the genLep collection is : %.5f"%(GenPart_genLeptons_phi[i][lep]))
#    print("The eta of the particle in the genLep collection is : %.5f"%(GenPart_genLeptons_eta[i][lep]))
    #if (abs(GenPart_genLeptons_pdgId[i][lep]) == 11):
    #  print("The event of this lepton is", i, "and index", LastGenLeptons[i])



nevt = nano_file_evt.num_entries

for jet_idx in range(len(Branches["Jet_pt"][965])):
  for part_idx in range(len(Branches["GenPart_pdgId"][965])):
    if abs(Branches["GenPart_pdgId"][965][part_idx]) == 11:
      if not (Branches["Jet_pt"][965][jet_idx] > jetPtMin and abs(Branches["Jet_eta"][965][jet_idx]) < jetEtaMax): continue
      if not (Branches["Jet_genJetIdx"][965][jet_idx] >= 0 and Branches["Jet_genJetIdx"][965][jet_idx] < len(Branches["GenJet_partonFlavour"][evt])): continue
      jet_p4 = ROOT.Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<double>")(Branches["Jet_pt"][965][jet_idx], Branches["Jet_eta"][965][jet_idx], Branches["Jet_phi"][965][jet_idx], Branches["Jet_mass"][965][jet_idx])
      part_p4 = ROOT.Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<double>")(Branches["GenPart_pt"][965][part_idx], Branches["GenPart_eta"][965][part_idx], Branches["GenPart_phi"][965][part_idx], Branches["GenPart_mass"][965][part_idx])
      if (ROOT.Math.VectorUtil.DeltaR(jet_p4,part_p4) < 0.3):
        print("The jet and the elec idx that are together are ", jet_idx, "with jet pt ", Branches["Jet_pt"][965][jet_idx], "and ", part_idx)

for evt in range(nevt):
  if len(Branches["Jet_pt"][evt]) > 0 :
    for jet_idx in range(len(Branches["Jet_pt"][evt])):
        has_genTau = False
        if (len(Branches["GenVisTau_pt"][evt])== 0): continue
        for tau_idx in range(len(Branches["GenVisTau_pt"][evt])):

            if not (Branches["Jet_pt"][evt][jet_idx] > jetPtMin and abs(Branches["Jet_eta"][evt][jet_idx]) < jetEtaMax): continue
            if not (Branches["Jet_genJetIdx"][evt][jet_idx] >= 0 and Branches["Jet_genJetIdx"][evt][jet_idx] < len(Branches["GenJet_partonFlavour"][evt])): continue
            jet_p4 = ROOT.Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<double>")(Branches["Jet_pt"][evt][jet_idx], Branches["Jet_eta"][evt][jet_idx], Branches["Jet_phi"][evt][jet_idx], Branches["Jet_mass"][evt][jet_idx])
            mthrTau_idx = Branches["GenVisTau_genPartIdxMother"][evt][tau_idx]
            assert(abs(Branches["GenPart_pdgId"][evt][mthrTau_idx])==15)
            if not (Branches["GenVisTau_pt"][evt][tau_idx] > genTauPtMin and abs( Branches["GenVisTau_eta"][evt][tau_idx] ) < genTauEtaMax): continue
            #if not (abs(Branches["GenPart_vertexZ"][evt][mthrTau_idx]) < genTauVtxZMax and Branches["GenPart_vertexRho"][evt][mthrTau_idx] < genTauVtxRhoMax): continue
            visTau_p4 = ROOT.Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<double>")(Branches["GenVisTau_pt"][evt][tau_idx], Branches["GenVisTau_eta"][evt][tau_idx], Branches["GenVisTau_phi"][evt][tau_idx],  Branches["GenVisTau_mass"][evt][tau_idx])

            if (ROOT.Math.VectorUtil.DeltaR(jet_p4,visTau_p4) < 0.3): 
              has_genTau = True
              if evt == 965:
                print("The jet and the tau idx that are together are ", jet_idx, "with jet pt ", Branches["Jet_pt"][evt][jet_idx], "and ", Branches["GenVisTau_genPartIdxMother"][evt][tau_idx])

        if has_genTau:
            tau_count+=1 
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
        jet_p4 = ROOT.Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<double>")(Branches["Jet_pt"][evt][jet_idx], Branches["Jet_eta"][evt][jet_idx], Branches["Jet_phi"][evt][jet_idx], Branches["Jet_mass"][evt][jet_idx])
        if len(GenLep_vis_p4[evt]) > 0:
          for tau_idx in range(len(GenLep_vis_p4[evt])):
            #visible_tau = ROOT.Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<double>")(GenPart_genLeptons_pt[evt][tau_idx], GenPart_genLeptons_eta[evt][tau_idx], GenPart_genLeptons_phi[evt][tau_idx], GenPart_genLeptons_mass[evt][tau_idx])
            #lep_dphi = abs(Branches["Jet_phi"][evt][jet_idx] - GenPart_genLeptons_phi[evt][tau_idx])
            #if (lep_dphi > math.pi) : dphi -= 2 * math.pi
            #lep_deta = Branches["Jet_eta"][evt][jet_idx] - GenPart_genLeptons_eta[evt][tau_idx]
            #lep_dR2 = lep_dphi ** 2 + lep_deta ** 2
            #if lep_dR2 < 0.4**2:
            #  lepVeto = True
            lep_dR = ROOT.Math.VectorUtil.DeltaR(jet_p4, GenLep_vis_p4[evt][tau_idx])
            if lep_dR < genLepton_iso_dR:
              lepVeto = True 
        if lepVeto: continue
#        print("For event", evt, "and jet", jet_idx, "this jet remains and has pt:", Branches["Jet_pt"][evt][jet_idx])
        jet_count += 1 
print(jet_count)
#for i in range(nevt):
#  if not (np.array_equal(Branches["Jet_matchedGenJetIdx"][i], Branches["Jet_genJetIdx"][i])):
#    print("Event ", i, " does not match")
#    print("With manual matching, ", Branches["Jet_matchedGenJetIdx"][i])
#    print("With auto matching, ", Branches["Jet_genJetIdx"][i])
#    print("The length of the genJet array for this event is ", len(Branches["GenJet_pt"][i]))
