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
#file = "SUS-RunIISummer20UL18GEN-stau100_lsp1_ctau100mm_v6_with-disTauTagScore.root"
#file = "Staus_M_400_100mm_13p6TeV_Run3Summer22_NanoAOD.root"
file = "Staus_M_100_100mm_13p6TeV_Run3Summer22_NanoAOD.root"
#file = "Stau_M_100_100mm_Summer18UL_NanoAOD.root"
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

print("Jet pt cut:", len(ak.flatten(Branches["Jet_pt"][Branches["Jet_pt"] > 20])))
print("Jet eta cut:", len(ak.flatten(Branches["Jet_pt"][abs(Branches["Jet_eta"]) < 2.4])))
print("Total taus:", len(ak.flatten(Branches["GenVisTau_pt"])))
print("Tau pt cut:", len(ak.flatten(Branches["GenVisTau_pt"][Branches["GenVisTau_pt"] > 20])))
print("Tau eta cut:", len(ak.flatten(Branches["GenVisTau_pt"][abs(Branches["GenVisTau_eta"]) < 2.4]))) 

mother_vertexZ = Branches["GenPart_vertexZ"][Branches["GenVisTau_genPartIdxMother"]]
mother_vertexRho = Branches["GenPart_vertexRho"][Branches["GenVisTau_genPartIdxMother"]]

print("Tau vtx z cut:", len(ak.flatten(mother_vertexZ[mother_vertexZ < 100])))
print("Tau vtx rho cut:", len(ak.flatten(mother_vertexRho[mother_vertexRho < 50]))) 
