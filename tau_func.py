import numpy as np
import math
import scipy
import os
import uproot
import awkward as ak
import matplotlib as mpl
from matplotlib import pyplot as plt 

#def tauIdx(evt, tauPartMother, genPart, genPartMother):
#  print("This genvistau corresponds with ", genPart[evt][tauPartMother])
#  if abs(genPart[evt][tauPartMother]) == 100015:
#    print("This tau's mother is a stau!")
#    return tauPartMother
#  if abs(genPart[evt][tauPartMother]) == 15:
#    print("This tau's mother is ", genPartMother[evt][tauPartMother])
#    motherIdx = genPartMother[evt][tauPartMother]
#    if abs(genPart[evt][motherIdx]) == 1000015:
#      return tauPartMother
#    elif abs(genPart[evt][motherIdx]) == 15:
#      return tauIdx(evt, motherIdx, genPart, genPartMother)
#    else:
#      print("Tau does not decay from stau")

def getTauVertexR(event, tauIdx):
  for partIdx in range(len(Branches["GenPart_genPartIdxMother"][event])):
    if Branches["GenPart_genPartIdxMother"][event][partIdx] == tauIdx: 
      return Branches["GenPart_vertexR"][event][partIdx]

def getTauVertexRho(event, tauIdx):
  for partIdx in range(len(Branches["GenPart_genPartIdxMother"][event])):
    if Branches["GenPart_genPartIdxMother"][event][partIdx] == tauIdx: 
      return Branches["GenPart_vertexRho"][event][partIdx]

def stauMother(event, partIdx, genPart, genPartIdxMother):
  if abs(genPart[event][partIdx]) == 15:
    motherIdx = genPartIdxMother[event][partIdx]
    if abs(genPart[event][motherIdx]) == 1000015: 
      return 1
    elif abs(genPart[event][motherIdx]) == 15:
      return stauMother(event, motherIdx, genPart, genPartIdxMother)
    else:
      return 0
  if partIdx == -1:
    return 0

def stauIdx(evt, tauPartMother, genPart, genPartMother):
  if abs(genPart[evt][tauPartMother]) == 1000015:
    return tauPartMother
  if abs(genPart[evt][tauPartMother]) == 15:
    motherIdx = genPartMother[evt][tauPartMother]
    if abs(genPart[evt][motherIdx]) == 1000015:
      return motherIdx
    elif abs(genPart[evt][motherIdx]) == 15:
      return stauIdx(evt, motherIdx, genPart, genPartMother)
    else:
      return -999

# Function tells me if any of the W bosons in the TT bg are leptonic in any capacity
def isLeptonic(event, genPart, genPartIdxMother):
  lepton_pdgId = [11, 12, 13, 14, 15, 16]
  leptonic_evt = 0
  for lepton in range(len(genPart[event])):
    print("Inb this event her are the pdgIds", genPart[event][lepton])
    if abs(genPart[event][lepton]) in lepton_pdgId:
      print("In this case, the particle is a lepton")
    #Get the index of the mother particle to the lepton
      motherIdx = genPartIdxMother[event][lepton]
      print("The mother of this lepton is", genPart[event][motherIdx])
      if abs(genPart[event][motherIdx]) == 24:
        leptonic_evt = 1
  return leptonic_evt

def isSUSY(event, genPart, genPartIdxMother):
  for lepton in range(len(genPart[event])):
    if abs(genPart[event][lepton]) == 15:
      motherIdx = genPartIdxMother[event][lepton]
      if abs(genPart[event][motherIdx]) == 15:
        if stauMother(event, motherIdx, genPart, genPartIdxMother):
          return 1
        else:
          return 0 

def vetoRecoLep(jet_phi, jet_eta, event, lep_collec):
  lepJet = 0
  if (len(Branches[lep_collec+"_pt"][event])) == 0:
    return 0
  else:
    for lep_idx in range(len(Branches[lep_collec+"_pt"][event])):
      dphi = abs(Branches[lep_collec+"_phi"][event][lep_idx] - jet_phi)
      if (dphi > math.pi) :
        dphi -= 2 * math.pi
      deta = Branches[lep_collec+"_eta"][event][lep_idx] - jet_eta
      dR2  = dphi ** 2 + deta ** 2
      if dR2 < 0.16:
        lepJet =  1
    return lepJet

def vetoGenLep(jet_phi, jet_eta, event, lep_collec):
  lepJet = 0
  lepPdgId = {
              "Electron":11,
              "eNeutrino":12,
              "Muon":13,
              "mNeutrino":14,
              "Tau":15,
              "tNeutrino":16
              }
  for lep_idx in range(len(Branches["GenPart_pt"][event])):
    if (Branches["GenPart_pdgId"][event][lep_idx] == lepPdgId[lep_collec]):
      dphi = abs(Branches["GenPart_phi"][event][lep_idx] - jet_phi)
      if (dphi > math.pi) :
        dphi -= 2 * math.pi
      deta = Branches["GenPart_eta"][event][lep_idx] - jet_eta
      dR2  = dphi ** 2 + deta ** 2
      if dR2 < 0.16:
        lepJet =  1
    return lepJet

def L(event, tau_idx, stau_idx):
  Lx = Branches["GenPart_vertexX"][event][tau_idx] - Branches["GenPart_vertexX"][event][stau_idx]
  Ly = Branches["GenPart_vertexY"][event][tau_idx] - Branches["GenPart_vertexY"][event][stau_idx]
  Lz = Branches["GenPart_vertexZ"][event][tau_idx] - Branches["GenPart_vertexZ"][event][stau_idx]
  return math.sqrt(Lx**2 + Ly**2 + Lz**2)

def Lxy(event, tau_idx, stau_idx, vertexX, vertexY):
  Lx = vertexX[event][tau_idx] - vertexX[event][stau_idx]
  Ly = vertexY[event][tau_idx] - vertexY[event][stau_idx]
  return math.sqrt(Lx**2 + Ly**2)
  
def ctau(event, tau_idx, stau_idx, mass, vertexX, vertexY, pt, phi, eta):
  Lx = vertexX[event][tau_idx] - vertexX[event][stau_idx]
  Ly = vertexY[event][tau_idx] - vertexY[event][stau_idx]
  L = np.sqrt(Lx**2 + Ly**2)

  px = pt[event][stau_idx] * np.cos(phi[event][stau_idx])
  py = pt[event][stau_idx] * np.sin(phi[event][stau_idx])
  pz = pt[event][stau_idx] * np.sinh(eta[event][stau_idx])
  p = np.sqrt(px**2 + py**2 + pz**2)

  m = mass 

  return L * m/p
