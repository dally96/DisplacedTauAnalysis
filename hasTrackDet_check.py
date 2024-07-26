import ROOT, sys, os, pdb, argparse
import numpy as np
from array import array
from collections import OrderedDict
from DataFormats.FWLite import Events, Handle
import math
from copy import deepcopy as dc
from treeVariables_PF import branches


# Get ahold of the events
files = [ #"ntuples/SUS-RunIISummer20UL18GEN-stau100_lsp1_ctau100mm_v6_cms-xrd-global.root",
          #"ntuples/Staus_M_100_100mm_13p6TeV_Run3Summer22EE_cms-xrd-global.root"
          "Staus_M_400_100mm_13p6TeV_Run3Summer22_cms-xrd-global.root",
        ] 
events = Events(files)

maxevents = -1 
totevents = events.size()

handle_jet    = Handle('std::vector<pat::Jet>')
handle_score  = Handle('edm::ValueMap<float> ')
handle_genPart= Handle('std::vector<reco::GenParticle>')  
handle_genJet = Handle('std::vector<reco::GenJet>')      
handle_PF     = Handle('std::vector<pat::PackedCandidate>')

handles = {}
if "22EE" in files[0]:
  handles['jets']     = [('slimmedJets', '', "RECO") , handle_jet, False]
  handles['genPart']  = [('prunedGenParticles', '', "RECO"), handle_genPart, False]
  handles["PF"]       = [('packedPFCandidates', '', "RECO"), handle_PF, False]
  handles["genJet"]   = [("slimmedGenJets", "", "RECO"), handle_genJet, False]        

else:
  handles['jets']     = [('slimmedJets', '', "PAT") , handle_jet, False]
  handles['genPart']  = [('prunedGenParticles', '', "PAT"), handle_genPart, False]
  handles["PF"]       = [('packedPFCandidates', '', "PAT"), handle_PF, False]
  handles["genJet"]   = [("slimmedGenJets", "", "PAT"), handle_genJet, False]        

handles['scores']   = [('disTauTag', 'score1', 'TAUTAGGER'), handle_score, False]

#can = ROOT.TCanvas()
h_charged_pt = ROOT.TH1F("h_charged_pt", "Charged PF Cands with pt > 1;hasTrackDetails;Number of PF Cands", 2, 0, 2)

for i, ev in enumerate(events):


  if maxevents>0 and i>maxevents:
    break
  if i%1000==0:
    print(('===> processing %d / %d event' %(i, totevents)))  
    
  for k, v in list(handles.items()):
    setattr(ev, k, None)
    v[2] = False
    try:
        ev.getByLabel(v[0], v[1])
        setattr(ev, k, v[1].product())
        v[2] = True
    except:    
        v[2] = False

  for pf, pfcand in enumerate(ev.PF):
    #if pfcand.pt() <  0.5 and pfcand.charge() != 0:
      #print("For event ", i, " and PF Cand number ", pf, " its pt is ", pfcand.pt(), " charge ", pfcand.charge(), " hasTrackDets ", pfcand.hasTrackDetails(), " eta is ", pfcand.eta(), "phi is ", pfcand.phi())
      #h_charged_pt.Fill(pfcand.hasTrackDetails())
    if not pfcand.hasTrackDetails() and pfcand.charge() != 0:
        print("For event ", i, " and PF Cand number ", pf, " its pt is ", pfcand.pt(), " charge ", pfcand.charge(), " hasTrackDets ", pfcand.hasTrackDetails(), " eta is ", pfcand.eta(), "phi is ", pfcand.phi())
#h_charged_pt.Draw("HISTE")
#can.SaveAs("charged_PFCand_pt1.pdf")
