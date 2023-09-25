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
          "ntuples/Staus_M_100_100mm_13p6TeV_Run3Summer22EE_cms-xrd-global.root"
        ] 
events = Events(files)

maxevents = -1 
totevents = events.size()


PFCandDict = {}


outfile_gen = ROOT.TFile(files[0].split("/")[-1].split(".")[0]+'_ntuple.root', 'recreate')
ntree = ROOT.TTree('tree', 'tree')
print((outfile_gen.GetName())) 


handle_jet    = Handle('std::vector<pat::Jet>')
handle_score  = Handle('edm::ValueMap<float> ')
handle_genPart= Handle('std::vector<reco::GenParticle>')  

handles = {}
if "22" in files[0]:
  handles['jets']     = [('slimmedJets', '', "RECO") , handle_jet, False]
  handles['genPart']  = [('prunedGenParticles', '', "RECO"), handle_genPart, False]
if "18" in files[0]:
  handles['jets']     = [('slimmedJets', '', "PAT") , handle_jet, False]
  handles['genPart']  = [('prunedGenParticles', '', "PAT"), handle_genPart, False]

handles['scores']   = [('disTauTag', 'score1', 'TAUTAGGER'), handle_score, False]



PF_pt                          = ROOT.std.vector(ROOT.std.vector(ROOT.std.vector('double')))()
PF_eta                         = ROOT.std.vector(ROOT.std.vector(ROOT.std.vector('double')))()
PF_phi                         = ROOT.std.vector(ROOT.std.vector(ROOT.std.vector('double')))()
PF_mass                        = ROOT.std.vector(ROOT.std.vector(ROOT.std.vector('double')))()
PF_energy                      = ROOT.std.vector(ROOT.std.vector(ROOT.std.vector('double')))()
PF_charge                      = ROOT.std.vector(ROOT.std.vector(ROOT.std.vector('double')))()
PF_puppiWeight                 = ROOT.std.vector(ROOT.std.vector(ROOT.std.vector('double')))()
PF_lostInnerHits               = ROOT.std.vector(ROOT.std.vector(ROOT.std.vector('double')))()
PF_nPixelHits                  = ROOT.std.vector(ROOT.std.vector(ROOT.std.vector('double')))()
PF_nHits                       = ROOT.std.vector(ROOT.std.vector(ROOT.std.vector('double')))()
PF_caloFraction                = ROOT.std.vector(ROOT.std.vector(ROOT.std.vector('double')))()
PF_hcalFraction                = ROOT.std.vector(ROOT.std.vector(ROOT.std.vector('double')))()
PF_hcalEnergy                  = ROOT.std.vector(ROOT.std.vector(ROOT.std.vector('double')))()
PF_ecalEnergy                  = ROOT.std.vector(ROOT.std.vector(ROOT.std.vector('double')))()
PF_hasTrackDetails             = ROOT.std.vector(ROOT.std.vector(ROOT.std.vector('double')))()
PF_dz                          = ROOT.std.vector(ROOT.std.vector(ROOT.std.vector('double')))()
PF_dzError                     = ROOT.std.vector(ROOT.std.vector(ROOT.std.vector('double')))()
PF_dxy                         = ROOT.std.vector(ROOT.std.vector(ROOT.std.vector('double')))()
PF_dxyError                    = ROOT.std.vector(ROOT.std.vector(ROOT.std.vector('double')))()
PF_deta                        = ROOT.std.vector(ROOT.std.vector(ROOT.std.vector('double')))()
PF_dphi                        = ROOT.std.vector(ROOT.std.vector(ROOT.std.vector('double')))()
PF_puppiWeightNoLep            = ROOT.std.vector(ROOT.std.vector(ROOT.std.vector('double')))()
PF_rawCaloFraction             = ROOT.std.vector(ROOT.std.vector(ROOT.std.vector('double')))()
PF_rawHcalFraction             = ROOT.std.vector(ROOT.std.vector(ROOT.std.vector('double')))()
PF_track_nchi2                  = ROOT.std.vector(ROOT.std.vector(ROOT.std.vector('double')))()
PF_track_chi2                  = ROOT.std.vector(ROOT.std.vector(ROOT.std.vector('double')))()
PF_track_ndof                  = ROOT.std.vector(ROOT.std.vector(ROOT.std.vector('double')))()

PF_jet_pt                      = ROOT.std.vector(ROOT.std.vector('double'))() 
PF_jet_eta                     = ROOT.std.vector(ROOT.std.vector('double'))() 
PF_jet_phi                     = ROOT.std.vector(ROOT.std.vector('double'))() 
PF_jet_score                   = ROOT.std.vector(ROOT.std.vector('double'))() 

PF_genPart_pt                  = ROOT.std.vector(ROOT.std.vector('double'))() 
PF_genPart_eta                 = ROOT.std.vector(ROOT.std.vector('double'))() 
PF_genPart_phi                 = ROOT.std.vector(ROOT.std.vector('double'))() 
PF_genPart_pdgId               = ROOT.std.vector(ROOT.std.vector('int'))() 
PF_genPart_genPartIdxMother    = ROOT.std.vector(ROOT.std.vector('int'))()
PF_genPart_vertexX             = ROOT.std.vector(ROOT.std.vector('double'))() 
PF_genPart_vertexY             = ROOT.std.vector(ROOT.std.vector('double'))() 
PF_genPart_vertexZ             = ROOT.std.vector(ROOT.std.vector('double'))() 
PF_genPart_vertexR             = ROOT.std.vector(ROOT.std.vector('double'))() 
PF_genPart_vertexRho           = ROOT.std.vector(ROOT.std.vector('double'))()

PF_genVisTau_genPartIdx        = ROOT.std.vector(ROOT.std.vector('int'))()


pt                             = ROOT.std.vector(ROOT.std.vector('double'))()
eta                            = ROOT.std.vector(ROOT.std.vector('double'))()
phi                            = ROOT.std.vector(ROOT.std.vector('double'))()
mass                           = ROOT.std.vector(ROOT.std.vector('double'))()
energy                         = ROOT.std.vector(ROOT.std.vector('double'))()
charge                         = ROOT.std.vector(ROOT.std.vector('double'))()
puppiWeight                    = ROOT.std.vector(ROOT.std.vector('double'))()
lostInnerHits                  = ROOT.std.vector(ROOT.std.vector('double'))()
nPixelHits                     = ROOT.std.vector(ROOT.std.vector('double'))()
nHits                          = ROOT.std.vector(ROOT.std.vector('double'))()
caloFraction                   = ROOT.std.vector(ROOT.std.vector('double'))()
hcalFraction                   = ROOT.std.vector(ROOT.std.vector('double'))()
hcalEnergy                     = ROOT.std.vector(ROOT.std.vector('double'))()
ecalEnergy                     = ROOT.std.vector(ROOT.std.vector('double'))()
hasTrackDetails                = ROOT.std.vector(ROOT.std.vector('double'))()
dz                             = ROOT.std.vector(ROOT.std.vector('double'))()
dzError                        = ROOT.std.vector(ROOT.std.vector('double'))()
dxy                            = ROOT.std.vector(ROOT.std.vector('double'))()
dxyError                       = ROOT.std.vector(ROOT.std.vector('double'))()
deta                           = ROOT.std.vector(ROOT.std.vector('double'))()
dphi                           = ROOT.std.vector(ROOT.std.vector('double'))()
puppiWeightNoLep               = ROOT.std.vector(ROOT.std.vector('double'))()
rawCaloFraction                = ROOT.std.vector(ROOT.std.vector('double'))()
rawHcalFraction                = ROOT.std.vector(ROOT.std.vector('double'))()
track_nchi2                     = ROOT.std.vector(ROOT.std.vector('double'))()
track_chi2                     = ROOT.std.vector(ROOT.std.vector('double'))()
track_ndof                     = ROOT.std.vector(ROOT.std.vector('double'))()

pt_dau                         = ROOT.std.vector('double')()
eta_dau                        = ROOT.std.vector('double')() 
phi_dau                        = ROOT.std.vector('double')() 
mass_dau                       = ROOT.std.vector('double')() 
energy_dau                     = ROOT.std.vector('double')() 
charge_dau                     = ROOT.std.vector('double')() 
deta_dau                       = ROOT.std.vector('double')() 
dphi_dau                       = ROOT.std.vector('double')() 
puppiWeight_dau                = ROOT.std.vector('double')() 
lostInnerHits_dau              = ROOT.std.vector('double')() 
nPixelHits_dau                 = ROOT.std.vector('double')() 
nHits_dau                      = ROOT.std.vector('double')() 
caloFraction_dau               = ROOT.std.vector('double')() 
hcalFraction_dau               = ROOT.std.vector('double')() 
hcalEnergy_dau                 = ROOT.std.vector('double')() 
ecalEnergy_dau                 = ROOT.std.vector('double')() 
hasTrackDetails_dau            = ROOT.std.vector('double')() 
dz_dau                         = ROOT.std.vector('double')() 
dzError_dau                    = ROOT.std.vector('double')() 
dxy_dau                        = ROOT.std.vector('double')() 
dxyError_dau                   = ROOT.std.vector('double')() 
puppiWeightNoLep_dau           = ROOT.std.vector('double')()
rawCaloFraction_dau            = ROOT.std.vector('double')()
rawHcalFraction_dau            = ROOT.std.vector('double')()
track_nchi2_dau                 = ROOT.std.vector('double')()
track_chi2_dau                 = ROOT.std.vector('double')()
track_ndof_dau                 = ROOT.std.vector('double')()

jet_pt                         = ROOT.std.vector('double')() 
jet_eta                        = ROOT.std.vector('double')() 
jet_phi                        = ROOT.std.vector('double')() 
jet_score                      = ROOT.std.vector('double')() 

genPart_pt                     = ROOT.std.vector('double')()
genPart_eta                    = ROOT.std.vector('double')()
genPart_phi                    = ROOT.std.vector('double')()
genPart_pdgId                  = ROOT.std.vector('int')()
genPart_genPartIdxMother       = ROOT.std.vector('int')()
genPart_vertexX                = ROOT.std.vector('double')() 
genPart_vertexY                = ROOT.std.vector('double')() 
genPart_vertexZ                = ROOT.std.vector('double')() 
genPart_vertexR                = ROOT.std.vector('double')() 
genPart_vertexRho              = ROOT.std.vector('double')() 

genVisTau_genPartIdx           = ROOT.std.vector('int')()



ntree.Branch("PF_pt", pt                         )
ntree.Branch("PF_eta", eta                         )
ntree.Branch("PF_phi", phi                        )
ntree.Branch("PF_mass", mass                        )
ntree.Branch("PF_energy", energy                        )
ntree.Branch("PF_charge", charge                      )
ntree.Branch("PF_puppiWeight", puppiWeight                 )
ntree.Branch("PF_lostInnerHits", lostInnerHits               )
ntree.Branch("PF_nPixelHits", nPixelHits                  )
ntree.Branch("PF_nHits", nHits                       )
ntree.Branch("PF_caloFraction", caloFraction                )
ntree.Branch("PF_hcalFraction", hcalFraction                )
ntree.Branch("PF_hcalEnergy", hcalEnergy                )
ntree.Branch("PF_ecalEnergy", ecalEnergy                )
ntree.Branch("PF_hasTrackDetails", hasTrackDetails             )
ntree.Branch("PF_dz", dz                          )
ntree.Branch("PF_dzError", dzError                     )
ntree.Branch("PF_dxy", dxy                         )
ntree.Branch("PF_dxyError", dxyError                    )
ntree.Branch("PF_deta", deta                        )
ntree.Branch("PF_dphi", dphi                        )
ntree.Branch("PF_puppiWeightNoLep", puppiWeightNoLep )
ntree.Branch("PF_rawCaloFraction",  rawCaloFraction  )  
ntree.Branch("PF_rawHcalFraction",  rawHcalFraction  )  
ntree.Branch("PF_track_nchi2",       track_nchi2       )  
ntree.Branch("PF_track_chi2",       track_chi2       )  
ntree.Branch("PF_track_ndof",       track_ndof       )  

ntree.Branch("jet_pt", jet_pt                      )
ntree.Branch("jet_eta", jet_eta                     )
ntree.Branch("jet_phi", jet_phi                     )
ntree.Branch("jet_score", jet_score                   )

ntree.Branch("genPart_pt", genPart_pt               )              
ntree.Branch("genPart_eta", genPart_eta             ) 
ntree.Branch("genPart_phi", genPart_phi             )
ntree.Branch("genPart_pdgId", genPart_pdgId         )  
ntree.Branch("genPart_genPartIdxMother", genPart_genPartIdxMother)
ntree.Branch("genVisTau_genPartIdx", genVisTau_genPartIdx)
ntree.Branch("genPart_vertexX", genPart_vertexX                  )
ntree.Branch("genPart_vertexY", genPart_vertexY                  )
ntree.Branch("genPart_vertexZ", genPart_vertexZ                  )
ntree.Branch("genPart_vertexR", genPart_vertexR                  )
ntree.Branch("genPart_vertexRho", genPart_vertexRho              )


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

  pt                         .clear()
  eta                        .clear()
  phi                        .clear()
  mass                       .clear()
  energy                     .clear()
  charge                     .clear()
  deta                       .clear() 
  dphi                       .clear() 
  puppiWeight                .clear() 
  lostInnerHits              .clear() 
  nPixelHits                 .clear() 
  nHits                      .clear() 
  caloFraction               .clear() 
  hcalFraction               .clear() 
  hcalEnergy                 .clear() 
  ecalEnergy                 .clear() 
  hasTrackDetails            .clear() 
  dz                         .clear() 
  dzError                    .clear() 
  dxy                        .clear() 
  dxyError                   .clear() 
  puppiWeightNoLep           .clear() 
  rawCaloFraction            .clear() 
  rawHcalFraction            .clear() 
  track_nchi2                 .clear() 
  track_chi2                 .clear() 
  track_ndof                 .clear() 

  genPart_pt                 .clear()
  genPart_phi                .clear()
  genPart_eta                .clear()
  genPart_pdgId              .clear()
  genPart_genPartIdxMother   .clear()
  genPart_vertexX            .clear() 
  genPart_vertexY            .clear()
  genPart_vertexZ            .clear()
  genPart_vertexR            .clear()
  genPart_vertexRho          .clear()

  genVisTau_genPartIdx       .clear()

  jet_pt                     .clear() 
  jet_eta                    .clear() 
  jet_phi                    .clear() 
  jet_score                  .clear() 
  
  for j, jet in enumerate(ev.jets):
    daughterPt = []
    nDaughters = jet.numberOfDaughters()
    
    for d in range(nDaughters):
      daughterPt.append(jet.daughter(d).polarP4().pt())
      #print(daughterPt)
    
    sortedIndex = sorted(range(len(daughterPt)), key=lambda k: daughterPt[k], reverse = True)
    daughterPt.sort(reverse=True)
      #print(sortedIndex)
      #print(daughterPt)
    
    if len(sortedIndex) > 50: 
      sortedIndex = sortedIndex[0:50]
      daughterPt = daughterPt[0:50]
    
    pt_dau                   .clear()
    eta_dau                  .clear() 
    phi_dau                  .clear() 
    mass_dau                 .clear() 
    energy_dau               .clear() 
    charge_dau               .clear() 
    puppiWeight_dau          .clear() 
    lostInnerHits_dau        .clear() 
    nPixelHits_dau           .clear() 
    nHits_dau                .clear() 
    caloFraction_dau         .clear() 
    hcalFraction_dau         .clear() 
    hcalEnergy_dau           .clear() 
    ecalEnergy_dau           .clear() 
    hasTrackDetails_dau      .clear() 
    dz_dau                   .clear() 
    dzError_dau              .clear() 
    dxy_dau                  .clear() 
    dxyError_dau             .clear() 
    deta_dau                 .clear() 
    dphi_dau                 .clear() 
    puppiWeightNoLep_dau     .clear() 
    rawCaloFraction_dau      .clear() 
    rawHcalFraction_dau      .clear() 
    track_nchi2_dau           .clear() 
    track_chi2_dau           .clear() 
    track_ndof_dau           .clear() 
      
    for ind, d in enumerate(sortedIndex):
      pt_dau                 .push_back(jet.daughter(d).polarP4().pt()                        ) 
      eta_dau                .push_back(jet.daughter(d).polarP4().eta()                      ) 
      phi_dau                .push_back(jet.daughter(d).polarP4().phi()                      ) 
      mass_dau               .push_back(jet.daughter(d).polarP4().mass()                     ) 
      energy_dau             .push_back(jet.daughter(d).polarP4().energy()                   ) 
      charge_dau             .push_back(jet.daughter(d).charge()                             ) 
      puppiWeight_dau        .push_back(jet.daughter(d).puppiWeight()                        ) 
      lostInnerHits_dau      .push_back(jet.daughter(d).lostInnerHits()                      ) 
      nPixelHits_dau         .push_back(jet.daughter(d).numberOfPixelHits()                  ) 
      nHits_dau              .push_back(jet.daughter(d).numberOfHits()                       ) 
      caloFraction_dau       .push_back(jet.daughter(d).caloFraction()                       ) 
      hcalFraction_dau       .push_back(jet.daughter(d).hcalFraction()                       ) 
      hcalEnergy_dau         .push_back(jet.daughter(d).hcalEnergy()                         ) 
      ecalEnergy_dau         .push_back(jet.daughter(d).ecalEnergy()                         ) 
      hasTrackDetails_dau    .push_back(jet.daughter(d).hasTrackDetails()                    ) 
      deta_dau               .push_back(jet.polarP4().phi() - jet.daughter(d).polarP4().phi()) 
      dphi_dau               .push_back(jet.polarP4().eta() - jet.daughter(d).polarP4().eta()) 
      puppiWeightNoLep_dau   .push_back(jet.daughter(d).puppiWeightNoLep()                   )
      rawCaloFraction_dau    .push_back(jet.daughter(d).rawCaloFraction()                    )
      rawHcalFraction_dau    .push_back(jet.daughter(d).rawHcalFraction()                    )

      if jet.daughter(d).hasTrackDetails():
        if math.isfinite(jet.daughter(d).dz()):       dz_dau       .push_back(jet.daughter(d).dz()      ) 
        else:                                         dz_dau       .push_back(-9999)
        if math.isfinite(jet.daughter(d).dzError()):  dzError_dau  .push_back(jet.daughter(d).dzError() ) 
        else:                                         dzError_dau  .push_back(-9999)
        if math.isfinite(jet.daughter(d).dxyError()): dxyError_dau .push_back(jet.daughter(d).dxyError()) 
        else:                                         dxyError_dau .push_back(-9999)
        dxy_dau                .push_back(jet.daughter(d).dxy()             ) 
        track_nchi2_dau         .push_back(jet.daughter(d).pseudoTrack().normalizedChi2())
        track_chi2_dau         .push_back(jet.daughter(d).pseudoTrack().chi2())
        track_ndof_dau         .push_back(jet.daughter(d).pseudoTrack().ndof())
      else:
        dxy_dau              .push_back(9999)
        dz_dau               .push_back(9999)
        dzError_dau          .push_back(9999)
        dxyError_dau         .push_back(9999) 
        track_nchi2_dau       .push_back(9999) 
        track_chi2_dau       .push_back(9999) 
        track_ndof_dau       .push_back(9999) 

    pt                         .push_back( pt_dau              ) 
    eta                        .push_back( eta_dau             )  
    phi                        .push_back( phi_dau             )  
    mass                       .push_back( mass_dau            )  
    energy                     .push_back( energy_dau          )  
    charge                     .push_back( charge_dau          )  
    puppiWeight                .push_back( puppiWeight_dau     )  
    lostInnerHits              .push_back( lostInnerHits_dau   )  
    nPixelHits                 .push_back( nPixelHits_dau      )  
    nHits                      .push_back( nHits_dau           )  
    caloFraction               .push_back( caloFraction_dau    )  
    hcalFraction               .push_back( hcalFraction_dau    )  
    hcalEnergy                 .push_back( hcalEnergy_dau      )  
    ecalEnergy                 .push_back( ecalEnergy_dau      )  
    hasTrackDetails            .push_back( hasTrackDetails_dau )
    dz                         .push_back( dz_dau              ) 
    dzError                    .push_back( dzError_dau         )      
    dxy                        .push_back( dxy_dau             )   
    dxyError                   .push_back( dxyError_dau        ) 
    deta                       .push_back( deta_dau            )
    dphi                       .push_back( dphi_dau            )
    puppiWeightNoLep           .push_back( puppiWeightNoLep_dau)           
    rawCaloFraction            .push_back( rawCaloFraction_dau )           
    rawHcalFraction            .push_back( rawHcalFraction_dau )           
    track_nchi2                 .push_back( track_nchi2_dau      )           
    track_chi2                 .push_back( track_chi2_dau      )           
    track_ndof                 .push_back( track_ndof_dau      )           

    jet_pt                     .push_back( jet.polarP4().pt()  )
    jet_eta                    .push_back( jet.polarP4().eta() )
    jet_phi                    .push_back( jet.polarP4().phi() )
    jet_score                  .push_back( ev.scores.get(j)    ) 


  PF_pt                        .push_back( pt                  )                        
  PF_eta                       .push_back( eta                 ) 
  PF_phi                       .push_back( phi                 ) 
  PF_mass                      .push_back( mass                ) 
  PF_energy                    .push_back( energy              ) 
  PF_charge                    .push_back( charge              ) 
  PF_deta                      .push_back( deta                ) 
  PF_dphi                      .push_back( dphi                ) 
  PF_puppiWeight               .push_back( puppiWeight         )        
  PF_lostInnerHits             .push_back( lostInnerHits       )        
  PF_nPixelHits                .push_back( nPixelHits          )        
  PF_nHits                     .push_back( nHits               )        
  PF_caloFraction              .push_back( caloFraction        )        
  PF_hcalFraction              .push_back( hcalFraction        )        
  PF_hcalEnergy                .push_back( hcalEnergy          )        
  PF_ecalEnergy                .push_back( ecalEnergy          )        
  PF_hasTrackDetails           .push_back( hasTrackDetails     )        
  PF_dz                        .push_back( dz                  )      
  PF_dzError                   .push_back( dzError             )      
  PF_dxy                       .push_back( dxy                 )      
  PF_dxyError                  .push_back( dxyError            )      
  PF_puppiWeightNoLep          .push_back( puppiWeightNoLep    )         
  PF_rawCaloFraction           .push_back( rawCaloFraction     )       
  PF_rawHcalFraction           .push_back( rawHcalFraction     )       
  PF_track_nchi2                .push_back( track_nchi2          )       
  PF_track_chi2                .push_back( track_chi2          )       
  PF_track_ndof                .push_back( track_ndof          )       

  PF_jet_pt                    .push_back( jet_pt              )          
  PF_jet_eta                   .push_back( jet_eta             )  
  PF_jet_phi                   .push_back( jet_phi             ) 
  PF_jet_score                 .push_back( jet_score           ) 


  for pp, part in enumerate(ev.genPart):

    genPart_pt                 .push_back(part.pt())
    genPart_phi                .push_back(part.phi())
    genPart_eta                .push_back(part.eta())
    genPart_pdgId              .push_back(part.pdgId())
    genPart_vertexX            .push_back(part.vertex().X()         )  
    genPart_vertexY            .push_back(part.vertex().Y()         )  
    genPart_vertexZ            .push_back(part.vertex().Z()         )  
    genPart_vertexR            .push_back(part.vertex().R()         )  
    genPart_vertexRho          .push_back(part.vertex().Rho()       )
    if part.numberOfMothers() > 0:
      genPart_genPartIdxMother .push_back(part.motherRef(0).key())
      
      if abs(part.pdgId()) != 11 and abs(part.pdgId()) != 12 and abs(part.pdgId()) != 13 and abs(part.pdgId()) != 14 and abs(part.pdgId()) != 15 and abs(part.pdgId()) != 16 and abs(part.pdgId()) != 22:
        if abs(part.motherRef(0).pdgId()) == 15:
          if part.motherRef(0).key() in genVisTau_genPartIdx: continue
          genVisTau_genPartIdx.push_back(part.motherRef(0).key())
    else:
      genPart_genPartIdxMother .push_back(-1)
    
    
    
  PF_genPart_pt                .push_back( genPart_pt               )  
  PF_genPart_phi               .push_back( genPart_phi              )  
  PF_genPart_eta               .push_back( genPart_eta              )  
  PF_genPart_pdgId             .push_back( genPart_pdgId            )  
  PF_genPart_genPartIdxMother  .push_back( genPart_genPartIdxMother )  
  PF_genPart_vertexX           .push_back( genPart_vertexX             ) 
  PF_genPart_vertexY           .push_back( genPart_vertexY             ) 
  PF_genPart_vertexZ           .push_back( genPart_vertexZ             ) 
  PF_genPart_vertexR           .push_back( genPart_vertexR             ) 
  PF_genPart_vertexRho         .push_back( genPart_vertexRho           ) 

  PF_genVisTau_genPartIdx      .push_back( genVisTau_genPartIdx     )
  ntree.Fill()
                  
ntree.Write()        
outfile_gen.Write()
outfile_gen.Close()
        
          


