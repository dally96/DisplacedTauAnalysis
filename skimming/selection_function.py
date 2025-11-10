import awkward as ak

SR = {
              "muon_pt_min":                30., ##GeV
              "muon_medium_ID_min":         0, ## > min and < max -> [0,2] to select mediumID, [-1,1] to select not-medium, [-999,999] to select any 
              "muon_medium_ID_max":         2, ## 1 is passing, 0 if not 
              "muon_tight_ID_min":         -999, ## > min and < max -> [0,2] to select tightID, [-1,1] to select not-tight, [-999,999] to select any 
              "muon_tight_ID_max":          999,  ## 
              "muon_dxy_min":               0.1, ##cm
              "muon_dxy_max":               10.,  ##cm
              "muon_iso_min":               -999,
              "muon_iso_max":               0.18,

              "jet_pt_min":                 32.,  ##GeV
              "jet_score_min":              0.9, 
              "jet_score_max":              999, 
              "jet_dxy_min":                0.02, ##cm   ## maybe, not clear if currently it's set or not
              "jet_dxy_max":                999, ##cm

              "MET_pt":                     105., ##GeV
             }

## defined but never used
loose_SR = {j:k for j,k in SR.items()}
loose_SR["muon_dxy_min"] =  50E-4 ##cm
loose_SR["muon_iso_max"] =  0.36
loose_SR["jet_score_min"] =  0.7

## defined but never used
loose_noIso_SR = {j:k for j,k in loose_SR.items()}
loose_noIso_SR["muon_iso_max"] =  999.

## validate w/ daniel
validation_daniel = {j:k for j,k in loose_noIso_SR.items()}
validation_daniel["muon_dxy_max"]  =  50E-4 ##cm
validation_daniel["muon_dxy_min"]  =  0 ##cm
validation_daniel["jet_score_min"] =  0 ##cm
validation_daniel["jet_score_max"] =  0.7 ##cm

## loose cuts to validate even with few events and in DY region
validation_prompt = {j:k for j,k in loose_noIso_SR.items()}
validation_prompt["jet_dxy_min"] =  -999
validation_prompt["muon_dxy_min"] = -999 ##cm
validation_prompt["muon_dxy_max"] = 999 ##cm
validation_prompt["MET_pt"] = 10 

## to be fixed, need Daniel
TT_CR = {j:k for j,k in SR.items()}
TT_CR["jet_score_min"] =  0.
TT_CR["jet_score_max"] =  0.7
TT_CR["jet_dxy_min"] =  -999

## there was a tight_TT_CR where muon had tight ID
## to be fixed, need Daniel
QCD_CR = {j:k for j,k in SR.items()}
## muon ID not clear, ask Daniel
QCD_CR["muon_dxy_min"] = -999 
QCD_CR["muon_dxy_max"] = 999  
QCD_CR["muon_iso_min"] =  0.18 ## invert iso cut
QCD_CR["muon_iso_max"] =  999.
QCD_CR["jet_score_min"] =  0.7
QCD_CR["jet_dxy_min"] =  -999

HPSTauMu = {
              "muon_pt_min":                26, ##GeV
              "muon_eta_max":               2.4, 
              "muon_medium_ID_min":         0., ## > min and < max -> [0,2] to select mediumID, [-1,1] to select not-medium, [-999,999] to select any 
              "muon_medium_ID_max":         2, ## 1 is passing, 0 if not 
              "muon_tight_ID_min":          -999, ## > min and < max -> [0,2] to select tightID, [-1,1] to select not-tight, [-999,999] to select any 
              "muon_tight_ID_max":          999,  ## 
              "muon_dxy_min":               -999, ##cm
              "muon_dxy_max":               0.045,  ##cm
              "muon_dz_min":                -999,
              "muon_dz_max":                0.2,
              "muon_isoid_min":             3, 
              "muon_iso_max":               0.15, 

              "tau_pt_min":                 20.,  ##GeV
              "tau_eta_max":                2.5, 
              "tau_dz_min":                -999,
              "tau_dz_max":                0.2,
              "tau_dm":                     0, ## > 0 -> pass new DM 
              "tau_vs_e_wp":                2, ## VVloose WP
              "tau_vs_jet_wp":              5, ## medium WP
              "tau_vs_mu_wp":               4, ## tight WP

             }

selections_dict = {
  'SR_selections' : SR,
  'loose_SR_selections' : loose_SR,
  'loose_noIso_SR_selections' : loose_noIso_SR,
  'validation_prompt' : validation_prompt,
  'validation_daniel' : validation_daniel,
  'TT_CR' : TT_CR,
  'QCD_CR' : QCD_CR,
  'HPSTauMu' : HPSTauMu,
}

def event_selection(events, selection):

    selections = selections_dict[selection]

    print ('in function selections: \n', selections)
    good_muons  =  ak.flatten(
                     (events.DisMuon.pt > selections["muon_pt_min"])  &\
                     (abs(events.DisMuon.dxy) > selections["muon_dxy_min"]) &\
                     (abs(events.DisMuon.dxy) < selections["muon_dxy_max"]) &\
                     (events.DisMuon.pfRelIso03_all > selections["muon_iso_min"]) &\
                     (events.DisMuon.pfRelIso03_all < selections["muon_iso_max"]) &\
                     (events.DisMuon.mediumId > selections["muon_medium_ID_min"]) &\
                     (events.DisMuon.mediumId < selections["muon_medium_ID_max"]) &\
                     (events.DisMuon.tightId > selections["muon_tight_ID_min"]) &\
                     (events.DisMuon.tightId < selections["muon_tight_ID_max"])
                   ) 
    
    good_jets   = ak.flatten(
                    (events.Jet.pt > selections["jet_pt_min"]) &\
                    (events.Jet.disTauTag_score1 > selections["jet_score_min"]) &\
                    (events.Jet.disTauTag_score1 < selections["jet_score_max"]) &\
                    (abs(events.Jet.dxy) > selections["jet_dxy_min"]) &\
                    (abs(events.Jet.dxy) < selections["jet_dxy_max"])   
                  )
    
    good_MET = (events.PFMET.pt > selections["MET_pt"])
    events = events[good_muons & good_jets & good_MET]

    return events


def event_selection_hpstau_mu(events, selection):

    selections = selections_dict[selection]

    good_muons  =  ak.flatten(
                     (events.Muon.pt > selections["muon_pt_min"])  &\
                     (abs(events.Muon.eta) < selections["muon_eta_max"]) &\
                     (abs(events.Muon.dxy) > selections["muon_dxy_min"]) &\
                     (abs(events.Muon.dxy) < selections["muon_dxy_max"]) &\
                     (abs(events.Muon.dz)  > selections["muon_dz_min"]) & \
                     (abs(events.Muon.dz)  < selections["muon_dz_max"]) & \
                     #(events.Muon.pfIsoId > selections["muon_isoid_min"]) &\
                     (events.Muon.pfRelIso04_all < selections["muon_iso_max"]) &\
                     (events.Muon.mediumId > selections["muon_medium_ID_min"]) &\
                     (events.Muon.mediumId < selections["muon_medium_ID_max"]) &\
                     (events.Muon.tightId > selections["muon_tight_ID_min"]) &\
                     (events.Muon.tightId < selections["muon_tight_ID_max"])
                   ) 
    
    good_taus   = ak.flatten(
                    (events.Tau.pt > selections["tau_pt_min"]) &\
                    (abs(events.Tau.eta) < selections["tau_eta_max"]) &\
                    (abs(events.Tau.dz)  > selections["tau_dz_min"]) & \
                    (abs(events.Tau.dz)  < selections["tau_dz_max"]) & \
                    (events.Tau.idDeepTau2018v2p5VSe >= selections["tau_vs_e_wp"]) &\
                    (events.Tau.idDeepTau2018v2p5VSjet >= selections["tau_vs_jet_wp"]) &\
                    (events.Tau.idDeepTau2018v2p5VSmu >= selections["tau_vs_mu_wp"]) 
                 )   
    
    events = events[good_muons & good_taus ]
    return events

#     if region == "keep_all":
#         events = events
    

def Zpeak_selection(events, selections): 
    good_muons  = (events.DisMuon.pt > selections["muon_pt"])           &\
                   (events.DisMuon.tightId == 1)                                    &\
                   (abs(events.DisMuon.dxy) > selections["muon_dxy_prompt_min"]) &\
                   (abs(events.DisMuon.dxy) < selections["muon_dxy_prompt_max"]) &\
                   (events.DisMuon.pfRelIso03_all < selections["muon_iso_max"])
                  

    good_jets   = (events.Jet.disTauTag_score1 > selections["jet_score"])   &\
                   (events.Jet.pt > selections["jet_pt"])                               &\
                   (abs(events.Jet.dxy) > selections["jet_dxy_displaced_min"])            #&\
                   #(abs(events.Jet.dxy) < selections["muon_dxy_prompt_max"])
                  
    events['DisMuon'] = ak.drop_none(events.DisMuon[good_muons])
    logger.info("Applied mask to DisMuon")
    num_good_muons = ak.count_nonzero(good_muons, axis=1)
    num_muon_mask = (num_good_muons > 0)

    events['Jet'] = ak.drop_none(events.Jet[good_jets])
    logger.info("Applied mask to Jet")
    num_good_jets = ak.count_nonzero(good_jets, axis=1)
    num_jet_mask = (num_good_jets > 0)


    events = events[num_muon_mask & num_jet_mask & good_events]

    return events
