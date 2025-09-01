import awkward as ak

SR_selections = {
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
loose_SR_selections = {j:k for j,k in SR_selections.items()}
loose_SR_selections["muon_dxy_min"] =  50E-4 ##cm
loose_SR_selections["muon_iso_max"] =  0.36
loose_SR_selections["jet_score_min"] =  0.7

## defined but never used
loose_noIso_SR_selections = {j:k for j,k in loose_SR_selections.items()}
loose_noIso_SR_selections["muon_iso_max"] =  999.

validation_prompt = {j:k for j,k in loose_noIso_SR_selections.items()}
validation_prompt["jet_dxy_min"] =  -999
validation_prompt["muon_dxy_min"] = -999 ##cm
validation_prompt["muon_dxy_max"] = 999 ##cm
validation_prompt["MET_pt"] = 10 

## to be fixed, need Daniel
TT_CR = {j:k for j,k in SR_selections.items()}
TT_CR["jet_score_min"] =  0.
TT_CR["jet_score_max"] =  0.7
TT_CR["jet_dxy_min"] =  -999

## there was a tight_TT_CR where muon had tight ID
## to be fixed, need Daniel
QCD_CR = {j:k for j,k in SR_selections.items()}
## muon ID not clear, ask Daniel
QCD_CR["muon_dxy_min"] = -999 
QCD_CR["muon_dxy_max"] = 999  
QCD_CR["muon_iso_min"] =  0.18 ## invert iso cut
QCD_CR["muon_iso_max"] =  999.

QCD_CR["jet_score_min"] =  0.7
QCD_CR["jet_dxy_min"] =  -999


selections_dict = {
  'SR_selections' : SR_selections,
  'loose_SR_selections' : loose_SR_selections,
  'loose_noIso_SR_selections' : loose_noIso_SR_selections,
  'validation_prompt' : validation_prompt,
  'TT_CR' : TT_CR,
  'QCD_CR' : QCD_CR,
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
