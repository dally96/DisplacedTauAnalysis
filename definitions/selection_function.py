import dask_awkward as dak


SR_selections = {
              "muon_pt":                    30., ##GeV
              "muon_ID":                    "DisMuon_mediumId",
              "muon_dxy_prompt_max":        50E-4, ##cm
              "muon_dxy_prompt_min":        0E-4, ##cm
              "muon_dxy_displaced_min":     0.1, ##cm
              "muon_dxy_displaced_max":     10.,  ##cm
              "muon_iso_max":               0.18,

              "jet_score":                  0.9, 
              "jet_pt":                     32.,  ##GeV
              "jet_dxy_displaced_min":      0.02, ##cm

              "MET_pt":                     105., ##GeV
             }

loose_SR_selections = {
              "muon_pt":                    30., ##GeV
              "muon_ID":                    "DisMuon_mediumId",
              "muon_dxy_prompt_max":        50E-4, ##cm
              "muon_dxy_prompt_min":        0E-4, ##cm
              "muon_dxy_displaced_min":     50E-4, ##cm
              "muon_dxy_displaced_max":     10.,  ##cm
              "muon_iso_max":               0.36,

              "jet_score":                  0.7, 
              "jet_pt":                     32.,  ##GeV
              "jet_dxy_displaced_min":      0.02, ##cm

              "MET_pt":                     105., ##GeV
             }

loose_noIso_SR_selections = {
              "muon_pt":                    30., ##GeV
              "muon_ID":                    "DisMuon_mediumId",
              "muon_dxy_prompt_max":        50E-4, ##cm
              "muon_dxy_prompt_min":        0E-4, ##cm
              "muon_dxy_displaced_min":     50E-4, ##cm
              "muon_dxy_displaced_max":     10.,  ##cm
              "muon_iso_max":               999,

              "jet_score":                  0.7, 
              "jet_pt":                     32.,  ##GeV
              "jet_dxy_displaced_min":      0.02, ##cm

              "MET_pt":                     105., ##GeV
             }

def event_selection(events, selections, region):

    if region == "SR":
        good_muons  = dak.flatten((events.DisMuon.pt > selections["muon_pt"])           &\
                       (events.DisMuon.mediumId == 1)                                     &\
                       (abs(events.DisMuon.dxy) > selections["muon_dxy_displaced_min"]) &\
                       (abs(events.DisMuon.dxy) < selections["muon_dxy_displaced_max"]) &\
                       (events.DisMuon.pfRelIso03_all < selections["muon_iso_max"])
                      )
        
        good_jets   = dak.flatten((events.Jet.disTauTag_score1 > selections["jet_score"])   &\
                       (events.Jet.pt > selections["jet_pt"])                               &\
                       (abs(events.Jet.dxy) > selections["jet_dxy_displaced_min"])          #&\
                       #(abs(events.Jet.dxy) < selections["muon_dxy_prompt_max"])
                      )
        
        good_events = (events.PFMET.pt > selections["MET_pt"])

        events = events[good_muons & good_jets & good_events]
    ###############################################################
    if region == "TT_CR":
        good_muons  = dak.flatten((events.DisMuon.pt > selections["muon_pt"])           &\
                       (events.DisMuon.mediumId == 1)                                    &\
                       (abs(events.DisMuon.dxy) > selections["muon_dxy_prompt_min"]) &\
                       (abs(events.DisMuon.dxy) < selections["muon_dxy_prompt_max"]) &\
                       (events.DisMuon.pfRelIso03_all < selections["muon_iso_max"])
                      )
        
        good_jets   = dak.flatten((events.Jet.disTauTag_score1 < selections["jet_score"])   &\
                       (events.Jet.pt > selections["jet_pt"])                               &\
                       (abs(events.Jet.dxy) > selections["jet_dxy_displaced_min"])          #&\
                       #(abs(events.Jet.dxy) < selections["muon_dxy_prompt_max"])
                      )
        
        good_events = (events.PFMET.pt > selections["MET_pt"])
        

            
        events = events[good_muons & good_jets & good_events]
    ###############################################################
    if region == "tight_TT_CR":
        good_muons  = dak.flatten((events.DisMuon.pt > selections["muon_pt"])           &\
                       (events.DisMuon.tightId == 1)                                    &\
                       (abs(events.DisMuon.dxy) > selections["muon_dxy_displaced_min"]) &\
                       (abs(events.DisMuon.dxy) < selections["muon_dxy_displaced_max"]) &\
                       (events.DisMuon.pfRelIso03_all < selections["muon_iso_max"])
                      )
        
        good_jets   = dak.flatten((events.Jet.disTauTag_score1 < selections["jet_score"])   &\
                       (events.Jet.pt > selections["jet_pt"])                               &\
                       (abs(events.Jet.dxy) > selections["jet_dxy_displaced_min"])          #&\
                       #(abs(events.Jet.dxy) < selections["muon_dxy_prompt_max"])
                      )
        
        good_events = (events.PFMET.pt > selections["MET_pt"])
        

            
        events = events[good_muons & good_jets & good_events]
    ###############################################################
    if region == "QCD_CR":
        good_muons  = dak.flatten((events.DisMuon.pt > selections["muon_pt"])           &\
                       (events[selections["muon_ID"]] == 1)                                    &\
                       (abs(events.DisMuon.dxy) > selections["muon_dxy_displaced_min"]) &\
                       (abs(events.DisMuon.dxy) < selections["muon_dxy_displaced_max"]) &\
                       (events.DisMuon.pfRelIso03_all > selections["muon_iso_max"])
                      )
        
        good_jets   = dak.flatten((events.Jet.disTauTag_score1 > selections["jet_score"])   &\
                       (events.Jet.pt > selections["jet_pt"])                               &\
                       (abs(events.Jet.dxy) > selections["jet_dxy_displaced_min"])          #&\
                       #(abs(events.Jet.dxy) < selections["muon_dxy_prompt_max"])
                      )
        
        good_events = (events.PFMET.pt > selections["MET_pt"])
        

            
        events = events[good_muons & good_jets & good_events]
    ###############################################################
    if region == "W_CR":
        good_muons  = dak.flatten((events.DisMuon.pt > selections["muon_pt"])           &\
                       (events.DisMuon.mediumId == 1)                                    &\
                       (abs(events.DisMuon.dxy) > selections["muon_dxy_prompt_min"]) &\
                       (abs(events.DisMuon.dxy) < selections["muon_dxy_prompt_max"]) &\
                       (events.DisMuon.pfRelIso03_all < selections["muon_iso_max"])
                      )
        
        good_jets   = dak.flatten((events.Jet.disTauTag_score1 > selections["jet_score"])   &\
                       (events.Jet.pt > selections["jet_pt"])                               &\
                       (abs(events.Jet.dxy) > selections["muon_dxy_prompt_min"])            &\
                       (abs(events.Jet.dxy) < selections["muon_dxy_prompt_max"])
                      )
        
        good_events = (events.PFMET.pt > selections["MET_pt"])
        

            
        events = events[good_muons & good_jets & good_events]
    ###############################################################
    
    return events

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
