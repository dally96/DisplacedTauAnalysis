import awkward as ak
import uproot
import os, time
from datetime import datetime

from coffea import processor
from coffea.nanoevents.methods import candidate
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema

import fsspec_xrootd
from  fsspec_xrootd import XRootDFileSystem

from dask.distributed import Client, wait, progress, LocalCluster

PFNanoAODSchema.warn_missing_crossrefs = False
PFNanoAODSchema.mixins["DisMuon"] = "Muon"

exclude_prefixes = ['Flag', 'JetSVs', 'GenJetAK8_', 'SubJet', 
                    'Photon', 'TrigObj', 'TrkMET', 'HLT',
                    'Puppi', 'OtherPV', 'GenJetCands',
                    'FsrPhoton', ''
                    ## tmp
                    'diele', 'LHE', 'dimuon', 'Rho', 'JetPFCands', 'GenJet', 'GenCands', 
                    'Electron'
                    ]
                    
include_prefixes = ['DisMuon', 'Jet', 'Tau', 'PFMET', 'SV', 'PV', 'GenPart', 'GenVisTau', 'GenVtx', 'Pileup',
                    'event', 'run', 'luminosityBlock', 'nDisMuon', 'nTau', 'nJet', 'nGenPart', 'nGenVisTau'
                   ]                    

good_hlts = [
  "HLT_PFMET120_PFMHT120_IDTight",                  
  "HLT_PFMET130_PFMHT130_IDTight",
  "HLT_PFMET140_PFMHT140_IDTight",
  "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight",
  "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight",
  "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight",
  "HLT_PFMET120_PFMHT120_IDTight_PFHT60",
  "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF",
  "HLT_PFMETTypeOne140_PFMHT140_IDTight",
  "HLT_MET105_IsoTrk50",
  "HLT_MET120_IsoTrk50",
  "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1",
  "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1",
  "HLT_Ele30_WPTight_Gsf",                                         
  "HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1",                 
  "HLT_DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1",     
  "HLT_DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1"
]

def is_included(name):
        return any(name.startswith(prefix) for prefix in include_prefixes)

def is_good_hlt(name):
        return (name in good_hlts)
   


## for old coffea
def is_rootcompat_old_coffea(a):
    """Check if the data is a flat or 1-d jagged array for compatibility with uproot."""
    t = ak.type(a)
    if isinstance(t, ak.types.NumpyType):
        return True
    if isinstance(t, ak.types.ListType) and isinstance(t.content, ak.types.NumpyType):
        return True
    return False

### for old coffea
def uproot_writeable_old_coffea(events):
    """Restrict to columns that uproot can write compactly."""
    out_event = events[
        [x for x in events.fields if (is_included(x)) or (is_good_hlt(x))]
    ]    
    for bname in events.fields:
        if events[bname].fields and ((is_included(bname) or is_good_hlt(bname))):
            print (bname)
            out_event[bname] = ak.zip(
                {
                    n: ak.without_parameters(events[bname][n])
                    for n in events[bname].fields
                    if is_rootcompat(events[bname][n])
                }
            )
    return out_event


def is_rootcompat(a):
    """Is it a flat or 1-d jagged array?"""
    t = ak.type(a)
    if isinstance(t, ak.types.ArrayType):
        if isinstance(t.content, ak.types.NumpyType):
            return True
        if isinstance(t.content, ak.types.ListType) and isinstance(t.content.content, ak.types.NumpyType):
            return True
    return False

## for new coffea and uproot.recreate    
## https://github.com/scikit-hep/coffea/discussions/735
# def uproot_writeable(events):
#     """Restrict to columns that uproot can write compactly"""
#     out = {}
# #     out = events[
# #         [x for x in events.fields if (is_included(x)) or (is_good_hlt(x))]
# #     ]    
#     for bname in events.fields:
#         if events[bname].fields and ((is_included(bname) or is_good_hlt(bname))):
#             out[bname] = ak.zip(
#                 {
#                     n: ak.to_packed(ak.without_parameters(events[bname][n])) 
#                     for n in events[bname].fields 
#                     if is_rootcompat(events[bname][n])
#                 }, depth_limit=1
#             )
#     return out    

def uproot_writeable(events):
    """Restrict to columns uproot can write compactly"""
    out = {}

    for bname in events.fields:
        if events[bname].fields and (is_included(bname) or is_good_hlt(bname)):
            # Flatten the collection: Tau -> Tau_pt, Tau_eta, ...
            for n in events[bname].fields:
                if is_rootcompat(events[bname][n]):
                    branch_name = f"{bname}_{n}"
                    out[branch_name] = ak.to_packed(
                        ak.without_parameters(events[bname][n])
                    )

        # handle simple top-level fields too
        elif is_included(bname) or is_good_hlt(bname):
            if is_rootcompat(events[bname]):
                out[bname] = ak.to_packed(ak.without_parameters(events[bname]))

    return out


class MyProcessor(processor.ProcessorABC):
    def __init__(self):
        pass
#         self._accumulator = {} 
#         for samp in fileset:
#             self._accumulator[samp] = dak.from_awkward(ak.Array([]), npartitions = 1)

    def process(self, events):
        
        if events is None: 
            return output

        # Determine if dataset is MC or Data
        is_MC = True if hasattr(events, "GenPart") else False
        if not is_MC:
            try:
                lumimask = select_lumis('2022', events)
                events = events[lumimask]
            except:
                print (f"[ lumimask ] Skip now! Unable to find year info of {dataset_name}")


        # Define the "good muon" condition for each muon per event
        good_muon_mask = (
            #((events.DisMuon.isGlobal == 1) | (events.DisMuon.isTracker == 1)) & # Equivalent to loose ID cut without isPFcand requirement. Reduce background from non-prompt muons
             (events.DisMuon.pt > 20)
            & (abs(events.DisMuon.eta) < 2.4) # Acceptance of the CMS muon system
        )
        events['DisMuon'] = events.DisMuon[good_muon_mask]
        num_evts = ak.num(events, axis=0)
        num_good_muons = ak.count_nonzero(good_muon_mask, axis=1)
        events = events[num_good_muons >= 1]

        good_jet_mask = (
            (events.Jet.pt > 20)
            & (abs(events.Jet.eta) < 2.4)
            & ~(ak.all(events.Jet.constituents.pf.charge == 0, axis = -1)) 
        )
        events['Jet'] = events.Jet[good_jet_mask]
        num_good_jets = ak.count_nonzero(good_jet_mask, axis=1)
        events = events[num_good_jets >= 1]

        #Noise filter
        noise_mask = (
                     (events.Flag.goodVertices == 1) 
                     & (events.Flag.globalSuperTightHalo2016Filter == 1)
                     & (events.Flag.EcalDeadCellTriggerPrimitiveFilter == 1)
                     & (events.Flag.BadPFMuonFilter == 1)
                     & (events.Flag.BadPFMuonDzFilter == 1)
                     & (events.Flag.hfNoisyHitsFilter == 1)
                     & (events.Flag.eeBadScFilter == 1)
                     & (events.Flag.ecalBadCalibFilter == 1)
                         )

        events = events[noise_mask] 

        n_jets = ak.num(events.Jet.pt)
        charged_sel = events.Jet.constituents.pf.charge != 0
        dxy = ak.where(ak.all(events.Jet.constituents.pf.charge == 0, axis = -1), -999, ak.flatten(events.Jet.constituents.pf[ak.argmax(events.Jet.constituents.pf[charged_sel].pt, axis=2, keepdims=True)].d0, axis = -1))
        events["Jet"] = ak.with_field(events.Jet, dxy, where = "dxy")
        # Write directly to ROOT
        events_to_write = uproot_writeable(events)
        
        # unique name: dataset name + chunk range
        fname = os.path.basename(events.metadata["filename"]).replace(".root", "")
        start = events.metadata["entrystart"]
        stop  = events.metadata["entrystop"]
        outname = f"{fname}_{start}_{stop}.root"

        with uproot.recreate(outname) as fout:
#         with uproot.recreate("out.root") as fout:
            fout["Events"] = events_to_write

        # You can also return a summary/histogram/etc.
        return {"entries_written": len(events_to_write)}


    def postprocess(self, accumulator):
        pass
#         return accumulator
  
  
  
# fileset = {
#     'signal': [
#         'root://cms-xrd-global.cern.ch///store/user/fiorendi/displacedTaus/inputFiles_fortesting/nano_10_0.root',
# #         'root://cms-xrd-global.cern.ch///store/user/fiorendi/displacedTaus/inputFiles_fortesting/nano_7_0.root',
# #         'root://cms-xrd-global.cern.ch///store/user/fiorendi/displacedTaus/inputFiles_fortesting/nano_8_0.root',
# #         'root://cms-xrd-global.cern.ch///store/user/fiorendi/displacedTaus/inputFiles_fortesting/nano_9_0.root',
#     ]
# }
# 

import cloudpickle
from dask.distributed import LocalCluster

if __name__ == "__main__":
## https://github.com/CoffeaTeam/coffea-hats/blob/master/04-processor.ipynb

    print("Time started:", datetime.now().strftime("%H:%M:%S"))
    tic = time.time()

    from fileset import fileset
    futures_run = processor.Runner(
        executor=processor.FuturesExecutor(compression=None, workers = 4),
        schema=PFNanoAODSchema,
        maxchunks=4,
    )
    
    out = futures_run(
        fileset,
        treename="Events",
        processor_instance=MyProcessor(),
    )
    print(out)

    elapsed = time.time() - tic
    print(f"Finished in {elapsed:.1f}s")
