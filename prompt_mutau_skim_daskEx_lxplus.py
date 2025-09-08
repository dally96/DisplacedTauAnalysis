import awkward as ak
import uproot
from coffea import processor
from coffea.nanoevents.methods import candidate
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema
from dask.distributed import Client, wait, progress, LocalCluster
from dask_lxplus import CernCluster
from dask import config as cfg
cfg.set({'distributed.scheduler.worker-ttl': None}) # Check if this solves some dask issues

import fsspec_xrootd
from  fsspec_xrootd import XRootDFileSystem

import os, argparse, importlib, pdb, socket
import time
from datetime import datetime

from itertools import islice
from collections import defaultdict
import json

PFNanoAODSchema.warn_missing_crossrefs = False
PFNanoAODSchema.mixins["DisMuon"] = "Muon"

parser = argparse.ArgumentParser(description="")
parser.add_argument(
	"--sample",
	choices=['QCD','DY', 'signal'],
	required=True,
	help='Specify the sample you want to process')
parser.add_argument(
	"--subsample",
	nargs='*',
	default='all',
	required=False,
	help='Specify the exact sample you want to process')
parser.add_argument(
	"--nfiles",
	default='-1',
	required=False,
	help='Specify the number of input files to process')
parser.add_argument(
	"--usePkl",
	default=True,
	required=False,
	help='Turn it to false to use the non-preprocessed samples')
args = parser.parse_args()


def take(n, iterable):
    """Return the first n items of the iterable as a list."""
    return list(islice(iterable, n))


out_folder = '/eos/cms/store/user/fiorendi/displacedTaus/skim/prompt_mutau/v1/'


all_fileset = {}
if args.usePkl:
    import pickle 
    with open(f"samples/{args.sample}_preprocessed.pkl", "rb") as  f:
        input_dataset = pickle.load(f)

if not args.usePkl:
    samples = {
        "Wto2Q": "samples.fileset_WTo2Q",
        "WtoLNu": "samples.fileset_WToLNu",
        "QCD": "samples.fileset_QCD",
        "DY": "samples.fileset_DY",
        "signal": "samples.fileset_signal",
        "TT": "samples.fileset_TT",
    }
  
    module = importlib.import_module(samples[args.sample])
    input_dataset = module.fileset   

## restrict to specific sub-samples
if args.subsample == 'all':
    fileset = input_dataset
else:  
    fileset = {k: input_dataset[k] for k in args.subsample}


## restrict to n files
nfiles = int(args.nfiles)
if nfiles != -1:
    for k in fileset.keys():
        if nfiles < len(fileset[k]['files']):
            fileset[k]['files'] = take(nfiles, fileset[k]['files'])

print("Will process {} files from the following samples:".format(nfiles), fileset.keys())


## exclude not used
exclude_prefixes = ['Flag', 'JetSVs', 'GenJetAK8_', 'SubJet', 
                    'Photon', 'TrigObj', 'TrkMET', 'HLT',
                    'Puppi', 'OtherPV', 'GenJetCands',
                    'FsrPhoton', ''
                    ## tmp
                    'diele', 'LHE', 'dimuon', 'Rho', 'JetPFCands', 'GenJet', 'GenCands', 
                    'Electron'
                    ]
                    

include_prefixes = ['DisMuon',  'Muon',  'Jet',  'Tau',   'PFMET', 'MET' , 'ChsMET', 'PuppiMET',   'PV', 'GenPart',   'GenVisTau', 'GenVtx',
                    'nDisMuon', 'nMuon', 'nJet', 'nTau', 'nPFMET', 'nMET', 'nChsMET','nPuppiMET', 'nPV', 'nGenPart', 'nGenVisTau', 'nGenVtx',
                    'nVtx', 'event', 'run', 'luminosityBlock', 'Pileup', 'weight', 'genWeight'
                   ]


good_hlts = [
#   "HLT_PFMET120_PFMHT120_IDTight",                  
#   "HLT_PFMET130_PFMHT130_IDTight",
#   "HLT_PFMET140_PFMHT140_IDTight",
#   "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight",
#   "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight",
#   "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight",
#   "HLT_PFMET120_PFMHT120_IDTight_PFHT60",
#   "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF",
#   "HLT_PFMETTypeOne140_PFMHT140_IDTight",
#   "HLT_MET105_IsoTrk50",
#   "HLT_MET120_IsoTrk50",
  "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1",
  "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1",
#   "HLT_Ele30_WPTight_Gsf",                                         
#   "HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1",                 
#   "HLT_DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1",     
#   "HLT_DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1"
]

def is_included(name):
        return any(name.startswith(prefix) for prefix in include_prefixes)

def is_good_hlt(name):
        return (name in good_hlts)
   

def is_rootcompat(a):
    """Is it a flat or 1-d jagged array?"""
    t = ak.type(a)
    if isinstance(t, ak.types.ArrayType):
        if isinstance(t.content, ak.types.NumpyType):
            return True
        if isinstance(t.content, ak.types.ListType) and isinstance(t.content.content, ak.types.NumpyType):
            return True
    return False


def uproot_writeable(events):
    """Restrict to columns that uproot can write compactly"""
    out = {}
    for bname in events.fields:
        if events[bname].fields and (is_included(bname) or is_good_hlt(bname)):
            out[bname] = ak.zip({n: ak.to_packed(ak.without_parameters(events[bname][n])) for n in events[bname].fields if is_rootcompat(events[bname][n])})
        elif is_included(bname) or is_good_hlt(bname):
            out[bname] = ak.to_packed(ak.without_parameters(events[bname]))
    return out



class SkimProcessor(processor.ProcessorABC):
    def __init__(self):
        pass
#         self._accumulator = {} 
#         for samp in fileset:
#             self._accumulator[samp] = dak.from_awkward(ak.Array([]), npartitions = 1)

    def process(self, events):
        
        if events is None: 
            return output
            
        dataset = events.metadata["dataset"]    

        # Determine if dataset is MC or Data
        is_MC = True if hasattr(events, "GenPart") else False
        if not is_MC:
            try:
                lumimask = select_lumis('2022', events)
                events = events[lumimask]
            except:
                print (f"[ lumimask ] Skip now! Unable to find year info of {dataset_name}")

        ## retrieve the list of run/lumi being processed        
        run_lumi_list = set(zip(events.run, events.luminosityBlock))
        run_dict = defaultdict(list)
        for run, lumi in sorted(run_lumi_list):
            run_dict[str(int(run))].append(int(lumi))

        # Define the "good muon" condition for each muon per event
        good_prompt_muon_mask = (
             (events.Muon.pt > 22)
            & (abs(events.Muon.eta) < 2.1) 
            & (events.Muon.tightId)
            & (events.Muon.pfIsoId >= 4)
        )
        num_evts = ak.num(events, axis=0)
        num_good_muons = ak.count_nonzero(good_prompt_muon_mask, axis=1)
        sel_muons = events.Muon[good_prompt_muon_mask]
        events['Muon'] = sel_muons
        events = events[num_good_muons >= 1]

        good_tau_mask = (
            (events.Tau.pt > 28)
            & (abs(events.Tau.eta) < 2.1)
            & (events.Tau.idDecayModeNewDMs)
            & (events.Tau.idDeepTau2018v2p5VSe >= 5)     ## Medium 
            & (events.Tau.idDeepTau2018v2p5VSjet >= 5)   ## Medium
            & (events.Tau.idDeepTau2018v2p5VSmu >= 3)    ## Medium
        )
        num_good_taus = ak.count_nonzero(good_tau_mask, axis=1)
        sel_taus = events.Tau[good_tau_mask]
        events['Tau'] = sel_taus
        events = events[num_good_taus >= 1]
        ## add mT cut to remove W contamination
        ## veto other leptons
        ## QCD from MC will not pass the selection
        ## same for W
        ## could take them from SS
        ## veto on bjets?
          

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
        dxy = ak.fill_none(dxy, -999)
        events["Jet"] = ak.with_field(events.Jet, dxy, where = "dxy")

        # Write directly to ROOT
        events_to_write = uproot_writeable(events)
        
        # unique name: dataset name + chunk range
        fname = os.path.basename(events.metadata["filename"]).replace(".root", "")
        start = events.metadata["entrystart"]
        stop  = events.metadata["entrystop"]
        outname = f"{out_folder}{dataset}/{fname}_{start}_{stop}.root"

        with uproot.recreate(outname) as fout:
            fout["Events"] = events_to_write

        # You can also return a summary/histogram/etc.
        return {
          "entries_written": len(events_to_write),
          "run_dict" : run_dict
        }


    def postprocess(self, accumulator):
        return accumulator
  
  
    
if __name__ == "__main__":
## https://github.com/CoffeaTeam/coffea-hats/blob/master/04-processor.ipynb

    print("Time started:", datetime.now().strftime("%H:%M:%S"))
    tic = time.time()


#     n_port = 8786
#     cluster = CernCluster(
#             cores=1,
#             memory='3000MB',
#             disk='1000MB',
#             death_timeout = '60',
#             container_runtime = "none",
#             log_directory = "/eos/user/f/fiorendi/condor/log/prompt_skim",
#             scheduler_options={
#                 'port': n_port,
#                 'host': socket.gethostname(),
#                 },
#             job_extra={
#                 '+JobFlavour': '"longlunch"',
#                 },
#             job_script_prologue=[
#                 "export XRD_RUNFORKHANDLER=1",  ### enables fork-safety in the XRootD client, to avoid deadlock when accessing EOS files
#                 f"export X509_USER_PROXY=/afs/cern.ch/user/f/fiorendi/x509up_u58808",
#                 "export PYTHONPATH=$PYTHONPATH:$_CONDOR_SCRATCH_DIR",
#             ],
#             extra = ['--worker-port 10000:10100']
#            )
#     cluster.adapt(minimum=1, maximum=2000)
#     print(cluster.job_script())
    cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    client = Client(cluster)

    lxplus_run = processor.Runner(
        executor=processor.DaskExecutor(client=client, compression=None),
        chunksize=30_000,
        skipbadfiles=True,
        schema=PFNanoAODSchema,
        savemetrics=True,
#         maxchunks=4,
    )
    
    out, proc_report = lxplus_run(
        fileset,
        treename="Events",
        processor_instance=SkimProcessor(),
        uproot_options={"allow_read_errors_with_report": (OSError, KeyError)}
    )
    print(out)

    # Save to JSON file
    with open(f'{out_folder}/result.json', 'w') as fp:  
        json.dump(proc_report,fp)

    ## save processed run/lumi to json file 
    run_dict = defaultdict(list)
    for run in out['run_dict']:
        run_dict[run] = [out['run_dict'][run]]
    with open(f'{out_folder}/processed_lumis.json', 'w') as fp:  
        json.dump(run_dict, fp, indent=2)
    elapsed = time.time() - tic
    print(f"Finished in {elapsed:.1f}s")