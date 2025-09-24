import awkward as ak
import uproot
from coffea import processor
from coffea.nanoevents.methods import candidate
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema
from dask.distributed import Client, wait, progress, LocalCluster, performance_report
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

from utils import process_n_files


PFNanoAODSchema.warn_missing_crossrefs = False
PFNanoAODSchema.mixins["DisMuon"] = "Muon"

parser = argparse.ArgumentParser(description="")
parser.add_argument(
	"--sample",
	choices=['QCD','DY', 'signal', 'WtoLNu', 'Wto2Q', 'TT', 'singleT'],
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
parser.add_argument(
	"--nanov",
	choices=['Summer22_CHS_v9', 'Summer22_CHS_v7'],
	default='Summer22_CHS_v9',
	required=False,
	help='Specify the custom nanoaod version to process')
parser.add_argument(
	"--testjob",
	default=False,
	required=False,
	help='Turn it to true to run a test job locally')
args = parser.parse_args()

out_folder = f'root://eoscms.cern.ch//store/user/fiorendi/displacedTaus/skim/{args.nanov}/mutau/v0/'
out_folder_json = out_folder.replace('root://eoscms.cern.ch/','/eos/cms')
custom_nano_v = args.nanov + '/'
custom_nano_v_p = args.nanov + '.'


all_fileset = {}
if args.usePkl == True:
    import pickle 
    with open(f"samples/{custom_nano_v}{args.sample}_preprocessed.pkl", "rb") as  f:
        input_dataset = pickle.load(f)
else:
    samples = {
        "Wto2Q": f"samples.{custom_nano_v_p}fileset_Wto2Q",
        "WtoLNu": f"samples.{custom_nano_v_p}fileset_WtoLNu",
        "QCD": f"samples.{custom_nano_v_p}fileset_QCD",
        "DY": f"samples.{custom_nano_v_p}fileset_DY",
        "signal": f"samples.{custom_nano_v_p}fileset_signal",
        "TT": f"samples.{custom_nano_v_p}fileset_TT",
        "singleT": f"samples.{custom_nano_v_p}fileset_singleT",
    }
  
    module = importlib.import_module(samples[args.sample])
    input_dataset = module.fileset   

## restrict to specific sub-samples
if args.subsample == 'all':
    fileset = input_dataset
else:  
    fileset = {k: input_dataset[k] for k in args.subsample}


## restrict to n files
process_n_files(int(args.nfiles), fileset)
print("Will process {} files from the following samples:".format(args.nfiles), fileset.keys())


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
#   "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1",
#   "HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1",
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
            return {
                "entries_written": 0,
                "run_dict" : defaultdict(list)
            }
            
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
        run_dict = defaultdict(list)
        dataset_run_dict = defaultdict(list)
        run_lumi_list = set(zip(events.run, events.luminosityBlock))
        for run, lumi in sorted(run_lumi_list):
            run_dict[str(int(run))].append(int(lumi))
        dataset_run_dict[dataset] = dict(run_dict)

        ## Trigger mask
        trigger_mask = (
                       events.HLT.PFMET120_PFMHT120_IDTight                                    |\
                       events.HLT.PFMET130_PFMHT130_IDTight                                    |\
                       events.HLT.PFMET140_PFMHT140_IDTight                                    |\
                       events.HLT.PFMETNoMu120_PFMHTNoMu120_IDTight                            |\
                       events.HLT.PFMETNoMu130_PFMHTNoMu130_IDTight                            |\
                       events.HLT.PFMETNoMu140_PFMHTNoMu140_IDTight                            |\
                       events.HLT.PFMET120_PFMHT120_IDTight_PFHT60                             |\
                       events.HLT.PFMETTypeOne140_PFMHT140_IDTight                             |\
                       events.HLT.PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF                   |\
                       events.HLT.MET105_IsoTrk50                                              |\
                       events.HLT.MET120_IsoTrk50                                              #|\
        )
        events = events[trigger_mask]

        # Define the "good muon" condition for each muon per event
        good_muon_mask = (
            (events.DisMuon.pt > 20)
            & (abs(events.DisMuon.eta) < 2.4) # Acceptance of the CMS muon system
        )
        num_good_muons = ak.count_nonzero(good_muon_mask, axis=1)
        sel_muons = events.DisMuon[good_muon_mask]
        events['DisMuon'] = sel_muons
        events = events[num_good_muons >= 1]

        good_jet_mask = (
            (events.Jet.pt > 20)
            & (abs(events.Jet.eta) < 2.4)
#             & ~(ak.all(events.Jet.constituents.pf.charge == 0, axis = -1)) 
        )
        num_good_jets = ak.count_nonzero(good_jet_mask, axis=1)
        sel_jets = events.Jet[good_jet_mask]
        events['Jet'] = sel_jets
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

        charged_sel = events.Jet.constituents.pf.charge != 0
        dxy = ak.where(ak.all(events.Jet.constituents.pf.charge == 0, axis = -1), -999, ak.flatten(events.Jet.constituents.pf[ak.argmax(events.Jet.constituents.pf[charged_sel].pt, axis=2, keepdims=True)].d0, axis = -1))
        dxy = ak.fill_none(dxy, -999)
        events["Jet"] = ak.with_field(events.Jet, dxy, where = "dxy")

        ## prevent writing out files with empty trees
        if not len(events) > 0:
            return {
                "entries_written": 0,
                "run_dict" : dataset_run_dict
            }
            
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
          "run_dict" : dataset_run_dict
        }


    def postprocess(self, accumulator):
        return accumulator
  
  
    
if __name__ == "__main__":
## https://github.com/CoffeaTeam/coffea-hats/blob/master/04-processor.ipynb

    print("Time started:", datetime.now().strftime("%H:%M:%S"))
    tic = time.time()

    test_job = args.testjob 
    
    if not test_job:
        n_port = 8786
        cluster = CernCluster(
                cores=1,
                memory='4000MB',
                disk='1000MB',
                death_timeout = '240',
                nanny=False,
                container_runtime = "none",
                log_directory = "root://eosuser.cern.ch//eos/user/f/fiorendi/condor/log/mutau_skim/v3",
                scheduler_options={
                    'port': n_port,
                    'host': socket.gethostname(),
                    },
                job_extra={
                    '+JobFlavour': '"workday"',
                    'transfer_input_files': 'utils.py',
                    'should_transfer_files': 'YES',
                    },
                job_script_prologue=[
                    "export XRD_RUNFORKHANDLER=1",  ### enables fork-safety in the XRootD client, to avoid deadlock when accessing EOS files
                    f"export X509_USER_PROXY=/afs/cern.ch/user/f/fiorendi/x509up_u58808",
                    "export PYTHONPATH=$PYTHONPATH:$_CONDOR_SCRATCH_DIR",
                ],
                extra = ['--worker-port 10000:10100']
               )
        cluster.adapt(minimum=1, maximum=300)#, wait_count=3)
        print(cluster.job_script())
    else:    
        cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    
    client = Client(cluster)
    lxplus_run = processor.Runner(
        executor=processor.DaskExecutor(client=client, compression=None),
        chunksize=50_000,
        skipbadfiles=True,
        schema=PFNanoAODSchema,
        savemetrics=True,
#         maxchunks=4,
    )
    
    myfileset = {
      "WLNu1J": {
        "files": {
            "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-800to1000_TuneCP5_13p6TeV_pythia8_ext/nano_0_0.root": "Events",
            "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-800to1000_TuneCP5_13p6TeV_pythia8_ext/nano_100_0.root": "Events",
            "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-800to1000_TuneCP5_13p6TeV_pythia8_ext/nano_10_0.root": "Events",
            "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-800to1000_TuneCP5_13p6TeV_pythia8_ext/nano_102_0.root": "Events",
            "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-800to1000_TuneCP5_13p6TeV_pythia8_ext/nano_103_0.root": "Events",
            "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-800to1000_TuneCP5_13p6TeV_pythia8_ext/nano_104_0.root": "Events",
            "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-800to1000_TuneCP5_13p6TeV_pythia8_ext/nano_105_0.root": "Events",
            "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-800to1000_TuneCP5_13p6TeV_pythia8_ext/nano_106_0.root": "Events",
            "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-800to1000_TuneCP5_13p6TeV_pythia8_ext/nano_107_0.root": "Events",
            "root://cmsxrootd.fnal.gov//store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/QCD_PT-800to1000_TuneCP5_13p6TeV_pythia8_ext/nano_108_0.root": "Events",
         }
      }
    }
    out, proc_report = lxplus_run(
        myfileset,
        treename="Events",
        processor_instance=SkimProcessor(),
        uproot_options={"allow_read_errors_with_report": (OSError, KeyError)}
    )

    # Save to JSON file
    sub_string = '_'.join(subs for subs in args.subsample)
    with open(f'{out_folder_json}/result_{args.sample}_{sub_string}.json', 'w') as fp:  
        json.dump(proc_report,fp)

    ## save processed run/lumi to json file 
    with open(f"{out_folder_json}/processed_lumis_{args.sample}_{sub_string}.json", "w") as fp:
        # Convert defaultdicts into normal dicts for clean JSON
        json.dump({k: dict(v) for k, v in out['run_dict'].items()}, fp, indent=2)

    elapsed = time.time() - tic
    print(f"Finished in {elapsed:.1f}s")