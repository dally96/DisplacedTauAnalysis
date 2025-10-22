import awkward as ak
import uproot
from coffea import processor
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema
import dask
from dask.distributed import Client, wait, progress, LocalCluster, performance_report
from lpcjobqueue import LPCCondorCluster, schedd
print("SCHEDD_POOL:", schedd.SCHEDD_POOL)
#from dask_lxplus import CernCluster
from dask import config as cfg
from dask_jobqueue import HTCondorCluster
cfg.set({'distributed.scheduler.worker-ttl': None}) # Check if this solves some dask issues

import fsspec_xrootd
from  fsspec_xrootd import XRootDFileSystem

import os, argparse, importlib, pdb, socket, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "./")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import time
from datetime import datetime

from collections import defaultdict
import json
from utils import process_n_files, is_rootcompat, is_good_hlt, is_included, uproot_writeable
from selections.lumi_selections import select_lumis

PFNanoAODSchema.warn_missing_crossrefs = False
PFNanoAODSchema.mixins["DisMuon"] = "Muon"

parser = argparse.ArgumentParser(description="")
parser.add_argument(
	"--sample",
	choices=['QCD','DY', 'signal', 'WtoLNu', 'Wto2Q', 'TT', 'singleT', 'all', 'JetMET'],
	nargs='*',
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
	choices=['Summer22_CHS_v10', 'Summer22_CHS_v7'],
	default='Summer22_CHS_v10',
	required=False,
	help='Specify the custom nanoaod version to process')
parser.add_argument(
	"--testjob",
        action="store_true",
        help="Run a test job locally")
args = parser.parse_args()

out_folder = f'root://cmseos.fnal.gov//store/group/lpcdisptau/dally/displacedTaus/test/skim/{args.nanov}/mutau/v_all_samples/'
out_folder_json = out_folder.replace('root://cmseos.fnal.gov/','/eos/uscms')
custom_nano_v = args.nanov + '/'
custom_nano_v_p = args.nanov + '.'

samples = {
    "Wto2Q": f"samples.{custom_nano_v_p}fileset_Wto2Q",
    "WtoLNu": f"samples.{custom_nano_v_p}fileset_WtoLNu",
    "QCD": f"samples.{custom_nano_v_p}fileset_QCD",
#    "DY": f"samples.{custom_nano_v_p}fileset_DY",
    "signal": f"samples.{custom_nano_v_p}fileset_signal",
#    "TT": f"samples.{custom_nano_v_p}fileset_TT",
    "singleT": f"samples.{custom_nano_v_p}fileset_singleT",
    "JetMET": f"samples.{custom_nano_v_p}fileset_JetMET_2022",
}

all_samples = args.sample
if args.sample[0] == 'all':
    all_samples = [k for k in samples.keys()]

fileset = {}
for isample in all_samples:
    if args.usePkl == True:
        import pickle 
        with open(f"samples/{custom_nano_v}{isample}_preprocessed.pkl", "rb") as  f:
            input_dataset = pickle.load(f)
    else:
        try:
            module = importlib.import_module(samples[isample])
            input_dataset = module.fileset   
        except:
            print ('file:', samples[isample], ' does not exists, skipping it' )    
            continue

    ## restrict to specific sub-samples
    if args.subsample == 'all':
        fileset.update(input_dataset)
    else:  
        fileset_tmp = {k: input_dataset[k] for k in args.subsample}
        fileset.update(fileset_tmp)
     

## restrict to n files
process_n_files(int(args.nfiles), fileset)
print("Will process {} files from the following samples:".format(args.nfiles), fileset.keys())

include_prefixes = ['DisMuon',  'Muon',  'Jet',  'Tau',   'PFMET', 'MET' , 'ChsMET', 'PuppiMET',   'PV', 'GenPart',   'GenVisTau', 'GenVtx',
                    'nDisMuon', 'nMuon', 'nJet', 'nTau', 'nPFMET', 'nMET', 'nChsMET','nPuppiMET', 'nPV', 'nGenPart', 'nGenVisTau', 'nGenVtx',
                    'nVtx', 'event', 'run', 'luminosityBlock', 'Pileup', 'weight', 'genWeight'#, 'HLT'
                   ]


good_hlts = [
  "PFMET120_PFMHT120_IDTight",                  
  "PFMET130_PFMHT130_IDTight",
  "PFMET140_PFMHT140_IDTight",
  "PFMETNoMu120_PFMHTNoMu120_IDTight",
  "PFMETNoMu130_PFMHTNoMu130_IDTight",
  "PFMETNoMu140_PFMHTNoMu140_IDTight",
  "PFMET120_PFMHT120_IDTight_PFHT60",
  "PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF",
  "PFMETTypeOne140_PFMHT140_IDTight",
  "MET105_IsoTrk50",
  "MET120_IsoTrk50",
#   "HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1",                 
#   "HLT_DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1",     
#   "HLT_DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1"
]



class SkimProcessor(processor.ProcessorABC):
    def __init__(self):
        pass
#         self._accumulator = {} 
#         for samp in fileset:
#             self._accumulator[samp] = dak.from_awkward(ak.Array([]), npartitions = 1)

    def process(self, events):
        
        import sys
        sys.path.append('.')
        from lumi_selections import select_lumis

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

        ## NB: to be double checked if the string identifying the buggy DY dataset is correct
        if is_MC and dataset == 'DYJetsToLL_M-50': 
            lhe_part = events.LHEPart
            outcoming = lhe_part[lhe_part.status > 0]
            lhe_z = outcoming[(outcoming.status == 2) & (outcoming.pdgId==23)]
            out_tau_tau = outcoming[(abs(outcoming.pdgId)==15)]
            counts_tautau = ak.num(out_tau_tau, axis=1)  
            mask_ztautau = (counts_tautau == 2)
            mask_zll = ~mask_ztautau
            events = events[mask_zll]

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
        events_to_write = uproot_writeable(events, good_hlts, include_prefixes)
        
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
        cluster = LPCCondorCluster(
                cores=4,
                memory='12000MB',
                disk='4000MB',
                death_timeout = '180',
                nanny=True,
                #container_runtime = "none",
                log_directory = "/uscms/home/dally/condor/log/mutau_skim/v3",
                #ship_env=True,
                transfer_input_files='utils.py',
                #scheduler_options={
                #    'port': n_port,
                #    'host': socket.gethostname(),
                #    },
                job_extra={
                    'should_transfer_files': 'YES',
                    '+JobFlavour': '"workday"',
                    },
                job_script_prologue=[
                    "export XRD_RUNFORKHANDLER=1",  ### enables fork-safety in the XRootD client, to avoid deadlock when accessing EOS files
                    f"export X509_USER_PROXY=$HOME/x509up_u57864",
                    "export PYTHONPATH=$PYTHONPATH:$_CONDOR_SCRATCH_DIR:$HOME",
                ],
                #worker_extra_args = ['--worker-port 10000:10100']
               )
        cluster.adapt(minimum=1, maximum=300)#, wait_count=3)
        print(cluster.job_script())
    else:    
        cluster = LocalCluster(n_workers=8, threads_per_worker=1)
    
    client = Client(cluster)
    client.run(lambda: __import__('os').system('ls -l $_CONDOR_SCRATCH_DIR'))
    lxplus_run = processor.Runner(
        executor=processor.DaskExecutor(client=client, compression=None),
        chunksize=100_000,
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
    
    ## save processed run/lumi to json file 
    for isubsample in out['run_dict'].keys():
        with open(f"{out_folder_json}/processed_lumis_{isubsample}.json", "w") as fp:
            ## convert to normal dict and dump as JSON
            json.dump({isubsample: v for v in out['run_dict'][isubsample].items()}, fp, indent=2)
        ## save proc report to json file (not useful for now)
        with open(f'{out_folder_json}/result_{isubsample}.json', 'w') as fp:  
            json.dump(proc_report,fp)

    elapsed = time.time() - tic
    print(f"Finished in {elapsed:.1f}s")
