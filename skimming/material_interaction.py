import awkward as ak
import uproot, os, sys
import numpy as np
import gzip, correctionlib, importlib, pickle

#sys.stdout.reconfigure(line_buffering=True)
#sys.stderr.reconfigure(line_buffering=True)
#os.environ["PYTHONUNBUFFERED"] = "1"

from coffea import processor
from coffea.nanoevents import PFNanoAODSchema
from coffea.lumi_tools import LumiData, LumiList, LumiMask

from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
from coffea.lookup_tools import extractor

import fsspec_xrootd
from  fsspec_xrootd import XRootDFileSystem

import dask
from dask import config as cfg
cfg.set({'distributed.scheduler.worker-ttl': None}) # Check if this solves some dask issues
#cfg.set({'distributed.scheduler.allowed-failures': 30}) # Check if this solves some dask issues
cfg.set({"distributed.logging.distributed": "debug"})
from dask.distributed import Client, LocalCluster, wait, progress, performance_report
#from dask_lxplus import CernCluster
from lpcjobqueue import LPCCondorCluster, schedd 
from dask import config as cfg
from dask_jobqueue import HTCondorCluster
import socket, time

import warnings
warnings.filterwarnings("ignore", module="coffea") # Suppress annoying deprecation warnings for coffea vector, c.f. https://github.com/CoffeaTeam/coffea/blob/master/src/coffea/nanoevents/methods/candidate.py
import logging

from selection_function import event_selection, event_selection_hpstau_mu
from utils import process_n_files, is_rootcompat, uproot_writeable_selected
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../input_jsons")))
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-m"    , "--muon"    , dest = "leading_muon_type"   , help = "Leading muon variable"    , default = "pt")
parser.add_argument("-j"    , "--jet"     , dest = "leading_jet_type"    , help = "Leading jet variable"     , default = "pt")
parser.add_argument(
	"--sample",
	choices=['QCD','DY', 'signal', 'WtoLNu', 'Wto2Q', 'TT', 'singleT', 'JetMET_2022', 'Muon'],
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
	"--skim",
	default='prompt_mutau',
	required=False,
	choices=['prompt_mutau','mutau'],
	help='Specify input skim, which objects, and selections (Muon and HPSTau, or DisMuon and Jet)')
parser.add_argument(
	"--skimversion",
	default='v0',
	required=False,
	help='If listing skimmed files, select which version of the inputs')
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

out_folder = f'root://cmseos.fnal.gov//store/user/dally/skim/{args.nanov}/{args.skim}/{args.skimversion}/material_veto/'


## define input samples
all_fileset = {}
if args.usePkl==True:
    ## to be made configurable
    with open(f"samples/{args.nanov}/{args.skim}/{args.skimversion}/{args.sample}_preprocessed.pkl", "rb") as  f:
        input_dataset = pickle.load(f)
        print(input_dataset.keys())
else:
    samples = {
        "JetMET": f"samples.{args.nanov}.{args.skim}.fileset_JetMET_2022",
    }
    module = importlib.import_module(samples[args.sample])
    input_dataset = module.fileset  #['Stau_100_0p1mm'] 
  

## restrict to specific sub-samples
if args.subsample == 'all':
    fileset = input_dataset
else:  
    fileset = {k: input_dataset[k] for k in args.subsample}


## restrict to n files
process_n_files(int(args.nfiles), fileset)
## add an else statement to prevent empty lists if nfiles > len(fileset)            
print("Will process {} files from the following samples:".format(args.nfiles), fileset.keys())


## branches to be included in the output files.
## tuned on prompt skim case
## will save all fields if in include_all, but only combinations of (include_prefixes,include_postfixes)
include_prefixes  = ['DisMuon',  'Muon',  'Jet', 'GenPart', 'GenVisTau']
include_postfixes = ['pt', 'eta', 'phi', 'pdgId', 'status', 'statusFlags', 'mass', 'dxy', 'charge', 'dz',
                     'mediumId', 'tightId', 'nTrackerLayers', 'tkRelIso', 'pfRelIso03_all', 'pfRelIso03_chg',
                     'disTauTag_score1', 'rawFactor', 'nConstituents',
                     'genPartIdxMother'
                    ]                       
include_all = ['Tau',  'PFMET',  'ChsMET', 'PuppiMET',         'GenVtx',
               'nTau', 'nPFMET', 'nChsMET','nPuppiMET', 'nPV', 'nGenVtx',
               'nVtx', 'event', 'run', 'luminosityBlock', 'Pileup', 'weights', 'genWeight', 'weight', 'HLT',
               'nDisMuon', 'nMuon', 'nJet',  'nGenPart', 'nGenVisTau', 'Stau', 'StauTau', 'mT', 'PV', 'mutau_mass',
               'CorrectedPuppiMET', 'L1MaterialDisMuon', 'L2MaterialDisMuon'
              ]

class MaterialProcessor(processor.ProcessorABC):
    def __init__(self):
        self._accumulator = {}
        
    def process(self, events):

        n_evts = len(events)  
        logger.info(f"starting process")
        if n_evts == 0: 
            logger.info(f"no input events")
            return {"entries_written": 0}
#         out = self._accumulator.identity() 

        logger.info(f"Start process for {events.metadata['dataset']}")
        dataset = events.metadata["dataset"]    
        num_dimuon = ak.count_nonzero(events.dimuon.pt, axis = 1)
        events = events[num_dimuon > 0]
        
        l1_pt = ak.cartesian([events.DisMuon.pt, events.dimuon.l1_pt])
        l1_charge =ak.cartesian([events.DisMuon.charge, events.dimuon.l1_charge])
        l1_idx = ak.argcartesian([events.DisMuon.pt, events.dimuon.l1_pt])

        l1_charge_mask = (l1_charge['0'] == l1_charge['1'])
        l1_pt_mask = (abs(l1_pt['0'] - l1_pt['1']) < 1)

        l1_dismuon_idx = l1_idx['0'][l1_charge_mask & l1_pt_mask]
        l1_dimuon_idx = l1_idx['1'][l1_charge_mask & l1_pt_mask]

        l1_vtx_x = events.dimuon.vtx_x[l1_dimuon_idx]
        l1_vtx_y = events.dimuon.vtx_y[l1_dimuon_idx]

        l2_pt = ak.cartesian([events.DisMuon.pt, events.dimuon.l2_pt])
        l2_charge = ak.cartesian([events.DisMuon.charge, events.dimuon.l2_charge])
        l2_idx = ak.argcartesian([events.DisMuon.pt, events.dimuon.l2_pt])

        l2_charge_mask = (l2_charge['0'] == l2_charge['1'])
        l2_pt_mask = (abs(l2_pt['0'] - l2_pt['1']) < 1)

        l2_dismuon_idx = l2_idx['0'][l2_charge_mask & l2_pt_mask]
        l2_dimuon_idx = l2_idx['1'][l2_charge_mask & l2_pt_mask]

        l2_vtx_x = events.dimuon.vtx_x[l2_dimuon_idx]
        l2_vtx_y = events.dimuon.vtx_y[l2_dimuon_idx]

        with uproot.open("Material_Map_HIST.root") as f:
            h2 = f["material_map"]
        values = h2.values()
        xedges = h2.axes[0].edges()
        yedges = h2.axes[1].edges()

        l1_x_index = ak.values_astype((30 + l1_vtx_x)//0.05, "int64")
        l1_y_index = ak.values_astype((30 + l1_vtx_y)//0.05, "int64")

        l1_x_index = ak.where(abs(l1_x_index) >= 1200, 0, l1_x_index)
        l1_y_index = ak.where(abs(l1_y_index) >= 1200, 0, l1_y_index)

        l1_x_counts = ak.num(l1_x_index)
        l1_y_counts = ak.num(l1_y_index)

        flat_l1_x_index = ak.flatten(l1_x_index, axis = None)
        flat_l1_y_index = ak.flatten(l1_y_index, axis = None)

        l1_material_interactions = values[flat_l1_x_index, flat_l1_y_index]
        l1_mat_muon = ak.unflatten(l1_material_interactions, l1_x_counts)

        l1_mat_mask = ak.where(l1_mat_muon > 0, True, False)
        l1_mat_dismuon = l1_dismuon_idx[l1_mat_mask]

        l2_x_index = ak.values_astype((30 + l2_vtx_x)//0.05, "int64")
        l2_y_index = ak.values_astype((30 + l2_vtx_y)//0.05, "int64")

        l2_x_index = ak.where(abs(l2_x_index) >= 1200, 0, l2_x_index)
        l2_y_index = ak.where(abs(l2_y_index) >= 1200, 0, l2_y_index)

        l2_x_counts = ak.num(l2_x_index)
        l2_y_counts = ak.num(l2_y_index)

        flat_l2_x_index = ak.flatten(l2_x_index, axis = None)
        flat_l2_y_index = ak.flatten(l2_y_index, axis = None)

        l2_material_interactions = values[flat_l2_x_index, flat_l2_y_index]
        l2_mat_muon = ak.unflatten(l2_material_interactions, l2_x_counts)

        l2_mat_mask = ak.where(l2_mat_muon > 0, True, False)
        l2_mat_dismuon = l2_dismuon_idx[l2_mat_mask]

        l1_dismuon = events.DisMuon[l1_mat_dismuon]
        l2_dismuon = events.DisMuon[l2_mat_dismuon]

        
        events = ak.with_field(events, l1_dismuon, "L1MaterialDisMuon")
        events = ak.with_field(events, l2_dismuon, "L2MaterialDisMuon")

        ## prevent writing out files with empty trees
        if not len(events) > 0:
            return {
                "entries_written": 0,
                 #"run_dict" : run_dict
            }

        # Write to ROOT
        events_to_write = uproot_writeable_selected(events, include_all, include_prefixes, include_postfixes)
        # unique name: dataset name + chunk range
        fname = os.path.basename(events.metadata["filename"]).replace(".root", "")
        outname = f"{out_folder}{dataset}/jet_dmu/{fname}_jet_dmu.root"

        with uproot.recreate(outname) as fout:
            fout["Events"] = events_to_write
#         skim = ak.to_parquet(events_to_write, outname.replace('.root', '.parquet'), extensionarray=False)
        return {"entries_written": len(events_to_write)}


    def postprocess(self, accumulator):
        return accumulator
    
    
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    tic = time.time()

    test_job = args.testjob 

    if not test_job:
        n_port = 8786
        cluster = LPCCondorCluster(
                cores=16,
                memory='32000MB',
                #disk='1000MB',
                #death_timeout = '600',
                #lcg = True,
                #nanny = False,
                #container_runtime = "none",
                log_directory = f"/uscmst1b_scratch/lpc1/3DayLifetime/condor/log/selected/{args.skimversion}",
                transfer_input_files = ["selection_function.py", "utils.py", "Cert_Collisions2022_355100_362760_Golden.json", "jec/", "Material_Map_HIST.root"],
                #scheduler_options={
                #    'port': n_port,
                #    'host': socket.gethostname(),
                #    },
                job_extra_directives={
                    "should_transfer_files": "YES",
                    '+JobFlavour': '"longlunch"',
                    },
                job_script_prologue=[
                    "export XRD_RUNFORKHANDLER=1",  ### enables fork-safety in the XRootD client, to avoid deadlock when accessing EOS files
                    f"export X509_USER_PROXY=$HOME/x509up_u57864",
                    "export PYTHONPATH=$PYTHONPATH:$_CONDOR_SCRATCH_DIR:$HOME",
                ],
                #worker_extra_args = ['--worker-port 10000:10100']
                )
         #minimum > 0: https://github.com/CoffeaTeam/coffea/issues/465
        cluster.adapt(minimum=1, maximum=200)
        print(cluster.job_script())
    
    else:
        cluster = LocalCluster(n_workers=10, threads_per_worker=1)

    client = Client(cluster)
    lxplus_run = processor.Runner(
        executor=processor.DaskExecutor(client=client, compression=None),
        ### alternative executors
        ## executor=processor.FuturesExecutor(compression=None, workers = 4),
        ## executor=processor.IterativeExecutor(compression=None),
        chunksize=30_000,
        skipbadfiles=True,
        schema=PFNanoAODSchema,
        savemetrics=True,
#         maxchunks=4,
    )
    
    out, proc_report = lxplus_run(
        fileset,
        treename="Events",
        processor_instance=MaterialProcessor(),
        uproot_options={"allow_read_errors_with_report": (OSError, KeyError)}
    )

    elapsed = time.time() - tic 
    print(f"Finished in {elapsed:.1f}s")
#     client.shutdown()
#     cluster.close()
        
        

