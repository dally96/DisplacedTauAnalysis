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


## define the folder where the input .pkl files are defined, as well as the output folder on eos for the final events
skim_folder = args.skim
## will change once "region" will become a flag
if skim_folder == 'prompt_mutau':
    mode_string = 'hpstau_mu' 
    selection_string = 'HPSTauMu'
elif skim_folder == 'mutau':
    mode_string = 'jet_dmu' 
    selection_string = 'validation_daniel'  ## FIXME
else:
    print ('make sure using the correct folder/selections')
    exit(0)
    

out_folder = f'root://cmseos.fnal.gov//store/user/dally/displacedTaus/skim/{args.nanov}/{skim_folder}/{args.skimversion}/selected/'


## define input samples
all_fileset = {}
if args.usePkl==True:
    ## to be made configurable
    with open(f"samples/{args.nanov}/{skim_folder}/v4/{args.sample}_preprocessed.pkl", "rb") as  f:
        input_dataset = pickle.load(f)
        print(input_dataset.keys())
else:
    samples = {
        "Wto2Q": f"samples.{args.nanov}.{skim_folder}.fileset_Wto2Q",
        "WtoLNu": f"samples.{args.nanov}.{skim_folder}.fileset_WtoLNu",
        "QCD": f"samples.{args.nanov}.{skim_folder}.fileset_QCD",
        "DY": f"samples.{args.nanov}.{skim_folder}.fileset_DY",
        "signal": f"samples.{args.nanov}.{skim_folder}.fileset_signal",
        "TT": f"samples.{args.nanov}.{skim_folder}.fileset_TT",
        "singleT": f"samples.{args.nanov}.{skim_folder}.fileset_singleT",
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
               'nDisMuon', 'nMuon', 'nJet',  'nGenPart', 'nGenVisTau', 'Stau', 'StauTau', 'mT', 'PV', 'mutau'
              ]

### FIXME: need to add Lxy and IP at GEN level                             


class SelectionProcessor(processor.ProcessorABC):
    def __init__(self, leading_muon_var, leading_jet_var, mode="jet_dmu"):
        self.leading_muon_var = args.leading_muon_type
        self.leading_jet_var  = args.leading_jet_type
        ## sara: to be better understood
        assert mode in ["hpstau_mu", "jet_dmu"]
        self._mode = mode

        self._accumulator = {}
        #for samp in skimmed_fileset:
        #    self._accumulator[samp] = dak.from_awkward(ak.Array([]), npartitions = 1)

       # Load pileup weights evaluators 
        jsonpog = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration"
        pileup_file = jsonpog + "/POG/LUM/2022_Summer22EE/puWeights.json.gz"

        with gzip.open(pileup_file, 'rt') as f:
            self.pileup_set = correctionlib.CorrectionSet.from_string(f.read().strip())

    def get_pileup_weights(self, events, also_syst=False):
        # Apply pileup weights
        evaluator = self.pileup_set["Collisions2022_359022_362760_eraEFG_GoldenJson"]
        sf = evaluator.evaluate(events.Pileup.nTrueInt, "nominal")
#         if also_syst:
#             sf_up = evaluator.evaluate(events.Pileup.nTrueInt, "up")
#             sf_down = evaluator.evaluate(events.Pileup.nTrueInt, "down")
#         return {'nominal': sf, 'up': sf_up, 'down': sf_down}
        return {'nominal': sf}


    def process_weight_corrs_and_systs(self, events, weights):
        pileup_weights = self.get_pileup_weights(events)
        # Compute nominal weight and systematic variations by multiplying relevant factors
        # For pileup, do not multiply nominal correction factor, as it is already included in the up/down variations
        # To see this, one can reproduce the ratio in
        # https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/blob/master/misc/LUM/2018_UL/puWeights.png?ref_type=heads
        # from the plain correctionset
        weight_dict = {
            'weight': weights * pileup_weights['nominal'] #* muon_weights['muon_trigger_SF'],
#             'weight_pileup_up': weights * pileup_weights['up'] * muon_weights['muon_trigger_SF'],
#             'weight_pileup_down': weights * pileup_weights['down'] * muon_weights['muon_trigger_SF'],
#             'weight_muon_trigger_up': weights * pileup_weights['nominal'] * (muon_weights['muon_trigger_SF'] + muon_weights['muon_trigger_SF_syst']),
#             'weight_muon_trigger_down': weights * pileup_weights['nominal'] * (muon_weights['muon_trigger_SF'] - muon_weights['muon_trigger_SF_syst']),
        }

        return weight_dict


    def process(self, events):
        n_evts = len(events)  
        logger.info(f"starting process")
        if n_evts == 0: 
            logger.info(f"no input events")
            return {"entries_written": 0}
#         out = self._accumulator.identity() 

        logger.info(f"Start process for {events.metadata['dataset']}")
        dataset = events.metadata["dataset"]    

        leading_muon_var = self.leading_muon_var
        leading_jet_var = self.leading_jet_var

        # Determine if dataset is MC or Data
        is_MC = True if hasattr(events, "GenPart") else False
        if is_MC and 'Stau' in dataset: 
            events["Stau"] = events.GenPart[(abs(events.GenPart.pdgId) == 1000015) &\
                                            (events.GenPart.hasFlags("isLastCopy"))]
            ## FIXME this needs some thoughts
#           events["StauTau"] = events.GenVisTau[(abs(events.GenVisTau.eta) < 2.4) & (events.GenVisTau.pt > 20)]
#           ## original Daniel
            events["StauTau"] = events.Stau.distinctChildren[(abs(events.Stau.distinctChildren.pdgId) == 15) &\
                                                   (events.Stau.distinctChildren.hasFlags("isLastCopy"))]
            ## here we take only the two highest pT taus to reject rare cases where more than two are found 
            events["StauTau"] = ak.firsts(events.StauTau[ak.argsort(events.StauTau.pt, ascending=False)], axis = 2) 
            events["StauTau"] = ak.flatten(ak.drop_none(events["StauTau"]), axis=0)


        ## IMPORTANT
        ## do we need to add selections before choosing the leading obj?
        if self._mode == "hpstau_mu":
            muons = events.Muon
            muons = muons[ak.argsort(muons[leading_muon_var], ascending=False, axis=1)]
            muons = ak.singletons(ak.firsts(muons))
            events["Muon"] = muons
            taus = events.Tau
            taus = events.Tau[ak.argsort(taus[leading_jet_var], ascending=False, axis = 1)]
            taus = ak.singletons(ak.firsts(taus))
            events["Tau"] = taus
            
            ## add transverse mass and mu+tau mass vars
            met = events.PFMET.pt            
            met_phi =  events.PFMET.phi        
            dphi = abs(muons.phi - met_phi)
            dphi = np.where(dphi > np.pi, 2*np.pi - dphi, dphi)  # wrap to [-pi, pi]
            mT = np.sqrt(2 * muons.pt * met * (1 - np.cos(dphi)))      
            events = ak.with_field(events, mT, "mT")
            #events = events[ak.ravel(events.mT < 65)]            
            #taus = taus[ak.ravel(events.mT < 65)]
            #muons = muons[ak.ravel(events.mT < 65)]
 
            mutau_cand = taus + muons
            mutau_mass = mutau_cand.mass 
            events = ak.with_field(events, mutau_mass, "mutau_mass")
            #events = events[ak.ravel(mutau_cand.charge == 0)]

            ## apply selections
            events = event_selection_hpstau_mu(events, selection_string)
        else:
            dismuons = events.DisMuon
            dismuons = dismuons[ak.argsort(dismuons[leading_muon_var], ascending=False, axis=1)]
            dismuons = ak.singletons(ak.firsts(dismuons))
            events["DisMuon"] = dismuons
            jets = events["Jet"]
            jets =  jets[ak.argsort(jets[leading_jet_var], ascending=False, axis = 1)]
            jets = ak.singletons(ak.firsts(jets))
            events["Jet"] = jets

            ## add transverse mass var
            met = events.PFMET.pt            
            met_phi =  events.PFMET.phi        
            dphi = abs(dismuons.phi - met_phi)
            dphi = np.where(dphi > np.pi, 2*np.pi - dphi, dphi)  # wrap to [-pi, pi]
            mT = np.sqrt(2 * dismuons.pt * met * (1 - np.cos(dphi)))      
            events = ak.with_field(events, mT, "mT")
            
            ## apply selections
            events = event_selection(events, selection_string)  

        logger.info(f"Chose leading objects & filtered events")

        weights = events.genWeight if is_MC else 1 * ak.ones_like(events.event) 
        logger.info("mc weights")
        # Handle systematics and weights
        if is_MC:
            weight_branches = self.process_weight_corrs_and_systs(events, weights)
        else:
            weight_branches = {'weight': weights}
        logger.info("all weights")
        events = ak.with_field(events, weight_branches["weight"], "weight")

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
        outname = f"{out_folder}{dataset}/{selection_string}/{fname}_{selection_string}.root"

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
                cores=4,
                memory='8000MB',
                #disk='1000MB',
                #death_timeout = '600',
                #lcg = True,
                #nanny = False,
                #container_runtime = "none",
                log_directory = "/uscmst1b_scratch/lpc1/3DayLifetime/condor/log/selected/v1",
                transfer_input_files = ["selection_function.py", "utils.py", "Cert_Collisions2022_355100_362760_Golden.json"],
                #scheduler_options={
                #    'port': n_port,
                #    'host': socket.gethostname(),
                #    },
                #job_extra_directives={
                #    "should_transfer_files": "YES",
                #    '+JobFlavour': '"longlunch"',
                #    },
                job_script_prologue=[
                    #"export XRD_RUNFORKHANDLER=1",  ### enables fork-safety in the XRootD client, to avoid deadlock when accessing EOS files
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
        processor_instance=SelectionProcessor(args.leading_muon_type, args.leading_jet_type, mode_string),
        uproot_options={"allow_read_errors_with_report": (OSError, KeyError)}
    )

    elapsed = time.time() - tic 
    print(f"Finished in {elapsed:.1f}s")
#     client.shutdown()
#     cluster.close()
