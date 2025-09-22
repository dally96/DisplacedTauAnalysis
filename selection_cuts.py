import awkward as ak
import uproot
import sys, argparse, os
import numpy as np
import pickle
import json, gzip, correctionlib, importlib

from coffea import processor
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema, NanoAODSchema
from coffea.lumi_tools import LumiData, LumiList, LumiMask

import fsspec_xrootd
from  fsspec_xrootd import XRootDFileSystem

import dask
from dask import config as cfg
cfg.set({'distributed.scheduler.worker-ttl': None}) # Check if this solves some dask issues
from dask.distributed import Client, wait, progress, LocalCluster
import socket, time

import warnings
warnings.filterwarnings("ignore", module="coffea") # Suppress annoying deprecation warnings for coffea vector, c.f. https://github.com/CoffeaTeam/coffea/blob/master/src/coffea/nanoevents/methods/candidate.py
import logging

from xsec import *
from selection_function import event_selection, event_selection_hpstau_mu

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-m"    , "--muon"    , dest = "leading_muon_type"   , help = "Leading muon variable"    , default = "pt")
parser.add_argument("-j"    , "--jet"     , dest = "leading_jet_type"    , help = "Leading jet variable"     , default = "pt")         
parser.add_argument(
	"--sample",
	choices=['QCD','DY', 'signal', 'WtoLNu', 'Wto2Q', 'TT'],
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

args = parser.parse_args()

## define the folder where the input .pkl files are defined, as well as the output folder on eos for the final events (not implemented yet)
skim_folder = args.skim

## will change once "region" will become a flag
if skim_folder == 'prompt_mutau':
    mode_string = 'hpstau_mu' 
    selection_string = 'HPSTauMu'
elif skim_folder == 'mutau':
    mode_string = 'jet_dmu' 
    selection_string = 'validation_prompt'  ## FIXME
else:
    print ('make sure using the correct folder/selections')
    exit(0)
    

out_folder = f'/eos/cms/store/user/fiorendi/displacedTaus/skim/{skim_folder}/selected/'


## move to an utils file
from itertools import islice
def take(n, iterable):
    """Return the first n items of the iterable as a list."""
    return list(islice(iterable, n))


all_fileset = {}
if args.usePkl==True:
    import pickle 
    ## to be made configurable
    with open(f"samples/{skim_folder}/{args.sample}_preprocessed.pkl", "rb") as  f:
        input_dataset = pickle.load(f)
else:
    samples = {
        "Wto2Q": f"samples.{skim_folder}.fileset_Wto2Q",
        "WtoLNu": f"samples.{skim_folder}.fileset_WtoLNu",
        "QCD": f"samples.{skim_folder}.fileset_QCD",
        "DY": f"samples.{skim_folder}.fileset_DY",
        "signal": f"samples.{skim_folder}.fileset_signal",
        "TT": f"samples.{skim_folder}.fileset_TT",
    }
    module = importlib.import_module(samples[args.sample])
    input_dataset = module.fileset  #['Stau_100_0p1mm'] 
  

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
## add an else statement to prevent empty lists if nfiles > len(fileset)            

print("Will process {} files from the following samples:".format(nfiles), fileset.keys())


## prompt skim case
include_prefixes = ['DisMuon',  'Muon',  'Jet',   
                    'GenPart',   
                    'GenVisTau', 'GenVtx'
                   ]

include_postfixes = ['pt', 'eta', 'phi', 'pdgId', 'status', 'statusFlags', 'mass', 'dxy', 'charge', 'dz',
                     'mediumId', 'tightId', 'nTrackerLayers', 'tkRelIso', 'pfRelIso03_all', 'pfRelIso03_chg',
                     'disTauTag_score1', 'rawFactor', 'nConstituents',
                     'genPartIdxMother'
                     ]           
                     
### need to add Lxy and IP at GEN level                             

include_all = ['Tau',  'PFMET',  'ChsMET', 'PuppiMET',         'GenVtx',     #'GenPart',  'GenVisTau', 'GenVtx',
               'nTau', 'nPFMET', 'nChsMET','nPuppiMET', 'nPV', 'nGenVtx',    #'nGenPart', 'nGenVisTau', 'nGenVtx',
               'nVtx', 'event', 'run', 'luminosityBlock', 'Pileup', 'weights', 'genWeight', 'weight', 
               'nDisMuon', 'nMuon', 'nJet',  'nGenPart', 'nGenVisTau', 'Stau', 'StauTau', 'mT', 'PV'
              ]

def is_rootcompat(a):
    """Is it a flat or 1-d jagged array?"""
    t = ak.type(a)
    if isinstance(t, ak.types.ArrayType):
        if isinstance(t.content, ak.types.NumpyType):
            return True
        if isinstance(t.content, ak.types.ListType) and isinstance(t.content.content, ak.types.NumpyType):
            return True
    return False


## please double check tomorrow
def uproot_writeable(events):
    '''
        Check columns that uproot can write out:
      - keep all branches starting with any prefix in include_all.
      - for branches starting with include_prefixes, keep only fields in include_postfixes.
    '''
    out = {}

    for bname in events.fields:
        keep_branch = any(bname.startswith(p) for p in include_all)
        reduced_branch = any(bname.startswith(p) for p in include_prefixes)

        # handle structured branches (records with subfields)
        if events[bname].fields:
            fields = {}
            for n in events[bname].fields:
                if is_rootcompat(events[bname][n]):
#                     print (bname, n)
                    if keep_branch or (reduced_branch and n in include_postfixes):
                        fields[n] = ak.to_packed(ak.without_parameters(events[bname][n]))

            if fields:  # avoid ak.zip({})
                out[bname] = ak.zip(fields)

        # handle flat branches
        else:
            if keep_branch or reduced_branch:
                out[bname] = ak.to_packed(ak.without_parameters(events[bname]))

    return out



class SelectionProcessor(processor.ProcessorABC):
    def __init__(self, leading_muon_var, leading_jet_var, mode="jet_dmu"):
        self.leading_muon_var = args.leading_muon_type
        self.leading_jet_var  = args.leading_jet_type
        ## sara: to be better understood
        assert mode in ["hpstau_mu", "jet_dmu"]
        self._mode = mode

        self._accumulator = {}
#         for samp in skimmed_fileset:
#             self._accumulator[samp] = dak.from_awkward(ak.Array([]), npartitions = 1)

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
#         out = self._accumulator.identity() 
#         if events is None: 
            logger.info(f"no input events")
            return {"entries_written": 0}

        logger.info(f"Start process for {events.metadata['dataset']}")
        dataset = events.metadata["dataset"]    

        leading_muon_var = self.leading_muon_var
        leading_jet_var = self.leading_jet_var

        # Determine if dataset is MC or Data
        is_MC = True if hasattr(events, "GenPart") else False
        if is_MC and 'Stau' in dataset: 
#             sumWeights = num_events[events.metadata["dataset"]]

            events["Stau"] = events.GenPart[(abs(events.GenPart.pdgId) == 1000015) &\
                                            (events.GenPart.hasFlags("isLastCopy"))]
            ## this needs some thoughts
#           events["StauTau"] = events.GenVisTau[(abs(events.GenVisTau.eta) < 2.4) & (events.GenVisTau.pt > 20)]
#           ## original Daniel
            events["StauTau"] = events.Stau.distinctChildren[(abs(events.Stau.distinctChildren.pdgId) == 15) &\
                                                   (events.Stau.distinctChildren.hasFlags("isLastCopy"))]
            ## here we take only the two highest pT taus to reject rare cases where more than two are found 
            events["StauTau"] = ak.firsts(events.StauTau[ak.argsort(events.StauTau.pt, ascending=False)], axis = 2) 
            events["StauTau"] = ak.flatten(ak.drop_none(events["StauTau"]), axis=0)



        ## IMPORTANT
        ## do we need to add selections before choosing the leading obj?
        ## these selections are not there anymore in the previous skimming step
        ## test to check the function in the SR
        if self._mode == "hpstau_mu":
#             print(ak.num(events.Muon.pt, axis=1))
#             print(ak.num(events.Muon.eta, axis=1))
#             print(ak.num(events.Muon.IPx, axis=1))
##             sort_indices = ak.argsort(ak.to_numpy(muons[leading_muon_var]), ascending=False, axis=1)
##             muons = muons[sort_indices]
            
            muons = events.Muon
            muons = muons[ak.argsort(muons[leading_muon_var], ascending=False, axis=1)]
            muons = ak.singletons(ak.firsts(muons))
            events["Muon"] = muons
            taus = events.Tau
            taus = events.Tau[ak.argsort(taus[leading_jet_var], ascending=False, axis = 1)]
            taus = ak.singletons(ak.firsts(taus))
            events["Tau"] = taus
            
            met = events.PFMET.pt            
            met_phi =  events.PFMET.phi        
            dphi = abs(muons.phi - met_phi)
            dphi = np.where(dphi > np.pi, 2*np.pi - dphi, dphi)  # wrap to [-pi, pi]
            mT = np.sqrt(2 * muons.pt * met * (1 - np.cos(dphi)))      
            events = ak.with_field(events, mT, "mT")
#             
            events = event_selection_hpstau_mu(events, selection_string)
        else:
            events["DisMuon"] = events.DisMuon[ak.argsort(events["DisMuon"][leading_muon_var], ascending=False, axis = 1)]
            events["DisMuon"] = ak.singletons(ak.firsts(events.DisMuon))
            events["Jet"] = events.Jet[ak.argsort(events["Jet"][leading_jet_var], ascending=False, axis = 1)]
            events["Jet"] = ak.singletons(ak.firsts(events.Jet))
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
#                 "run_dict" : run_dict
            }

        # Write to ROOT
        events_to_write = uproot_writeable(events)
        # unique name: dataset name + chunk range
        fname = os.path.basename(events.metadata["filename"]).replace(".root", "")
        outname = f"{out_folder}{dataset}/{fname}_{selection_string}.root"

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

#     n_port = 8786
#     cluster = CernCluster(
#             cores=1,
#             memory='3000MB',
#             disk='1000MB',
#             death_timeout = '60',
#             lcg = True,
#             nanny = False,
#             container_runtime = "none",
#             log_directory = "/eos/user/f/fiorendi/condor/log",
#             scheduler_options={
#                 'port': n_port,
#                 'host': socket.gethostname(),
#                 },
#             job_extra={
#                 '+JobFlavour': '"longlunch"',
#                 },
#             extra = ['--worker-port 10000:10100']
#             )
#      #minimum > 0: https://github.com/CoffeaTeam/coffea/issues/465
#     cluster.adapt(minimum=1, maximum=10000)
    
    cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    client = Client(cluster)

    lxplus_run = processor.Runner(
#         executor=processor.FuturesExecutor(compression=None, workers = 4),
#         executor=processor.IterativeExecutor(compression=None),
        executor=processor.DaskExecutor(client=client, compression=None),
        chunksize=30_000,
        skipbadfiles=True,
        schema=PFNanoAODSchema,
        savemetrics=True,
#         maxchunks=4,
    )
    
#     myfileset = {
#       "WLNu0J": {
#         "files": {
#           "root://eoscms.cern.ch//store/user/fiorendi/displacedTaus/skim/prompt_mutau/v1/WtoLNu-2Jets_1J/nano_2644_0_0_9429.root" : "Events",
# #           "root://eoscms.cern.ch//store/user/fiorendi/displacedTaus/skim/prompt_mutau/v1/WtoLNu-2Jets_0J/nano_10022_0_0_5251.root": "Events",
# # #           "root://eoscms.cern.ch//store/user/fiorendi/displacedTaus/skim/prompt_mutau/v1/WtoLNu-2Jets_0J/nano_10038_0_0_4011.root": "Events",
#          }
#       }
#     }
#     out = lxplus_run(
    out, proc_report = lxplus_run(
        fileset,
        treename="Events",
        processor_instance=SelectionProcessor(args.leading_muon_type, args.leading_jet_type, mode_string),
        uproot_options={"allow_read_errors_with_report": (OSError, KeyError)}
    )
#     print(out)

    elapsed = time.time() - tic 
    print(f"Finished in {elapsed:.1f}s")
#     client.shutdown()
#     cluster.close()
