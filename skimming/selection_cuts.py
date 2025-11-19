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
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory, CorrectedMETFactory
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
    

out_folder = f'root://cmseos.fnal.gov//store/group/lpcdisptau/dally/displacedTaus/skim/{args.nanov}/{skim_folder}/{args.skimversion}/selected/'


## define input samples
all_fileset = {}
if args.usePkl==True:
    ## to be made configurable
    with open(f"samples/{args.nanov}/{skim_folder}/{args.skimversion}/{args.sample}_preprocessed.pkl", "rb") as  f:
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
               'nDisMuon', 'nMuon', 'nJet',  'nGenPart', 'nGenVisTau', 'Stau', 'StauTau', 'mT', 'PV', 'mutau',
               'CorrectedPuppiMET'
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

        PFNanoAODSchema.mixins["CandidateMuon"] = "Muon"
        PFNanoAODSchema.mixins["CandidateElectron"] = "Electron"
        PFNanoAODSchema.mixins["DoubleMuon"] = "Muon"
        PFNanoAODSchema.mixins["DoubleElectron"] = "Electron"

        n_evts = len(events)  
        print("Before any selections, the number of events is", n_evts)
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
            muons = muons[ak.argsort(muons["pfRelIso04_all"], ascending=True, axis=1)]
            muon_iso_run_lengths = ak.run_lengths(muons.pfRelIso04_all)
            muon_iso_run_lengths_first = ak.firsts(muon_iso_run_lengths)
            muon_same_leadingIso_mask = ak.where(ak.local_index(muons.pfRelIso04_all) < muon_iso_run_lengths_first, True, False)
            muons = muons[muon_same_leadingIso_mask]
            muons = muons[ak.argsort(muons.pt, ascending = False, axis =1)]
            muons = ak.singletons(ak.firsts(muons))
            events["Muon"] = muons

            taus = events.Tau
            taus = events.Tau[ak.argsort(taus["rawIso"], ascending=True, axis = 1)]
            tau_iso_run_lengths = ak.run_lengths(taus.rawIso)
            tau_iso_run_lengths_first = ak.firsts(tau_iso_run_lengths)
            tau_same_leadingIso_mask = ak.where(ak.local_index(taus.rawIso) < tau_iso_run_lengths_first, True, False)
            taus = taus[tau_same_leadingIso_mask]
            taus = taus[ak.argsort(taus.pt, ascending = False, axis =1)]
            taus = ak.singletons(ak.firsts(taus))
            events["Tau"] = taus

            extra_electron_veto = (
                (ak.flatten(events.CandidateElectron.metric_table(muons), axis = 2) <= 0.5)
                | (ak.flatten(events.CandidateElectron.metric_table(taus), axis = 2)  <= 0.5)
            )
            num_extra_electron = ak.count_nonzero(extra_electron_veto, axis = 1)
            events = events[num_extra_electron == 0]
            muons  = muons[num_extra_electron == 0]
            taus   = taus[num_extra_electron == 0]
            print("Extra electron veto", ak.num(events, axis = 0))
 
            extra_muon_veto = (
                ((ak.flatten(events.CandidateMuon.metric_table(muons), axis = 2) > 0)
                & (ak.flatten(events.CandidateMuon.metric_table(muons), axis = 2) <= 0.5))
                | (ak.flatten(events.CandidateMuon.metric_table(taus), axis = 2)  <= 0.5)
            )
            num_extra_muon = ak.count_nonzero(extra_muon_veto, axis = 1)
            events = events[num_extra_muon == 0]
            muons = muons[num_extra_muon == 0]
            taus = taus[num_extra_muon == 0]
            print("Extra  muon veto", ak.num(events, axis = 0))

            ### Dilepton Veto
            mu_pair = ak.combinations(events.DoubleMuon, 2, axis = 1)
            el_pair = ak.combinations(events.DoubleElectron, 2, axis = 1)

            mu1,mu2 = ak.unzip(mu_pair)
            el1,el2 = ak.unzip(el_pair)

            presel_mask = lambda leps1, leps2: ((leps1.charge * leps2.charge < 0) & (leps1.delta_r(leps2) > 0.15))

            dlveto_mu_mask = presel_mask(mu1,mu2)
            dlveto_el_mask = presel_mask(el1,el2)
        
            dl_mu_veto = ak.sum(dlveto_mu_mask, axis=1) == 0
            dl_el_veto = ak.sum(dlveto_el_mask, axis=1) == 0

            dl_veto    = dl_mu_veto & dl_el_veto

            events = events[dl_veto]    
            print("DL Veto", ak.num(events, axis = 0)) 
            ###

            #bjets = events.LooseJet[(events.LooseJet.btagDeepFlavB < 0.0614)]
            bjet_veto = (
                (events.Jet.pt > 30)
                & (abs(events.Jet.eta) < 2.4)
                & (events.Jet.btagDeepFlavB >= 0.3196)
            )
            num_bjet = ak.count_nonzero(bjet_veto, axis = 1)
            events = events[num_bjet == 0]
            muons = muons[num_bjet == 0]
            taus = taus[num_bjet == 0]
            print("b-jet veto", ak.num(events, axis = 0)) 
            
            ## add transverse mass and mu+tau mass vars
            met = events.PFMET.pt            
            met_phi =  events.PFMET.phi     
            dphi = abs(muons.phi - met_phi)
            dphi = np.where(dphi > np.pi, 2*np.pi - dphi, dphi)  # wrap to [-pi, pi]
            mT = np.sqrt(2 * muons.pt * met * (1 - np.cos(dphi)))      
            events = ak.with_field(events, mT, "mT")
            dR = muons.metric_table(taus)

            events = events[ak.ravel(dR > 0.5)]
            taus = taus[ak.ravel(dR > 0.5)]
            muons = muons[ak.ravel(dR > 0.5)]
            print("dR veto", ak.num(events, axis = 0))
 
            mutau_cand = taus + muons
            events = events[ak.ravel(mutau_cand.charge == 0)]
            muons = muons[ak.ravel(mutau_cand.charge == 0)]
            taus = taus[ak.ravel(mutau_cand.charge == 0)]
            print("opposite charge veto", ak.num(events, axis = 0))

            mutau_cand = mutau_cand[ak.ravel(mutau_cand.charge == 0)]
            mutau_mass = mutau_cand.mass 
            events = ak.with_field(events, mutau_mass, "mutau_mass")

            events = events[ak.ravel(mutau_mass > 40)]
            print("mutau mass veto", ak.num(events, axis = 0))

            ## apply selections
            events = event_selection_hpstau_mu(events, selection_string)
            print("function veto", ak.num(events, axis = 0))
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
        print("Weights have been added to event fields")

        # JEC/JERC
        if is_MC:
            print("For JEC JERC, is MC")
            ext = extractor()
            print("Extractor initialized")
            ext.add_weight_sets([
                "* * Summer22EE_22Sep2023_V2_MC_L1FastJet_AK4PFPuppi.jec.txt",
                "* * Summer22EE_22Sep2023_V2_MC_L2Relative_AK4PFPuppi.jec.txt",
                "* * Summer22EE_22Sep2023_V2_MC_L2L3Residual_AK4PFPuppi.jec.txt",
                "* * Summer22EE_22Sep2023_V2_MC_L3Absolute_AK4PFPuppi.jec.txt",
                "* * Summer22EE_JRV1_MC_PtResolution_AK4PFPuppi.jer.txt",
                "* * Summer22EE_JRV1_MC_SF_AK4PFPuppi.jer.txt",
            ])
            print("Added weight sets")
            ext.finalize()

            jet_stack_names = [
                "Summer22EE_22Sep2023_V2_MC_L1FastJet_AK4PFPuppi",
                "Summer22EE_22Sep2023_V2_MC_L2Relative_AK4PFPuppi",
                "Summer22EE_22Sep2023_V2_MC_L2L3Residual_AK4PFPuppi",
                "Summer22EE_22Sep2023_V2_MC_L3Absolute_AK4PFPuppi",
                "Summer22EE_JRV1_MC_PtResolution_AK4PFPuppi",
                "Summer22EE_JRV1_MC_SF_AK4PFPuppi"
            ]
            print(x for x in jet_stack_names)

            evaluator = ext.make_evaluator()
            jec_inputs = {name: evaluator[name] for name in jet_stack_names}
            jec_stack = JECStack(jec_inputs)

            name_map = jec_stack.blank_name_map
            name_map['JetPt'] = 'pt'
            name_map['JetMass'] = 'mass'
            name_map['JetEta'] = 'eta'
            name_map['JetA'] = 'area'
            print("Created Jet name map")

            jets = events.Jet
            jets['pt_raw'] = (1 - jets['rawFactor']) * jets['pt']
            jets['mass_raw'] = (1 - jets['rawFactor']) * jets['mass']
            jets['pt_gen'] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
            jets['rho'] = ak.broadcast_arrays(events.Rho.fixedGridRhoFastjetAll, jets.pt)[0]    

            name_map['ptGenJet'] = 'pt_gen'
            name_map['ptRaw'] = 'pt_raw'
            name_map['massRaw'] = 'mass_raw'
            name_map['Rho'] = 'rho'

            jet_factory = CorrectedJetsFactory(name_map, jec_stack)
            corrected_jets = jet_factory.build(jets)
            print("Corrected jets done")

            met = events.PuppiMET
            met['pt_raw'] = events.RawPuppiMET.pt
            met['unclustEDeltaX'] = met.ptUnclusteredUp * np.cos(met.phiUnclusteredUp)
            met['unclustEDeltaY'] = met.ptUnclusteredUp * np.sin(met.phiUnclusteredUp)

            met_name_map = {}
            print("MET name map created")
            met_name_map['METpt'] = 'pt'
            met_name_map['METphi'] = 'phi'
            met_name_map['JetPt'] = 'pt'
            met_name_map['JetPhi'] = 'phi'
            met_name_map['ptRaw'] = 'pt_raw'
            met_name_map['UnClusteredEnergyDeltaX'] = 'unclustEDeltaX'
            met_name_map['UnClusteredEnergyDeltaY'] = 'unclustEDeltaY'

            met_factory = CorrectedMETFactory(met_name_map)
            CorrectedPuppiMET = met_factory.build(met, corrected_jets) 
            print("Corrected PuppiMET", CorrectedPuppiMET.fields)
            print("Events", type(events))
            print("CPM", type(CorrectedPuppiMET.pt.compute()))
            print("Events npartitions", events.npartitions)
            events = ak.with_field(events, CorrectedPuppiMET, "CorrectedPuppiMET")
            #events["CorrectedPuppiMET"] = CorrectedPuppiMET
            print("Adding CorrectedPuppiMET to events", events.CorrectedPuppiMET.pt.compute())

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
                cores=6,
                memory='12000MB',
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
