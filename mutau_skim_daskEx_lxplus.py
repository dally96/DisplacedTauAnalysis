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

import os, argparse, importlib, pdb
import time
from datetime import datetime

from itertools import islice

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
args = parser.parse_args()


def take(n, iterable):
    """Return the first n items of the iterable as a list."""
    return list(islice(iterable, n))


out_folder = '/eos/cms/store/user/fiorendi/displacedTaus/skim/mutau/v106/'

samples = {
#     "Wto2Q": "fileset_WTo2Q.py",
#     "WtoLNu": "fileset_WToLNu.py",
    "QCD": "samples.fileset_QCD",
    "DY": "samples.fileset_DY",
    "signal": "samples.fileset_signal",
#     "TTto": "fileset_TT.py",
}

module = importlib.import_module(samples[args.sample])
all_fileset = module.fileset  #['Stau_100_0p1mm'] 


if args.subsample == 'all':
    fileset = all_fileset
else:  
    fileset = {k: all_fileset[k] for k in args.subsample}

nfiles = int(args.nfiles)
if nfiles != -1:
    for k in fileset.keys():
        if nfiles < len(fileset[k]['files']):
            fileset[k]['files'] = take(nfiles, fileset[k]['files'])

print("Will process {} files from the following samples:".format(nfiles), fileset.keys())

exclude_prefixes = ['Flag', 'JetSVs', 'GenJetAK8_', 'SubJet', 
                    'Photon', 'TrigObj', 'TrkMET', 'HLT',
                    'Puppi', 'OtherPV', 'GenJetCands',
                    'FsrPhoton', ''
                    ## tmp
                    'diele', 'LHE', 'dimuon', 'Rho', 'JetPFCands', 'GenJet', 'GenCands', 
                    'Electron'
                    ]
                    
include_prefixes = ['DisMuon', 'Jet', 'Tau', 'PFMET', 'MET', 'ChsMET', 'PuppiMET', 'SV', 'PV', 'GenPart', 'GenVisTau', 'GenVtx', 'Pileup',
                    'event', 'run', 'luminosityBlock', 'nDisMuon', 'nTau', 'nJet', 'nGenPart', 'nGenVisTau', 'nVtx'
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

        # Combine into one record (uproot expects a single awkward Array)
#         ak_array = ak.zip(events_to_write)
#         ak_array = ak.Array(events_to_write)


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
# #         logger.info(f"Filled dictionary")

#         out_file = uproot.dask_write(ak.zip(events_to_write), "out.root", compute=False, tree_name='Events')
#         return out_file
# #         uproot.dask_write(
# #         skimmed,
# #         destination="skimtest/",
# #         prefix=f"{dataset}/skimmed",
# #     )

#         logger.info(f"Dictionary zipped: {events.metadata['dataset']}: {out_dict}")
#         return out_file
#         out_file = uproot.dask_write(out_dict, "/eos/cms/store/user/fiorendi/displacedTaus/skim/mutau/v3/"+events.metadata['dataset'], compute=False, tree_name='Events')
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
        return {"entries_written": len(events_to_write)}


    def postprocess(self, accumulator):
        pass
#         return accumulator
  
  
    
if __name__ == "__main__":
## https://github.com/CoffeaTeam/coffea-hats/blob/master/04-processor.ipynb

    print("Time started:", datetime.now().strftime("%H:%M:%S"))
    tic = time.time()

    import socket
    n_port = 8786
    cluster = CernCluster(
            cores=1,
            memory='3000MB',
            disk='1000MB',
            death_timeout = '60',
            container_runtime = "none",
            log_directory = "/eos/user/f/fiorendi/condor/log",
            scheduler_options={
                'port': n_port,
                'host': socket.gethostname(),
                },
            job_extra={
                '+JobFlavour': '"longlunch"',
                },
            job_script_prologue=[
                "export XRD_RUNFORKHANDLER=1",  ### enables fork-safety in the XRootD client, to avoid deadlock when accessing EOS files
                f"export X509_USER_PROXY=/afs/cern.ch/user/f/fiorendi/x509up_u58808",
                "export PYTHONPATH=$PYTHONPATH:$_CONDOR_SCRATCH_DIR",
            ],
            extra = ['--worker-port 10000:10100']
           )
    cluster.adapt(minimum=1, maximum=2000)
    print(cluster.job_script())
    client = Client(cluster)

#     cluster = LocalCluster(n_workers=4, threads_per_worker=1)
#     client = Client(cluster)

    iterative_run = processor.Runner(
        executor=processor.DaskExecutor(client=client, compression=None),
        chunksize=30_000,
        skipbadfiles=True,
        schema=PFNanoAODSchema,
        savemetrics=True,
#         maxchunks=4,
    )
    
    out = iterative_run(
        fileset,
        treename="Events",
        processor_instance=SkimProcessor(),
    )
    print(out)
    elapsed = time.time() - tic
    print(f"Finished in {elapsed:.1f}s")


#     with open("signal_test_preprocessed_fileset.pkl", "rb") as  f:
#         Stau_dataset_runnable = pickle.load(f)    
#     dataset_runnable = Stau_dataset_runnable  
# #    print(dataset_runnable.keys())
# 
#     for samp in dataset_runnable.keys(): 
#         if samp not in os.listdir("/eos/cms/store/user/fiorendi/displacedTaus/skim/mutau/v3/"):
# #             if "TT" in samp or "DY" in samp or "Stau" in samp or "QCD" in samp or "MET" in samp: continue
#             
#             samp_runnable = {}
#             samp_runnable[samp] = dataset_runnable[samp]
#             print("Time before comupute:", datetime.now().strftime("%H:%M:%S")) 
#             to_compute = apply_to_fileset(
#                      MyProcessor(),
#                      max_chunks(samp_runnable, 100000),
#                      schemaclass=PFNanoAODSchema,
#                      uproot_options={"coalesce_config": uproot.source.coalesce.CoalesceConfig(max_request_ranges=10, max_request_bytes=1024*1024),
#                                      "allow_read_errors_with_report": True}
#             )
# #             dak.necessary_columns(to_compute)
# 
#             #failed_chunks = get_failed_steps_for_fileset(
#             print(to_compute)
#             #outfile = uproot.dask_write(to_compute[0][samp], "root://cmseos.fnal.gov//store/user/dally/first_skim_muon_root_test/"+samp, compute=False, tree_name='Events')
#             dask.compute(to_compute)
#     
#             elapsed = time.time() - tic
#             print(f"Finished in {elapsed:.1f}s")
#         else:    
#             print(f"Folder already exists in /eos/cms/store/user/fiorendi/displacedTaus/skim/mutau/v3/")
#         
# 
#     client.shutdown()
#     cluster.close()
#     
#     
