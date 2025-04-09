import sys
import os
import fsspec_xrootd
from  fsspec_xrootd import XRootDFileSystem
import uproot
import uproot.exceptions
from uproot.exceptions import KeyInFileError
import dill as pickle

import argparse
import numpy as np
from coffea import processor
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema, NanoAODSchema
from coffea.dataset_tools import (
    apply_to_fileset,
    max_chunks,
    preprocess,
)
from coffea.lumi_tools import LumiData, LumiList, LumiMask
import awkward as ak
import dask
import hist
import hist.dask as hda
import matplotlib as mpl
from matplotlib import pyplot as plt
from dask import config as cfg
cfg.set({'distributed.scheduler.worker-ttl': None}) # Check if this solves some dask issues
import dask_awkward as dak
from dask_jobqueue import HTCondorCluster
from dask.distributed import Client, wait, progress, LocalCluster
from fileset import *

import time
from distributed import Client
from lpcjobqueue import LPCCondorCluster

import warnings
warnings.filterwarnings("ignore", module="coffea") # Suppress annoying deprecation warnings for coffea vector, c.f. https://github.com/CoffeaTeam/coffea/blob/master/src/coffea/nanoevents/methods/candidate.py
import logging 

lep_list = ["e", "mu", "tau", "reco_e", "reco_mu", "reco_tau"]


SAMP = [
      'QCD',
      "DYJetsToLL",  
      "TTtoLNu2Q",
      "TTto4Q",
      "TTto2L2Nu",
      ]

datasets = [ 
      'QCD50_80',
      'QCD80_120',
      'QCD120_170',
      'QCD170_300',
      'QCD300_470',
      'QCD470_600',
      'QCD600_800',
      'QCD800_1000',
      'QCD1000_1400',
      'QCD1400_1800',
      'QCD1800_2400',
      'QCD2400_3200',
      'QCD3200',
      'DYJetsToLL',  
      'TTtoLNu2Q',
      'TTto4Q',
      'TTto2L2Nu',
      ]


NanoAODSchema.mixins["DisMuon"] = "Muon"

class ZProcessor(processor.ProcessorABC):
    def __init__(self, lep_list, datasets):
        self.lep_list = lep_list
        self.samp     = datasets

        histos = {}
        for var in self.lep_list:
            histos[var] = hda.hist.Hist(hist.axis.Regular(500, -2.4, 2.4, name="eta", label = r"$\eta$"))
        self._accumulator = histos

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events): 
        
        logger.info(f"Now processing {events.metadata['dataset']}")

        histograms = {}
        for var in self.lep_list:
            histograms[var] = hda.hist.Hist(hist.axis.Regular(100, -2.4, 2.4, name="eta", label = r"$\eta$"))

        electrons = events.GenPart[(abs(events.GenPart.pdgId) == 11) & (events.GenPart.hasFlags("isLastCopy")) & (events.GenPart.pt > 20) & (abs(events.GenPart.eta) < 2.4)] 
        recoelec = events.Electron[(events.Electron.pt > 20) & (abs(events.Electron.eta) < 2.4) & (abs(events.Electron.matched_gen.pdgId) == 11)]
        muons = events.GenPart[(abs(events.GenPart.pdgId) == 13) & (events.GenPart.hasFlags("isLastCopy")) & (events.GenPart.pt > 20) & (abs(events.GenPart.eta) < 2.4)] 
        recomuon = events.DisMuon[(events.DisMuon.pt > 20) & (abs(events.DisMuon.eta) < 2.4) & (abs(events.GenPart[events.DisMuon.genPartIdx].pdgId) == 13)]
        taus = events.GenVisTau.parent[(events.GenVisTau.parent.pt > 20) & (abs(events.GenVisTau.parent.eta) < 2.4)]
        recotaus = taus.nearest(events.Jet, threshold = 0.3)
        recotaus = recotaus[(recotaus.pt > 20) & (abs(recotaus.eta) < 2.4)]

        histograms["e"].fill( dak.flatten(electrons["eta"], axis = None))
        histograms["reco_e"].fill( dak.flatten(recoelec["eta"], axis = None))
        logger.info("electron hists filled")
        histograms["mu"].fill( dak.flatten(muons["eta"], axis = None))
        histograms["reco_mu"].fill( dak.flatten(recomuon["eta"], axis = None))
        logger.info("muon hists filled")
        histograms["tau"].fill( dak.flatten(taus["eta"], axis = None))
        histograms["reco_tau"].fill( dak.flatten(recotaus["eta"], axis = None))
        logger.info("tau hists filled")

        return histograms

    def postprocess(self):
        return accumulator


if __name__ == "__main__":
    
    dask.config.set({"distributed.worker.memory.target": 0.8})

    client = Client()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    tic = time.time()
    cluster = LPCCondorCluster(ship_env=True, transfer_input_files='/uscms/home/dally/x509up_u57864')
    # minimum > 0: https://github.com/CoffeaTeam/coffea/issues/465
    #cluster.scale(10)
    cluster.adapt(minimum=1, maximum=1000)
    client = Client(cluster)
    
    tic = time.time()

    with open("preprocessed_fileset.pkl",  "rb") as f:
        dataset_runnable = pickle.load(f)

    elapsed = time.time() - tic
    print(f"preprocess took {elapsed:.1f}s")
    to_compute = apply_to_fileset(
                 ZProcessor(lep_list, datasets),
                 max_chunks(dataset_runnable, 100000),
                 schemaclass=NanoAODSchema
                 )

    logger.info(to_compute)

    for samp in to_compute.keys():
        print(f"Plotting for {samp}")
        #samp_hists = dask.compute(to_compute[samp]['e'], to_compute[samp]['mu'], to_compute[samp]['tau'], to_compute[samp]['reco_e'], to_compute[samp]['reco_mu'], to_compute[samp]['reco_tau'])
        samp_hists = dask.compute(to_compute[samp]['e'])
        plt.clf()
        plt.cla()
        samp_hists[0]['e'].plot(label="Gen electrons")
        #samp_hists[0]['reco_e'].plot(label="Reco electrons")
        plt.title(f"{samp}")
        plt.xlabel(r"$\eta$")
        plt.legend()
        plt.savefig(f"{samp}_electrons.pdf")
        print("Electron hist plotted")

        #print("Starting muon hist")
        #plt.clf()
        #plt.cla()
        #samp_hists[samp]['mu'].plot(label="Gen muons")
        #samp_hists[samp]['reco_mu'].plot(label="Reco muons")
        #plt.title(f"{samp}")
        #plt.xlabel(r"$\eta$")
        #plt.legend()
        #plt.savefig(f"{samp}_muons.pdf")
        #print("Muon hist plotted")

        #print("Starting tau hist")
        #plt.clf()
        #plt.cla()
        #samp_hists[samp]['tau'].plot(label="Gen had taus")
        #samp_hists[samp]['reco_tau'].plot(label="Reco jets")
        #plt.title(f"{samp}")
        #plt.xlabel(r"$\eta$")
        #plt.legend()
        #plt.savefig(f"{samp}_taus.pdf")
        #print("Tau hist plotted")

    client.shutdown()
    cluster.close()





