import numpy as np 
import math
import scipy
import array

import awkward as ak 
import uproot
import dask
import dask_awkward as dak
from coffea import processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, PFNanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection
from coffea.dataset_tools import (
    apply_to_fileset,
    max_chunks,
    preprocess,
)

import warnings
warnings.filterwarnings("ignore")

from skimmed_fileset import *

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-m"    , "--muon"    , dest = "muon"   , help = "Leading muon variable"    , default = "pt")
parser.add_argument("-j"    , "--jet"     , dest = "jet"    , help = "Leading jet variable"     , default = "pt")         

leading_var = parser.parse_args()

selections = {
              "muon_pt":                    30, ##GeV
              "muon_ID":                    "muon_tightId",
              "muon_dxy_prompt_max":        50E-4, ##cm
              "muon_dxy_prompt_min":        0E-4, ##cm
              "muon_dxy_displaced_min":     0.0, ##cm
              "muon_dxy_displaced_max":     10, ##cm
              "muon_iso_max":               0.19,

              "jet_score":                  0.9, 
              "jet_pt":                     32, ##GeV

              "MET_pt":                     105, ##GeV
             }

class skimProcessor(processor.ProcessorABC):
    def __init__(self, leading_muon_var, leading_jet_var):
        self.leading_muon_var = leading_var.muon
        self.leading_jet_var  = leading_var.jet

    def process(self, events):
        
        leading_muon_var = self.leading_muon_var
        leading_jet_var = self.leading_jet_var
        
        events["Stau"] = events.GenPart[(abs(events.GenPart.pdgId) == 1000015) &\
                                        (events.GenPart.hasFlags("isLastCopy"))]

        events["StauTau"] = events.Stau.distinctChildren[(abs(events.Stau.distinctChildren.pdgId) == 15) &\
                                                         (events.Stau.distinctChildren.hasFlags("isLastCopy"))]

        events["StauTau"] = ak.firsts(events.StauTau[ak.argsort(events.StauTau.pt, ascending=False)], axis = 2) 

        events["DisMuon"] = events.DisMuon[ak.argsort(events["DisMuon"][leading_muon_var], ascending=False, axis = 1)]
        events["Jet"] = events.Jet[ak.argsort(events["Jet"][leading_jet_var], ascending=False, axis = 1)]

        events["DisMuon"] = ak.singletons(ak.firsts(events.DisMuon))
        events["Jet"] = ak.singletons(ak.firsts(events.Jet))

        good_muons  = dak.flatten((events.DisMuon.pt > selections["muon_pt"])           &\
                       (events.DisMuon.tightId == 1)                                    &\
                       (abs(events.DisMuon.dxy) > selections["muon_dxy_displaced_min"]) &\
                       (abs(events.DisMuon.dxy) < selections["muon_dxy_displaced_max"]) &\
                       (events.DisMuon.pfRelIso03_all < selections["muon_iso_max"])
                      )

        good_jets   = dak.flatten((events.Jet.disTauTag_score1 > selections["jet_score"])   &\
                       (events.Jet.pt > selections["jet_pt"])
                      )

        good_events = (events.MET.pt > selections["MET_pt"])

        events = events[good_muons & good_jets & good_events]

        meta = ak.Array([0], backend = "typetracer")
        event_counts = events.map_partitions(lambda part: ak.num(part, axis = 0), meta = meta)
        partition_counts = event_counts.compute()
        non_empty_partitions = [
                                events.partitions[i] for i in range(len(partition_counts)) if partition_counts[i] > 0
                               ]
        if non_empty_partitions:
            events = dak.concatenate(non_empty_partitions) 
        else:
            with uproot.recreate("skimmed_muon_"+ events.metadata["dataset"] + "/part0.root") as fout:
                fout["Events"] = events.compute()
            return fout

        out_dict = {}
        muon_vars =  [ 
                     "pt",
                     "eta",
                     "phi",
                     "charge",
                     "dxy",
                     "dxyErr",
                     "dz",
                     "dzErr",
                     "looseId",
                     "mediumId",
                     "tightId",
                     "pfRelIso03_all",
                     "pfRelIso03_chg",
                     "pfRelIso04_all",
                     ]

        jet_vars =   [
                     "pt",
                     "eta",
                     "phi",
                     "disTauTag_score1",
                     ]

        gpart_vars = [
                     "genPartIdxMother", 
                     "statusFlags", 
                     "pdgId",
                     "status", 
                     "eta", 
                     "mass", 
                     "phi", 
                     "pt", 
                     "vertexR", 
                     "vertexRho", 
                     "vertexX", 
                     "vertexY", 
                     "vertexZ",
                     ]

        gvist_vars = [
                     "genPartIdxMother", 
                     "charge",
                     "status", 
                     "eta", 
                     "mass", 
                     "phi", 
                     "pt", 
                     ]

        tau_vars   = events.Tau.fields                        
        MET_vars   = events.MET.fields  
        JetPF_vars = events.JetPFCands.fields
        
        for branch in muon_vars:
            out_dict["DisMuon_"   + branch]  = ak.drop_none(events["DisMuon"][branch])
        for branch in jet_vars:
            out_dict["Jet_"       + branch]  = ak.drop_none(events["Jet"][branch])
        for branch in gpart_vars:
            out_dict["GenPart_"   + branch]  = ak.drop_none(events["GenPart"][branch])
        for branch in gpart_vars:
            out_dict["Stau_"      + branch]  = ak.drop_none(events["Stau"][branch])
        for branch in gpart_vars:
            out_dict["StauTau_"   + branch]  = ak.drop_none(events["StauTau"][branch])
        for branch in gvist_vars:
            out_dict["GenVisTau_" + branch]  = ak.drop_none(events["GenVisTau"][branch])
        for branch in tau_vars:
            out_dict["Tau_"       + branch]  = ak.drop_none(events["Tau"][branch])
        for branch in MET_vars: 
            out_dict["MET_"       + branch]  = ak.drop_none(events["MET"][branch])    
        for branch in JetPF_vars:
            out_dict["JetPFCands_" + branch] = ak.drop_none(events["JetPFCands"][branch])

        out_dict["event"]           = ak.drop_none(events.event)
        out_dict["run"]             = ak.drop_none(events.run)
        out_dict["luminosityBlock"] = ak.drop_none(events.luminosityBlock)
        out_dict["nDisMuon"]        = dak.num(ak.drop_none(events.DisMuon))
        out_dict["nJet"]            = dak.num(ak.drop_none(events.Jet))
        out_dict["nGenPart"]        = dak.num(ak.drop_none(events.GenPart))
        out_dict["nGenVisTau"]      = dak.num(ak.drop_none(events.GenVisTau))
        out_dict["nTau"]            = dak.num(ak.drop_none(events.Tau))
        try:
            out_dict = dak.zip(out_dict, depth_limit = 1)

            print(f"The filename should be skimmed_muon_"+events.metadata["dataset"]+"+_root")
            return uproot.dask_write(out_dict, "skimmed_muon_" + events.metadata["dataset"] + "_root", tree_name="Events")

        except Exception as e:
            print(f"Error processing {events.metadata['dataset']}: {e}")        


    def postprocess(self, accumulator):
        pass

dataset_runnable, dataset_updated = preprocess(
    skimmed_fileset,
    align_clusters=False,
    step_size=100_000_000,
    files_per_batch=1,
    skip_bad_files=True,
    save_form=False,
)
to_compute = apply_to_fileset(
            skimProcessor(leading_var.muon, leading_var.jet),
             max_chunks(dataset_runnable, 10000000),
             schemaclass=PFNanoAODSchema
)
(out,) = dask.compute(to_compute)
print(out)
    
