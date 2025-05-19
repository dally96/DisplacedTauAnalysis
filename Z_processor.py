import hist
import dask
import awkward as ak
import hist.dask as hda
import dask_awkward as dak
import matplotlib  as mpl
from  matplotlib import pyplot as plt
from xsec import *

from coffea import processor
from coffea.nanoevents.methods import candidate
from coffea.dataset_tools import (
    apply_to_fileset,
    max_chunks,
    preprocess,
)
from distributed import Client


client = Client()

lumi = 38.01 ##fb-1

SAMP = [
      #['Stau_100_1000mm', 'SIG'],
      #['Stau_100_100mm', 'SIG'],
      #['Stau_100_10mm', 'SIG'],
      #['Stau_100_1mm', 'SIG'],
      #['Stau_200_1000mm', 'SIG'],
      #['Stau_200_100mm', 'SIG'],
      #['Stau_200_10mm', 'SIG'],
      #['Stau_200_1mm', 'SIG'],
      #['Stau_300_1000mm', 'SIG'],
      #['Stau_300_100mm', 'SIG'],
      #['Stau_300_10mm', 'SIG'],
      #['Stau_300_1mm', 'SIG'],
      #['Stau_500_1000mm', 'SIG'],
      #['Stau_500_100mm', 'SIG'],
      #['Stau_500_10mm', 'SIG'],
      #['Stau_500_1mm', 'SIG'],
      #['QCD50_80', 'QCD'],
      #['QCD80_120','QCD'],
      #['QCD120_170','QCD'],
      #['QCD170_300','QCD'],
      #['QCD300_470','QCD'],
      #['QCD470_600','QCD'],
      #['QCD600_800','QCD'],
      #['QCD800_1000','QCD'],
      #['QCD1000_1400','QCD'],
      #['QCD1400_1800','QCD'],
      #['QCD1800_2400','QCD'],
      #['QCD2400_3200','QCD'],
      #['QCD3200','QCD'],
      #["DYJetsToLL", 'EWK'],  
      #["WtoLNu2Jets", 'EWK'],
      ["TTtoLNu2Q",  'TT'],
      ["TTto4Q", 'TT'],
      ["TTto2L2Nu", 'TT'],
      ]

class MyProcessor(processor.ProcessorABC):
    def __init__(self):
        pass

    def process(self, events, weights):
        dataset = events.metadata['dataset']
        taus = ak.zip(
            {
                "pt": events.Tau_pt,
                "eta": events.Tau_eta,
                "phi": events.Tau_phi,
                "mass": events.Tau_mass,
                "charge": events.Tau_charge,
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )

        h_mass = (
            hda.Hist.new
            .StrCat(["opposite"], name="sign")
            .Regular(130, 20., 150., name="mass", label="$m_{\tau\tau}$ [GeV]")
            .Int64()
        )

        cut = (ak.num(taus) == 2) & (ak.sum(taus.charge, axis=1) == 0)
        # add first and second muon in every event together
        ditau = taus[cut][:, 0] + taus[cut][:, 1]
        h_mass.fill(sign="opposite", mass=ditau.mass, weight = weights)

        return {
            dataset: {
                "entries": ak.num(events, axis=0),
                "mass": h_mass,
            }
        }

    def postprocess(self, accumulator):
        pass


background_samples = {} 
background_samples["TT"] = []

for samples in SAMP:
    if "TT" in samples[0]:
        background_samples["TT"].append(  ("/eos/uscms/store/user/dally/second_skim_muon_root/merged/merged_prompt_score" + samples[0] + "/*.root", xsecs[samples[0]] * lumi * 1000 * 1/num_events[samples[0]]))


for background, samples in background_samples.items(): 
    for sample_file, sample_weight in samples:
        events = NanoEventsFactory.from_root({sample_file:"Events"}, schemaclass= PFNanoAODSchema).events() 
        print(f'Starting {sample_file} histogram')
        processor_instance = MyProcessor()
        output = processor_instance.process(events, sample_weight)
        print(f'{sample_file} finished successfully')



