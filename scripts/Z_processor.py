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
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema

#client = Client()

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
      ["DYJetsToLL", 'EWK'],  
      #["WtoLNu2Jets", 'EWK'],
      #["TTtoLNu2Q",  'TT'],
      #["TTto4Q", 'TT'],
      #["TTto2L2Nu", 'TT'],
      ]

class MyProcessor(processor.ProcessorABC):
    def __init__(self):
        pass

    def process(self, events, weights):
        events["Tau"] = events.Tau[(events.Tau.pt > 20) & (abs(events.Tau.eta) < 2.4)]
        print(events.Tau.decayMode.compute())
    
        taus = ak.zip(
            {
                "pt": events.Tau.pt,
                "eta": events.Tau.eta,
                "phi": events.Tau.phi,
                "mass": events.Tau.mass,
                "charge": events.Tau.charge,
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )

        h_mass = hda.hist.Hist(
            hist.axis.Regular(280, 20., 300., name="mass", label=r"$m_{\tau\tau}$ [GeV]")
        )

        cut = (ak.num(taus) == 2) & (ak.sum(taus.charge, axis=1) == 0)
        print(ak.any(cut, axis = -1).compute())
        # add first and second muon in every event together
        ditau = taus[cut][:, 0] + taus[cut][:, 1]

        h_mass.fill(mass=ditau.mass, weight = weights)

        return {
            "DY": {
                "entries": ak.num(events, axis=0),
                "mass": h_mass,
            }
        }

    def postprocess(self, accumulator):
        pass


background_samples = {} 
background_samples["DY"] = []

for samples in SAMP:
    if "DY" in samples[0]:
        background_samples["DY"].append(  ("/eos/uscms/store/user/dally/second_skim_muon_root_pileup_genvtx/Z_peak_" + samples[0] + "/*.root", xsecs[samples[0]] * lumi * 1000 * 1/num_events[samples[0]]))


hist_mass =  hda.hist.Hist(
            hist.axis.Regular(280, 20., 300., name="mass", label=r"$m_{\tau\tau}$ [GeV]")
        ).compute()


for background, samples in background_samples.items(): 
    for sample_file, sample_weight in samples:
        events = NanoEventsFactory.from_root({sample_file:"Events"}, schemaclass= PFNanoAODSchema).events() 
        print(f'Starting {sample_file} histogram')
        processor_instance = MyProcessor()
        output = processor_instance.process(events, sample_weight)
        print(f'{sample_file} finished successfully')

        (computed,) = dask.compute(output)
        print(computed)
        print(computed["DY"]["mass"].show())
        hist_mass += computed["DY"]["mass"]

hist_mass.plot1d()
plt.savefig("ditau_mass.png")

