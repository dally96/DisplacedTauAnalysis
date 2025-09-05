import uproot, importlib
import awkward as ak
import pdb
import json, argparse

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

import json
def remap_keys(mapping):
    return [{'lumisections':k, 'sumgenw': v[0], 'ngen': v[1]} for k, v in mapping.items()]


samples = {
    "Wto2Q": "samples.fileset_WTo2Q",
    "WtoLNu": "samples.fileset_WToLNu",
    "QCD": "samples.fileset_QCD",
    "DY": "samples.fileset_DY",
    "signal": "samples.fileset_signal",
    "TT": "samples.fileset_TT",
}

the_sample = args.sample
module = importlib.import_module(samples[the_sample])
input_dataset = module.fileset  

if args.subsample == 'all':
    fileset = input_dataset
else:  
    fileset = {k: input_dataset[k] for k in args.subsample}
  
ls_weight_dict = {}
for idataset in fileset.keys():
    files = fileset[idataset]['files']
    print ('n files:', len(files))
    for i,ifile in enumerate(files): 
#         if i > 2:  break   
        if (i % 10 == 0):  print ('processing file', i)
        file = uproot.open(ifile)
        lumiTree = file['LuminosityBlocks']
        lumis = lumiTree["luminosityBlock"].array().to_list()
    
        genTree = file['Runs']
        sumGenW = genTree['genEventSumw'].array().to_list()[0]
        sumGenN = genTree['genEventCount'].array().to_list()[0]
        ls_weight_dict[tuple(lumis)] = [sumGenW,sumGenN]

    with open(f'samples/ls_sumw_dict_{the_sample}_{idataset}.json', 'w') as fp:  
        json.dump(remap_keys(ls_weight_dict), fp) 