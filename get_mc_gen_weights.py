import uproot, importlib
import awkward as ak
import json
import argparse
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--sample",
    choices=['QCD', 'DY', 'signal', 'WtoLNu', 'Wto2Q', 'TT'],
    required=True,
    help="Specify the sample you want to process")
parser.add_argument(
    "--subsample",
    nargs="*",
    default="all",
    required=False,
    help="Specify the exact sample you want to process")
parser.add_argument(
    "--nfiles",
    type=int,
    default=-1,
    required=False,
    help="Specify the number of input files to process")
parser.add_argument(
	"--nanov",
	choices=['Summer22_CHS_v9', 'Summer22_CHS_v7'],
	default='Summer22_CHS_v9',
	required=False,
	help='Specify the custom nanoaod version to process')
args = parser.parse_args()
custom_nano_v = args.nanov + '/'
custom_nano_v_p = args.nanov + '.'

# def remap_keys(mapping):
#     return [{'lumisections': k, 'sumgenw': v[0], 'ngen': v[1]} for k, v in mapping.items()]

samples = {
    "Wto2Q": f"samples.{custom_nano_v_p}fileset_Wto2Q",
    "WtoLNu": f"samples.{custom_nano_v_p}fileset_WtoLNu",
    "QCD": f"samples.{custom_nano_v_p}fileset_QCD",
    "DY": f"samples.{custom_nano_v_p}fileset_DY",
    "signal": f"samples.{custom_nano_v_p}fileset_signal",
    "TT": f"samples.{custom_nano_v_p}fileset_TT",
    "singleT": f"samples.{custom_nano_v_p}fileset_singleT",
}

the_sample = args.sample
module = importlib.import_module(samples[the_sample])
input_dataset = module.fileset  

if args.subsample == "all":
    fileset = input_dataset
else:
    fileset = {k: input_dataset[k] for k in args.subsample}

def process_file(ifile):
    """Read luminosity + gen info from a single ROOT file"""
    try:
        with uproot.open(ifile) as file:
            lumis = file["LuminosityBlocks/luminosityBlock"].array(library="np")
            runs = file["Runs"]
            sumGenW = runs["genEventSumw"].array(library="np")[0]
            sumGenN = runs["genEventCount"].array(library="np")[0]
        # Store lumis as a list (JSON-friendly)
        return {"lumisections": lumis.tolist(),
                "sumgenw": float(sumGenW),
                "ngen": int(sumGenN)}
    except Exception as e:
        print(f"Error processing {ifile}: {e}")
        return None

for idataset, dataset_info in fileset.items():
    files = dataset_info["files"]
    if args.nfiles > 0:
        files = files[:args.nfiles]
    print(f"Dataset {idataset}, n files: {len(files)}")

    results = []
    # ThreadPoolExecutor is usually better for uproot (I/O bound)
    with ThreadPoolExecutor() as executor:
        for idx, result in enumerate(executor.map(process_file, files)):
            if idx % 100 == 0:
                print("Processing file", idx)
            if result:
                results.append(result)

    out_path = f"samples/{custom_nano_v}processed_LS_from_crab/ls_sumw_dict_{the_sample}_{idataset}.json"
    with open(out_path, "w") as fp:
        json.dump(results, fp)

    print(f"{out_path} written")
