import uproot, importlib
import awkward as ak
import json, pdb
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

'''
Read a list of input files corresponding to the processed data (custom NANO),
and creates a JSON file with all the processed Run:LS, reading from the LuminosityBlock tree.
'''
parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--sample",
    choices=['JetMET'],
    required=True,
    help="Specify the sample you want to process")
parser.add_argument(
    "--year",
    choices=['2022'],
    required=True,
    help="Specify the data taking year")
parser.add_argument(
    "--era",
    nargs="*",
    default="all",
    choices=['E', 'F', 'G'],
    required=False,
    help="Specify the exact era you want to process")
parser.add_argument(
    "--nfiles",
    type=int,
    default=-1,
    required=False,
    help="Specify the number of input files to process")
parser.add_argument(
	"--nanov",
	choices=['Summer22_CHS_v9', 'Summer22_CHS_v7'],
	default='Summer22_CHS_v7',
	required=False,
	help='Specify the custom nanoaod version to process')
args = parser.parse_args()
custom_nano_v = args.nanov + '/'
custom_nano_v_p = args.nanov + '.'


samples = {
    "JetMET_2022": f"samples.{custom_nano_v_p}fileset_JetMET_2022",
}

the_sample = args.sample
module = importlib.import_module(samples[the_sample+'_'+args.year])
input_dataset = module.fileset  

if args.era == "all":
    fileset = input_dataset
else:
    for k in args.era:
        era_str = args.sample+'_Run'+args.year+k
        fileset = {era_str: input_dataset[era_str]}

def find_consecutive_ls(lumis):
    nums = sorted(set(lumis))
    ranges = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
        else:
            ranges.append([start, prev])
            start = prev = n
    ranges.append([start, prev])
    return sorted(ranges)


def process_file(ifile):
    """Read luminosity + gen info from a single ROOT file"""
    try:
        with uproot.open(ifile) as file:
            lumis = file["LuminosityBlocks/luminosityBlock"].array(library="np")
            runs = file["LuminosityBlocks/run"].array(library="np")
        return {"lumisections": lumis.tolist(),
                "runs": runs.tolist()
                }
    except Exception as e:
        print(f"Error processing {ifile}: {e}")
        return None

for idataset, dataset_info in fileset.items():
    files = dataset_info["files"]
    if args.nfiles > 0:
        files = dict(list(files.items())[:args.nfiles]) 
#         files = files[:args.nfiles]
    print(f"Dataset {idataset}, n files: {len(files)}")

    results = []
    ## this to run as a sequential loop (debug)
#     for idx, file in enumerate(files):
#         result = process_file(file)
#         if idx % 100 == 0:
#             print("Processing file", idx)
#         if result:
#             results.append(result)

    ## this to run faster
    with ThreadPoolExecutor() as executor:
        for idx, result in enumerate(executor.map(process_file, files)):
            if idx % 100 == 0:
                print("Processing file", idx)
            if result:
                results.append(result)


    ## now transform to have all processed lumisections per run
    run_dict = defaultdict(list)
    for entry in results:
        for lumi, run in zip(entry["lumisections"], entry["runs"]):
            run_dict[run].append(lumi)
            run_dict[run] = sorted(run_dict[run])
    
    ## find possible contiguous ranges of LSs
    final_dict = {run: find_consecutive_ls(lumis) for run, lumis in run_dict.items()}
    ## dump into json file
    out_path = f"samples/{custom_nano_v}processed_LS_from_crab/run_LS_dict_{the_sample}_{idataset}.json"
    with open(out_path, "w") as fp:
        json.dump(final_dict, fp)
    print(f"{out_path} written")
