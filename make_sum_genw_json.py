import json, pdb, glob, os
import argparse
from pathlib import Path


parser = argparse.ArgumentParser(description="")
parser.add_argument(
	"--nanov",
	choices=['Summer22_CHS_v10', 'Summer22_CHS_v7'],
	default='Summer22_CHS_v10',
	required=False,
	help='Specify the custom nanoaod version to process')
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

args = parser.parse_args()

processed_json_folder = f'/eos/cms/store/user/fiorendi/displacedTaus/skim/{args.nanov}/{args.skim}/{args.skimversion}/'

outdir = Path(f'plots_config/{args.nanov}')
if not outdir.exists():
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Folder '{outdir}' created.")

final_results = {}
final_number_gen = {}

## find the JSON files with processed lumis in the given directory
subsamples_list = []
for filepath in glob.glob(os.path.join(processed_json_folder, "processed*.json")):
    filename = os.path.basename(filepath)  
    if "_" in filename:
        subsample = filename.split("_", 2)[2].replace(".json", "")
        subsamples_list.append(subsample)

## per each sample, calculate the sum of gen weights of processed events by getting the intersection with the 
## general dict of sumw previously created with the get_mc_gen_weights.py script
for isample in subsamples_list:
    with open(f'{processed_json_folder}processed_lumis_{isample}.json') as processed_file:
        processed_dict = json.load(processed_file)
        keys_list = list(processed_dict.keys()) ## should be a one element list
        if len(subsamples_list) > 1 :
            print ('len(subsamples_list) = ', len(subsamples_list), ' should have been one. \nPlease check!')
            
        isubsample = keys_list[0]

        total_sum_w = 0
        total_sum_n = 0 ## can be used as a cross check
        track_summed_genw = set() ## faster than a list

        ls_sumw_filename = f'ls_sumw_dict_{isubsample}.json'
        with open(f'samples/{args.nanov}/processed_LS_from_crab/{ls_sumw_filename}') as sumw_file:
            sumgenw_dict = json.load(sumw_file)
            ## this one above is a list of dictionaries with keys lumisections, sumgenw, ngen
          
            pdb.set_trace()
            processed_lumis = set(processed_dict[isubsample][1])  # make a set for O(1) lookups
        
            for i, ibunch in enumerate(sumgenw_dict):
                ## do not sum if this lumiblock was already accounted before
                if i in track_summed_genw:
                    continue
                # check intersection and, in case, sum 
                if processed_lumis.intersection(ibunch['lumisections']):
                    total_sum_w += ibunch['sumgenw']
                    total_sum_n += ibunch['ngen']
                    track_summed_genw.add(i)
        
        final_results[isubsample] = total_sum_w
        final_number_gen[isubsample] = total_sum_n
        print(f"{isubsample} : total_sum_w = {total_sum_w}")

## dump to a single JSON file
with open(f"{outdir}/all_total_sumw.json", "w") as outfile:
    json.dump(final_results, outfile, indent=2)

with open(f"{outdir}/all_total_event_number.json", "w") as outfile:
    json.dump(final_number_gen, outfile, indent=2)

print(f"Results written to {outdir}/all_total_sumw.json")
print(f"Results written to {outdir}/all_total_event_number.json")