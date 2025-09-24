import json, pdb
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument(
	"--nanov",
	choices=['Summer22_CHS_v9', 'Summer22_CHS_v7'],
	default='Summer22_CHS_v9',
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

samples_list = ['DY', 'QCD', 'WtoLNu', 'Wto2Q', 'TT', 'singleT']
 
processed_json_folder = f'/eos/cms/store/user/fiorendi/displacedTaus/skim/{args.nanov}/{args.skim}/{args.skimversion}/'
# processed_json_folder = ''

final_results = {}
final_number_gen = {}


for isample in samples_list:

    with open(f'{processed_json_folder}processed_lumis_{isample}_a_l_l.json') as processed_file:
        processed_dict = json.load(processed_file)
        subsamples_list = list(processed_dict.keys())
        
        for isubsample in subsamples_list:

            total_sum_w = 0
            total_sum_n = 0 ## can be used as a cross check
            track_summed_genw = set() ## faster than a list

            ls_sumw_filename = f'ls_sumw_dict_{isample}.json'
            if 'QCD' in isubsample or 'Wto' in isubsample or 'TT' in isubsample:
                ls_sumw_filename = f'ls_sumw_dict_{isample}_{isubsample}.json'
            with open(f'samples/{args.nanov}/processed_LS_from_crab/{ls_sumw_filename}') as sumw_file:
                sumgenw_dict = json.load(sumw_file)
                ## this one above is a list of dictionaries with keys lumisections, sumgenw, ngen
              
                processed_lumis = set(processed_dict[isubsample]['1'])  # make a set for O(1) lookups
            
                for i, ibunch in enumerate(sumgenw_dict):
                    ## do not sum if this lumiblock was already accounted before
                    if i in track_summed_genw:
                        continue
                    # check intersection and, in case, sum 
                    if processed_lumis.intersection(ibunch['lumisections']):
                        total_sum_w += ibunch['sumgenw']
                        total_sum_n += ibunch['ngen']
                        track_summed_genw.add(i)
        
            if isample not in final_results:
                final_results[isample] = {}
            final_results[isample][isubsample] = total_sum_w

            if isample not in final_number_gen:
                final_number_gen[isample] = {}
            final_number_gen[isample][isubsample] = total_sum_n
                          
            print(f"{isample} {isubsample}: total_sum_w = {total_sum_w}")

## dump to a single JSON file
with open("plots_config/all_total_sumw.json", "w") as outfile:
    json.dump(final_results, outfile, indent=2)

with open("plots_config/all_total_event_number.json", "w") as outfile:
    json.dump(final_number_gen, outfile, indent=2)

print("Results written to plots_config/all_total_sumw.json")
print("Results written to plots_config/all_total_event_number.json")