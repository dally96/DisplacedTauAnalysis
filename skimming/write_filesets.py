import os
import subprocess
import pdb, json, argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="")
parser.add_argument(
	"--skim",
	default='',
	required=False,
	choices=['', 'prompt_mutau','mutau'],
	help='If providing list of the skimmed files, select which of the possible skims')
parser.add_argument(
	"--skimversion",
	default='v0',
	required=False,
	help='If listing skimmed files, select which version of the inputs')
args = parser.parse_args()

# directory on EOS with input files
## to replace with v8 once available
BASE_DIRS = [
#  "/store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v10/",
#  "/store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v10_resubmit_v2",
#  "/store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v10_resubmit_v2",
#  "/store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v10/",
#  "/store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v10_data",
#   "/store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/",
#   "/store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/", 
#   "/store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7_v2/",
    "/store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v10_data/",
    "/store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v10_data/"
]
custom_nano_v = 'Summer22_CHS_v10'

XROOTD_PREFIX = "root://cmseos.fnal.gov/"
EOS_LOC = 'root://cmseos.fnal.gov'
outdir = 'samples/' + custom_nano_v + '/'


if args.skim != '':
    skim_folder = args.skim
    skim_version = args.skimversion
    BASE_DIRS = [
      f"/store/group/lpcdisptau/dally/displacedTaus/skim/{custom_nano_v}/{skim_folder}/{skim_version}"
    ]
    
    XROOTD_PREFIX = "root://cmseos.fnal.gov/"
    EOS_LOC = 'root://cmseos.fnal.gov/'
    outdir = f'samples/{custom_nano_v}/{skim_folder}/{skim_version}/'
    
outfolder = Path(f'{outdir}')
if not outfolder.exists():
    outfolder.mkdir(parents=True, exist_ok=True)
    ## create init file to be able to import modules later
    init_file = outfolder / "__init__.py"
    init_file.touch(exist_ok=True)    
    print(f"Folder '{outdir}' created.")

# patterns for grouping different processes
GROUPS = {
    "SMS-TStauStau": f"{outdir}fileset_signal.py",
    "Stau"         : f"{outdir}fileset_signal.py",
    "Wto2Q"        : f"{outdir}fileset_Wto2Q.py",
    "WtoLNu"       : f"{outdir}fileset_WtoLNu.py",
    "QCD_PT"       : f"{outdir}fileset_QCD.py",
    "DYJetsToLL"   : f"{outdir}fileset_DY.py",
    "TTto"         : f"{outdir}fileset_TT.py",
    "T"            : f"{outdir}fileset_singleT.py",  ## more on this later
    "JetMET"       : f"{outdir}fileset_JetMET_2022.py",  ## more on this later
    "Muon"       : f"{outdir}fileset_Muon_2022.py",  ## more on this later
}


def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"Error running command: {cmd}\n{result.stderr}")
        return []
    return result.stdout.strip().split("\n")

def list_dirs(base):
    cmd = f"xrdfs {EOS_LOC} ls {base}"
    return run_cmd(cmd)

def list_root_files(path):
    cmd = f"xrdfs {EOS_LOC} ls {path}"
    files = run_cmd(cmd)
    return [f for f in files if f.endswith(".root")]

def make_key(dirname):
    if args.skim == '':
        if "QCD_PT-" in dirname:
            part = dirname.split("_")[0]+'_'+dirname.split("_")[1]
            return part
        elif "SMS-TStauStau" in dirname:
            part = 'Stau_'+dirname.split('_')[1].split('-')[1]+'_'+dirname.split('_')[2].split('-')[1]
            return part
        elif "Stau_" in dirname or 'JetMET' in dirname or  'Muon' in dirname:
            return dirname
        else:    
            return dirname.split("Tune")[0][:-1]
    return dirname      


import importlib
def count_events(outfile, total_events):
    
    module = importlib.import_module(outfile.replace('/', '.').replace('.py',''))
    all_fileset = module.fileset      

    for isample in all_fileset.keys():
        total_events[isample] = 0
        for ifile in all_fileset[isample]['files'].keys():
            n_ev_this_file = int(ifile.split('/nano_')[1].replace('.root','').split('_')[-1]) - int(ifile.split('/nano_')[1].replace('.root','').split('_')[-2])
            total_events[isample] += n_ev_this_file
    return total_events


def write_filesets(grouped):

    for base in BASE_DIRS:
        print(base)
        dirs = list_dirs(base)
        print(dirs)
        for d in dirs:
            print(d)
            dirname = os.path.basename(d)
            for key, outfile in GROUPS.items():
                match = False
                if key == "T":  ## special case for single tops, as don't want to include TT into them
                    if dirname.startswith(("TB", "Tb", "TW")):
                        match = True
                elif dirname.startswith(key):
                    match = True
                if not match:
                    continue
                if "WtoLNu" in dirname and "ext" in dirname:
                   continue

                sample_key = make_key(dirname)
                rootfiles = list_root_files(d)
                if not rootfiles:
                    continue
                if sample_key not in grouped[key]:
                    grouped[key][sample_key] = {"files": {}}
                for rf in rootfiles:
                    grouped[key][sample_key]["files"][XROOTD_PREFIX + rf] = "Events"

    ## write out into fileset files, as dictionary
    for key, outfile in GROUPS.items():
        if not grouped[key]:
            continue
        with open(outfile, "w") as f:
            f.write("fileset = {\n")
            for sample, content in grouped[key].items():
                f.write(f"    '{sample}': {{\n")
                f.write("        \"files\": {\n")
                for path, events in content["files"].items():
                    f.write(f"            \"{path}\": \"{events}\",\n")
                f.write("        }\n")
                f.write("    },\n")
            f.write("}\n")
        print(f"Wrote {outfile}")



def main():

    grouped = {g: {} for g in GROUPS}
    write_filesets(grouped)

    ## write out number of events processed by the previous step
    if args.skim:
        total_events = {}
        for key, outfile in GROUPS.items():
            if grouped[key]:
                count_events(outfile, total_events)
        with open(f'{outdir}/event_counts.json', "w") as file: 
            json.dump(total_events, file)
        

if __name__ == "__main__":
    main()

