import os
import subprocess
import pdb

# directory on EOS with input files
## to replace with v8 once available
BASE_DIRS = [
  "/store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/",
  "/store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/",
]

XROOTD_PREFIX = "root://cmsxrootd.fnal.gov/"
# outdir = 'samples/'
outdir = ''

# patterns for grouping different processes
GROUPS = {
    "SMS-TStauStau": f"{outdir}fileset_signal.py",
    "Wto2Q"        : f"{outdir}fileset_WTo2Q.py",
    "WtoLNu"       : f"{outdir}fileset_WToLNu.py",
    "QCD_PT"       : f"{outdir}fileset_QCD.py",
    "DYJetsToLL"   : f"{outdir}fileset_DY.py",
    "TTto"         : f"{outdir}fileset_TT.py",
}

def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"Error running command: {cmd}\n{result.stderr}")
        return []
    return result.stdout.strip().split("\n")

def list_dirs(base):
    cmd = f"xrdfs root://cmseos.fnal.gov ls {base}"
    return run_cmd(cmd)

def list_root_files(path):
    cmd = f"xrdfs root://cmseos.fnal.gov ls {path}"
    files = run_cmd(cmd)
    return [f for f in files if f.endswith(".root")]

def make_key(dirname):
    if "QCD_PT-" in dirname:
        part = dirname.split("_")[0]+'_'+dirname.split("_")[1]
        return part
    elif "SMS-TStauStau" in dirname:
        part = 'Stau_'+dirname.split('_')[1].split('-')[1]+'_'+dirname.split('_')[2].split('-')[1]
        return part
    elif "Wto" in dirname or "TTto" in dirname:
        return dirname.split("Tune")[0][:-1]
    else:
        return dirname



def main():

    grouped = {g: {} for g in GROUPS}

    for base in BASE_DIRS:
        dirs = list_dirs(base)
        for d in dirs:
            dirname = os.path.basename(d)
            for key, outfile in GROUPS.items():
                if dirname.startswith(key):
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

if __name__ == "__main__":
    main()

