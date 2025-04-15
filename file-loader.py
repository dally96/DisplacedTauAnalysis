with open("root_files.txt") as f:
    lines = [line.strip() for line in f if line.strip()]

xrootd_prefix    = 'root://cmseos.fnal.gov/'
base_prefix_full = '/eos/uscms/store/user/dally/DisplacedTauAnalysis/skimmed_muon_'
base_prefix      = '/eos/uscms'
base_suffix      = '_root:'

paths = []
sets  = []

i = 0
while i < len(lines):
    if '.root' in lines[i]:
        i += 1
        continue

    # Base path line
    base_path = lines[i]
    dataset_name = base_path.removeprefix(base_prefix_full).removesuffix(base_suffix)
    sets.append(dataset_name)
    xrootd_path = base_path.removeprefix(base_prefix).removesuffix(':')

    # Look ahead for .root file
    if i + 1 < len(lines) and '.root' in lines[i + 1]:
        root_file = lines[i + 1]
        paths.append(xrootd_prefix + xrootd_path + root_file)
        i += 2  # Move past both lines
    else:
        i += 1  # Only move past base path

print(paths)
print(sets)
