import pickle
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument(
	"--inputs",
	nargs='*',
	required=True,
	help='Specify the sample you want to process')
parser.add_argument(
	"--output",
	required=True,
	help='Specify the output file')
args = parser.parse_args()

pkl_files = args.inputs

merged_dict = {}
for file in pkl_files:
    with open(file, "rb") as f:
        fileset = pickle.load(f)
        if not isinstance(fileset, dict):
            raise ValueError(f"{file} does not contain a dictionary")
        merged_dict.update(fileset)

# Save the merged dictionary
with open(args.output, "wb") as f:
    pickle.dump(merged_dict, f)

print(f"Merged {len(pkl_files)} files into {args.output}")

