import json
import os, argparse, importlib, pdb, socket

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--inputFile",
    required=True,
    type=str,
    help='Specify the JSON file you want to reformat')

args = parser.parse_args()

if ".json" not in args.inputFile:
    print('Not the right type of file!')
    

filename = args.inputFile
jsonFile = open(filename, 'r')
compactList = json.load(jsonFile)

for run in compactList.keys():
    newLumis = []
    newNewLumis = []
    tmp = []
    for lumi in sorted(compactList[run]):
        if tmp:
            if newNewLumis:
                if lumi <= newNewLumis[-1][1] + 1:
                    newNewLumis[-1][1] = lumi
                    if lumi == sorted(compactList[run])[-1]:
                        newLumis.append(newNewLumis[-1])
                else:
                    newLumis.append(newNewLumis[-1])
                    tmp = [lumi, lumi]
                    newNewLumis.append(tmp)
                    if lumi == sorted(compactList[run])[-1]:
                        newLumis.append(newNewLumis[-1])
        else:
            tmp = [lumi, lumi]
            newNewLumis.append(tmp)
            if lumi == sorted(compactList[run])[-1]:
                newLumis.append(newNewLumis[-1])
    compactList[run] = newLumis
 
with open(f'formatted_{filename}', 'w') as f:
    json.dump(compactList, f)

