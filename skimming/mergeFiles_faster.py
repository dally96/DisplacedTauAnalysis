#!/usr/bin/env python3

"""
2025/04/10
Daniel Ally

Merge script to deal with data for displaced tau analysis
Adapted from script from Tova Holmes who adapted it from Jonathan Long
"""

import os
import subprocess
from math import floor
from concurrent.futures import ThreadPoolExecutor
from ROOT import TFile
import pdb

mergeDir = "faster_trial"
doMerge = True
doFinalMerge = True
includeEmptyFiles = False
splitHaddLimit = 200
skipExistingMerges = True
skipMergedFiles = True
treeName = "Events"

# Source directory
# sourceDir = "/eos/cms/store/user/fiorendi/displacedTaus/skim/Summer22_CHS_v7/mutau/daniel/selected/"
sourceDir = "/eos/cms/store/user/fiorendi/displacedTaus/skim/Summer22_CHS_v7/mutau/v_sync/selected/QCD_PT-800to1000/"
sampleDir = sourceDir

## create merged directory if missing
os.makedirs(os.path.join(sourceDir, mergeDir), exist_ok=True)


def safe_run(cmd):
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error running: {cmd}")

def hadd_files(output, files):
    """Run hadd for a list of files into output."""
    if not files:
        return
    cmd = "hadd -f {} {}".format(output, " ".join(files))
    safe_run(cmd)


if doMerge:
    for folder in os.listdir(sampleDir):
#         pdb.set_trace()
        if mergeDir in folder:
            continue
        folderPath = os.path.join(sampleDir, folder)
        if not os.path.isdir(folderPath):
            continue

        allFiles = os.listdir(folderPath)
        rootFiles = [f for f in allFiles if f.endswith(".root")]

        mergeName = mergeDir + "_" + folder
        fullMergeDir = os.path.join(sourceDir, mergeDir, mergeName)
        os.makedirs(fullMergeDir, exist_ok=True)

        finalName = f"{mergeName}.root"
        finalPath = os.path.join(fullMergeDir, finalName)
        finalMetadataPath = os.path.join(fullMergeDir, f"{mergeName}.info")

        # Skip files that have already been included in the final merge
        if skipMergedFiles and os.path.exists(finalMetadataPath):
            with open(finalMetadataPath) as f:
                merged_set = set(f.read().splitlines())
            rootFiles = list(set(rootFiles) - merged_set)
            print(f" {folder}: Skipping {len(merged_set)} already merged files.")

        if not rootFiles:
            print(f"  {folder}: No new files to merge.")
            continue

        ## Handle files with no branches separately
        emptyFiles, goodFiles = [], []
        for fname in rootFiles:
            fpath = os.path.join(folderPath, fname)
            if os.path.getsize(fpath) == 0:
                emptyFiles.append(fname)
                continue
            f = TFile.Open(fpath)
            if not f or not f.Get(treeName) or f.Get(treeName).GetNbranches() == 0:
                emptyFiles.append(fname)
            else:
                goodFiles.append(fname)
            f.Close()

        ## Split files up into smaller chunks
        fDict = {}
        for i, fFile in enumerate(goodFiles):
            idx = floor(i / splitHaddLimit)
            fDict.setdefault(idx, []).append(os.path.join(folderPath, fFile))

        ## hadd chunks
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for idx, files in fDict.items():
                outFile = os.path.join(fullMergeDir, f"{mergeName}_{idx}.root")
                if skipExistingMerges and os.path.exists(outFile):
                    print(f"  Skipping existing {outFile}")
                    continue
                futures.append(executor.submit(hadd_files, outFile, files))
            for fut in futures:
                fut.result()

        ## Final merge
        if doFinalMerge:
            subfiles = [os.path.join(fullMergeDir, f"{mergeName}_{i}.root") for i in fDict]
            if len(subfiles) == 1:
                os.rename(subfiles[0], finalPath)
            elif subfiles:
                hadd_files(f"{finalPath}.tmp", subfiles)
                for sf in subfiles:
                    os.remove(sf)
                os.rename(f"{finalPath}.tmp", finalPath)

        ## Create metadata file
        with open(finalMetadataPath, "w") as f:
            for fname in allFiles:
                if fname.endswith(".root"):
                    f.write(fname + "\n")

        print(f"  Finished merging {folder} -> {finalPath}")

