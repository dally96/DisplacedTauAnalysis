#!/usr/bin/env python3

"""
2025/04/10
Daniel Ally

Merge script to deal with data for displaced tau analysis
Adapted from script from Tova Holmes who adapted it from Jonathan Long
"""

import sys, os, subprocess
from math import floor
from ROOT import *

mergeDir = "merged"

doMerge = True           # This really doesn't need to be an option, but is here in case I decide to separate data and MC at some point


doFinalMerge = False    # Hadds together all the chunks for one dataset and deletes the partial files
includeEmptyFiles = False   # Hadds histograms from files without trees

#haddTemplateMC = "/afs/cern.ch/user/t/tholmes/LLP/scripts/ntuple_macros/haddTemplate_slimmed_mc_sfs.root"
#haddTemplateData = "/afs/cern.ch/user/t/tholmes/LLP/scripts/ntuple_macros/haddTemplate_slimmed_data_sfs.root"

splitHaddLimit   = 100      # only for data, will split into subhadds of N files
skipExistingMerges = True   # this will skip re-hadding files if their 'tempX.root' file exists, useful if the job was stopped midway, but make sure to remove any partially written files, i.e., the last one
skipMergedFiles = True      # This will check the metadata of the final file and skip individual files that are included in it
                            # WARNING: Make sure you don't have subfiles hanging around from an old job before you use this!

treeName = "Events"
#sampleDir = "/eos/uscms/store/user/dally/first_skim/noLepVeto"
sourceDir = "/eos/uscms/store/group/lpcdisptau/dally/second_skim/all_trig_SR/"
sampleDir = sourceDir

# Create directory for merged output
if not mergeDir in os.listdir(sourceDir): os.mkdir(os.path.join(sourceDir, mergeDir))

if doMerge:

    # Get list of files; assume we're in the directory we rucio downloaded to
    for folder in os.listdir(sampleDir):
        if mergeDir in folder: continue
        fFiles = os.listdir(os.path.join(sampleDir,folder))
        mergeName = mergeDir + "_" + folder
        print(mergeName)

        if not mergeName in os.listdir(os.path.join(sourceDir, mergeDir)): os.mkdir(os.path.join(sourceDir, mergeDir, mergeName))

        print("\nWorking with sample:", folder)
        print("Merging with name:", mergeName)
        print("Found %d files"%len(fFiles))

        # Get some paths
        fullMergeDir = os.path.join(sourceDir, mergeDir, mergeName)
        finalName = "%s.root"%mergeName
        finalPath = "%s/%s"%(fullMergeDir, finalName)
        finalMetadataName = "%s.info"%mergeName
        finalMetadataPath = "%s/%s"%(fullMergeDir, finalMetadataName)

        # Skip files that have already been included in the final merge
        if skipMergedFiles and finalMetadataName in os.listdir(fullMergeDir):
            if not doFinalMerge:
                print("WARNING: Skipping merged files doesn't work if you have old submerges in your directory!")
                #inp = input("Are you sure you want to continue? (y/n) ")
                #if inp != "y": exit()
            print("Found metadata, and will skip already included files.")
            fFilesToUse = []
            fFilesToSkip = []
            with open(finalMetadataPath, "r") as f:
                for line in f: fFilesToSkip.append(line.strip("\n"))
            for f in fFiles:
                if f not in fFilesToSkip: fFilesToUse.append(f)
            print("Merging %d unmerged files."%len(fFilesToUse))
        else: fFilesToUse = fFiles

        if len(fFilesToUse) < 1:
            print("Didn't find any files. Continuing.")
            continue

        # Handle files with no branches separately
        fFilesEmpty = []
        for fFile in fFilesToUse:
            if os.path.getsize(os.path.join(sampleDir, folder, fFile)) == 0:
                fFilesEmpty.append(fFile)
                continue
            f = TFile("%s/%s/%s"%(sampleDir, folder, fFile), "READ")
            try:
                if f.Get(treeName).GetNbranches() == 0:
                    fFilesEmpty.append(fFile)
                    print("Tree with no branches found in file", fFile, "; skipping for now.")
            except:
                print("Couldn't open tree in file", fFile, "; skipping for now.")
                fFilesEmpty.append(fFile)
            f.Close()

        # Split files up into smaller chunks
        fDict = {}
        for i, fFile in enumerate(fFilesToUse):
            if fFile in fFilesEmpty: continue
            index = floor(i/splitHaddLimit)
            if index not in fDict: fDict[index] = []
            fDict[index].append(fFile)

        # Hadd each chunk
        for i in fDict:
            mergePath = "%s/%s_%d.root"%(fullMergeDir, mergeName, i)
            if skipExistingMerges and "%s_%d.root"%(mergeName, i) in os.listdir(fullMergeDir):
                print("Merged file %s already exists. Skipping."%mergePath)
                continue

            command = "hadd %s"%(mergePath)
            for fFile in fDict[i]:
                command += " %s/%s/%s"%(sampleDir, folder, fFile)
            os.system(command)

        # Do final merge
        if doFinalMerge:
            if finalName in os.listdir(fullMergeDir) and not skipMergedFiles:
                print("Merged file already exists. Skipping final merge.")
                continue
            filesToMerge = []
            for i in fDict: filesToMerge.append("%s/%s_%d.root"%(mergeDir, mergeName, i))
            if finalName in os.listdir(fullMergeDir) and skipMergedFiles: filesToMerge.append(finalPath)
            if len(filesToMerge)==1:
                print("Only have one subfile; just renaming instead of hadding.")
                os.rename(filesToMerge[0], finalPath)
            else:
                print("Merging subfiles into final dataset file.")
                command = "hadd %s.tmp"%finalPath
                for name in filesToMerge: command += " %s"%name
                os.system(command)
                for i in fDict:
                    os.remove("%s/%s_%d.root"%(mergeDir, mergeName, i))
                os.rename("%s.tmp"%finalPath, finalPath)

        # Now add in the histograms from any files without trees
        if includeEmptyFiles:
            for fFile in fFilesEmpty:
                print("Adding histograms for file with empty tree,", fFile)
                tmpHists = {}
                f = TFile("%s/%s"%(folder,fFile), "READ")
                for k in f.GetListOfKeys():
                    if k.GetName() == treeName: continue
                    tmpHists[k.GetName()] = f.Get(k.GetName())
                fFinal = TFile(finalPath, "UPDATE")
                for k in tmpHists:
                    h = fFinal.Get(k)
                    h.Add(tmpHists[k])
                fFinal.Write("", TObject.kOverwrite)
                fFinal.Close()

        # Create metadata file
        with open(finalMetadataPath, "w") as f:
            for fname in fFiles: f.write(fname+"\n")
