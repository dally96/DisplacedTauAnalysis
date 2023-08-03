import uproot
import os
import numpy as np 
import awkward as ak

file = "TTToHadronic_TuneCP5_13TeV.root"
sample = uproot.open(file)

eventDict = {
"event" : sample["Events"]["event"].array(),
"lumiBlock" : sample["Events"]["luminosityBlock"].array(),
"run" : sample["Events"]["run"].array()
}

for var in eventDict:
  eventDict[var] = eventDict[var][0:1000]

filename = "TTToHadronic_Events.txt"

with open(filename, 'w') as f:
  f.write("\t".join(eventDict.keys()) + "\n")
  
  for i in range(1000):
    f.write(str(eventDict["event"][i]) + "\t" + str(eventDict["lumiBlock"][i]) + "\t" + str(eventDict["run"][i]) + "\n")

 
