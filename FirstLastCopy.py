import uproot 
import awkward as ak
import scipy
import math
from matplotlib import pyplot as plt 
import matplotlib as mpl 
import pandas as pd
import numpy as np
from array import array

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{textgreek}')

files = ["SUS-RunIISummer20UL18GEN-stau100_lsp1_ctau100mm_v6_with-disTauTagScore.root",
        ]

file = uproot.open(files[0])
Events = file["Events"]

GenPart = Events.arrays(filter_name="GenPart*")

isLastCopy  = (GenPart["GenPart_statusFlags"] & (2 ** 13) == (2 ** 13))

GenPart_isLastCopy  = GenPart[isLastCopy]

def isFirstCopy(evt, motherIdx, pdgId):
  if ((GenPart["GenPart_statusFlags"][evt][motherIdx] & (2 ** 12) == (2 ** 12)) and 
     (GenPart["GenPart_pdgId"][evt][motherIdx] == pdgId)):
    return motherIdx
  if (GenPart["GenPart_pdgId"][evt][motherIdx] != pdgId):
    return -1 
  else:
    return isFirstCopy(evt, GenPart["GenPart_genPartIdxMother"][evt][motherIdx], pdgId)

originalParent = []

for event in range(Events.num_entries):

  originalParent_evt = []
  if len(GenPart_isLastCopy["GenPart_genPartIdxMother"][event]) == 0:
    continue
  for partIdx in range(len(GenPart_isLastCopy["GenPart_genPartIdxMother"][event])):
    motherIdx = GenPart_isLastCopy["GenPart_genPartIdxMother"][event][partIdx]
    pdgId = GenPart_isLastCopy["GenPart_pdgId"][event][partIdx]
    originalParent_evt.append(isFirstCopy(event, motherIdx, pdgId))
  
  originalParent.append(originalParent_evt)

GenPart_res = {}
for key in ak.fields(GenPart):
  GenPart_res[key] = []

for event in range(len(originalParent)):
  if len(originalParent[event]) == 0: continue

  for partIdx in range(len(originalParent[event])):
    if originalParent[event][partIdx] == -1: continue
    for key in ak.fields(GenPart):
      GenPart_res[key].append(GenPart_isLastCopy[key][event][partIdx] - GenPart[key][event][originalParent[event][partIdx]])



##          Input variable            bins                        xlabel            ylabel
histDict = {
           "pt"               : [np.linspace(-100, 100, 51),    r"\textDelta p$_{t}$ [GeV]",         "A.U."        ], 
           "eta"              : [np.linspace(-5, 5, 51),        r"\textDelta \texteta",         "A.U."        ],
           "phi"              : [np.linspace(-6.4, 6.4, 65),    r"\textDelta \textphi",         "A.U."        ],
           "vertexX"          : [np.linspace(-0.1, 0.1, 21),        r"\textDelta vtx [cm]",         "A.U."        ],
           "vertexY"          : [np.linspace(-0.1, 0.1, 21),        r"\textDelta vty [cm]",         "A.U."        ],
           "vertexZ"          : [np.linspace(-0.1, 0.1, 21),        r"\textDelta vtz [cm]",         "A.U."        ], 
           "vertexR"          : [np.linspace(-0.1, 0.1, 21),        r"\textDelta vtR [cm]",         "A.U."        ],
           "vertexRho"        : [np.linspace(-0.1, 0.1, 21),        r"\textDelta vtRho [cm]",       "A.U."        ], 
           }


for branch in histDict:
  plt.cla()
  plt.clf()
  y, bins, other = plt.hist(GenPart_res["GenPart_"+branch], bins=histDict[branch.split("_")[-1]][0], weights=[1/len(GenPart_res["GenPart_"+branch]),]*len(GenPart_res["GenPart_"+branch]), histtype = 'step', label=files[0] + "i, RMS = " + "{:.1f}".format(np.std(GenPart_res["GenPart_"+branch])))
  bincenters = 0.5*(bins[1:]+bins[:-1])
  yerr = np.sqrt(y * len(GenPart_res["GenPart_"+branch]))/len(GenPart_res["GenPart_"+branch])
  
  plt.errorbar(bincenters, y, yerr = yerr, ls = 'none')
  plt.title(branch.split("_")[-1] + " difference between isFirstCopy and isLastCopy"); plt.xlabel(histDict[branch.split("_")[-1]][1]); plt.ylabel(histDict[branch.split("_")[-1]][2])
  #if "pt" in branch or "Frac" in branch or "Weight" in branch or "Error" in branch or "mass" in branch or "dxy" in branch or "dz" in branch:
  plt.yscale("log")
  legend = plt.legend(fontsize="6.5")
  legend.get_frame().set_alpha(0)
  plt.show()
  plt.savefig("FirstLastRes/Last-First_" + branch + ".png")

